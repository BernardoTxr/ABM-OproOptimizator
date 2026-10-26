import datetime
import unittest

import numpy as np
import pandas as pd

from lob_sim.loss.facts import acf_abs_returns
from lob_sim.loss.real_facts import (
    _assemble_facts,
    _filter_measurement_window,
    _reconstruction_input,
    _sample_frame,
    resolve_date_split,
    resolve_measurement_windows,
    select_window_assignments,
)


class RealFactWindowTests(unittest.TestCase):
    def setUp(self):
        self.start = datetime.time(14, 0)
        self.end = datetime.time(15, 0)

    def test_measurement_window_requires_exact_date(self):
        frame = pd.DataFrame({
            "dat_tim": [
                "2025-10-13 14:30:00",
                "2025-10-14 13:59:59",
                "2025-10-14 14:00:00",
                "2025-10-14 14:59:59",
                "2025-10-14 15:00:00",
            ]
        })
        result = _filter_measurement_window(frame, "20251014", self.start, self.end)
        self.assertEqual(result.index.tolist(), [2, 3])

    def test_reconstruction_keeps_full_bootstrap_and_target_updates(self):
        frame = pd.DataFrame({
            "dat_tim": [
                "2025-10-12 09:00:00",
                "2025-10-13 18:00:00",
                "2025-10-14 13:00:00",
                "2025-10-14 14:59:59",
                "2025-10-14 15:00:00",
                "2025-10-13 14:30:00",
            ],
            "msg_tpe": [
                "snapshot", "snapshot", "incremental", "incremental",
                "incremental", "incremental",
            ],
        })
        result = _reconstruction_input(frame, "20251014", self.end)
        self.assertEqual(result.index.tolist(), [0, 1, 2, 3])

    def test_second_sampling_uses_previous_state_and_exact_duration(self):
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2025-10-14 13:59:59.500Z",
                "2025-10-14 14:00:01.200Z",
            ]),
            "midprice": [10.0, 11.0],
        })
        sampled = _sample_frame(
            frame, "20251014", self.start, datetime.time(14, 0, 3)
        )
        self.assertEqual(sampled["elapsed_seconds"].tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(sampled["midprice"].tolist(), [10.0, 11.0, 11.0])

    def test_second_sampling_refuses_post_boundary_backfill(self):
        frame = pd.DataFrame({
            "timestamp": pd.to_datetime(["2025-10-14 14:00:01.200Z"]),
            "midprice": [11.0],
        })
        with self.assertRaisesRegex(ValueError, "before"):
            _sample_frame(
                frame, "20251014", self.start, datetime.time(14, 0, 3)
            )

    def test_acf_is_finite_for_sparse_constant_returns(self):
        result = acf_abs_returns(np.array([0.0, 0.0, 0.0, 1.0]))
        self.assertTrue(np.isfinite(result).all())

    def test_real_coverage_is_assembled_from_measured_windows(self):
        facts = _assemble_facts(
            [], [], [], [], [], [], [], [1.0, 0.75]
        )
        self.assertEqual(facts["book_coverage"].tolist(), [1.0, 0.75])

    def test_pooled_facts_preserve_event_mass_but_average_window_profiles(self):
        facts = _assemble_facts(
            [np.array([0.0, 0.1, 0.0]), np.array([0.0, 0.2, 0.0])],
            [np.array([0.01]), np.array([0.02])],
            [np.array([-0.1]), np.array([0.1])],
            [np.array([100.0]), np.array([200.0, 300.0, 400.0])],
            [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0, 6.0])],
            [np.array([10.0, 20.0]), np.array([30.0, 40.0])],
            [np.array([12.0, 22.0]), np.array([32.0, 42.0])],
            [1.0, 1.0],
        )
        self.assertEqual(facts["trade_sizes"].tolist(), [100.0, 200.0, 300.0, 400.0])
        self.assertEqual(facts["inter_arrival"].tolist(), [1.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(facts["depth_bid"], [20.0, 30.0])
        np.testing.assert_allclose(facts["depth_ask"], [22.0, 32.0])

    def test_named_date_split_is_validated_against_available_dates(self):
        available = ["20251013", "20251014", "20251020"]
        config = {"date_splits": {"development": ["20251013", "20251020"]}}
        self.assertEqual(
            resolve_date_split(available, config, "development"),
            ["20251013", "20251020"],
        )
        with self.assertRaises(ValueError):
            resolve_date_split(available, config, "test")

    def test_tiled_windows_must_remain_inside_safe_session(self):
        config = {
            "session_start_utc": "14:00:00",
            "session_end_utc": "16:00:00",
            "window_duration_hours": 0.5,
            "window_starts_utc": ["14:00:00", "14:30:00", "15:00:00", "15:30:00"],
        }
        windows = resolve_measurement_windows(config)
        self.assertEqual(len(windows), 4)
        self.assertEqual(windows[0], (datetime.time(14, 0), datetime.time(14, 30)))
        self.assertEqual(windows[-1], (datetime.time(15, 30), datetime.time(16, 0)))
        config["window_starts_utc"].append("16:00:00")
        with self.assertRaises(ValueError):
            resolve_measurement_windows(config)

    def test_tiled_windows_must_not_overlap(self):
        config = {
            "window_duration_hours": 0.5,
            "window_starts_utc": ["14:00:00", "14:15:00"],
        }
        with self.assertRaisesRegex(ValueError, "overlap"):
            resolve_measurement_windows(config)

    def test_window_assignments_are_balanced_and_reproducible(self):
        dates = ["20251013", "20251014", "20251020"]
        windows = [
            (datetime.time(14, 0), datetime.time(14, 30)),
            (datetime.time(14, 30), datetime.time(15, 0)),
        ]
        first = select_window_assignments(dates, windows, 7, 42)
        second = select_window_assignments(dates, windows, 7, 42)
        self.assertEqual(first, second)
        counts = {
            start: sum(assigned_start == start for _, assigned_start in first)
            for start, _ in windows
        }
        self.assertEqual(sorted(counts.values()), [3, 4])
        date_counts = {
            date: sum(assigned_date == date for assigned_date, _ in first)
            for date in dates
        }
        self.assertLessEqual(max(date_counts.values()) - min(date_counts.values()), 1)


if __name__ == "__main__":
    unittest.main()
