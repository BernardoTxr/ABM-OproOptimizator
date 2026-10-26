import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from lob_sim.optimization.evaluator import (
    _evaluation_fingerprint,
    _pooled_aggregate_for_loss,
    apply_simulation_output_retention,
    validate_smm_analysis_archive,
    validate_simulation,
)


class OptimizationEvaluatorTests(unittest.TestCase):
    def test_smm_archive_guard_requires_all_plot_ready_rows(self):
        names = ("moment_a", "moment_b")
        details = {
            key: {"moment_a": 1.0, "moment_b": 2.0}
            for key in (
                "simulated_moments",
                "target_moments",
                "raw_differences",
                "standardized_residuals",
            )
        }
        details["scenario_moments"] = [{
            "scenario": "scenario_0000",
            "date": "20260101",
            "window_start_utc": "14:00:00",
            "moment_a": 1.0,
            "moment_b": 2.0,
        }]
        schedule = [{
            "scenario_idx": 0,
            "real_date": "20260101",
            "window_start_utc": "14:00:00",
            "random_seed": 7,
        }]
        validity = {"scenarios": [{"scenario": "scenario_0000"}]}
        record = validate_smm_analysis_archive(
            details, schedule, validity, names, expected_scenarios=1
        )
        self.assertEqual(record["moment_count"], 2)
        self.assertEqual(record["scenario_count"], 1)

        details["scenario_moments"] = []
        with self.assertRaisesRegex(ValueError, "every scenario"):
            validate_smm_analysis_archive(
                details, schedule, validity, names, expected_scenarios=1
            )

    def test_smm_loss_skips_legacy_pooled_aggregation(self):
        with patch("lob_sim.optimization.evaluator.aggregate_all") as aggregate:
            result = _pooled_aggregate_for_loss(Path("simulation"), "smm_v4")

        self.assertIsNone(result)
        aggregate.assert_not_called()

    def test_legacy_loss_still_builds_pooled_aggregation(self):
        expected = {"kde_data": {}}
        with patch(
            "lob_sim.optimization.evaluator.aggregate_all",
            return_value=expected,
        ) as aggregate:
            result = _pooled_aggregate_for_loss(Path("simulation"), "pooled_v2")

        self.assertIs(result, expected)
        aggregate.assert_called_once_with("simulation", force=True)

    def test_summary_only_retention_removes_only_explicit_simulation_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory) / "trial"
            simulation = trial / "simulation"
            simulation.mkdir(parents=True)
            (simulation / "raw.parquet").write_bytes(b"raw")
            self.assertTrue(
                apply_simulation_output_retention(simulation, "summary_only")
            )
            self.assertFalse(simulation.exists())
            self.assertTrue(trial.exists())

    def test_full_retention_preserves_outputs_and_unsafe_target_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "other"
            target.mkdir()
            self.assertFalse(apply_simulation_output_retention(target, "full"))
            self.assertTrue(target.exists())
            with self.assertRaisesRegex(ValueError, "unsafe simulation path"):
                apply_simulation_output_retention(target, "summary_only")

    def test_evaluation_fingerprint_changes_with_source_or_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lob_sim").mkdir()
            source = root / "lob_sim" / "model.py"
            config = root / "config.yaml"
            data = root / "events.parquet"
            source.write_text("VERSION = 1\n")
            config.write_text("seed: 1\n")
            data.write_bytes(b"data-v1")
            first = _evaluation_fingerprint(root, [config], [data], {"split": "dev"})
            source.write_text("VERSION = 2\n")
            second = _evaluation_fingerprint(root, [config], [data], {"split": "dev"})
            self.assertNotEqual(first, second)
            data.write_bytes(b"data-v2-longer")
            third = _evaluation_fingerprint(root, [config], [data], {"split": "dev"})
            self.assertNotEqual(second, third)

    def test_validity_gate_rejects_sparse_or_crossed_scenarios(self):
        with tempfile.TemporaryDirectory() as directory:
            scenario = Path(directory) / "scenario_0000"
            scenario.mkdir()
            pd.DataFrame({
                "mid_price": [10.0, None, 10.0],
                "best_bid": [9.99, None, 10.01],
                "best_ask": [10.01, None, 10.00],
            }).to_parquet(scenario / "mid_price.parquet")
            pd.DataFrame({"quantity": [100.0]}).to_parquet(
                scenario / "trade_tape.parquet"
            )
            result = validate_simulation(
                Path(directory), expected_scenarios=1,
                min_book_coverage=0.99, min_trades_per_scenario=2,
            )
            self.assertFalse(result["valid"])
            joined = " ".join(result["violations"])
            self.assertIn("coverage", joined)
            self.assertIn("trades", joined)
            self.assertIn("crossed_or_locked_book", joined)

    def test_validity_gate_requires_complete_depth_when_configured(self):
        with tempfile.TemporaryDirectory() as directory:
            scenario = Path(directory) / "scenario_0000"
            scenario.mkdir()
            pd.DataFrame({
                "mid_price": [10.0, 10.0],
                "best_bid": [9.99, 9.99],
                "best_ask": [10.01, 10.01],
            }).to_parquet(scenario / "mid_price.parquet")
            pd.DataFrame({"quantity": [100.0, 100.0]}).to_parquet(
                scenario / "trade_tape.parquet"
            )
            pd.DataFrame({
                "timestamp": [0, 0, 0, 0, 1, 1, 1, 1, 1],
                "level": [1, 2, 3, 4, 1, 2, 3, 4, 5],
                "bid_qty": [100.0] * 9,
                "ask_qty": [100.0] * 9,
            }).to_parquet(scenario / "lob_snapshots.parquet")
            result = validate_simulation(
                Path(directory), expected_scenarios=1,
                min_book_coverage=0.99, min_trades_per_scenario=1,
                required_depth_levels=5, min_depth_coverage=0.99,
            )
            self.assertFalse(result["valid"])
            self.assertIn("depth_coverage", " ".join(result["violations"]))


if __name__ == "__main__":
    unittest.main()
