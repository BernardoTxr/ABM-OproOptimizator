import tempfile
import unittest
from pathlib import Path

import numpy as np

from lob_sim.loss.compare import LossSpec, compute_loss, save_statistics_report
from lob_sim.loss.bootstrap import aggregate_facts_for_loss


class LossAlignmentTests(unittest.TestCase):
    def test_acf_ignores_non_finite_pairs(self):
        sim = {"acf_abs_returns": {"mean": np.array([1.0, np.nan, 0.2])}}
        real = {"acf_abs_returns": {"mean": np.array([1.0, 0.4, 0.0])}}
        loss = compute_loss(sim, real)
        self.assertAlmostEqual(loss.per_fact["acf_abs_returns"], 0.2)
        self.assertTrue(np.isfinite(loss.total))

    def test_depth_loss_is_relative_and_dimensionless(self):
        sim = {
            "depth_bid": {"mean": np.array([50.0, 100.0])},
            "depth_ask": {"mean": np.array([50.0, 100.0])},
        }
        real = {
            "depth": {
                "bid_mean": np.array([100.0, 200.0]),
                "ask_mean": np.array([100.0, 200.0]),
            }
        }
        loss = compute_loss(sim, real)
        self.assertAlmostEqual(loss.per_fact["depth"], 0.5)

    def test_pooled_v2_uses_rmse_for_acf_and_relative_depth(self):
        sim = {
            "acf_abs_returns": {"mean": np.array([1.0, 0.4])},
            "depth_bid": {"mean": np.array([50.0, 100.0])},
            "depth_ask": {"mean": np.array([50.0, 100.0])},
        }
        real = {
            "acf_abs_returns": {"mean": np.array([1.0, 0.0])},
            "depth": {
                "bid_mean": np.array([100.0, 200.0]),
                "ask_mean": np.array([100.0, 200.0]),
            },
        }
        specification = LossSpec.from_dict({"version": "pooled_v2"})
        loss = compute_loss(sim, real, specification=specification)
        self.assertAlmostEqual(loss.per_fact["acf_abs_returns"], 0.4)
        self.assertAlmostEqual(loss.per_fact["depth"], 0.5)
        self.assertEqual(loss.specification["acf_distance"], "rmse_nonzero_lags")

    def test_missing_fact_receives_a_penalty(self):
        loss = compute_loss({}, {})
        self.assertEqual(loss.per_fact["depth"], 1.0)
        self.assertEqual(loss.per_fact["book_coverage"], 1.0)
        self.assertEqual(loss.total, 1.0)

    def test_versioned_loss_spec_controls_penalty_and_is_reported(self):
        specification = LossSpec.from_dict({
            "version": "pooled_v2",
            "winsor_quantiles": [0.05, 0.95],
            "missing_fact_penalty": 2.0,
        })
        loss = compute_loss({}, {}, specification=specification)
        self.assertEqual(loss.total, 2.0)
        self.assertEqual(loss.specification["version"], "pooled_v2")
        self.assertEqual(loss.specification["winsor_quantiles"], [0.05, 0.95])

    def test_loss_spec_rejects_unsupported_versions_and_weight_typos(self):
        with self.assertRaises(ValueError):
            LossSpec.from_dict({"version": "pooled_v1"})
        with self.assertRaises(ValueError):
            LossSpec.from_dict({
                "version": "pooled_v2",
                "weights": {"interarrival_typo": 1.0},
            })

    def test_statistics_report_uses_pooled_samples(self):
        sim = {
            "spread": {"samples": np.array([0.01, 0.02])},
            "book_coverage": {"mean": 1.0},
        }
        real = {
            "spread": {"samples": np.array([0.01, 0.01])},
            "book_coverage": {"mean": 1.0},
        }
        with tempfile.TemporaryDirectory() as directory:
            save_statistics_report(sim, real, directory)
            report = (Path(directory) / "statistics_report.md").read_text()
        self.assertIn("Mean spread | 0.015 | 0.01", report)
        self.assertIn("Two-sided book coverage", report)

    def test_bootstrap_aggregation_keeps_scenario_clusters(self):
        facts = []
        for shift in (0.0, 1.0):
            facts.append({
                "log_returns": np.array([shift, shift + 0.1]),
                "spread": np.array([1.0 + shift]),
                "imbalance": np.array([shift]),
                "trade_sizes": np.array([100.0 + shift]),
                "inter_arrival": np.array([1.0 + shift]),
                "acf_abs_returns": np.array([1.0, 0.1 + shift]),
                "depth_bid": np.array([10.0 + shift]),
                "depth_ask": np.array([20.0 + shift]),
                "book_coverage": np.array([1.0]),
            })
        aggregate = aggregate_facts_for_loss(facts)
        self.assertEqual(aggregate["log_returns"]["samples"].tolist(), [0.0, 0.1, 1.0, 1.1])
        self.assertTrue(np.allclose(aggregate["depth_bid"]["mean"], [10.5]))
        self.assertEqual(aggregate["book_coverage"]["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
