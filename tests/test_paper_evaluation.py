import unittest
from pathlib import Path

from lob_sim.loss.smm import SMMSpec, SMMTarget
from lob_sim.optimization.evaluator import use_frozen_smm_weights
from workflows.benchmark.evaluate import (
    METHODS,
    _bootstrap_later,
)


ROOT = Path(__file__).resolve().parents[1]


class PaperEvaluationTests(unittest.TestCase):
    def test_test_means_keep_calibration_standard_errors(self):
        def target(value, standard_error, signature):
            return SMMTarget(
                moment_names=("return_mean",),
                target={"return_mean": value},
                standard_errors={"return_mean": standard_error},
                covariance=[[standard_error ** 2]],
                date_moments=[],
                metadata={"signature": signature},
            )

        combined = use_frozen_smm_weights(
            target(2.0, 9.0, "test"),
            target(1.0, 0.5, "calibration"),
            ROOT / "results/real/ASAI3/calibration.json",
        )
        self.assertEqual(combined.target["return_mean"], 2.0)
        self.assertEqual(combined.standard_errors["return_mean"], 0.5)
        self.assertEqual(combined.metadata["signature"], "test")
        self.assertEqual(combined.metadata["weight_source"], "calibration_target")
        self.assertEqual(combined.metadata["weight_target_signature"], "calibration")

    def test_whole_date_bootstrap_preserves_paired_method_ordering(self):
        spec = SMMSpec.from_dict({
            "version": "smm_v4",
            "groups": {
                "returns": ["return_mean"],
                "spread": ["spread_p1"],
                "imbalance": ["imbalance_mean"],
                "trade_size": ["trade_size_q50"],
                "inter_arrival": ["inter_arrival_q90"],
                "depth": ["depth_bid_l1"],
            },
            "standard_errors": {"bootstrap_draws": 2},
        })
        target = SMMTarget(
            moment_names=spec.moment_names,
            target={name: 0.0 for name in spec.moment_names},
            standard_errors={name: 1.0 for name in spec.moment_names},
            covariance=[[1.0 if i == j else 0.0 for j in range(6)] for i in range(6)],
            date_moments=[
                {"date": date, **{name: value for name in spec.moment_names}}
                for date, value in (("d1", 0.0), ("d2", 0.2))
            ],
            metadata={},
        )
        offsets = {
            "random": 2.0, "tpe": 1.5,
            "opro_plain": 1.0, "opro_heuristic": 0.5,
        }
        results = {
            method: {"smm": {"scenario_moments": [
                {"date": date, **{
                    name: empirical + offset for name in spec.moment_names
                }}
                for date, empirical in (("d1", 0.0), ("d2", 0.2))
                for _ in range(2)
            ]}}
            for method, offset in offsets.items()
        }
        report = _bootstrap_later(results, target, spec, draws=50, seed=7)
        self.assertEqual(set(report["methods"]), set(METHODS))
        self.assertEqual(
            report["paired_comparisons"]["opro_heuristic_minus_opro_plain"]
            ["probability_left_better"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
