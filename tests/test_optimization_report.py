import json
import tempfile
import unittest
from pathlib import Path

from lob_sim.optimization.report import (
    load_policy_runs,
    paired_policy_comparisons,
    summarize_policy_runs,
)


class OptimizationReportTests(unittest.TestCase):
    def test_repetition_is_the_aggregation_unit(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "policies" / "random_repeat00"
            trials = run / "trials"
            trials.mkdir(parents=True)
            (run / "summary.json").write_text(json.dumps({"budget": 3}))
            for index, objective in enumerate((0.3, 0.2, 0.25)):
                (trials / f"trial_{index:04d}.json").write_text(json.dumps({
                    "policy_trial": index,
                    "sampler": "random",
                    "repetition": 0,
                    "objective": objective,
                    "status": "complete",
                    "duplicate": False,
                    "candidate": {"x": index},
                    "controller_runtime_seconds": 1.0,
                    "proposal_metadata": {},
                    "policy_feedback_mode": "none",
                }))
            runs = load_policy_runs(directory)
            self.assertEqual(runs[0]["best_so_far"], [0.3, 0.2, 0.2])
            self.assertAlmostEqual(runs[0]["mean_best_so_far"], 0.7 / 3)
            summary = summarize_policy_runs(runs)["random"]
            self.assertEqual(summary["repetitions"], 1)
            self.assertEqual(summary["complete_repetitions"], 1)
            self.assertAlmostEqual(summary["best_objective_mean"], 0.2)
            self.assertEqual(summary["policy_feedback_modes"], ["none"])

    def test_pairwise_comparison_uses_matched_repetitions(self):
        runs = [
            {"sampler": "random", "repetition": 0, "best_objective": 0.3,
             "mean_best_so_far": 0.4},
            {"sampler": "random", "repetition": 1, "best_objective": 0.2,
             "mean_best_so_far": 0.3},
            {"sampler": "tpe", "repetition": 0, "best_objective": 0.1,
             "mean_best_so_far": 0.2},
        ]
        comparison = paired_policy_comparisons(runs)[0]
        self.assertEqual(comparison["matched_repetitions"], [0])
        self.assertAlmostEqual(comparison["final_loss_a_minus_b_mean"], 0.2)
        self.assertEqual(comparison["method_b_final_wins"], 1)


if __name__ == "__main__":
    unittest.main()
