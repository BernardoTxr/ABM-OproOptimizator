import json
import tempfile
import unittest
from pathlib import Path

from workflows.benchmark.score_predictions import _visible_prompt_trials, score_run


class OPROPredictionScoringTests(unittest.TestCase):
    def _write(self, directory, trial, objective, component, signed=None, metadata=None):
        metadata = dict(metadata or {})
        if metadata.get("predictions") and "prompt" not in metadata:
            metadata["prompt"] = json.dumps({
                "observations": [{"trial_id": index} for index in range(trial)]
            })
        record = {
            "policy_trial": trial,
            "objective": objective,
            "proposal_metadata": metadata,
            "evaluation": {
                "component_losses": component,
                "signed_diagnostics": signed or {},
            },
        }
        path = Path(directory) / f"trial_{trial:04d}.json"
        path.write_text(json.dumps(record))
        return path

    def test_uses_best_cited_evidence_and_scores_component_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [
                self._write(directory, 0, 0.30, {"spread": 0.20}),
                self._write(directory, 1, 0.20, {"spread": 0.10}),
                self._write(
                    directory, 2, 0.19, {"spread": 0.08},
                    metadata={
                        "evidence_trial_ids": [0, 1],
                        "predictions": [{
                            "metric": "spread", "direction": "decrease",
                            "rationale": "narrower",
                        }],
                    },
                ),
            ]
            rows = score_run(paths)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["reference_trial"], 1)
            self.assertAlmostEqual(rows[0]["observed_change"], -0.02)
            self.assertTrue(rows[0]["correct"])

    def test_scores_signed_diagnostic_and_reports_unknown_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [
                self._write(
                    directory, 0, 0.30, {}, {"imbalance_mean_scaled_error": -0.2}
                ),
                self._write(
                    directory, 1, 0.20, {}, {"imbalance_mean_scaled_error": 0.1},
                    metadata={
                        "evidence_trial_ids": [0],
                        "predictions": [
                            {
                                "metric": "imbalance_mean_scaled_error",
                                "direction": "increase", "rationale": "toward zero",
                            },
                            {
                                "metric": "unknown", "direction": "decrease",
                                "rationale": "not stored",
                            },
                        ],
                    },
                ),
            ]
            rows = score_run(paths)
            self.assertTrue(rows[0]["correct"])
            self.assertEqual(rows[0]["metric_source"], "signed_diagnostic")
            self.assertFalse(rows[1]["scorable"])
            self.assertEqual(
                rows[1]["unscorable_reason"], "metric_missing_from_evaluation"
            )

    def test_no_change_uses_declared_tolerance(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [
                self._write(directory, 0, 0.30, {"depth": 0.20}),
                self._write(
                    directory, 1, 0.20, {"depth": 0.2005},
                    metadata={
                        "evidence_trial_ids": [0],
                        "predictions": [{
                            "metric": "depth", "direction": "no_change",
                            "rationale": "stable",
                        }],
                    },
                ),
            ]
            self.assertTrue(score_run(paths)[0]["correct"])

    def test_hidden_ledger_evidence_is_an_unscorable_protocol_violation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [
                self._write(directory, 0, 0.30, {"depth": 0.20}),
                self._write(directory, 1, 0.25, {"depth": 0.18}),
                self._write(
                    directory, 2, 0.20, {"depth": 0.15},
                    metadata={
                        "prompt": json.dumps({"observations": [{"trial_id": 1}]}),
                        "evidence_trial_ids": [0],
                        "predictions": [{
                            "metric": "depth", "direction": "decrease",
                            "rationale": "claimed hidden evidence",
                        }],
                    },
                ),
            ]
            row = score_run(paths)[0]
            self.assertFalse(row["scorable"])
            self.assertFalse(row["evidence_visible_in_prompt"])
            self.assertEqual(row["hidden_evidence_trial_ids"], [0])
            self.assertEqual(
                row["unscorable_reason"], "evidence_trial_not_visible_in_prompt"
            )

    def test_visible_prompt_trial_parser_is_strict(self):
        metadata = {"prompt": json.dumps({
            "observations": [{"trial_id": 1}, {"trial_id": 3}]
        })}
        self.assertEqual(_visible_prompt_trials(metadata), {1, 3})
        self.assertIsNone(_visible_prompt_trials({"prompt": "not json"}))


if __name__ == "__main__":
    unittest.main()
