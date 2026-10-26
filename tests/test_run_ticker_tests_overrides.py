"""Regression test for a real bug hit during the Foundry follow-up: Gate 8
(run_ticker_tests) reconstructs the calibration TrialEvaluator to read back
an existing ledger's incumbents. If that reconstruction omits an operational
--simulation-workers override the calibration run actually used, the fresh
evaluator_fingerprint differs from the one recorded on every ledger trial
(since it's folded into runtime_overrides), and Gate 8 fails with
"evaluator differs from its calibration ledgers" even though nothing
scientific changed.
"""
import unittest
from unittest.mock import patch

from workflows.benchmark import evaluate_component_aware


class _StopEarly(Exception):
    """Raised by the fake TrialEvaluator right after recording its call."""


class RunTickerTestsCalibrationOverridesTests(unittest.TestCase):
    def _captured_overrides(self, calibration_simulation_workers):
        calls = []

        def fake_trial_evaluator(config_path, runtime_overrides=None):
            calls.append(dict(runtime_overrides or {}))
            raise _StopEarly()

        with patch.object(
            evaluate_component_aware, "experiment_paths",
            return_value=(
                "calibration.yaml", "test.yaml", "real.yaml",
                {"llm_model_key": "claude_sonnet5_foundry_v1"},
                {}, {}, {},
            ),
        ), patch.object(
            evaluate_component_aware, "validate_component_contract",
        ), patch.object(
            evaluate_component_aware, "build_ticker_tasks",
            return_value=(
                type("Task", (), {
                    "symbol": "ASAI3",
                    "calibration_overrides": staticmethod(
                        lambda model_key, proposal_runtime=None: {
                            "study_name": "s", "symbol": "ASAI3",
                        }
                    ),
                })(),
            ),
        ), patch.object(
            evaluate_component_aware, "resolved_proposal_runtime", return_value=None,
        ), patch.object(
            evaluate_component_aware, "TrialEvaluator", side_effect=fake_trial_evaluator,
        ):
            with self.assertRaises(_StopEarly):
                evaluate_component_aware.run_ticker_tests(
                    "comparison.yaml",
                    ("ASAI3",),
                    methods=("opro_plain", "opro_heuristic"),
                    calibration_simulation_workers=calibration_simulation_workers,
                )
        return calls[0]

    def test_override_is_threaded_through_when_given(self):
        overrides = self._captured_overrides(calibration_simulation_workers=2)
        self.assertEqual(overrides.get("simulation_workers"), 2)

    def test_override_is_absent_by_default(self):
        overrides = self._captured_overrides(calibration_simulation_workers=None)
        self.assertNotIn("simulation_workers", overrides)


if __name__ == "__main__":
    unittest.main()
