from __future__ import annotations

import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class OptimizerExtensionWorkflowTests(unittest.TestCase):
    def test_checked_in_extension_is_historical_only(self):
        extension = yaml.safe_load(
            (ROOT / "configs/paper/optimizer-extension.yaml").read_text()
        )
        self.assertEqual(extension["status"], "historical_only")
        self.assertEqual(
            tuple(extension["methods"]), ("sobol", "gp_bo", "cma_es", "bads")
        )
        self.assertEqual(
            extension["method_settings"]["gp_bo"]["deterministic_objective"],
            True,
        )
        self.assertEqual(
            extension["method_settings"]["bads"]["invalid_penalty"],
            1_000_000.0,
        )
        self.assertEqual(
            extension["method_settings"]["sobol"]["sequence_seed"],
            yaml.safe_load(
                (ROOT / "configs/paper/initial-design.yaml").read_text()
            )["seed"],
        )
        self.assertTrue(all(
            extension["method_settings"][method]["rationale"]
            for method in extension["methods"]
        ))

    def test_new_protocol_does_not_resume_extension_study(self):
        extension = yaml.safe_load(
            (ROOT / "configs/paper/optimizer-extension.yaml").read_text()
        )
        optimizer = yaml.safe_load(
            (ROOT / "configs/paper/optimizer.yaml").read_text()
        )
        self.assertNotEqual(extension["comparison_study_name"], optimizer["study_name"])
        self.assertNotEqual(extension["status"], "ready_to_run")


if __name__ == "__main__":
    unittest.main()
