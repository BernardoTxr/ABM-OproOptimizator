import unittest
from pathlib import Path

from workflows.benchmark.ticker_books import TickerTask, resolved_proposal_runtime


class ResolvedProposalRuntimeTests(unittest.TestCase):
    def test_none_when_comparison_has_no_proposal_runtime(self):
        self.assertIsNone(
            resolved_proposal_runtime(Path("/tmp/comparison.yaml"), {})
        )

    def test_relative_config_paths_resolve_against_comparison_directory(self):
        comparison_path = Path(
            "/repo/configs/experiments/followup_foundry_sonnet5/comparison.yaml"
        )
        comparison = {
            "proposal_runtime": {
                "kind": "anthropic_foundry",
                "config": "anthropic_foundry.yaml",
                "model_registry": "models.yaml",
                "model_key": "claude_sonnet5_foundry_v1",
            }
        }
        resolved = resolved_proposal_runtime(comparison_path, comparison)
        self.assertEqual(resolved["kind"], "anthropic_foundry")
        self.assertEqual(resolved["model_key"], "claude_sonnet5_foundry_v1")
        self.assertEqual(
            resolved["config"],
            "/repo/configs/experiments/followup_foundry_sonnet5/anthropic_foundry.yaml",
        )
        self.assertEqual(
            resolved["model_registry"],
            "/repo/configs/experiments/followup_foundry_sonnet5/models.yaml",
        )

    def test_absolute_config_paths_are_left_unchanged(self):
        comparison_path = Path("/repo/configs/experiments/x/comparison.yaml")
        comparison = {
            "proposal_runtime": {
                "kind": "anthropic_foundry",
                "config": "/elsewhere/anthropic_foundry.yaml",
                "model_registry": "models.yaml",
                "model_key": "k",
            }
        }
        resolved = resolved_proposal_runtime(comparison_path, comparison)
        self.assertEqual(resolved["config"], "/elsewhere/anthropic_foundry.yaml")


class TickerTaskCalibrationOverridesTests(unittest.TestCase):
    def _task(self) -> TickerTask:
        return TickerTask(
            symbol="ASAI3",
            calibration_study_name="study/ASAI3/calibration",
            test_study_name="study/ASAI3/test",
            calibration_target=Path("/tmp/cal_target.json"),
            calibration_windows=Path("/tmp/cal_windows.parquet"),
            test_target=Path("/tmp/test_target.json"),
            test_windows=Path("/tmp/test_windows.parquet"),
            books_per_partition=90,
        )

    def test_proposal_runtime_key_absent_by_default(self):
        overrides = self._task().calibration_overrides("qwen_qwen3_8_flash_8192_routed")
        self.assertNotIn("proposal_runtime", overrides)

    def test_proposal_runtime_key_present_when_given(self):
        overrides = self._task().calibration_overrides(
            "claude_sonnet5_foundry_v1",
            proposal_runtime={"kind": "anthropic_foundry"},
        )
        self.assertEqual(overrides["proposal_runtime"], {"kind": "anthropic_foundry"})


if __name__ == "__main__":
    unittest.main()
