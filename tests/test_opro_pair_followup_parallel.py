import unittest
from pathlib import Path

from workflows.benchmark.opro_pair_followup import pair_plan
from workflows.benchmark.opro_pair_followup_parallel import build_ticker_command
from workflows.benchmark.ticker_books import comparison_output_root, experiment_paths


ROOT = Path(__file__).resolve().parents[1]
COMPARISON_PATH = ROOT / "configs/experiments/followup_foundry_sonnet5/comparison.yaml"


class ParallelLauncherRootTests(unittest.TestCase):
    def test_launcher_root_matches_the_single_ticker_workflows_root(self):
        # Regression test: the launcher once passed comparison_path where
        # calibration_path was required, resolving output_directory (a path
        # relative to configs/paper/) against configs/experiments/.../
        # instead, silently writing into configs/results/... rather than
        # results/....
        (calibration_path, _, _, comparison, calibration, _, _) = experiment_paths(
            COMPARISON_PATH
        )
        launcher_root = comparison_output_root(comparison, calibration_path, calibration)
        single_ticker_root = Path(pair_plan(COMPARISON_PATH)["comparison_root"])
        self.assertEqual(launcher_root.resolve(), single_ticker_root.resolve())
        self.assertNotIn("configs", launcher_root.parts)


class BuildTickerCommandTests(unittest.TestCase):
    def test_command_uses_declared_flags_without_shell_interpolation(self):
        command = build_ticker_command(
            comparison_path=COMPARISON_PATH,
            ticker="ASAI3",
            methods=("opro_plain",),
            repetitions=(0, 1),
            stage_first_adaptive_slot=True,
            simulation_workers=2,
        )
        self.assertIn("--stage-first-adaptive-slot", command)
        self.assertEqual(
            command[command.index("--ticker") + 1], "ASAI3",
        )
        self.assertEqual(
            command[command.index("--simulation-workers") + 1], "2",
        )
        self.assertEqual(command.count("--method"), 1)
        self.assertEqual(command.count("--repetition"), 2)


if __name__ == "__main__":
    unittest.main()
