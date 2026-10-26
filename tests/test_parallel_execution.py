import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from workflows.benchmark.parallel_execution import (
    format_cpu_set,
    load_execution_config,
    parse_cpu_set,
)
from workflows.benchmark.run_component_aware_parallel import (
    _write_execution_summary,
    build_ticker_command,
)


ROOT = Path(__file__).resolve().parents[1]


class ParallelExecutionTests(unittest.TestCase):
    def test_checked_in_xeon_configuration_reserves_one_cpu(self):
        values = load_execution_config(
            ROOT / "configs/paper/xeon-e2146g-execution.yaml"
        )
        execution = values["execution"]
        self.assertEqual(execution["reserved_logical_cpus"], [11])
        self.assertEqual(len(execution["cpu_affinity"]), 11)
        self.assertEqual(execution["ticker_workers"], 5)
        self.assertEqual(execution["simulation_workers_per_ticker"], 3)

    def test_linux_cpu_list_parser(self):
        self.assertEqual(parse_cpu_set("0-3,6,8-9"), (0, 1, 2, 3, 6, 8, 9))
        self.assertEqual(format_cpu_set((0, 2, 4)), "0,2,4")
        with self.assertRaises(ValueError):
            parse_cpu_set("3-1")

    def test_ticker_child_command_is_isolated_and_affined(self):
        command = build_ticker_command(
            comparison_path=ROOT / "configs/paper/component-aware-comparison.yaml",
            ticker="ASAI3",
            methods=("sobol",),
            repetitions=(),
            stage_first_adaptive_slot=True,
            provenance=None,
            schema_preflight=None,
            simulation_workers=3,
            cpu_affinity=tuple(range(11)),
        )
        self.assertIn("ASAI3", command)
        self.assertIn("--stage-first-adaptive-slot", command)
        self.assertEqual(command[command.index("--simulation-workers") + 1], "3")
        self.assertEqual(
            command[command.index("--cpu-set") + 1],
            ",".join(str(cpu) for cpu in range(11)),
        )

    def test_execution_summary_ignores_full_workflow_journal(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "parallel.json").write_text(json.dumps({
                "version": "component_aware_parallel_execution_v1",
                "mode": "stage",
                "status": "complete",
                "started_at_utc": "2026-09-11T20:00:00+00:00",
                "finished_at_utc": "2026-09-11T20:10:00+00:00",
                "wall_seconds": 600.0,
                "methods": ["sobol"],
            }))
            (directory / "full.json").write_text(json.dumps({
                "version": "full_component_experiment_v1",
                "status": "failed",
                "steps": [],
            }))

            output = _write_execution_summary(directory)
            summary = json.loads(output.read_text())

            self.assertEqual(len(summary["records"]), 1)
            self.assertEqual(summary["records"][0]["mode"], "stage")
            self.assertEqual(summary["by_mode"]["stage"]["invocations"], 1)


if __name__ == "__main__":
    unittest.main()
