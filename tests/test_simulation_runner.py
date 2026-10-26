import tempfile
import unittest
from pathlib import Path

import yaml

from lob_sim.simulation.runner import SimulationRunner


class SimulationRunnerTests(unittest.TestCase):
    def test_diagnostics_can_be_disabled_without_disabling_state_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agents = root / "agents.yaml"
            environment = root / "environment.yaml"
            scenario = root / "scenario.yaml"
            agents.write_text("agents: []\n")
            environment.write_text(yaml.safe_dump({
                "tick_size": 0.01,
                "min_order_size": 100.0,
                "start_time": 0.0,
                "end_time": 1.0,
                "lob_depth": 1,
                "book_storage_depth": 2,
                "seed_order_lifetime": None,
                "initial_snapshot": {
                    "bids": [[10.00, 1000.0]],
                    "asks": [[10.02, 1000.0]],
                },
            }))
            scenario.write_text(yaml.safe_dump({
                "scenario_name": "quiet",
                "n_scenarios": 1,
                "agents_yaml": str(agents),
                "environment_yaml": str(environment),
                "time_step": 1,
                "random_seed": 1,
                "n_workers": 1,
                "write_diagnostics": False,
                "output_directory": str(root / "output"),
            }))

            SimulationRunner().run(str(scenario))

            output = root / "output" / "quiet" / "scenario_0000"
            self.assertTrue((output / "mid_price.parquet").is_file())
            self.assertTrue((output / "lob_snapshots.parquet").is_file())
            self.assertTrue((output / "metadata.yaml").is_file())
            self.assertFalse((output / "report.md").exists())
            self.assertFalse(list(output.glob("plot_*.png")))


if __name__ == "__main__":
    unittest.main()
