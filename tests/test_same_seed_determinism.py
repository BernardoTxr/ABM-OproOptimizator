import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lob_sim.agents.factory import AgentFactory
from lob_sim.environment import Environment, EnvironmentConfig


class SameSeedDeterminismTests(unittest.TestCase):
    @staticmethod
    def _run(agent_path: Path, seed: int) -> dict[str, pd.DataFrame]:
        config = EnvironmentConfig(
            tick_size=0.01,
            min_order_size=100.0,
            start_time=0.0,
            end_time=60.0,
            lob_depth=5,
            book_storage_depth=50,
            seed_order_lifetime=1.0,
            initial_snapshot={
                "bids": [[10.00, 5000.0], [9.99, 5000.0]],
                "asks": [[10.02, 5000.0], [10.03, 5000.0]],
            },
        )
        agents = AgentFactory.from_yaml(str(agent_path), 1.0, seed)
        environment = Environment(config)
        shuffle_rng = np.random.default_rng(seed)
        for timestamp in np.arange(0.0, 60.0, 1.0):
            environment.advance_time(float(timestamp))
            shuffled = list(agents)
            shuffle_rng.shuffle(shuffled)
            for agent in shuffled:
                for action in agent.act(environment.get_observation(agent.agent_id)):
                    environment.submit_action(agent.agent_id, action)
            environment.record_state(float(timestamp))
        return environment.get_state_dataframes()

    def test_same_seed_reproduces_market_path_ignoring_opaque_ids(self):
        specification = {
            "agents": [
                {
                    "archetype": "NoiseTrader",
                    "id": "flow",
                    "count": 1,
                    "parameters": {
                        "side": "random",
                        "activity": 0.5,
                        "order_size": 200.0,
                        "buy_probability": 0.5,
                        "side_persistence": 0.3,
                    },
                },
                {
                    "archetype": "MultiLevelMarketMaker",
                    "id": "maker",
                    "count": 1,
                    "parameters": {
                        "levels": 5,
                        "spread": 0.01,
                        "tick_size": 0.01,
                        "order_size": 2000.0,
                        "inventory_limit": 10000.0,
                        "inventory_skew_ticks": 1,
                        "wide_spread_probability": 0.2,
                        "spread_state_persistence": 0.9,
                    },
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agents.yaml"
            path.write_text(yaml.safe_dump(specification, sort_keys=False))
            left = self._run(path, 1234)
            right = self._run(path, 1234)

        pd.testing.assert_frame_equal(left["mid_price"], right["mid_price"])
        pd.testing.assert_frame_equal(left["lob_snapshots"], right["lob_snapshots"])
        trade_columns = [
            "timestamp", "buyer_agent_id", "seller_agent_id", "price",
            "quantity", "aggressor_side",
        ]
        self.assertFalse(left["trade_tape"].empty)
        pd.testing.assert_frame_equal(
            left["trade_tape"][trade_columns], right["trade_tape"][trade_columns]
        )
        order_columns = [
            "timestamp", "agent_id", "event_type", "side", "price",
            "quantity", "filled_qty",
        ]
        pd.testing.assert_frame_equal(
            left["agent_orders"][order_columns], right["agent_orders"][order_columns]
        )


if __name__ == "__main__":
    unittest.main()
