import unittest
import tempfile
from pathlib import Path

import numpy as np

from lob_sim.archetypes.multi_level_market_maker import MultiLevelMarketMakerAgent
from lob_sim.archetypes.mean_reversion_trader import MeanReversionTraderAgent
from lob_sim.archetypes.momentum_trader import MomentumTraderAgent
from lob_sim.archetypes.noise_trader import NoiseTraderAgent
from lob_sim.archetypes.random_limit_order_trader import RandomLimitOrderTraderAgent
from lob_sim.archetypes.value_trader import ValueTraderAgent
from lob_sim.agents.factory import AgentFactory
from lob_sim.core.models import (
    Action, ActionType, ExecutionReport, MarketObservation, Order, ReportType,
    Side,
)
from lob_sim.environment import Environment, EnvironmentConfig


def observation(*, bids=None, asks=None, open_orders=None):
    return MarketObservation(
        timestamp=0.0,
        bid_levels=bids or [(8.07, 1000.0)],
        ask_levels=asks or [(8.08, 1000.0)],
        execution_reports=[],
        open_orders=open_orders or [],
    )


class MinimalMarketAgentTests(unittest.TestCase):
    def test_fixed_seed_reproduces_population_action_values(self):
        config = Path(__file__).resolve().parents[1] / "configs" / "paper" / "agents.yaml"
        left = AgentFactory.from_yaml(str(config), 1.0, 812)
        right = AgentFactory.from_yaml(str(config), 1.0, 812)

        def normalized(actions):
            return [(
                action.action_type.value,
                action.side.value if action.side is not None else None,
                action.price,
                action.qty,
            ) for action in actions]

        for step in range(40):
            bid = 8.07 + 0.01 * (step % 3)
            obs = observation(bids=[(bid, 1000.0)], asks=[(bid + 0.01, 1000.0)])
            for left_agent, right_agent in zip(left, right):
                self.assertEqual(
                    normalized(left_agent.act(obs)),
                    normalized(right_agent.act(obs)),
                )

    def test_factory_requires_unique_stable_population_ids(self):
        missing = """agents:
  - archetype: NoiseTrader
    parameters: {side: random, activity: 1.0, order_size: 100.0}
"""
        duplicate = """agents:
  - archetype: NoiseTrader
    id: flow
    parameters: {side: random, activity: 1.0, order_size: 100.0}
  - archetype: NoiseTrader
    id: flow
    parameters: {side: random, activity: 1.0, order_size: 100.0}
"""
        with tempfile.TemporaryDirectory() as directory:
            for name, content in (("missing", missing), ("duplicate", duplicate)):
                path = Path(directory) / f"{name}.yaml"
                path.write_text(content)
                with self.assertRaises(ValueError):
                    AgentFactory.from_yaml(str(path), 1.0, 91)

    def test_factory_streams_are_stable_to_population_reordering(self):
        first = """agents:
  - archetype: NoiseTrader
    id: flow_a
    parameters: {side: random, activity: 1.0, order_size: 100.0}
  - archetype: NoiseTrader
    id: flow_b
    parameters: {side: random, activity: 1.0, order_size: 100.0}
"""
        second = """agents:
  - archetype: NoiseTrader
    id: flow_b
    parameters: {side: random, activity: 1.0, order_size: 100.0}
  - archetype: NoiseTrader
    id: flow_a
    parameters: {side: random, activity: 1.0, order_size: 100.0}
"""
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.yaml"
            second_path = Path(directory) / "second.yaml"
            first_path.write_text(first)
            second_path.write_text(second)
            left = {agent.agent_id: agent for agent in AgentFactory.from_yaml(
                str(first_path), 1.0, 91
            )}
            right = {agent.agent_id: agent for agent in AgentFactory.from_yaml(
                str(second_path), 1.0, 91
            )}
        self.assertEqual(set(left), set(right))
        for agent_id in left:
            self.assertEqual(left[agent_id]._rng.random(), right[agent_id]._rng.random())

    def test_one_noise_agent_generates_both_market_order_sides(self):
        agent = NoiseTraderAgent(
            "noise", None, order_frequency=5.0, order_size=100.0,
            rng=np.random.default_rng(7), buy_probability=0.5,
        )
        actions = [action for _ in range(30) for action in agent.act(observation())]
        self.assertEqual({action.side for action in actions}, {Side.BUY, Side.SELL})
        self.assertTrue(all(action.action_type == ActionType.PLACE_MARKET for action in actions))

    def test_persistent_noise_flow_keeps_the_sampled_side(self):
        agent = NoiseTraderAgent(
            "flow", None, order_frequency=8.0, order_size=100.0,
            rng=np.random.default_rng(19), buy_probability=0.5,
            side_persistence=1.0,
        )
        actions = [action for _ in range(8) for action in agent.act(observation())]
        self.assertGreater(len(actions), 1)
        self.assertEqual(len({action.side for action in actions}), 1)

    def test_market_maker_replenishes_a_missing_level_without_mid_move(self):
        agent = MultiLevelMarketMakerAgent(
            "maker", n_levels=2, half_spread=0.005, level_spacing=0.01,
            order_size=600.0, max_inventory=10_000.0, tick_size=0.01,
        )
        initial = agent.act(observation())
        resting = [
            Order.create(
                agent_id="maker", side=action.side, price=action.price,
                qty=action.qty, timestamp=0.0, order_id=action.order_id,
            )
            for action in initial
        ]

        actions = agent.act(observation(open_orders=resting[1:]))
        placements = [a for a in actions if a.action_type == ActionType.PLACE_LIMIT]
        self.assertEqual(len(placements), 1)
        self.assertEqual((placements[0].side, placements[0].price),
                         (resting[0].side, resting[0].price))

    def test_initialized_market_maker_recovers_a_one_sided_book(self):
        agent = MultiLevelMarketMakerAgent(
            "maker", n_levels=1, half_spread=0.005, level_spacing=0.01,
            order_size=600.0, max_inventory=10_000.0, tick_size=0.01,
        )
        agent.act(observation())
        one_sided = MarketObservation(
            timestamp=1.0,
            bid_levels=[(8.07, 1000.0)],
            ask_levels=[],
            execution_reports=[],
            open_orders=[],
        )
        actions = agent.act(one_sided)
        self.assertEqual(
            {action.side for action in actions if action.action_type == ActionType.PLACE_LIMIT},
            {Side.BUY, Side.SELL},
        )

    def test_random_limit_provider_can_improve_a_wide_spread(self):
        agent = RandomLimitOrderTraderAgent(
            "depth", order_frequency=10.0, order_size=100.0, max_offset=0.05,
            rng=np.random.default_rng(11), tick_size=0.01,
            inside_probability=1.0, cancel_probability=0.0,
        )
        actions = agent.act(observation(bids=[(8.00, 1000.0)], asks=[(8.10, 1000.0)]))
        placements = [a for a in actions if a.action_type == ActionType.PLACE_LIMIT]
        self.assertTrue(placements)
        self.assertTrue(all(8.00 < action.price < 8.10 for action in placements))

    def test_passive_provider_replenishes_the_thinner_side(self):
        agent = RandomLimitOrderTraderAgent(
            "depth", order_frequency=20.0, order_size=100.0, max_offset=0.01,
            rng=np.random.default_rng(23), tick_size=0.01,
            inside_probability=0.0, cancel_probability=0.0,
            imbalance_sensitivity=1.0,
        )
        actions = agent.act(observation(
            bids=[(8.07, 1000.0)], asks=[(8.08, 0.0)]
        ))
        placements = [a for a in actions if a.action_type == ActionType.PLACE_LIMIT]
        self.assertTrue(placements)
        self.assertTrue(all(action.side == Side.SELL for action in placements))

    def test_value_trader_buys_when_ask_is_below_private_value(self):
        agent = ValueTraderAgent(
            "value", fundamental_price=8.20, value_volatility=0.0,
            signal_noise=0.0, order_frequency=1.0, order_size=200.0,
            threshold=0.01, max_inventory=1000.0,
            rng=np.random.default_rng(29),
        )
        actions = agent.act(observation())
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].side, Side.BUY)
        self.assertEqual(actions[0].price, 8.08)

    def test_value_trader_can_initialize_from_observed_midpoint(self):
        agent = ValueTraderAgent(
            agent_id="value_dynamic",
            fundamental_price=None,
            value_volatility=0.0,
            signal_noise=0.0,
            order_frequency=1.0,
            order_size=100.0,
            threshold=0.01,
            max_inventory=1000.0,
            rng=np.random.default_rng(4),
        )
        actions = agent.act(observation(
            bids=[(9.99, 1000.0)], asks=[(10.01, 1000.0)]
        ))
        self.assertEqual(actions, [])
        self.assertEqual(agent.fundamental_value, 10.0)

    def test_momentum_trader_buys_after_a_positive_lagged_return(self):
        agent = MomentumTraderAgent(
            "momentum", lookback_steps=1, order_size=200.0,
            entry_threshold=0.001, order_frequency=1.0,
            max_inventory=1000.0, rng=np.random.default_rng(13),
        )
        self.assertEqual(agent.act(observation()), [])
        actions = agent.act(observation(
            bids=[(8.17, 1000.0)], asks=[(8.18, 1000.0)]
        ))
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].side, Side.BUY)
        self.assertEqual(actions[0].price, 8.18)

    def test_momentum_trader_sells_after_a_negative_lagged_return(self):
        agent = MomentumTraderAgent(
            "momentum", lookback_steps=1, order_size=200.0,
            entry_threshold=0.001, order_frequency=1.0,
            max_inventory=1000.0, rng=np.random.default_rng(17),
        )
        self.assertEqual(agent.act(observation(
            bids=[(8.17, 1000.0)], asks=[(8.18, 1000.0)]
        )), [])
        actions = agent.act(observation())
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].side, Side.SELL)
        self.assertEqual(actions[0].price, 8.07)

    def test_mean_reversion_trader_sells_a_positive_displacement(self):
        agent = MeanReversionTraderAgent(
            "contrarian", reference_horizon=20, order_frequency=1.0,
            order_size=200.0, threshold=0.01, max_inventory=1000.0,
            rng=np.random.default_rng(31),
        )
        self.assertEqual(agent.act(observation()), [])
        actions = agent.act(observation(
            bids=[(8.17, 1000.0)], asks=[(8.18, 1000.0)]
        ))
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].side, Side.SELL)
        self.assertEqual(actions[0].price, 8.17)

    def test_market_maker_optionally_recenters_on_the_observed_mid(self):
        agent = MultiLevelMarketMakerAgent(
            "maker", n_levels=1, half_spread=0.005, level_spacing=0.01,
            order_size=600.0, max_inventory=10_000.0, tick_size=0.01,
            recenter_rate=1.0, recenter_threshold_ticks=0.5,
        )
        initial = agent.act(observation())
        resting = [
            Order.create(
                agent_id="maker", side=action.side, price=action.price,
                qty=action.qty, timestamp=0.0, order_id=action.order_id,
            )
            for action in initial
        ]
        actions = agent.act(observation(
            bids=[(8.09, 1000.0)], asks=[(8.10, 1000.0)],
            open_orders=resting,
        ))
        placements = [a for a in actions if a.action_type == ActionType.PLACE_LIMIT]
        self.assertEqual(
            {(action.side, action.price) for action in placements},
            {(Side.BUY, 8.09), (Side.SELL, 8.10)},
        )

    def test_long_market_maker_inventory_shifts_and_tilts_quotes_downward(self):
        agent = MultiLevelMarketMakerAgent(
            "maker", n_levels=1, half_spread=0.005, level_spacing=0.01,
            order_size=600.0, max_inventory=1000.0, tick_size=0.01,
            inventory_skew_ticks=2, inventory_size_skew=0.5,
        )
        initial = agent.act(observation())
        bid = next(action for action in initial if action.side == Side.BUY)
        filled_bid = ExecutionReport(
            order_id=bid.order_id, agent_id="maker", report_type=ReportType.FILL,
            filled_qty=500.0, remaining_qty=0.0, price=bid.price, timestamp=1.0,
        )
        obs = observation()
        obs.execution_reports = [filled_bid]
        placements = [
            action for action in agent.act(obs)
            if action.action_type == ActionType.PLACE_LIMIT
        ]
        new_bid = next(action for action in placements if action.side == Side.BUY)
        new_ask = next(action for action in placements if action.side == Side.SELL)
        self.assertLess(new_bid.price, bid.price)
        self.assertLess(new_bid.qty, new_ask.qty)

    def test_visible_depth_does_not_delete_deeper_book_levels(self):
        env = Environment(EnvironmentConfig(
            tick_size=1.0,
            min_order_size=1.0,
            start_time=0.0,
            end_time=1.0,
            lob_depth=1,
            book_storage_depth=3,
            initial_snapshot={
                "bids": [[10.0, 1.0], [9.0, 1.0], [8.0, 1.0]],
                "asks": [[11.0, 1.0]],
            },
        ))
        env.submit_action("seller", Action(
            action_type=ActionType.PLACE_MARKET, side=Side.SELL, qty=2.0,
        ))
        obs = env.get_observation("observer")
        self.assertEqual(obs.bid_levels, [(8.0, 1.0)])

    def test_market_order_accepts_odd_lot_completion_residual(self):
        env = Environment(EnvironmentConfig(
            tick_size=0.01,
            min_order_size=100.0,
            start_time=0.0,
            end_time=1.0,
            lob_depth=1,
            initial_snapshot={"bids": [], "asks": [[8.11, 50.0]]},
        ))
        trades, _ = env.submit_action("buyer", Action(
            action_type=ActionType.PLACE_MARKET, side=Side.BUY, qty=50.0,
        ))
        self.assertEqual(sum(trade.qty for trade in trades), 50.0)

    def test_matching_engine_preserves_fifo_within_a_price_level(self):
        env = Environment(EnvironmentConfig(
            tick_size=0.01,
            min_order_size=1.0,
            start_time=0.0,
            end_time=2.0,
            lob_depth=1,
        ))
        for agent_id in ("first_seller", "second_seller"):
            env.submit_action(agent_id, Action(
                action_type=ActionType.PLACE_LIMIT,
                side=Side.SELL,
                price=8.10,
                qty=100.0,
            ))
        trades, _ = env.submit_action("buyer", Action(
            action_type=ActionType.PLACE_MARKET,
            side=Side.BUY,
            qty=150.0,
        ))
        self.assertEqual(
            [(trade.seller_agent_id, trade.qty) for trade in trades],
            [("first_seller", 100.0), ("second_seller", 50.0)],
        )

    def test_seed_book_expires_at_configured_lifetime(self):
        env = Environment(EnvironmentConfig(
            tick_size=0.01,
            min_order_size=1.0,
            start_time=0.0,
            end_time=2.0,
            lob_depth=1,
            seed_order_lifetime=1.0,
            initial_snapshot={"bids": [[8.09, 100.0]], "asks": [[8.10, 100.0]]},
        ))
        env.advance_time(1.0)
        trades, _ = env.submit_action("buyer", Action(
            action_type=ActionType.PLACE_MARKET,
            side=Side.BUY,
            qty=100.0,
        ))
        self.assertEqual(trades, [])


if __name__ == "__main__":
    unittest.main()
