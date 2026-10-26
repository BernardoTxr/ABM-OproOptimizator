from __future__ import annotations

import uuid

import numpy as np

from lob_sim.archetypes.base import BaseArchetype
from lob_sim.core.models import (
    Action,
    ActionType,
    ExecutionReport,
    MarketObservation,
    ReportType,
    Side,
)


class ValueTraderAgent(BaseArchetype):
    """Trades deviations from a private, noisy latent value.

    The latent value follows a random walk generated entirely inside the
    simulation. It never reads future market data. Orders cross at the current
    best quote and are bounded by the agent's inventory capacity.
    """

    def __init__(
        self,
        agent_id: str,
        fundamental_price: float | None,
        value_volatility: float,
        signal_noise: float,
        order_frequency: float,
        order_size: float,
        threshold: float,
        max_inventory: float,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(agent_id)
        self._initial_fundamental = (
            max(float(fundamental_price), 1e-12)
            if fundamental_price is not None else None
        )
        self._fundamental = self._initial_fundamental
        self._value_volatility = max(0.0, float(value_volatility))
        self._signal_noise = max(0.0, float(signal_noise))
        self._order_frequency = max(0.0, float(order_frequency))
        self._order_size = max(0.0, float(order_size))
        self._threshold = max(0.0, float(threshold))
        self._max_inventory = max(0.0, float(max_inventory))
        self._rng = rng
        self._inventory = 0.0
        self._order_sides: dict[str, Side] = {}

    @property
    def fundamental_value(self) -> float | None:
        return self._fundamental

    @property
    def inventory(self) -> float:
        return self._inventory

    def act(self, obs: MarketObservation) -> list[Action]:
        self._process_reports(obs.execution_reports)
        if not obs.bid_levels or not obs.ask_levels:
            return []
        best_bid = float(obs.bid_levels[0][0])
        best_ask = float(obs.ask_levels[0][0])
        if self._fundamental is None:
            self._fundamental = 0.5 * (best_bid + best_ask)
        self._fundamental = max(
            1e-12,
            self._fundamental + float(self._rng.normal(0.0, self._value_volatility)),
        )
        if self._rng.random() >= min(1.0, self._order_frequency):
            return []

        perceived_value = self._fundamental + float(
            self._rng.normal(0.0, self._signal_noise)
        )
        if perceived_value > best_ask + self._threshold:
            capacity = self._max_inventory - self._inventory
            return self._crossing_order(Side.BUY, best_ask, capacity)
        if perceived_value < best_bid - self._threshold:
            capacity = self._max_inventory + self._inventory
            return self._crossing_order(Side.SELL, best_bid, capacity)
        return []

    def reset(self) -> None:
        self._fundamental = self._initial_fundamental
        self._inventory = 0.0
        self._order_sides.clear()

    def _crossing_order(
        self, side: Side, price: float, inventory_capacity: float
    ) -> list[Action]:
        quantity = min(self._order_size, max(0.0, inventory_capacity))
        if quantity <= 0:
            return []
        order_id = str(uuid.uuid4())
        self._order_sides[order_id] = side
        return [Action(
            action_type=ActionType.PLACE_LIMIT,
            side=side,
            price=price,
            qty=quantity,
            order_id=order_id,
        )]

    def _process_reports(self, reports: list[ExecutionReport]) -> None:
        for report in reports:
            side = self._order_sides.get(report.order_id)
            if report.report_type in (ReportType.FILL, ReportType.PARTIAL_FILL):
                if side == Side.BUY:
                    self._inventory += report.filled_qty
                elif side == Side.SELL:
                    self._inventory -= report.filled_qty
            if report.remaining_qty <= 0:
                self._order_sides.pop(report.order_id, None)
