from __future__ import annotations
import math
import uuid
from collections import deque

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


class MomentumTraderAgent(BaseArchetype):
    """
    Trend-following directional trader.

    Signal: cumulative log-return of mid-price over `lookback_steps`:

        r = log(P_t / P_{t - lookback_steps})

    Buy if r > +entry_threshold, sell if r < -entry_threshold.
    Uses adaptive limit-then-market order logic with `limit_timeout`.
    """

    def __init__(
        self,
        agent_id: str,
        lookback_steps: int,
        order_size: float,
        entry_threshold: float,
        order_frequency: float = 1.0,
        max_inventory: float = float("inf"),
        limit_timeout: int = 5,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(agent_id)
        self._lookback_steps = lookback_steps
        self._order_size = order_size
        self._entry_threshold = entry_threshold
        self._order_frequency = min(1.0, max(0.0, float(order_frequency)))
        self._max_inventory = max(0.0, float(max_inventory))
        self._limit_timeout = max(1, int(limit_timeout))
        self._rng = rng if rng is not None else np.random.default_rng()

        # Ring buffer: holds lookback_steps+1 mid-prices for exact lookback_steps-period return
        self._mid_history: deque[float] = deque(maxlen=lookback_steps + 1)
        self._ticks_since_limit: int = 0
        self._inventory: float = 0.0
        self._order_sides: dict[str, Side] = {}

    # ------------------------------------------------------------------

    def act(self, obs: MarketObservation) -> list[Action]:
        self._process_reports(obs.execution_reports)
        # Record current mid-price
        if obs.bid_levels and obs.ask_levels:
            mid = (obs.bid_levels[0][0] + obs.ask_levels[0][0]) / 2.0
            self._mid_history.append(mid)

        # Handle active limit order
        has_open = len(obs.open_orders) > 0

        if has_open:
            self._ticks_since_limit += 1
            if self._ticks_since_limit >= self._limit_timeout:
                actions: list[Action] = [
                    Action(action_type=ActionType.CANCEL, order_id=o.order_id)
                    for o in obs.open_orders
                ]
                mkt_qty = sum(o.remaining_qty for o in obs.open_orders)
                if obs.open_orders and mkt_qty > 0:
                    side = obs.open_orders[0].side
                    opposite_levels = (
                        obs.ask_levels if side == Side.BUY else obs.bid_levels
                    )
                    if opposite_levels:
                        actions.extend(self._crossing_order(
                            side, opposite_levels[0][0], mkt_qty
                        ))
                self._ticks_since_limit = 0
                return actions
            return []

        self._ticks_since_limit = 0

        # Need a full history window
        if len(self._mid_history) < self._lookback_steps + 1:
            return []

        p_now = self._mid_history[-1]
        p_past = self._mid_history[0]   # oldest element = exactly lookback_steps ago
        if not (p_now > 0 and p_past > 0):
            return []

        r = math.log(p_now / p_past)

        if self._rng.random() >= self._order_frequency:
            return []

        if r > self._entry_threshold:
            if not obs.ask_levels:
                return []
            return self._crossing_order(
                Side.BUY,
                obs.ask_levels[0][0],
                self._max_inventory - self._inventory,
            )

        if r < -self._entry_threshold:
            if not obs.bid_levels:
                return []
            return self._crossing_order(
                Side.SELL,
                obs.bid_levels[0][0],
                self._max_inventory + self._inventory,
            )

        return []

    def reset(self) -> None:
        self._mid_history.clear()
        self._ticks_since_limit = 0
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
