from __future__ import annotations

import math
import uuid

import numpy as np

from lob_sim.archetypes.base import BaseArchetype
from lob_sim.core.models import (
    Action, ActionType, ExecutionReport, MarketObservation, ReportType, Side,
)


class MultiLevelMarketMakerAgent(BaseArchetype):
    """Simple aggregate market maker that continuously maintains a ladder.

    On every step the agent computes its desired bid/ask levels, cancels quotes
    that are no longer at those levels, and replenishes missing or partially
    filled quotes. Prices are constructed in integer ticks, avoiding threshold
    errors at half-tick midpoints.

    Inventory shifts the ladder by at most ``inventory_skew_ticks`` and tilts
    bid/ask sizes as absolute inventory approaches ``max_inventory``. The
    configured inventory value is a response scale, not a hard quote cutoff;
    keeping both sides present prevents a one-sided-book deadlock.
    """

    def __init__(
        self,
        agent_id: str,
        n_levels: int,
        half_spread: float,
        level_spacing: float,
        order_size: float,
        max_inventory: float,
        tick_size: float = 0.01,
        inventory_skew_ticks: int = 0,
        level_size_growth: float = 0.0,
        rng: np.random.Generator | None = None,
        wide_spread_probability: float = 0.0,
        spread_state_persistence: float = 0.95,
        level_sizes: list[float] | None = None,
        inventory_size_skew: float = 0.0,
        recenter_rate: float = 0.0,
        recenter_threshold_ticks: float = 1.0,
    ) -> None:
        super().__init__(agent_id)
        self._n_levels = max(1, int(n_levels))
        self._half_spread = float(half_spread)
        self._level_spacing = float(level_spacing)
        self._order_size = float(order_size)
        self._max_inventory = float(max_inventory)
        self._tick_size = float(tick_size)
        self._inventory_skew_ticks = max(0, int(inventory_skew_ticks))
        self._level_size_growth = max(0.0, float(level_size_growth))
        self._rng = rng if rng is not None else np.random.default_rng()
        self._wide_spread_probability = min(
            1.0, max(0.0, float(wide_spread_probability))
        )
        self._spread_state_persistence = min(
            1.0, max(0.0, float(spread_state_persistence))
        )
        self._wide_spread = False
        self._wide_direction = 0
        self._reference_tick: float | None = None
        self._level_sizes = [float(size) for size in level_sizes] if level_sizes else None
        self._inventory_size_skew = max(0.0, float(inventory_size_skew))
        self._recenter_rate = min(1.0, max(0.0, float(recenter_rate)))
        self._recenter_threshold_ticks = max(
            0.0, float(recenter_threshold_ticks)
        )

        self._inventory: float = 0.0
        self._order_side: dict[str, Side] = {}

    def act(self, obs: MarketObservation) -> list[Action]:
        self._process_reports(obs.execution_reports)
        if self._reference_tick is None:
            if not obs.bid_levels or not obs.ask_levels:
                return []
            self._reference_tick = (
                self._to_ticks(obs.bid_levels[0][0])
                + self._to_ticks(obs.ask_levels[0][0])
            ) / 2.0
        elif obs.bid_levels and obs.ask_levels and self._recenter_rate > 0:
            observed_mid_tick = (
                self._to_ticks(obs.bid_levels[0][0])
                + self._to_ticks(obs.ask_levels[0][0])
            ) / 2.0
            displacement = observed_mid_tick - self._reference_tick
            if abs(displacement) >= self._recenter_threshold_ticks:
                self._reference_tick += self._recenter_rate * displacement

        desired = self._desired_prices()
        actions: list[Action] = []
        kept: dict[Side, set[int]] = {Side.BUY: set(), Side.SELL: set()}

        # Keep at most one full-size order per desired level. Partial orders are
        # replaced so every quoted level returns to its configured target depth.
        for order in obs.open_orders:
            price_tick = self._to_ticks(order.price)
            target_size = desired[order.side].get(price_tick)
            should_keep = (
                target_size is not None
                and price_tick not in kept[order.side]
                and math.isclose(order.remaining_qty, target_size, rel_tol=1e-9, abs_tol=1e-9)
            )
            if should_keep:
                kept[order.side].add(price_tick)
                self._order_side[order.order_id] = order.side
            else:
                actions.append(Action(action_type=ActionType.CANCEL, order_id=order.order_id))
                self._order_side.pop(order.order_id, None)

        # Replenish every missing target level.
        for side in (Side.BUY, Side.SELL):
            for price_tick in sorted(desired[side], reverse=(side == Side.BUY)):
                if price_tick in kept[side]:
                    continue
                order_id = str(uuid.uuid4())
                actions.append(Action(
                    action_type=ActionType.PLACE_LIMIT,
                    side=side,
                    price=price_tick * self._tick_size,
                    qty=desired[side][price_tick],
                    order_id=order_id,
                ))
                self._order_side[order_id] = side

        return actions

    def reset(self) -> None:
        self._inventory = 0.0
        self._order_side.clear()
        self._wide_spread = False
        self._wide_direction = 0
        self._reference_tick = None

    def _process_reports(self, reports: list[ExecutionReport]) -> None:
        for report in reports:
            if report.report_type in (ReportType.FILL, ReportType.PARTIAL_FILL):
                side = self._order_side.get(report.order_id)
                if side == Side.BUY:
                    self._inventory += report.filled_qty
                elif side == Side.SELL:
                    self._inventory -= report.filled_qty
            if report.remaining_qty <= 0:
                self._order_side.pop(report.order_id, None)

    def _desired_prices(self) -> dict[Side, dict[int, float]]:
        assert self._reference_tick is not None
        centre_tick = self._reference_tick

        if self._max_inventory > 0 and self._inventory_skew_ticks:
            inventory_ratio = max(-1.0, min(1.0, self._inventory / self._max_inventory))
            centre_tick -= round(inventory_ratio * self._inventory_skew_ticks)

        if self._rng.random() > self._spread_state_persistence:
            was_wide = self._wide_spread
            self._wide_spread = self._rng.random() < self._wide_spread_probability
            if self._wide_spread and not was_wide:
                self._wide_direction = -1 if self._rng.random() < 0.5 else 1
        base_spread = max(1, int(round(2.0 * self._half_spread / self._tick_size)))
        target_spread = max(base_spread, 2) if self._wide_spread else base_spread
        spacing = max(1, int(round(self._level_spacing / self._tick_size)))
        inner_bid = math.floor(centre_tick - target_spread / 2.0 + 1e-12)
        if self._wide_spread and self._wide_direction > 0:
            inner_bid += 1
        inner_ask = inner_bid + target_spread

        return {
            Side.BUY: {
                inner_bid - i * spacing: self._size_at_level(i, Side.BUY)
                for i in range(self._n_levels)
            },
            Side.SELL: {
                inner_ask + i * spacing: self._size_at_level(i, Side.SELL)
                for i in range(self._n_levels)
            },
        }

    def _to_ticks(self, price: float) -> int:
        return int(round(price / self._tick_size))

    def _size_at_level(self, level: int, side: Side) -> float:
        if self._level_sizes and level < len(self._level_sizes):
            base_size = self._level_sizes[level]
        else:
            base_size = self._order_size * (1.0 + self._level_size_growth * level)
        if self._max_inventory <= 0 or self._inventory_size_skew == 0:
            return base_size
        ratio = max(-1.0, min(1.0, self._inventory / self._max_inventory))
        direction = -1.0 if side == Side.BUY else 1.0
        multiplier = max(0.1, 1.0 + direction * ratio * self._inventory_size_skew)
        return base_size * multiplier
