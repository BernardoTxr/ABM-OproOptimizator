from __future__ import annotations

import numpy as np

from lob_sim.archetypes.base import BaseArchetype
from lob_sim.core.models import Action, ActionType, MarketObservation, Side


class RandomLimitOrderTraderAgent(BaseArchetype):
    """
    Aggregate random limit-order provider that replenishes book depth.

    Each timestep:
    - Randomly cancels resting orders, giving them a finite lifetime.
    - Cancels orders that are too far from the current best quote.
    - With probability 0.5 chooses buy, 0.5 sell.
    - Usually places at or behind the best quote, but may improve a spread
      wider than one tick.

    Parameters
    ----------
    order_frequency : mean number of orders per step (Poisson draw)
    order_size      : fixed quantity per order
    max_offset      : maximum price distance from the best quote (in price units)
    tick_size       : minimum price increment
    inside_probability : probability of improving a spread wider than one tick
    cancel_probability : per-step cancellation probability for each open order
    best_probability : probability a passive order joins the best quote
    """

    def __init__(
        self,
        agent_id: str,
        order_frequency: float,
        order_size: float,
        max_offset: float,
        rng: np.random.Generator,
        tick_size: float = 0.01,
        inside_probability: float = 0.1,
        cancel_probability: float = 0.01,
        best_probability: float = 0.0,
        imbalance_sensitivity: float = 0.0,
    ) -> None:
        super().__init__(agent_id)
        self._order_frequency = order_frequency
        self._order_size      = float(order_size)
        self._max_offset      = float(max_offset)
        self._rng             = rng
        self._tick_size       = float(tick_size)
        self._inside_probability = min(1.0, max(0.0, float(inside_probability)))
        self._cancel_probability = min(1.0, max(0.0, float(cancel_probability)))
        self._best_probability = min(1.0, max(0.0, float(best_probability)))
        self._imbalance_sensitivity = min(
            1.0, max(0.0, float(imbalance_sensitivity))
        )

        self._open_order_ids: set[str] = set()

    # ------------------------------------------------------------------

    def act(self, obs: MarketObservation) -> list[Action]:
        actions: list[Action] = []

        # Skip if book is empty — no reference price to quote around
        if not obs.bid_levels or not obs.ask_levels:
            return actions

        best_bid = obs.bid_levels[0][0]
        best_ask = obs.ask_levels[0][0]

        # The observation is authoritative, including depth- and fill-driven
        # removals that happened since our previous action.
        self._open_order_ids = {o.order_id for o in obs.open_orders}

        # Cancel old or distant orders.
        for o in obs.open_orders:
            too_far = (
                (o.side == Side.BUY  and o.price < best_bid - self._max_offset) or
                (o.side == Side.SELL and o.price > best_ask + self._max_offset)
            )
            if too_far or self._rng.random() < self._cancel_probability:
                actions.append(Action(action_type=ActionType.CANCEL, order_id=o.order_id))
                self._open_order_ids.discard(o.order_id)

        bid_depth = sum(quantity for _, quantity in obs.bid_levels)
        ask_depth = sum(quantity for _, quantity in obs.ask_levels)
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0.0
        # Positive imbalance means excess bid depth, so a stabilizing provider
        # becomes more likely to replenish the ask. A zero sensitivity retains
        # the original symmetric side choice.
        buy_probability = min(
            1.0,
            max(0.0, 0.5 - 0.5 * self._imbalance_sensitivity * imbalance),
        )

        # Place new limit orders
        n_orders = int(self._rng.poisson(self._order_frequency))
        for _ in range(n_orders):
            side = Side.BUY if self._rng.random() < buy_probability else Side.SELL
            spread_ticks = max(1, int(round((best_ask - best_bid) / self._tick_size)))
            improve = spread_ticks > 1 and self._rng.random() < self._inside_probability

            if improve:
                # Choose a valid price strictly inside the spread.
                improvement = int(self._rng.integers(1, spread_ticks))
                price = (
                    best_bid + improvement * self._tick_size
                    if side == Side.BUY
                    else best_ask - improvement * self._tick_size
                )
            else:
                max_ticks = max(0, int(round(self._max_offset / self._tick_size)))
                if max_ticks == 0 or self._rng.random() < self._best_probability:
                    offset_ticks = 0
                else:
                    offset_ticks = int(self._rng.integers(1, max_ticks + 1))
                price = (
                    best_bid - offset_ticks * self._tick_size
                    if side == Side.BUY
                    else best_ask + offset_ticks * self._tick_size
                )
            if price <= 0:
                continue
            oid = str(__import__("uuid").uuid4())
            actions.append(Action(
                action_type=ActionType.PLACE_LIMIT,
                side=side,
                price=price,
                qty=self._order_size,
                order_id=oid,
            ))
            self._open_order_ids.add(oid)

        return actions

    def reset(self) -> None:
        self._open_order_ids.clear()
