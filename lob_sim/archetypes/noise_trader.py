from __future__ import annotations

import numpy as np

from lob_sim.archetypes.base import BaseArchetype
from lob_sim.core.models import Action, ActionType, MarketObservation, Side


class NoiseTraderAgent(BaseArchetype):
    """
    Aggregate market-order flow generator.

    Each timestep:
    - Draws the number of new market orders from Poisson(order_frequency).
    - Uses a fixed side when one is configured; otherwise independently draws
      buy/sell direction using ``buy_probability``.
    - Order size is fixed by default or lognormally distributed when
      ``order_size_sigma`` is positive.

    Allowing ``side=None`` lets one aggregate noise-flow agent represent both
    sides of the market without creating large populations of identical agents.
    """

    def __init__(
        self,
        agent_id: str,
        side: Side | None,
        order_frequency: float,
        order_size: float,
        rng: np.random.Generator,
        buy_probability: float = 0.5,
        order_size_sigma: float = 0.0,
        max_order_size: float | None = None,
        lot_size: float = 100.0,
        burst_size: float = 0.0,
        pause_probability: float = 0.0,
        mean_pause_steps: float = 1.0,
        side_persistence: float = 0.0,
    ) -> None:
        super().__init__(agent_id)
        self._side = side
        self._order_frequency = order_frequency
        self._order_size = order_size
        self._rng = rng
        self._buy_probability = min(1.0, max(0.0, float(buy_probability)))
        self._order_size_sigma = max(0.0, float(order_size_sigma))
        self._max_order_size = float(max_order_size) if max_order_size is not None else None
        self._lot_size = max(float(lot_size), 1e-12)
        self._burst_size = max(0.0, float(burst_size))
        self._pause_probability = min(1.0, max(0.0, float(pause_probability)))
        self._mean_pause_steps = max(1.0, float(mean_pause_steps))
        self._side_persistence = min(1.0, max(0.0, float(side_persistence)))
        self._pause_steps_remaining = 0
        self._previous_side: Side | None = None

    # ------------------------------------------------------------------

    def act(self, obs: MarketObservation) -> list[Action]:
        actions: list[Action] = []

        if self._pause_steps_remaining > 0:
            self._pause_steps_remaining -= 1
            return actions
        if self._rng.random() < self._pause_probability:
            probability = 1.0 / self._mean_pause_steps
            self._pause_steps_remaining = max(0, int(self._rng.geometric(probability)) - 1)
            return actions

        # Draw number of new market orders
        if self._burst_size > 0:
            expected_burst = self._burst_size + float(np.exp(-self._burst_size))
            activation_probability = min(1.0, self._order_frequency / expected_burst)
            n_orders = (
                max(1, int(self._rng.poisson(self._burst_size)))
                if self._rng.random() < activation_probability else 0
            )
        else:
            n_orders = int(self._rng.poisson(self._order_frequency))

        for _ in range(n_orders):
            side = self._side
            if side is None:
                if (
                    self._side_persistence > 0
                    and self._previous_side is not None
                    and self._rng.random() < self._side_persistence
                ):
                    side = self._previous_side
                else:
                    side = (
                        Side.BUY
                        if self._rng.random() < self._buy_probability
                        else Side.SELL
                    )
                self._previous_side = side
            quantity = self._order_size
            if self._order_size_sigma > 0:
                quantity *= float(np.exp(self._rng.normal(0.0, self._order_size_sigma)))
            if self._max_order_size is not None:
                quantity = min(quantity, self._max_order_size)
            quantity = max(self._lot_size, round(quantity / self._lot_size) * self._lot_size)
            actions.append(Action(
                action_type=ActionType.PLACE_MARKET,
                side=side,
                qty=quantity,
            ))

        return actions

    def reset(self) -> None:
        self._pause_steps_remaining = 0
        self._previous_side = None
