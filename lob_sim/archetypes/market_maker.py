from __future__ import annotations
import uuid
from typing import Optional

from lob_sim.archetypes.base import BaseArchetype
from lob_sim.core.models import Action, ActionType, ExecutionReport, MarketObservation, ReportType, Side


class MarketMakerAgent(BaseArchetype):
    """
    Continuous two-sided quoter.

    Every timestep:
    1. Cancels any stale outstanding quotes.
    2. Computes mid = (best_bid + best_ask) / 2.
    3. Posts bid at mid - half_spread and ask at mid + half_spread,
       subject to inventory constraints.

    Inventory constraint:
    - If inventory >= max_inventory: stops posting bids (only posts asks).
    - If inventory <= -max_inventory: stops posting asks (only posts bids).
    """

    def __init__(
        self,
        agent_id: str,
        half_spread: float,
        order_size: float,
        max_inventory: float,
    ) -> None:
        super().__init__(agent_id)
        self._half_spread = half_spread
        self._order_size = order_size
        self._max_inventory = max_inventory

        # State
        self._inventory: float = 0.0
        self._bid_order_id: Optional[str] = None
        self._ask_order_id: Optional[str] = None

    # ------------------------------------------------------------------

    def act(self, obs: MarketObservation) -> list[Action]:
        actions: list[Action] = []

        # Process fills to update inventory and replenish countdowns
        self._process_reports(obs.execution_reports)

        # Cancel all stale open quotes and refresh tracked IDs
        if obs.open_orders:
            stale_ids = {o.order_id for o in obs.open_orders}
            for o in obs.open_orders:
                actions.append(Action(action_type=ActionType.CANCEL, order_id=o.order_id))
            if self._bid_order_id in stale_ids:
                self._bid_order_id = None
            if self._ask_order_id in stale_ids:
                self._ask_order_id = None

        # Need both sides to compute mid
        if not obs.bid_levels or not obs.ask_levels:
            return actions

        mid = (obs.bid_levels[0][0] + obs.ask_levels[0][0]) / 2.0
        bid_price = mid - self._half_spread
        ask_price = mid + self._half_spread

        can_post_bid = self._inventory < self._max_inventory
        can_post_ask = self._inventory > -self._max_inventory

        if can_post_bid and bid_price > 0:
            new_id = str(uuid.uuid4())
            actions.append(Action(
                action_type=ActionType.PLACE_LIMIT,
                side=Side.BUY,
                price=bid_price,
                qty=self._order_size,
                order_id=new_id,
            ))
            self._bid_order_id = new_id

        if can_post_ask:
            new_id = str(uuid.uuid4())
            actions.append(Action(
                action_type=ActionType.PLACE_LIMIT,
                side=Side.SELL,
                price=ask_price,
                qty=self._order_size,
                order_id=new_id,
            ))
            self._ask_order_id = new_id

        return actions

    def reset(self) -> None:
        self._inventory = 0.0
        self._bid_order_id = None
        self._ask_order_id = None

    # ------------------------------------------------------------------

    def _process_reports(self, reports: list[ExecutionReport]) -> None:
        for report in reports:
            if report.report_type not in (ReportType.FILL, ReportType.PARTIAL_FILL):
                continue

            if report.order_id == self._bid_order_id:
                self._inventory += report.filled_qty
                if report.remaining_qty <= 0:
                    self._bid_order_id = None
            elif report.order_id == self._ask_order_id:
                self._inventory -= report.filled_qty
                if report.remaining_qty <= 0:
                    self._ask_order_id = None
