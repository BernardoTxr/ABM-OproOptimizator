from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from lob_sim.core.lob import LimitOrderBook
from lob_sim.core.models import (
    Action,
    ActionType,
    ExecutionReport,
    MarketObservation,
    Order,
    ReportType,
    Side,
    Trade,
)


@dataclass
class EnvironmentConfig:
    tick_size: float
    min_order_size: float
    start_time: float
    end_time: float
    lob_depth: int
    book_storage_depth: int = 50
    seed_order_lifetime: float | None = 1.0
    initial_snapshot: dict = field(default_factory=dict)
    # expected shape: {"bids": [[price, qty], ...], "asks": [[price, qty], ...]}


class Environment:
    """
    Wraps the LOB matching engine and provides:
      - Order submission with price validation and tick-snapping
      - Execution report routing per agent
      - Agent observation construction (LOB snapshot + reports + open orders)
      - State recording for output
    """

    SEED_AGENT_ID = "__seed__"

    def __init__(self, config: EnvironmentConfig) -> None:
        self._cfg = config
        self._lob = LimitOrderBook()
        self._current_time: float = config.start_time

        # Per-agent pending execution reports (drained on get_observation)
        self._pending_reports: dict[str, list[ExecutionReport]] = defaultdict(list)

        # Per-agent list of currently resting Order objects (mutated in-place by LOB)
        self._open_orders: dict[str, list[Order]] = defaultdict(list)

        # Flat registry: order_id → Order (for resting-side report generation)
        self._env_registry: dict[str, Order] = {}
        self._seed_order_ids: list[str] = []
        self._seed_orders_expired = False

        # State recording buffers
        self._lob_records: list[dict] = []
        self._trade_records: list[dict] = []
        self._mid_records: list[dict] = []
        self._order_records: list[dict] = []

        self._bootstrap(config.initial_snapshot)

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def _bootstrap(self, snapshot: dict) -> None:
        t = self._cfg.start_time
        for entry in snapshot.get("bids", []):
            price, qty = float(entry[0]), float(entry[1])
            order = Order.create(
                agent_id=self.SEED_AGENT_ID,
                side=Side.BUY,
                price=self._snap(price),
                qty=qty,
                timestamp=t,
                is_seed=True,
            )
            self._lob.place_limit_order(order, t)
            self._seed_order_ids.append(order.order_id)

        for entry in snapshot.get("asks", []):
            price, qty = float(entry[0]), float(entry[1])
            order = Order.create(
                agent_id=self.SEED_AGENT_ID,
                side=Side.SELL,
                price=self._snap(price),
                qty=qty,
                timestamp=t,
                is_seed=True,
            )
            self._lob.place_limit_order(order, t)
            self._seed_order_ids.append(order.order_id)

        self._enforce_book_depth(t)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def advance_time(self, t: float) -> None:
        self._current_time = t
        lifetime = self._cfg.seed_order_lifetime
        if (
            not self._seed_orders_expired
            and lifetime is not None
            and t >= self._cfg.start_time + lifetime
        ):
            for order_id in self._seed_order_ids:
                self._lob.cancel_order(order_id)
            self._seed_order_ids.clear()
            self._seed_orders_expired = True

    def submit_action(self, agent_id: str, action: Action) -> tuple[list[Trade], list[ExecutionReport]]:
        """Apply one agent action to the LOB. Returns (trades, reports_for_agent)."""
        t = self._current_time

        if action.action_type == ActionType.PLACE_LIMIT:
            return self._place_limit(agent_id, action, t)
        elif action.action_type == ActionType.PLACE_MARKET:
            return self._place_market(agent_id, action, t)
        elif action.action_type == ActionType.CANCEL:
            return self._cancel(agent_id, action, t)
        return [], []

    def get_observation(self, agent_id: str) -> MarketObservation:
        """Return the current market state from this agent's perspective."""
        bid_levels, ask_levels = self._lob.get_snapshot(self._cfg.lob_depth)

        # Drain pending reports
        reports = self._pending_reports.pop(agent_id, [])

        # Sync open orders: filter out fully filled ones (remaining_qty updated in-place by LOB)
        live_orders = [o for o in self._open_orders.get(agent_id, []) if o.remaining_qty > 0]
        self._open_orders[agent_id] = live_orders

        return MarketObservation(
            timestamp=self._current_time,
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            execution_reports=reports,
            open_orders=list(live_orders),
        )

    def record_state(self, timestamp: float) -> None:
        """Snapshot LOB and mid-price into recording buffers."""
        bid_levels, ask_levels = self._lob.get_snapshot(self._cfg.lob_depth)
        depth = self._cfg.lob_depth

        for i in range(depth):
            self._lob_records.append({
                "timestamp": timestamp,
                "level": i + 1,
                "bid_price": bid_levels[i][0] if i < len(bid_levels) else None,
                "bid_qty":   bid_levels[i][1] if i < len(bid_levels) else None,
                "ask_price": ask_levels[i][0] if i < len(ask_levels) else None,
                "ask_qty":   ask_levels[i][1] if i < len(ask_levels) else None,
            })

        self._mid_records.append({
            "timestamp": timestamp,
            "mid_price": self._lob.mid_price,
            "best_bid":  self._lob.best_bid,
            "best_ask":  self._lob.best_ask,
            "spread":    self._lob.spread,
        })

    def get_state_dataframes(self) -> dict[str, pd.DataFrame]:
        return {
            "lob_snapshots": pd.DataFrame(self._lob_records),
            "mid_price":     pd.DataFrame(self._mid_records),
            "trade_tape":    pd.DataFrame(self._trade_records),
            "agent_orders":  pd.DataFrame(self._order_records),
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_time(self) -> float:
        return self._current_time

    @property
    def start_time(self) -> float:
        return self._cfg.start_time

    @property
    def end_time(self) -> float:
        return self._cfg.end_time

    @property
    def tick_size(self) -> float:
        return self._cfg.tick_size

    @property
    def lob_depth(self) -> int:
        return self._cfg.lob_depth

    # ------------------------------------------------------------------
    # Private action handlers
    # ------------------------------------------------------------------

    def _place_limit(self, agent_id: str, action: Action, t: float) -> tuple[list[Trade], list[ExecutionReport]]:
        if action.side is None or action.price is None or action.qty is None:
            return [], []
        if action.qty < self._cfg.min_order_size:
            return [], []
        if not pd.notna(action.price) or action.price <= 0:
            return [], []

        price = self._snap(action.price)
        order = Order.create(
            agent_id=agent_id,
            side=action.side,
            price=price,
            qty=action.qty,
            timestamp=t,
            order_id=action.order_id,   # use proposed ID if provided
        )

        self._record_order_event(order, "placed", 0.0, t)

        trades = self._lob.place_limit_order(order, t)

        filled_qty = order.qty - order.remaining_qty
        aggressor_reports: list[ExecutionReport] = []

        if filled_qty > 0:
            avg_price = (
                sum(tr.price * tr.qty for tr in trades) / sum(tr.qty for tr in trades)
                if trades else price
            )
            report_type = ReportType.FILL if order.remaining_qty <= 0 else ReportType.PARTIAL_FILL
            report = ExecutionReport(
                order_id=order.order_id,
                agent_id=agent_id,
                report_type=report_type,
                filled_qty=filled_qty,
                remaining_qty=order.remaining_qty,
                price=avg_price,
                timestamp=t,
            )
            aggressor_reports.append(report)
            self._record_order_event(order, report_type.value, filled_qty, t)

        # Track in environment registry if still resting
        if order.remaining_qty > 0:
            self._open_orders[agent_id].append(order)
            self._env_registry[order.order_id] = order

        # Route fill reports to resting agents and record trades
        self._process_trades(trades, t)
        self._enforce_book_depth(t)

        self._pending_reports[agent_id].extend(aggressor_reports)
        return trades, aggressor_reports

    def _place_market(self, agent_id: str, action: Action, t: float) -> tuple[list[Trade], list[ExecutionReport]]:
        if action.side is None or action.qty is None:
            return [], []
        # Parent-order completion may leave an odd-lot residual after partial
        # fills. Accept any positive market quantity so it is not stranded.
        if action.qty <= 0:
            return [], []

        # Market orders get a temporary order object (never enters book)
        import uuid
        order = Order.create(
            agent_id=agent_id,
            side=action.side,
            price=float("inf") if action.side == Side.BUY else 0.0,
            qty=action.qty,
            timestamp=t,
        )

        trades = self._lob.place_market_order(order, t)

        filled_qty = order.qty - order.remaining_qty
        aggressor_reports: list[ExecutionReport] = []

        if filled_qty > 0:
            avg_price = sum(tr.price * tr.qty for tr in trades) / sum(tr.qty for tr in trades)
            report_type = ReportType.FILL if order.remaining_qty <= 0 else ReportType.PARTIAL_FILL
            report = ExecutionReport(
                order_id=order.order_id,
                agent_id=agent_id,
                report_type=report_type,
                filled_qty=filled_qty,
                remaining_qty=order.remaining_qty,
                price=avg_price,
                timestamp=t,
            )
            aggressor_reports.append(report)

        self._process_trades(trades, t)
        self._pending_reports[agent_id].extend(aggressor_reports)
        return trades, aggressor_reports

    def _cancel(self, agent_id: str, action: Action, t: float) -> tuple[list[Trade], list[ExecutionReport]]:
        if action.order_id is None:
            return [], []

        cancelled = self._lob.cancel_order(action.order_id)
        reports: list[ExecutionReport] = []

        if cancelled:
            self._open_orders[agent_id] = [
                o for o in self._open_orders[agent_id] if o.order_id != action.order_id
            ]
            self._env_registry.pop(action.order_id, None)

            report = ExecutionReport(
                order_id=action.order_id,
                agent_id=agent_id,
                report_type=ReportType.CANCEL,
                filled_qty=0.0,
                remaining_qty=0.0,
                price=0.0,
                timestamp=t,
            )
            reports.append(report)
            self._order_records.append({
                "timestamp": t,
                "order_id": action.order_id,
                "agent_id": agent_id,
                "event_type": "cancelled",
                "side": "",
                "price": 0.0,
                "quantity": 0.0,
                "filled_qty": 0.0,
            })
            self._pending_reports[agent_id].extend(reports)

        return [], reports

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enforce_book_depth(self, t: float) -> None:
        """Cap internal storage without truncating the visible book depth."""
        removed = self._lob.trim_to_depth(self._cfg.book_storage_depth)
        for order in removed:
            remaining_qty = order.remaining_qty
            self._env_registry.pop(order.order_id, None)
            if order.agent_id == self.SEED_AGENT_ID:
                order.remaining_qty = 0.0
                continue

            self._open_orders[order.agent_id] = [
                live for live in self._open_orders[order.agent_id]
                if live.order_id != order.order_id
            ]
            order.remaining_qty = 0.0
            self._pending_reports[order.agent_id].append(ExecutionReport(
                order_id=order.order_id,
                agent_id=order.agent_id,
                report_type=ReportType.CANCEL,
                filled_qty=0.0,
                remaining_qty=0.0,
                price=order.price,
                timestamp=t,
            ))
            self._order_records.append({
                "timestamp": t,
                "order_id": order.order_id,
                "agent_id": order.agent_id,
                "event_type": "auto_cancelled_depth",
                "side": order.side.value,
                "price": order.price,
                "quantity": remaining_qty,
                "filled_qty": 0.0,
            })

    def _process_trades(self, trades: list[Trade], t: float) -> None:
        """Record trades and deliver fill reports to resting agents."""
        for trade in trades:
            self._trade_records.append({
                "timestamp":        trade.timestamp,
                "price":            trade.price,
                "quantity":         trade.qty,
                "aggressor_side":   trade.aggressor_side.value,
                "buyer_agent_id":   trade.buyer_agent_id,
                "seller_agent_id":  trade.seller_agent_id,
            })

            resting_agent_id = (
                trade.seller_agent_id
                if trade.aggressor_side == Side.BUY
                else trade.buyer_agent_id
            )

            if resting_agent_id == self.SEED_AGENT_ID:
                continue

            resting_order = self._env_registry.get(trade.resting_order_id)
            if resting_order is not None:
                remaining = resting_order.remaining_qty
                report_type = ReportType.FILL if remaining <= 0 else ReportType.PARTIAL_FILL
                if remaining <= 0:
                    self._env_registry.pop(trade.resting_order_id, None)
            else:
                remaining = 0.0
                report_type = ReportType.FILL

            self._pending_reports[resting_agent_id].append(ExecutionReport(
                order_id=trade.resting_order_id,
                agent_id=resting_agent_id,
                report_type=report_type,
                filled_qty=trade.qty,
                remaining_qty=remaining,
                price=trade.price,
                timestamp=t,
            ))

    def _snap(self, price: float) -> float:
        """Round price to the nearest valid tick multiple."""
        tick = self._cfg.tick_size
        return round(round(price / tick) * tick, 10)

    def _record_order_event(self, order: Order, event_type: str, filled_qty: float, t: float) -> None:
        self._order_records.append({
            "timestamp":  t,
            "order_id":   order.order_id,
            "agent_id":   order.agent_id,
            "event_type": event_type,
            "side":       order.side.value,
            "price":      order.price,
            "quantity":   order.qty,
            "filled_qty": filled_qty,
        })
