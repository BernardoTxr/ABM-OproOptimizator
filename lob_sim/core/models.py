from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


class ReportType(Enum):
    FILL = "fill"
    PARTIAL_FILL = "partial_fill"
    CANCEL = "cancel"


class ActionType(Enum):
    PLACE_LIMIT = "place_limit"
    PLACE_MARKET = "place_market"
    CANCEL = "cancel"


@dataclass
class Order:
    order_id: str
    agent_id: str
    side: Side
    price: float
    qty: float
    timestamp: float
    is_seed: bool = False
    remaining_qty: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.remaining_qty = self.qty

    @classmethod
    def create(
        cls,
        agent_id: str,
        side: Side,
        price: float,
        qty: float,
        timestamp: float,
        order_id: Optional[str] = None,
        is_seed: bool = False,
    ) -> "Order":
        return cls(
            order_id=order_id or str(uuid.uuid4()),
            agent_id=agent_id,
            side=side,
            price=price,
            qty=qty,
            timestamp=timestamp,
            is_seed=is_seed,
        )


@dataclass
class Trade:
    trade_id: str
    aggressor_order_id: str
    resting_order_id: str
    buyer_agent_id: str
    seller_agent_id: str
    price: float
    qty: float
    timestamp: float
    aggressor_side: Side

    @classmethod
    def create(
        cls,
        aggressor_order_id: str,
        resting_order_id: str,
        buyer_agent_id: str,
        seller_agent_id: str,
        price: float,
        qty: float,
        timestamp: float,
        aggressor_side: Side,
    ) -> "Trade":
        return cls(
            trade_id=str(uuid.uuid4()),
            aggressor_order_id=aggressor_order_id,
            resting_order_id=resting_order_id,
            buyer_agent_id=buyer_agent_id,
            seller_agent_id=seller_agent_id,
            price=price,
            qty=qty,
            timestamp=timestamp,
            aggressor_side=aggressor_side,
        )


@dataclass
class ExecutionReport:
    order_id: str
    agent_id: str
    report_type: ReportType
    filled_qty: float
    remaining_qty: float
    price: float
    timestamp: float


@dataclass
class Action:
    """
    Represents an action an agent can submit to the environment.

    For PLACE_LIMIT: side, price, qty required; order_id is an optional
    proposed order ID (the environment uses it if provided, otherwise
    generates a UUID). This lets archetypes track their own orders by ID.

    For PLACE_MARKET: side, qty required.

    For CANCEL: order_id of the resting order to cancel.
    """
    action_type: ActionType
    side: Optional[Side] = None
    price: Optional[float] = None
    qty: Optional[float] = None
    order_id: Optional[str] = None


@dataclass
class MarketObservation:
    timestamp: float
    bid_levels: list[tuple[float, float]]   # (price, qty), lob_depth levels, best first
    ask_levels: list[tuple[float, float]]   # (price, qty), lob_depth levels, best first
    execution_reports: list[ExecutionReport]
    open_orders: list[Order]                # this agent's currently resting orders
