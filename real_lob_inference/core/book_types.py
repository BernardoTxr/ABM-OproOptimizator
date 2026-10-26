from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# Individual price level entry tracked during reconstruction
@dataclass
class BookEntry:
    """Individual order/entry in the book."""
    px: float
    size: int


# Snapshot aggregated by price level (MBP - Market by Price)
@dataclass(frozen=True)
class BookMBPLevelSnapshot:
    px: float
    size: int


# Snapshot with individual orders (MBO - Market by Order)
@dataclass(frozen=True)
class BookOrderSnapshot:
    px: float
    size: int
    position: int


# Complete book snapshot output
@dataclass(frozen=True)
class BookMBPSnapshot:
    feed_type: str
    template_id: int
    sec_id: int
    sen_tim: int
    hw_mke_tim: int
    time: str
    date: str
    seq_num: int
    upd_act: int
    snapshot_sequence_id: int
    rpt_seq: int
    buy_entries: List[BookMBPLevelSnapshot]
    sell_entries: List[BookMBPLevelSnapshot]
    buy_orders: List[BookOrderSnapshot]
    sell_orders: List[BookOrderSnapshot]


# Event output during reconstruction
@dataclass
class BookEvent:
    row_index: int
    seq_num: Optional[int]
    message_type: Optional[str]
    feed: Optional[str]
    applied: bool
    reason: Optional[str]
    data: Dict[str, Any]
    snapshot: BookMBPSnapshot

    def summary(self) -> Dict[str, Any]:
        return {
            "row_index": self.row_index,
            "seq_num": self.seq_num,
            "message_type": self.message_type,
            "feed": self.feed,
            "applied": self.applied,
            "reason": self.reason,
            "data": self.data,
        }
