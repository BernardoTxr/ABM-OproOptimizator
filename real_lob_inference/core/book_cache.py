from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .book_types import BookEntry, BookMBPLevelSnapshot, BookMBPSnapshot, BookOrderSnapshot
from .time_utils import current_hw_timestamp, format_date_time_from_timestamp

@dataclass
class BookSide:
    """One side of the book (buy or sell) - position-based like Rust Vec"""
    entries: List[BookEntry] = field(default_factory=list)

    def insert(self, pos: int, entry: BookEntry) -> None:
        """Insert entry at position (1-indexed like Rust, converted to 0-indexed)"""
        # pos is 1-indexed from B3, convert to 0-indexed
        self.entries.insert(pos - 1, entry)

    def remove(self, pos: int) -> None:
        """Remove entry at position (1-indexed like Rust)"""
        if 0 < pos <= len(self.entries):
            del self.entries[pos - 1]

    def change(self, pos: int, new_size: int, new_px: Optional[float] = None) -> None:
        """Update existing entry at position (update action 1)"""
        if 0 < pos <= len(self.entries):
            entry = self.entries[pos - 1]
            entry.size = new_size
            if new_px is not None:
                entry.px = new_px


    def delete_thru(self, pos: int) -> None:
        """Delete from position to end (update action 3)"""
        if 0 < pos <= len(self.entries):
            # Delete from pos-1 onwards
            del self.entries[pos - 1 :]

    def delete_from(self, pos: int, book_inverted: bool) -> None:
        """Delete from position onwards or delete up to position (update action 4)"""
        if book_inverted:
            # If inverted: delete from pos to end
            if 0 < pos <= len(self.entries):
                del self.entries[pos - 1 :]
        else:
            # If not inverted: delete from start to pos (inclusive)
            if 0 < pos <= len(self.entries):
                del self.entries[:pos]

    def overlay(self, pos: int, px_value: Optional[float], new_size: Optional[int]) -> None:
        """Update or delete based on price (update action 5)"""
        if px_value is not None and px_value > 0:
            # Update or insert at position
            if 0 < pos <= len(self.entries):
                self.entries[pos - 1].px = px_value
                if new_size is not None:
                    self.entries[pos - 1].size = new_size
            elif pos == len(self.entries) + 1:
                # Insert at end if position is just beyond current length
                entry = BookEntry(px=px_value, size=new_size or 0)
                self.entries.append(entry)
        elif px_value == 0:
            # Delete at position
            if 0 < pos <= len(self.entries):
                del self.entries[pos - 1]

    def clear(self) -> None:
        """Clear all entries"""
        self.entries.clear()

    def get_top_n(self, n: int) -> Tuple[List[BookEntry], Dict[float, int]]:
        """Get entries spanning top N price levels and aggregated size by price (MBP)."""
        if n <= 0:
            return [], {}

        entries: List[BookEntry] = []
        price_levels: Dict[float, int] = {}

        for entry in self.entries:
            px = entry.px
            if px not in price_levels and len(price_levels) >= n:
                break

            entries.append(entry)
            price_levels[px] = price_levels.get(px, 0) + entry.size

        return entries, price_levels


@dataclass
class InstrumentBook:
    """Complete order book state for a single instrument (Book from Rust)"""
    security_id: int
    price_exponent: int
    depth: int
    bids: BookSide = field(default_factory=BookSide)
    offers: BookSide = field(default_factory=BookSide)
    last_feed_type: str = ""
    last_sequence_number: int = 0
    last_sending_time: int = 0
    last_update_action: int = 0
    last_rpt_seq: int = 0
    last_transact_time: int = 0
    snapshot_sequence_id: int = 0
    book_inverted: bool = False  # Flag for inverted book ordering

    def reset(self) -> None:
        """Clear entire book"""
        self.bids.clear()
        self.offers.clear()

    def clear_book(self) -> None:
        """Alias for reset() - clear entire book (Type J clearing message)"""
        self.reset()


    def apply_update_action(
        self,
        side: str,
        upd_act: int,
        pos: Optional[int],
        px: Optional[float],
        size: Optional[int],
    ) -> None:
        """Apply one of the six Standard Event Schema update actions (0-5)."""
        normalized_side = side.lower()
        if normalized_side == "bid":
            book_side = self.bids
        elif normalized_side in {"offer", "ask"}:
            book_side = self.offers
        else:
            raise ValueError(f"Unsupported side '{side}'")

        if upd_act == 0:  # Add
            if pos is not None and px is not None and size is not None:
                entry = BookEntry(px=float(px), size=size)
                book_side.insert(pos, entry)

        elif upd_act == 1:  # Change
            if pos is not None and size is not None:
                book_side.change(pos, size, px if px is not None else None)

        elif upd_act == 2:  # Delete
            if pos is not None:
                book_side.remove(pos)

        elif upd_act == 3:  # Delete Thru (delete from position to end)
            if pos is not None:
                book_side.delete_thru(pos)

        elif upd_act == 4:  # Delete From
            if pos is not None:
                book_side.delete_from(pos, self.book_inverted)

        elif upd_act == 5:  # Overlay (conditional update/delete)
            if pos is not None:
                book_side.overlay(pos, px, size)

    def update_metadata(
        self,
        *,
        feed_type: Optional[str] = None,
        sequence_number: Optional[int] = None,
        sending_time: Optional[int] = None,
        update_action: Optional[int] = None,
        rpt_seq: Optional[int] = None,
        transact_time: Optional[int] = None,
    ) -> None:
        """Update book metadata"""
        if feed_type is not None:
            self.last_feed_type = feed_type
        if sequence_number is not None:
            self.last_sequence_number = sequence_number
        if sending_time is not None:
            self.last_sending_time = sending_time
        if update_action is not None:
            self.last_update_action = update_action
        if rpt_seq is not None:
            self.last_rpt_seq = rpt_seq
        if transact_time is not None:
            self.last_transact_time = transact_time

    def to_snapshot(self) -> BookMBPSnapshot:
        """Generate MBP (aggregated by price) and MBO (individual orders) snapshot"""
        buy_entries: List[BookMBPLevelSnapshot] = []
        sell_entries: List[BookMBPLevelSnapshot] = []
        buy_orders: List[BookOrderSnapshot] = []
        sell_orders: List[BookOrderSnapshot] = []

        # Process buy side (bids) - reverse order for display
        buy_entry_list, buy_price_agg = self.bids.get_top_n(self.depth)
        for px_value in sorted(buy_price_agg.keys(), reverse=True):
            agg_size = buy_price_agg[px_value]
            buy_entries.append(BookMBPLevelSnapshot(px=float(px_value), size=agg_size))

        # Collect individual buy orders for MBO
        for idx, entry in enumerate(buy_entry_list, start=1):
            if len(buy_orders) < self.depth * 10:  # Limit MBO entries
                buy_orders.append(
                    BookOrderSnapshot(
                        px=float(entry.px),
                        size=entry.size,
                        position=idx,
                    )
                )

        # Process sell side (offers)
        sell_entry_list, sell_price_agg = self.offers.get_top_n(self.depth)
        for px_value in sorted(sell_price_agg.keys()):
            agg_size = sell_price_agg[px_value]
            sell_entries.append(BookMBPLevelSnapshot(px=float(px_value), size=agg_size))

        # Collect individual sell orders for MBO
        for idx, entry in enumerate(sell_entry_list, start=1):
            if len(sell_orders) < self.depth * 10:
                sell_orders.append(
                    BookOrderSnapshot(
                        px=float(entry.px),
                        size=entry.size,
                        position=idx,
                    )
                )

        hw_timestamp, default_date, default_time = current_hw_timestamp()

        if self.last_transact_time:
            _, date_str, time_str = format_date_time_from_timestamp(self.last_transact_time)
        else:
            date_str, time_str = default_date, default_time

        return BookMBPSnapshot(
            feed_type=self.last_feed_type,
            template_id=1001,
            sec_id=self.security_id,
            sen_tim=self.last_sending_time,
            hw_mke_tim=hw_timestamp,
            time=time_str,
            date=date_str,
            seq_num=self.last_sequence_number,
            upd_act=self.last_update_action,
            snapshot_sequence_id=self.snapshot_sequence_id,
            rpt_seq=self.last_rpt_seq,
            buy_entries=buy_entries,
            sell_entries=sell_entries,
            buy_orders=buy_orders,
            sell_orders=sell_orders,
        )

