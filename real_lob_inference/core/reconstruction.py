from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from .book_cache import InstrumentBook
from .book_types import BookEvent, BookMBPSnapshot
from .utils import normalize_event_dataframe, validate_event_dataframe


class ReconstructionError(RuntimeError):
    """Raised when order book reconstruction cannot proceed."""


def reconstruct_from_parquet(path: str, depth: int = 10) -> BookMBPSnapshot:
    dataframe = pd.read_parquet(path)
    return reconstruct_from_dataframe(dataframe, depth=depth)


def iter_book_events_from_parquet(path: str, depth: int = 10) -> Iterator[BookEvent]:
    dataframe = pd.read_parquet(path)
    yield from iter_book_events_from_dataframe(dataframe, depth=depth)


def reconstruct_from_dataframe(
    dataframe: pd.DataFrame,
    depth: int = 10,
    *,
    security_id: Optional[int] = None,
) -> BookMBPSnapshot:
    snapshot: Optional[BookMBPSnapshot] = None
    for event in iter_book_events_from_dataframe(
        dataframe,
        depth=depth,
        security_id=security_id,
    ):
        snapshot = event.snapshot
    if snapshot is None:
        raise ReconstructionError("No events produced from dataframe")
    return snapshot


def iter_book_events_from_dataframe(
    dataframe: pd.DataFrame,
    depth: int = 10,
    *,
    security_id: Optional[int] = None,
) -> Iterator[BookEvent]:
    if dataframe.empty:
        raise ReconstructionError("DataFrame is empty")

    normalized = normalize_event_dataframe(dataframe)
    validate_event_dataframe(normalized)

    sec_id = security_id if security_id is not None else _extract_security_id(normalized)
    instrument_book = InstrumentBook(
        security_id=sec_id,
        price_exponent=0,
        depth=depth,
    )

    ordered = normalized.sort_values(["rpt_seq", "pos"], kind="mergesort", na_position="last")
    ordered = ordered.reset_index(drop=True)

    # Pre-compute column index map for fast positional access from itertuples
    col_names = list(ordered.columns)
    _ci = {name: idx for idx, name in enumerate(col_names)}

    pending_trade_left = 0

    for row_index, row in enumerate(ordered.itertuples(index=False, name=None)):
        # Fast field extraction via positional index
        side_raw = row[_ci["side"]] if "side" in _ci else None
        upd_act_raw = row[_ci["upd_act"]] if "upd_act" in _ci else None
        size_raw = row[_ci["size"]] if "size" in _ci else None
        pos_raw = row[_ci["pos"]] if "pos" in _ci else None
        px_raw = row[_ci["px"]] if "px" in _ci else None
        rpt_seq_raw = row[_ci["rpt_seq"]] if "rpt_seq" in _ci else None
        msg_tpe_raw = row[_ci["msg_tpe"]] if "msg_tpe" in _ci else None
        dat_tim_raw = row[_ci["dat_tim"]] if "dat_tim" in _ci else None

        normalized_side = _normalize_side(side_raw)
        update_action = _safe_int(upd_act_raw)
        size = _safe_int(size_raw)
        position = _safe_int(pos_raw)
        price = _safe_float(px_raw)

        if normalized_side == "trade":
            trade_size = size or 0
            if trade_size < 0:
                raise ReconstructionError("Negative trade size encountered during reconstruction")
            if trade_size > 0:
                pending_trade_left += trade_size
            trade_consumed_candidate = 0
        else:
            trade_consumed_candidate = 0
            if pending_trade_left > 0 and normalized_side in {"bid", "ask"}:
                if update_action in {1, 2, 3, 4, 5}:
                    trade_consumed_candidate = _compute_trade_consumption(
                        instrument_book,
                        normalized_side,
                        update_action,
                        position,
                        size,
                        price,
                    )

        applied, reason = _apply_standard_event(
            instrument_book,
            normalized_side,
            update_action,
            position,
            size,
            price,
        )

        trade_consumed = 0
        if normalized_side != "trade" and applied and pending_trade_left > 0 and trade_consumed_candidate > 0:
            trade_consumed = min(trade_consumed_candidate, pending_trade_left)
            pending_trade_left -= trade_consumed

        if normalized_side == "trade" and not applied and update_action not in {0, 1, 2, 3, 4, 5, None}:
            # Ensure trade events do not mutate pending state unexpectedly
            trade_consumed = 0

        seq_num = _safe_int(rpt_seq_raw)
        timestamp_ns = _timestamp_to_ns(dat_tim_raw)

        instrument_book.update_metadata(
            feed_type=str(msg_tpe_raw or ""),
            sequence_number=seq_num,
            sending_time=timestamp_ns,
            update_action=update_action if applied else None,
            rpt_seq=seq_num,
            transact_time=timestamp_ns,
        )
        instrument_book.snapshot_sequence_id += 1

        snapshot = instrument_book.to_snapshot()

        # Build payload dict once at yield time (avoids per-row dict + Series allocation)
        payload: Dict[str, object] = {}
        for ci_name, ci_idx in _ci.items():
            payload[ci_name] = _to_python_native(row[ci_idx])
        payload["trade_consumed_size"] = trade_consumed
        payload["left_size_traded"] = pending_trade_left

        yield BookEvent(
            row_index=row_index,
            seq_num=seq_num,
            message_type=str(msg_tpe_raw or ""),
            feed=str(side_raw or ""),
            applied=applied,
            reason=reason,
            data=payload,
            snapshot=snapshot,
        )

def _compute_trade_consumption(
    book: InstrumentBook,
    side: Optional[str],
    update_action: Optional[int],
    position: Optional[int],
    size: Optional[int],
    price: Optional[float],
) -> int:
    if side not in {"bid", "ask"}:
        return 0
    if update_action not in {1, 2, 3, 4, 5}:
        return 0

    book_side = _select_book_side(book, side)
    if book_side is None:
        return 0

    if position is None or position <= 0:
        return 0

    if update_action == 1:
        entry = _get_book_entry(book_side, position)
        if entry is None or size is None:
            return 0
        return max(entry.size - size, 0)

    if update_action == 2:
        entry = _get_book_entry(book_side, position)
        return entry.size if entry is not None else 0

    if update_action == 3:
        return _sum_book_entries(book_side, position - 1, len(book_side.entries))

    if update_action == 4:
        if book.book_inverted:
            return _sum_book_entries(book_side, position - 1, len(book_side.entries))
        return _sum_book_entries(book_side, 0, position)

    if update_action == 5:
        entry = _get_book_entry(book_side, position)
        if price is not None and price == 0:
            return entry.size if entry is not None else 0
        if entry is None or size is None:
            return 0
        return max(entry.size - size, 0)

    return 0


def _select_book_side(book: InstrumentBook, side: str):
    if side == "bid":
        return book.bids
    if side == "ask":
        return book.offers
    return None


def _get_book_entry(book_side, position: int):
    index = position - 1
    if index < 0:
        return None
    entries = getattr(book_side, "entries", [])
    if index >= len(entries):
        return None
    return entries[index]


def _sum_book_entries(book_side, start_index: int, end_index: int) -> int:
    entries = getattr(book_side, "entries", [])
    if not entries:
        return 0

    start = max(0, start_index)
    stop = min(len(entries), max(start, end_index))

    total = 0
    for idx in range(start, stop):
        total += entries[idx].size
    return total


def _apply_standard_event(
    book: InstrumentBook,
    side: Optional[str],
    update_action: Optional[int],
    position: Optional[int],
    size: Optional[int],
    price: Optional[float],
) -> Tuple[bool, Optional[str]]:
    if side is None:
        return False, "Unsupported side"

    if side == "trade":
        return False, "Trade event (no book update)"

    if update_action not in {0, 1, 2, 3, 4, 5}:
        return False, "Unsupported update action"

    if position is None or position <= 0:
        return False, "Missing book position"

    if update_action == 0 and (price is None or size is None):
        return False, "Add action requires price and size"

    if update_action == 1 and size is None:
        return False, "Change action requires size"

    book.apply_update_action(
        side=side,
        upd_act=update_action,
        pos=position,
        px=price,
        size=size,
    )

    return True, None


def _extract_security_id(dataframe: pd.DataFrame) -> int:
    if "sec_id" in dataframe.columns and dataframe["sec_id"].notna().any():
        return int(dataframe["sec_id"].dropna().iloc[0])
    return 0


def _normalize_side(value: Optional[object]) -> Optional[str]:
    if value is None:
        return None
    side = str(value).lower()
    if side == "bid":
        return "bid"
    if side in {"offer", "ask"}:
        return "ask"
    if side == "trade":
        return "trade"
    return None


def _safe_int(value: object) -> Optional[int]:
    """Convert a pandas nullable Int64 / numpy int / float value to plain int."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return int(round(value))
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> Optional[float]:
    """Convert a pandas nullable Float64 / numpy float value to plain float."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, (int, np.integer)):
        return float(value)
    try:
        result = float(value)  # type: ignore[arg-type]
        if np.isnan(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _to_python_native(value: object) -> Optional[object]:
    """Convert numpy/pandas scalars to native Python types for payload dict."""
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            return value.tz_localize("UTC")
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return None if np.isnan(v) else v
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _timestamp_to_ns(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        ts = value.tz_convert("UTC") if value.tzinfo else value.tz_localize("UTC")
    else:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    return int(ts.value)
