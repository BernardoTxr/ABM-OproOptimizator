from __future__ import annotations

from datetime import datetime, timezone
from typing import Tuple


def current_hw_timestamp() -> Tuple[int, str, str]:
    now = datetime.now(timezone.utc)
    timestamp = int(now.strftime("%Y%m%d%H%M%S%f")[:17])
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S%f")[:9]
    return timestamp, date_str, time_str


def format_date_time_from_timestamp(timestamp_ns: int) -> Tuple[int, str, str]:
    dt = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc)
    timestamp = int(dt.strftime("%Y%m%d%H%M%S%f")[:17])
    date_str = dt.strftime("%Y%m%d")
    time_str = dt.strftime("%H%M%S%f")[:9]
    return timestamp, date_str, time_str
