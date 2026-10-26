"""Compute stylized facts from real BTG-OBD order-book event data.

Produces the same eight-key dict[str, np.ndarray] format as
``lob_sim.loss.facts.compute_facts`` so that both outputs are directly
comparable by the loss engine.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from lob_sim.loss.facts import acf_abs_returns

logger = logging.getLogger(__name__)

_REAL_FACTS_CACHE_VERSION = 6


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_real_facts(
    symbol: str,
    dates: list[str],
    config: dict,
    cache_path: Optional[str] = None,
) -> dict[str, np.ndarray]:
    """Extract stylized facts from real BTG-OBD events, pooling across dates.

    Parameters
    ----------
    symbol:
        Ticker (e.g. ``'ASAI3'``).
    dates:
        List of trading dates as ``YYYYMMDD`` strings.
    config:
        Parsed ``real_analysis.yaml`` dict.  Required keys:
        ``data_dir``, ``depth``, ``lob_depth``,
        ``window_start_utc``, ``window_duration_hours``.
    cache_path:
        Optional ``.npz`` path.  If the file exists and ``dates`` is the
        same set used to build it, the cached arrays are returned instead
        of recomputing.  Pass ``None`` to skip caching.

    Returns
    -------
    dict with keys identical to :func:`lob_sim.loss.facts.compute_facts`:
    ``log_returns``, ``acf_abs_returns``, ``spread``, ``depth_bid``,
    ``depth_ask``, ``imbalance``, ``trade_sizes``, ``inter_arrival``.
    """
    # --- cache hit ---
    if cache_path is not None:
        cached = _try_load_cache(cache_path, symbol, dates, config)
        if cached is not None:
            logger.info("Real facts loaded from cache: %s", cache_path)
            return cached

    data_dir = Path(config["data_dir"])
    depth = int(config.get("depth", 10))
    lob_depth = int(config.get("lob_depth", 5))
    windows = resolve_measurement_windows(config)
    replay_end = windows[-1][1]
    depth_levels = list(range(1, lob_depth + 1))

    # Per-date accumulators
    all_log_returns: list[np.ndarray] = []
    all_spread: list[np.ndarray] = []
    all_imbalance: list[np.ndarray] = []
    all_trade_sizes: list[np.ndarray] = []
    all_trade_times_sec: list[np.ndarray] = []
    all_depth_bid: list[np.ndarray] = []
    all_depth_ask: list[np.ndarray] = []
    all_book_coverage: list[float] = []
    all_level_prices: list[pd.DataFrame] = []

    for date in dates:
        events_path = data_dir / symbol / date / f"d{depth}" / "events.parquet"
        if not events_path.exists():
            logger.warning("Events file not found — skipping: %s", events_path)
            continue

        df_raw = pd.read_parquet(events_path)
        # Reconstruct the book only once per date, then sample every configured
        # intraday window from that common event-state path.
        try:
            snapshot_records = _extract_snapshot_records(
                df_raw, date, depth, depth_levels, replay_end
            )
        except Exception as exc:
            logger.warning("Snapshot extraction failed for %s / %s: %s", symbol, date, exc)
            continue

        for window_start, window_end in windows:
            df_measurement = _filter_measurement_window(
                df_raw, date, window_start, window_end
            )
            if df_measurement.empty:
                logger.warning(
                    "No events in window for %s / %s / %s", symbol, date, window_start
                )
                continue

            # Trades are kept window-local so an overnight or window-boundary
            # gap can never enter the inter-arrival distribution.
            trade_mask = _get_trade_mask(df_measurement)
            df_trades = df_measurement[trade_mask]
            if not df_trades.empty and "size" in df_trades.columns:
                all_trade_sizes.append(df_trades["size"].values.astype(float))
            if not df_trades.empty and "dat_tim" in df_trades.columns:
                ts_sec = _timestamps_to_seconds(df_trades["dat_tim"])
                all_trade_times_sec.append(np.sort(ts_sec))

            snap_df = _sample_frame(
                snapshot_records, date, window_start, window_end
            )
            if snap_df.empty:
                continue

            two_sided = snap_df["midprice"].notna()
            all_book_coverage.append(float(two_sided.mean()))
            valid_snapshots = snap_df.loc[two_sided]
            if valid_snapshots.empty:
                continue

            mp = valid_snapshots["midprice"].replace(0, np.nan).dropna().values
            if len(mp) > 1:
                all_log_returns.append(np.diff(np.log(mp.clip(min=1e-10))))

            if "spread" in valid_snapshots.columns:
                spread = valid_snapshots["spread"].dropna().values
                spread = spread[spread > 0]
                if spread.size:
                    all_spread.append(spread)

            if "obi" in valid_snapshots.columns:
                all_imbalance.append(valid_snapshots["obi"].dropna().values)

            bid_depths, ask_depths = [], []
            previous_bid = np.zeros(len(valid_snapshots), dtype=float)
            previous_ask = np.zeros(len(valid_snapshots), dtype=float)
            for lvl in depth_levels:
                bid_col, ask_col = f"depth_{lvl}_bid", f"depth_{lvl}_ask"
                if bid_col in snap_df.columns:
                    cumulative = valid_snapshots[bid_col].to_numpy(dtype=float)
                    bid_depths.append(float(np.mean(cumulative - previous_bid)))
                    previous_bid = cumulative
                if ask_col in snap_df.columns:
                    cumulative = valid_snapshots[ask_col].to_numpy(dtype=float)
                    ask_depths.append(float(np.mean(cumulative - previous_ask)))
                    previous_ask = cumulative
            if bid_depths:
                all_depth_bid.append(np.array(bid_depths))
            if ask_depths:
                all_depth_ask.append(np.array(ask_depths))
            price_columns = [
                column for column in snap_df.columns
                if column.startswith("bid_l") or column.startswith("ask_l")
            ]
            if price_columns:
                all_level_prices.append(
                    valid_snapshots[["elapsed_seconds", *price_columns]].copy()
                )

    # ---- assemble final fact dict ----
    facts = _assemble_facts(
        all_log_returns,
        all_spread,
        all_imbalance,
        all_trade_sizes,
        all_trade_times_sec,
        all_depth_bid,
        all_depth_ask,
        all_book_coverage,
    )
    if all_level_prices:
        level_prices = (
            pd.concat(all_level_prices, ignore_index=True)
            .groupby("elapsed_seconds", as_index=False)
            .mean(numeric_only=True)
        )
        for column in level_prices.columns:
            facts[f"level_price_{column}"] = level_prices[column].to_numpy(dtype=float)

    if cache_path is not None:
        _save_cache(cache_path, facts, symbol, dates, config)

    return facts


def compute_real_window_fact_panel(
    symbol: str,
    dates: list[str],
    config: dict,
) -> list[dict]:
    """Return one fact dictionary per date and intraday measurement window.

    Unlike :func:`compute_real_facts`, this function preserves the date/window
    sampling units needed by clustered SMM weighting.  Each raw date is read
    and reconstructed once, after which all configured windows are sampled
    from the same reconstructed state path.
    """
    data_dir = Path(config["data_dir"])
    depth = int(config.get("depth", 10))
    lob_depth = int(config.get("lob_depth", 5))
    windows = resolve_measurement_windows(config)
    replay_end = windows[-1][1]
    depth_levels = list(range(1, lob_depth + 1))
    panel: list[dict] = []

    for date in dates:
        events_path = data_dir / symbol / date / f"d{depth}" / "events.parquet"
        if not events_path.exists():
            raise FileNotFoundError(f"Events file not found: {events_path}")
        raw = pd.read_parquet(events_path)
        records = _extract_snapshot_records(
            raw, date, depth, depth_levels, replay_end
        )
        for window_start, window_end in windows:
            facts = _compute_real_window_facts(
                raw,
                records,
                date,
                window_start,
                window_end,
                depth_levels,
            )
            panel.append({
                "date": str(date),
                "window_start_utc": window_start.isoformat(),
                "facts": facts,
            })
    return panel


def discover_dates(symbol: str, config: dict) -> list[str]:
    """Return all YYYYMMDD dates with event data available for *symbol*.

    Glob pattern: ``{data_dir}/{symbol}/*/d{depth}/events.parquet``.
    """
    data_dir = Path(config["data_dir"])
    depth = int(config.get("depth", 10))
    pattern = f"*/d{depth}/events.parquet"
    dates = sorted(
        p.parent.parent.name
        for p in (data_dir / symbol).glob(pattern)
    )
    return dates


def resolve_date_split(
    available_dates: list[str], config: dict, split: str = "all"
) -> list[str]:
    """Resolve and validate a named date split from the real-data config."""
    if split == "all":
        return list(available_dates)
    configured = config.get("date_splits", {}).get(split)
    if configured is None:
        raise ValueError(f"Unknown or unconfigured real-data split: {split}")
    dates = [str(date) for date in configured]
    if not dates:
        raise ValueError(f"Real-data split '{split}' is empty")
    if len(dates) != len(set(dates)):
        raise ValueError(f"Real-data split '{split}' contains duplicate dates")
    missing = sorted(set(dates) - set(available_dates))
    if missing:
        raise ValueError(
            f"Real-data split '{split}' contains unavailable dates: {missing}"
        )
    return dates


def resolve_measurement_windows(
    config: dict,
) -> list[tuple[datetime.time, datetime.time]]:
    """Resolve fixed or explicitly tiled intraday measurement windows.

    Existing configurations with only ``window_start_utc`` retain their exact
    behavior.  A multi-window configuration declares ``window_starts_utc``;
    every window is checked against the configured safe session interval.
    """
    duration = float(config["window_duration_hours"])
    if duration <= 0:
        raise ValueError("window_duration_hours must be positive")
    starts_raw = config.get("window_starts_utc")
    if starts_raw is None:
        starts_raw = [config["window_start_utc"]]
    if not isinstance(starts_raw, list) or not starts_raw:
        raise ValueError("window_starts_utc must be a non-empty list")

    starts = [_parse_time(str(value)) for value in starts_raw]
    if len(starts) != len(set(starts)):
        raise ValueError("window_starts_utc contains duplicate times")
    if starts != sorted(starts):
        raise ValueError("window_starts_utc must be sorted")

    windows = [(start, _add_hours(start, duration)) for start in starts]
    for (_, previous_end), (next_start, _) in zip(windows, windows[1:]):
        if _seconds_of_day(next_start) < _seconds_of_day(previous_end):
            raise ValueError("Measurement windows must not overlap")
    if "session_start_utc" in config and "session_end_utc" in config:
        session_start = _seconds_of_day(_parse_time(config["session_start_utc"]))
        session_end = _seconds_of_day(_parse_time(config["session_end_utc"]))
        if session_end <= session_start:
            raise ValueError("Overnight session intervals are not supported")
        for start, end in windows:
            start_seconds = _seconds_of_day(start)
            end_seconds = _seconds_of_day(end)
            if start_seconds < session_start or end_seconds > session_end:
                raise ValueError(
                    f"Measurement window {start}--{end} lies outside "
                    f"{config['session_start_utc']}--{config['session_end_utc']}"
                )
    return windows


def compute_real_level_prices(
    symbol: str,
    dates: list[str],
    config: dict,
    levels: tuple[int, ...] = (1, 3, 5),
    real_facts: Optional[dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """Mean intraday level-price paths aligned by elapsed session second."""
    if real_facts is not None:
        columns = ["elapsed_seconds", *[
            f"{side}_l{level}" for side in ("bid", "ask") for level in levels
        ]]
        cached_columns = {
            column: real_facts.get(f"level_price_{column}") for column in columns
        }
        if all(value is not None for value in cached_columns.values()):
            return pd.DataFrame(cached_columns)

    data_dir = Path(config["data_dir"])
    depth = int(config.get("depth", 10))
    windows = resolve_measurement_windows(config)
    replay_end = windows[-1][1]
    per_date: list[pd.DataFrame] = []

    for date in dates:
        events_path = data_dir / symbol / date / f"d{depth}" / "events.parquet"
        if not events_path.exists():
            continue
        raw = pd.read_parquet(events_path)
        records = _extract_snapshot_records(
            raw, date, depth, list(levels), replay_end
        )
        for window_start, window_end in windows:
            day = _sample_frame(records, date, window_start, window_end)
            if day.empty:
                continue
            columns = [
                "elapsed_seconds",
                *[
                    f"{side}_l{level}"
                    for side in ("bid", "ask")
                    for level in levels
                ],
            ]
            per_date.append(day[columns])

    if not per_date:
        return pd.DataFrame()
    return (
        pd.concat(per_date, ignore_index=True)
        .groupby("elapsed_seconds", as_index=False)
        .mean(numeric_only=True)
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_time(time_str: str) -> datetime.time:
    """Parse ``'HH:MM:SS'`` into :class:`datetime.time`."""
    return datetime.time.fromisoformat(time_str)


def _add_hours(t: datetime.time, hours: float) -> datetime.time:
    """Add *hours* to a :class:`datetime.time` (wraps at midnight)."""
    total_seconds = t.hour * 3600 + t.minute * 60 + t.second + int(hours * 3600)
    total_seconds %= 86400
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return datetime.time(h, m, s)


def _seconds_of_day(value: datetime.time) -> int:
    return value.hour * 3600 + value.minute * 60 + value.second


def _filter_time_window(
    df: pd.DataFrame,
    start: datetime.time,
    end: datetime.time,
) -> pd.DataFrame:
    """Keep rows where ``dat_tim`` falls in ``[start, end)``."""
    if "dat_tim" not in df.columns or df.empty:
        return df
    ts = pd.to_datetime(df["dat_tim"])
    tod = ts.dt.time
    mask = (tod >= start) & (tod < end)
    return df[mask].copy()


def _window_bounds(
    date: str,
    start: datetime.time,
    end: datetime.time,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return naive UTC timestamps matching the BTG-OBD ``dat_tim`` values."""
    day = pd.Timestamp(datetime.datetime.strptime(date, "%Y%m%d"))
    start_ts = day + pd.Timedelta(
        hours=start.hour, minutes=start.minute, seconds=start.second
    )
    end_ts = day + pd.Timedelta(hours=end.hour, minutes=end.minute, seconds=end.second)
    if end_ts <= start_ts:
        end_ts += pd.Timedelta(days=1)
    return start_ts, end_ts


def _filter_measurement_window(
    df: pd.DataFrame,
    date: str,
    start: datetime.time,
    end: datetime.time,
) -> pd.DataFrame:
    """Keep only rows in the exact requested calendar-date interval."""
    if "dat_tim" not in df.columns or df.empty:
        return df.iloc[0:0].copy()
    start_ts, end_ts = _window_bounds(date, start, end)
    ts = pd.to_datetime(df["dat_tim"], utc=True).dt.tz_localize(None)
    return df[(ts >= start_ts) & (ts < end_ts)].copy()


def _reconstruction_input(
    df: pd.DataFrame,
    date: str,
    window_end: datetime.time,
) -> pd.DataFrame:
    """Keep the full bootstrap snapshot and target-day events through the window.

    Snapshot rows carry the original timestamps of resting orders, often from
    earlier dates.  They must all be retained to initialize the book.  Only
    incremental rows from the requested date are replayed, including events
    before the measurement window so that the opening state is correct.
    """
    if df.empty or "dat_tim" not in df.columns:
        return df.copy()
    ts = pd.to_datetime(df["dat_tim"], utc=True).dt.tz_localize(None)
    day = pd.Timestamp(datetime.datetime.strptime(date, "%Y%m%d"))
    end_ts = day + pd.Timedelta(
        hours=window_end.hour, minutes=window_end.minute, seconds=window_end.second
    )
    if end_ts <= day:
        end_ts += pd.Timedelta(days=1)
    msg = df.get("msg_tpe", pd.Series("", index=df.index)).astype(str).str.lower()
    bootstrap = msg.isin({"snapshot", "w"})
    target_updates = (~bootstrap) & (ts >= day) & (ts < end_ts)
    return df[bootstrap | target_updates].copy()


def _get_trade_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask identifying trade events.

    Checks both ``side == 'trade'`` (normalised B3 UMDF schema) and
    ``msg_tpe`` containing the word *trade* (fallback for raw message types).
    """
    mask = pd.Series(False, index=df.index)
    if "side" in df.columns:
        mask |= df["side"].astype(str).str.lower() == "trade"
    if "msg_tpe" in df.columns:
        mask |= df["msg_tpe"].astype(str).str.lower().str.contains("trade", na=False)
    return mask


def _timestamps_to_seconds(series: pd.Series) -> np.ndarray:
    """Convert a series of timestamps to float seconds since epoch."""
    arr = pd.to_datetime(series).values.astype("datetime64[ns]").astype(np.int64)
    return arr.astype(float) / 1e9


def _compute_real_window_facts(
    raw: pd.DataFrame,
    snapshot_records: pd.DataFrame,
    date: str,
    window_start: datetime.time,
    window_end: datetime.time,
    depth_levels: list[int],
) -> dict[str, np.ndarray]:
    """Compute facts for one exact window from an already reconstructed date."""
    measurement = _filter_measurement_window(raw, date, window_start, window_end)
    trade_sizes = np.array([], dtype=float)
    trade_times = np.array([], dtype=float)
    if not measurement.empty:
        trades = measurement[_get_trade_mask(measurement)]
        if not trades.empty and "size" in trades:
            trade_sizes = trades["size"].to_numpy(dtype=float)
        if not trades.empty and "dat_tim" in trades:
            trade_times = np.sort(_timestamps_to_seconds(trades["dat_tim"]))

    sampled = _sample_frame(
        snapshot_records, date, window_start, window_end
    )
    if sampled.empty:
        return {
            "log_returns": np.array([], dtype=float),
            "acf_abs_returns": np.array([1.0]),
            "spread": np.array([], dtype=float),
            "depth_bid": np.array([], dtype=float),
            "depth_ask": np.array([], dtype=float),
            "imbalance": np.array([], dtype=float),
            "trade_sizes": trade_sizes,
            "inter_arrival": (
                np.diff(trade_times) if trade_times.size > 1 else np.array([], dtype=float)
            ),
            "book_coverage": np.array([0.0]),
        }

    two_sided = sampled["midprice"].notna()
    coverage = float(two_sided.mean())
    valid = sampled.loc[two_sided]
    mid = valid["midprice"].replace(0, np.nan).dropna().to_numpy(dtype=float)
    log_returns = (
        np.diff(np.log(mid.clip(min=1e-10)))
        if mid.size > 1 else np.array([], dtype=float)
    )
    spread = (
        valid["spread"].dropna().to_numpy(dtype=float)
        if "spread" in valid else np.array([], dtype=float)
    )
    spread = spread[spread > 0]
    imbalance = (
        valid["obi"].dropna().to_numpy(dtype=float)
        if "obi" in valid else np.array([], dtype=float)
    )

    bid_depth: list[float] = []
    ask_depth: list[float] = []
    previous_bid = np.zeros(len(valid), dtype=float)
    previous_ask = np.zeros(len(valid), dtype=float)
    for level in depth_levels:
        bid_column = f"depth_{level}_bid"
        ask_column = f"depth_{level}_ask"
        if bid_column in valid:
            cumulative = valid[bid_column].to_numpy(dtype=float)
            bid_depth.append(float(np.mean(cumulative - previous_bid)))
            previous_bid = cumulative
        if ask_column in valid:
            cumulative = valid[ask_column].to_numpy(dtype=float)
            ask_depth.append(float(np.mean(cumulative - previous_ask)))
            previous_ask = cumulative

    return {
        "log_returns": log_returns,
        "acf_abs_returns": acf_abs_returns(log_returns),
        "spread": spread,
        "depth_bid": np.asarray(bid_depth, dtype=float),
        "depth_ask": np.asarray(ask_depth, dtype=float),
        "imbalance": imbalance,
        "trade_sizes": trade_sizes,
        "inter_arrival": (
            # The simulator acts and records trades once per second. Quantize
            # empirical timestamps before differencing so SMM does not reward
            # an impossible match to sub-second timing.
            np.diff(np.floor(trade_times))
            if trade_times.size > 1 else np.array([], dtype=float)
        ),
        "book_coverage": np.array([coverage]),
    }


def _iter_non_bootstrap_events(df: pd.DataFrame, depth: int):
    """Reconstruct all rows while yielding only live incremental events."""
    from real_lob_inference.core.reconstruction import iter_book_events_from_dataframe

    for event in iter_book_events_from_dataframe(df, depth=depth):
        if event.message_type.lower() not in {"snapshot", "w"}:
            yield event


def _sample_frame(
    frame: pd.DataFrame,
    date: str,
    window_start: datetime.time,
    window_end: datetime.time,
) -> pd.DataFrame:
    """Last-observation-carried-forward sampling on an exact one-second grid."""
    if frame.empty or "timestamp" not in frame.columns:
        return pd.DataFrame()

    start_ts, end_ts = _window_bounds(date, window_start, window_end)
    start_utc = start_ts.tz_localize("UTC")
    end_utc = end_ts.tz_localize("UTC")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    source = frame.copy()
    source["timestamp"] = timestamps.dt.floor("s")
    source = source[source["timestamp"] < end_utc]
    if source.empty:
        return pd.DataFrame()

    source = source.sort_values("timestamp").groupby("timestamp", as_index=True).last()
    target_index = pd.date_range(start_utc, end_utc, freq="1s", inclusive="left")
    expanded_index = source.index.union(target_index).sort_values()
    if not np.any(source.index < start_utc):
        raise ValueError(
            f"No reconstructed two-sided book exists before {start_utc.isoformat()}"
        )
    sampled = source.reindex(expanded_index).ffill().reindex(target_index)
    sampled.index.name = "timestamp"
    sampled.insert(0, "elapsed_seconds", np.arange(len(sampled), dtype=float))
    return sampled.reset_index()


def _extract_sampled_snapshots(
    df_raw: pd.DataFrame,
    symbol: str,
    date: str,
    depth: int,
    depth_levels: list[int],
    window_start: datetime.time,
    window_end: datetime.time,
) -> pd.DataFrame:
    """Reconstruct a complete book and sample target-window facts each second."""
    extracted = _extract_snapshot_records(
        df_raw, date, depth, depth_levels, window_end
    )
    return _sample_frame(extracted, date, window_start, window_end)


def _extract_snapshot_records(
    df_raw: pd.DataFrame,
    date: str,
    depth: int,
    depth_levels: list[int],
    replay_end: datetime.time,
) -> pd.DataFrame:
    """Reconstruct one date once and retain event-level book snapshots."""
    replay = _reconstruction_input(df_raw, date, replay_end)
    max_depth = max(depth_levels)
    # The downstream measurement grid is one second. Retaining the last book
    # state per source second is therefore exactly equivalent to collecting
    # every event and later applying ``groupby(...).last()``, while avoiding a
    # large event-level memory buffer.
    records_by_second: dict[int, dict[str, object]] = {}
    for event in _iter_non_bootstrap_events(replay, depth):
        snapshot = event.snapshot
        bids = list(snapshot.buy_entries)
        asks = list(snapshot.sell_entries)
        bid_volume = sum(entry.size for entry in bids[:max_depth])
        ask_volume = sum(entry.size for entry in asks[:max_depth])
        total_volume = bid_volume + ask_volume
        two_sided = bool(bids and asks)
        row: dict[str, object] = {
            "timestamp": pd.Timestamp(snapshot.sen_tim, unit="ns", tz="UTC"),
            "midprice": (bids[0].px + asks[0].px) / 2.0 if two_sided else np.nan,
            "spread": asks[0].px - bids[0].px if two_sided else np.nan,
            "obi": (bid_volume - ask_volume) / total_volume if total_volume else 0.0,
        }
        for level in depth_levels:
            row[f"depth_{level}_bid"] = sum(entry.size for entry in bids[:level])
            row[f"depth_{level}_ask"] = sum(entry.size for entry in asks[:level])
            idx = level - 1
            row[f"bid_l{level}"] = float(bids[idx].px) if len(bids) > idx else np.nan
            row[f"ask_l{level}"] = float(asks[idx].px) if len(asks) > idx else np.nan
        records_by_second[int(snapshot.sen_tim // 1_000_000_000)] = row
    return pd.DataFrame.from_records(list(records_by_second.values()))


def _extract_sampled_level_prices(
    df_raw: pd.DataFrame,
    date: str,
    depth: int,
    levels: tuple[int, ...],
    window_start: datetime.time,
    window_end: datetime.time,
) -> pd.DataFrame:
    """Extract requested price levels from correctly reconstructed snapshots."""
    sampled = _extract_sampled_snapshots(
        df_raw, "", date, depth, list(levels), window_start, window_end
    )
    columns = [
        "elapsed_seconds",
        *[f"{side}_l{level}" for side in ("bid", "ask") for level in levels],
    ]
    return sampled[columns]


def extract_initial_snapshot(
    symbol: str,
    date: str,
    config: dict,
    lob_depth: int,
    window_start_utc: str | datetime.time | None = None,
) -> dict[str, list[list[float]]]:
    """Return the correctly reconstructed book at the analysis-window start."""
    data_dir = Path(config["data_dir"])
    raw_depth = int(config.get("depth", 10))
    path = data_dir / symbol / date / f"d{raw_depth}" / "events.parquet"
    raw = pd.read_parquet(path)
    if window_start_utc is None:
        window_start = resolve_measurement_windows(config)[0][0]
    elif isinstance(window_start_utc, datetime.time):
        window_start = window_start_utc
    else:
        window_start = _parse_time(str(window_start_utc))
    window_end = _add_hours(window_start, float(config["window_duration_hours"]))
    start_ts, _ = _window_bounds(date, window_start, window_end)
    start_ns = int(start_ts.tz_localize("UTC").value)
    replay = _reconstruction_input(raw, date, window_end)

    fallback = None
    for event in _iter_non_bootstrap_events(replay, raw_depth):
        snapshot = event.snapshot
        bids = list(snapshot.buy_entries)
        asks = list(snapshot.sell_entries)
        if not bids or not asks:
            continue
        candidate = {
            "bids": [[float(x.px), float(x.size)] for x in bids[:lob_depth]],
            "asks": [[float(x.px), float(x.size)] for x in asks[:lob_depth]],
        }
        if snapshot.sen_tim >= start_ns:
            if fallback is not None:
                return fallback
            break
        fallback = candidate
    if fallback is not None:
        return fallback
    raise ValueError(
        f"No usable two-sided snapshot exists before the window for {symbol} on {date}"
    )


def select_window_assignments(
    dates: list[str],
    windows: list[tuple[datetime.time, datetime.time]],
    n_scenarios: int,
    seed: int,
) -> list[tuple[str, datetime.time]]:
    """Select deterministic, time-stratified date/window assignments.

    Each intraday start receives either ``floor(n / n_windows)`` or
    ``ceil(n / n_windows)`` scenarios. Date usage is balanced globally and,
    within each clock stratum, every date is exhausted before any is repeated.
    """
    if not dates:
        raise ValueError("At least one date is required for window sampling")
    if not windows:
        raise ValueError("At least one measurement window is required")
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive")
    rng = np.random.default_rng(seed)
    base_count, remainder = divmod(n_scenarios, len(windows))
    assignments: list[tuple[str, datetime.time]] = []
    date_counts = {str(date): 0 for date in dates}
    for window_idx, (window_start, _) in enumerate(windows):
        required = base_count + (1 if window_idx < remainder else 0)
        selected: list[str] = []
        local_counts = {str(date): 0 for date in dates}
        while len(selected) < required:
            least_local = min(local_counts.values())
            eligible = [
                str(date) for date in dates
                if local_counts[str(date)] == least_local
            ]
            least_global = min(date_counts[date] for date in eligible)
            tied = [date for date in eligible if date_counts[date] == least_global]
            chosen = tied[int(rng.integers(len(tied)))]
            selected.append(chosen)
            local_counts[chosen] += 1
            date_counts[chosen] += 1
        assignments.extend((date, window_start) for date in selected)
    order = rng.permutation(len(assignments))
    return [assignments[index] for index in order]


def extract_initial_snapshots(
    symbol: str,
    assignments: list[tuple[str, datetime.time]],
    config: dict,
    lob_depth: int,
) -> list[dict[str, list[list[float]]]]:
    """Batch-extract opening books for date/window assignments.

    Event reconstruction is performed once per distinct date. The selected
    book is the last two-sided state strictly before the window boundary,
    avoiding look-ahead into the measured interval.
    """
    data_dir = Path(config["data_dir"])
    raw_depth = int(config.get("depth", 10))
    duration = float(config["window_duration_hours"])
    outputs: list[dict[str, list[list[float]]] | None] = [None] * len(assignments)
    grouped: dict[str, list[tuple[int, datetime.time]]] = {}
    for index, (date, window_start) in enumerate(assignments):
        grouped.setdefault(str(date), []).append((index, window_start))

    for date, indexed_starts in grouped.items():
        path = data_dir / symbol / date / f"d{raw_depth}" / "events.parquet"
        raw = pd.read_parquet(path)
        latest_start = max(start for _, start in indexed_starts)
        replay_end = _add_hours(latest_start, duration)
        replay = _reconstruction_input(raw, date, replay_end)
        targets = sorted(
            (
                int(_window_bounds(date, start, _add_hours(start, duration))[0]
                    .tz_localize("UTC").value),
                index,
            )
            for index, start in indexed_starts
        )
        target_idx = 0
        previous: dict[str, list[list[float]]] | None = None
        for event in _iter_non_bootstrap_events(replay, raw_depth):
            snapshot = event.snapshot
            bids = list(snapshot.buy_entries)
            asks = list(snapshot.sell_entries)
            if not bids or not asks:
                continue
            candidate = {
                "bids": [
                    [float(entry.px), float(entry.size)]
                    for entry in bids[:lob_depth]
                ],
                "asks": [
                    [float(entry.px), float(entry.size)]
                    for entry in asks[:lob_depth]
                ],
            }
            while target_idx < len(targets) and event.snapshot.sen_tim >= targets[target_idx][0]:
                _, output_idx = targets[target_idx]
                outputs[output_idx] = previous
                target_idx += 1
            previous = candidate
            if target_idx >= len(targets):
                break
        while target_idx < len(targets) and previous is not None:
            _, output_idx = targets[target_idx]
            outputs[output_idx] = previous
            target_idx += 1

    missing = [index for index, snapshot in enumerate(outputs) if snapshot is None]
    if missing:
        raise ValueError(f"No usable two-sided snapshots for assignments: {missing}")
    return [snapshot for snapshot in outputs if snapshot is not None]


def cached_initial_snapshots(
    symbol: str,
    assignments: list[tuple[str, datetime.time]],
    config: dict,
    lob_depth: int,
    cache_path: str,
) -> list[dict[str, list[list[float]]]]:
    """Load or batch-extract matched window-opening books with validation."""
    data_dir = Path(config["data_dir"]).resolve()
    raw_depth = int(config.get("depth", 10))
    files = []
    for date in sorted({date for date, _ in assignments}):
        path = data_dir / symbol / date / f"d{raw_depth}" / "events.parquet"
        stat = path.stat()
        files.append((str(path), stat.st_size, stat.st_mtime_ns))
    signature_payload = {
        "version": _REAL_FACTS_CACHE_VERSION,
        "symbol": symbol,
        "lob_depth": int(lob_depth),
        "assignments": [(date, start.isoformat()) for date, start in assignments],
        "files": files,
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = Path(cache_path)
    if path.exists():
        try:
            payload = json.loads(path.read_text())
            if payload.get("signature") == signature:
                snapshots = payload.get("snapshots", [])
                if len(snapshots) == len(assignments):
                    return snapshots
        except (OSError, ValueError, TypeError):
            logger.info("Invalid initial-snapshot cache; rebuilding %s", path)

    snapshots = extract_initial_snapshots(
        symbol, assignments, config, lob_depth
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"signature": signature, "snapshots": snapshots}))
    os.replace(temporary, path)
    return snapshots


def _assemble_facts(
    all_log_returns: list[np.ndarray],
    all_spread: list[np.ndarray],
    all_imbalance: list[np.ndarray],
    all_trade_sizes: list[np.ndarray],
    all_trade_times_sec: list[np.ndarray],
    all_depth_bid: list[np.ndarray],
    all_depth_ask: list[np.ndarray],
    all_book_coverage: list[float],
) -> dict[str, np.ndarray]:
    facts: dict[str, np.ndarray] = {}

    log_returns = np.concatenate(all_log_returns) if all_log_returns else np.array([])
    facts["log_returns"] = log_returns

    # Compute within-day ACFs first so no artificial overnight return enters.
    daily_acfs = [acf_abs_returns(day) for day in all_log_returns if len(day) > 2]
    if daily_acfs:
        min_len = min(map(len, daily_acfs))
        facts["acf_abs_returns"] = np.mean(
            np.array([acf[:min_len] for acf in daily_acfs]), axis=0
        )
    else:
        facts["acf_abs_returns"] = np.array([1.0])

    facts["spread"] = np.concatenate(all_spread) if all_spread else np.array([])
    facts["imbalance"] = np.concatenate(all_imbalance) if all_imbalance else np.array([])
    facts["trade_sizes"] = np.concatenate(all_trade_sizes) if all_trade_sizes else np.array([])

    inter_arrivals: list[np.ndarray] = []
    for ts_sec in all_trade_times_sec:
        if len(ts_sec) > 1:
            inter_arrivals.append(np.diff(ts_sec))
    facts["inter_arrival"] = np.concatenate(inter_arrivals) if inter_arrivals else np.array([])
    facts["book_coverage"] = np.asarray(all_book_coverage, dtype=float)

    if all_depth_bid:
        min_len = min(len(a) for a in all_depth_bid)
        facts["depth_bid"] = np.mean([a[:min_len] for a in all_depth_bid], axis=0)
    else:
        facts["depth_bid"] = np.array([])

    if all_depth_ask:
        min_len = min(len(a) for a in all_depth_ask)
        facts["depth_ask"] = np.mean([a[:min_len] for a in all_depth_ask], axis=0)
    else:
        facts["depth_ask"] = np.array([])

    return facts


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

_CACHE_DATES_KEY = "__dates__"
_CACHE_SIGNATURE_KEY = "__signature__"


def _cache_signature(symbol: str, dates: list[str], config: dict) -> str:
    data_dir = Path(config["data_dir"]).resolve()
    depth = int(config.get("depth", 10))
    files = []
    for date in sorted(dates):
        path = data_dir / symbol / date / f"d{depth}" / "events.parquet"
        if path.exists():
            stat = path.stat()
            files.append((str(path), stat.st_size, stat.st_mtime_ns))
        else:
            files.append((str(path), None, None))
    payload = {
        "version": _REAL_FACTS_CACHE_VERSION,
        "symbol": symbol,
        "dates": sorted(dates),
        "config": {
            "data_dir": str(data_dir),
            "depth": depth,
            "lob_depth": int(config.get("lob_depth", 5)),
            "window_start_utc": config.get("window_start_utc"),
            "window_starts_utc": config.get("window_starts_utc"),
            "window_duration_hours": float(config["window_duration_hours"]),
        },
        "files": files,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _try_load_cache(
    cache_path: str,
    symbol: str,
    dates: list[str],
    config: dict,
) -> Optional[dict[str, np.ndarray]]:
    p = Path(cache_path)
    if not p.exists():
        return None
    try:
        raw = np.load(p, allow_pickle=True)
        if _CACHE_SIGNATURE_KEY not in raw.files:
            logger.info("Legacy real-facts cache found — recomputing.")
            return None
        cached_dates = list(raw[_CACHE_DATES_KEY])
        if sorted(cached_dates) != sorted(dates):
            logger.info("Cache date mismatch — recomputing.")
            return None
        if str(raw[_CACHE_SIGNATURE_KEY].item()) != _cache_signature(symbol, dates, config):
            logger.info("Real-facts cache inputs changed — recomputing.")
            return None
        metadata = {_CACHE_DATES_KEY, _CACHE_SIGNATURE_KEY}
        return {k: raw[k] for k in raw.files if k not in metadata}
    except Exception as exc:
        logger.warning("Failed to load cache %s: %s", cache_path, exc)
        return None


def _save_cache(
    cache_path: str,
    facts: dict[str, np.ndarray],
    symbol: str,
    dates: list[str],
    config: dict,
) -> None:
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        **facts,
        **{
            _CACHE_DATES_KEY: np.array(dates, dtype=str),
            _CACHE_SIGNATURE_KEY: np.array(_cache_signature(symbol, dates, config)),
        },
    )
    logger.info("Real facts cached to: %s", cache_path)
