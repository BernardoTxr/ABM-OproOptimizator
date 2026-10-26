from __future__ import annotations
import os

import numpy as np
import pandas as pd


def acf_abs_returns(log_returns: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """Finite ACF of absolute returns, using zero for undefined correlations."""
    values = np.asarray(log_returns, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 2:
        return np.array([1.0])
    absolute = np.abs(values)
    lag_count = min(max_lag, len(absolute) - 1)
    acf = np.ones(lag_count + 1)
    for lag in range(1, lag_count + 1):
        left, right = absolute[:-lag], absolute[lag:]
        if np.std(left) <= np.finfo(float).eps or np.std(right) <= np.finfo(float).eps:
            acf[lag] = 0.0
        else:
            value = float(np.corrcoef(left, right)[0, 1])
            acf[lag] = value if np.isfinite(value) else 0.0
    return acf


def compute_facts(scenario_dir: str) -> dict:
    """
    Load Parquet output from a single scenario directory and compute
    all seven stylized facts. Returns a dict of numpy arrays.

    Keys:
      log_returns      : 1-D array of log-returns of mid-price
      acf_abs_returns  : 1-D array of ACF of |log-returns|, lags 0..MAX_LAG
      spread           : 1-D array of bid-ask spread values
      depth_bid        : 1-D array — average bid qty at levels 1..N
      depth_ask        : 1-D array — average ask qty at levels 1..N
      imbalance        : 1-D array of imbalance over all retained levels
      trade_sizes      : 1-D array of matched trade quantities
      inter_arrival    : 1-D array of seconds between consecutive trades
    """
    mid_path    = os.path.join(scenario_dir, "mid_price.parquet")
    trade_path  = os.path.join(scenario_dir, "trade_tape.parquet")
    lob_path    = os.path.join(scenario_dir, "lob_snapshots.parquet")

    mid_df   = pd.read_parquet(mid_path)   if os.path.exists(mid_path)   else pd.DataFrame()
    trade_df = pd.read_parquet(trade_path) if os.path.exists(trade_path) else pd.DataFrame()
    lob_df   = pd.read_parquet(lob_path)   if os.path.exists(lob_path)   else pd.DataFrame()

    facts: dict[str, np.ndarray] = {}

    # 1. Log-returns of mid-price
    if not mid_df.empty and "mid_price" in mid_df.columns:
        mp = mid_df["mid_price"].dropna()
        log_ret = np.diff(np.log(mp.values.clip(min=1e-10)))
        facts["log_returns"] = log_ret
        facts["book_coverage"] = np.array([float(mid_df["mid_price"].notna().mean())])
    else:
        facts["log_returns"] = np.array([])
        facts["book_coverage"] = np.array([0.0])

    # 2. ACF of |log-returns|
    lr = facts["log_returns"]
    facts["acf_abs_returns"] = acf_abs_returns(lr)

    # 3. Spread
    if not mid_df.empty and "spread" in mid_df.columns:
        facts["spread"] = mid_df["spread"].dropna().values
    else:
        facts["spread"] = np.array([])

    # 4 & 5. Depth profile (average qty at each level)
    if not lob_df.empty:
        max_level = int(lob_df["level"].max())
        depth_bid, depth_ask = [], []
        for lvl in range(1, max_level + 1):
            lvl_df = lob_df[lob_df["level"] == lvl]
            depth_bid.append(float(lvl_df["bid_qty"].mean()))
            depth_ask.append(float(lvl_df["ask_qty"].mean()))
        facts["depth_bid"] = np.array(depth_bid)
        facts["depth_ask"] = np.array(depth_ask)
    else:
        facts["depth_bid"] = np.array([])
        facts["depth_ask"] = np.array([])

    # 6. Imbalance over all retained book levels (normally L1-L5)
    if not lob_df.empty:
        grouped = lob_df.groupby("timestamp", sort=True)[["bid_qty", "ask_qty"]].sum()
        bq = grouped["bid_qty"].fillna(0.0)
        aq = grouped["ask_qty"].fillna(0.0)
        total = bq + aq
        imb = np.where(total > 0, (bq - aq) / total, np.nan)
        facts["imbalance"] = imb[~np.isnan(imb)]
    else:
        facts["imbalance"] = np.array([])

    # 7. Trade sizes
    if not trade_df.empty and "quantity" in trade_df.columns:
        facts["trade_sizes"] = trade_df["quantity"].values
    else:
        facts["trade_sizes"] = np.array([])

    # 8. Inter-arrival times between trades
    if not trade_df.empty and len(trade_df) > 1 and "timestamp" in trade_df.columns:
        times = np.sort(trade_df["timestamp"].values)
        facts["inter_arrival"] = np.diff(times)
    else:
        facts["inter_arrival"] = np.array([])

    return facts
