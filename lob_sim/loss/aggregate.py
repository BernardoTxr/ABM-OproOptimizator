from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from lob_sim.loss.facts import compute_facts

GRID_SIZE = 200
# Facts whose cross-scenario distribution is captured as a KDE band
KDE_FACTS = ["log_returns", "spread", "imbalance", "trade_sizes", "inter_arrival"]

_CACHE_SCALARS = "aggregate_cache.parquet"
_CACHE_KDE = "kde_cache.npz"
_CACHE_META = "aggregate_cache_meta.json"
_AGGREGATE_CACHE_VERSION = 4


def _aggregate_signature(scenario_dirs: list[str]) -> str:
    files = []
    for scenario_dir in scenario_dirs:
        for name in ("mid_price.parquet", "trade_tape.parquet", "lob_snapshots.parquet"):
            path = Path(scenario_dir) / name
            if path.exists():
                stat = path.stat()
                files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
            else:
                files.append((str(path.resolve()), None, None))
    payload = {"version": _AGGREGATE_CACHE_VERSION, "files": files}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Per-scenario summary
# ---------------------------------------------------------------------------

def compute_scenario_summary(scenario_dir: str) -> dict:
    """
    Compute scalar statistics and raw fact arrays for one scenario directory.

    Returns
    -------
    dict with keys:
      "scalars" : flat dict of scalar floats (one value per scenario)
      "arrays"  : raw facts dict (same keys as compute_facts)
    """
    facts = compute_facts(scenario_dir)
    lr  = facts["log_returns"]
    sp  = facts["spread"]
    imb = facts["imbalance"]
    ts  = facts["trade_sizes"]
    ia  = facts["inter_arrival"]
    acf = facts["acf_abs_returns"]
    db  = facts["depth_bid"]
    da  = facts["depth_ask"]

    def _s(arr: np.ndarray, fn, min_len: int = 1) -> float:
        return float(fn(arr)) if arr.size >= min_len else float("nan")

    scalars: dict[str, float] = {
        # Log returns
        "n_obs_returns":   float(lr.size),
        "mean_return":     _s(lr, np.mean),
        "vol":             _s(lr, np.std),
        "skew_returns":    _s(lr, stats.skew,     min_len=3),
        "kurt_returns":    _s(lr, stats.kurtosis, min_len=4),
        # Volatility clustering
        "mean_acf_1_10":   float(np.mean(acf[1:11])) if acf.size > 10 else float("nan"),
        # Spread
        "mean_spread":     _s(sp, np.mean),
        "std_spread":      _s(sp, np.std),
        "median_spread":   _s(sp, np.median),
        # Depth (level 1)
        "mean_depth_bid_l1": float(db[0]) if db.size >= 1 else float("nan"),
        "mean_depth_ask_l1": float(da[0]) if da.size >= 1 else float("nan"),
        # Imbalance
        "mean_imbalance":  _s(imb, np.mean),
        "std_imbalance":   _s(imb, np.std),
        # Trade sizes
        "mean_trade_size": _s(ts, np.mean),
        "std_trade_size":  _s(ts, np.std),
        # Inter-arrival
        "mean_inter_arrival": _s(ia, np.mean),
        "cv_inter_arrival": (
            float(np.std(ia) / np.mean(ia))
            if ia.size > 1 and float(np.mean(ia)) > 0
            else float("nan")
        ),
    }
    return {"scalars": scalars, "arrays": facts}


# ---------------------------------------------------------------------------
# Full aggregation
# ---------------------------------------------------------------------------

def aggregate_all(output_dir: str, force: bool = False) -> dict:
    """
    Aggregate stylized facts across all ``scenario_NNNN`` sub-directories.

    Parameters
    ----------
    output_dir : str
        Root directory that contains ``scenario_NNNN`` sub-directories.
    force : bool
        Re-compute even if cache files already exist.

    Returns
    -------
    dict with keys:

    ``"scalars_df"``
        DataFrame with one row per scenario and one column per scalar metric.

    ``"kde_data"``
        Nested dict with per-fact KDE/summary arrays:

        * KDE facts (log_returns, spread, imbalance, trade_sizes, inter_arrival)::

            {fact: {"x": ndarray[200], "kdes": ndarray[N, 200]}}

        * ACF fact::

            {"acf_abs_returns": {"lags": ndarray, "mean": ndarray, "std": ndarray}}

        * Depth facts::

            {"depth_bid": {"mean": ndarray, "std": ndarray},
             "depth_ask": {"mean": ndarray, "std": ndarray}}
    """
    cache_scalars = os.path.join(output_dir, _CACHE_SCALARS)
    cache_kde = os.path.join(output_dir, _CACHE_KDE)
    cache_meta = os.path.join(output_dir, _CACHE_META)

    # --- discover scenario directories ---
    scenario_dirs = sorted([
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("scenario_") and os.path.isdir(os.path.join(output_dir, d))
    ])
    if not scenario_dirs:
        raise FileNotFoundError(f"No scenario_NNNN directories found in: {output_dir}")

    signature = _aggregate_signature(scenario_dirs)
    if (
        not force
        and os.path.exists(cache_scalars)
        and os.path.exists(cache_kde)
        and os.path.exists(cache_meta)
    ):
        try:
            metadata = json.loads(Path(cache_meta).read_text())
            if metadata.get("signature") == signature:
                return _load_cache(cache_scalars, cache_kde)
        except (OSError, ValueError):
            pass

    # --- pass 1: collect per-scenario scalars and raw fact arrays ---
    all_scalars: list[dict] = []
    raw_arrays: dict[str, list[np.ndarray]] = {
        key: [] for key in KDE_FACTS + [
            "acf_abs_returns", "depth_bid", "depth_ask", "book_coverage"
        ]
    }

    for sdir in tqdm(scenario_dirs, desc="Computing facts"):
        summary = compute_scenario_summary(sdir)
        all_scalars.append(summary["scalars"])
        for key in raw_arrays:
            raw_arrays[key].append(summary["arrays"][key])

    scalars_df = pd.DataFrame(all_scalars)

    # --- pass 2: build KDE / summary arrays on shared grids ---
    kde_data: dict = {}

    # KDE-based facts — shared grid from 1st–99th percentile of pooled data
    for key in KDE_FACTS:
        arrays = [a for a in raw_arrays[key] if a.size > 1]
        if not arrays:
            continue
        all_data = np.concatenate(arrays)
        # Keep pooled empirical samples for a robust distribution distance.
        # This also supports discrete facts for which gaussian_kde is singular.
        kde_data[key] = {"samples": all_data}
        x_min = float(np.percentile(all_data, 1))
        x_max = float(np.percentile(all_data, 99))
        if x_min >= x_max:
            continue
        x = np.linspace(x_min, x_max, GRID_SIZE)
        kde_list = []
        for arr in arrays:
            try:
                kde_list.append(stats.gaussian_kde(arr)(x))
            except Exception:
                pass
        if kde_list:
            kde_data[key].update({"x": x, "kdes": np.array(kde_list)})

    # ACF — stacked mean / std across scenarios
    acfs = [a for a in raw_arrays["acf_abs_returns"] if a.size > 1]
    if acfs:
        min_len = min(len(a) for a in acfs)
        acf_mat = np.array([a[:min_len] for a in acfs])
        kde_data["acf_abs_returns"] = {
            "lags": np.arange(min_len, dtype=float),
            "mean": np.nanmean(acf_mat, axis=0),
            "std":  np.nanstd(acf_mat, axis=0),
        }

    # Depth — mean / std per level
    for key in ("depth_bid", "depth_ask"):
        arrays = [a for a in raw_arrays[key] if a.size > 0]
        if arrays:
            min_len = min(len(a) for a in arrays)
            mat = np.array([a[:min_len] for a in arrays])
            kde_data[key] = {"mean": mat.mean(axis=0), "std": mat.std(axis=0)}

    coverages = [a[0] for a in raw_arrays["book_coverage"] if a.size]
    if coverages:
        kde_data["book_coverage"] = {"mean": float(np.mean(coverages))}

    # --- persist cache ---
    scalars_df.to_parquet(cache_scalars)
    npz_dict: dict[str, np.ndarray] = {}
    for fact_key, val in kde_data.items():
        for subkey, arr in val.items():
            # double-underscore separator; fact keys use single underscores
            npz_dict[f"{fact_key}__{subkey}"] = arr
    np.savez(cache_kde, **npz_dict)
    Path(cache_meta).write_text(json.dumps({"signature": signature}, indent=2) + "\n")

    return {"scalars_df": scalars_df, "kde_data": kde_data}


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _load_cache(cache_scalars: str, cache_kde: str) -> dict:
    scalars_df = pd.read_parquet(cache_scalars)

    raw = np.load(cache_kde, allow_pickle=False)
    kde_data: dict = {}
    for flat_key in raw.files:
        # flat_key format: "fact_key__subkey"
        fact_key, subkey = flat_key.split("__", 1)
        kde_data.setdefault(fact_key, {})[subkey] = raw[flat_key]

    return {"scalars_df": scalars_df, "kde_data": kde_data}
