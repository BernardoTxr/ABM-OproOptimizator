"""
lob_sim/simulation/visualize.py
---------------------------------
After each scenario run, generate individual diagnostic PNGs + a markdown
report.  All files land in the scenario directory alongside the Parquet data.

Plot files
----------
plot_mid_price.png       Mid-price with bid/ask band
plot_spread.png          Spread over time
plot_depth_profile.png   Time-averaged LOB depth (bid vs ask, per level)
plot_level_prices.png    L1/L3/L5 bid and ask prices over time
plot_imbalance.png       Best-level imbalance (raw + rolling mean)
plot_trade_activity.png  Traded volume per 60-step window
plot_returns.png         Log-return distribution with normal fit
plot_volatility.png      Rolling 60-step volatility (std of log-returns)
plot_agent_pnl.png       Realized PnL by archetype (mean +- std)
plot_trade_prices.png    Trade price scatter coloured by aggressor side

Report
------
report.md                Markdown summary: run parameters, market stats,
                         agent table, and plot index
"""
from __future__ import annotations

import os
import warnings
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

_RC: dict = {
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linewidth":    0.6,
    "font.size":         8,
    "axes.titlesize":    9,
    "axes.titleweight":  "bold",
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "lines.linewidth":   1.2,
    "figure.dpi":        130,
}

_BID   = "#2ecc71"
_ASK   = "#e74c3c"
_MID   = "#2c3e50"
_TRADE = "#3498db"
_POS   = "#27ae60"
_NEG   = "#c0392b"

_ARCHETYPE_NAMES: dict[str, str] = {
    "NoiseTraderAgent":    "Noise Trader",
    "MarketMakerAgent":    "Market Maker",
    "MomentumTraderAgent": "Momentum Trader",
}

_PLOT_DESCRIPTIONS: dict[str, str] = {
    "plot_mid_price.png":      "Mid-price evolution with live bid/ask band",
    "plot_spread.png":         "Bid-ask spread over time with median reference line",
    "plot_depth_profile.png":  "Time-averaged LOB depth at each price level (bid vs ask)",
    "plot_level_prices.png":   "Bid and ask prices at levels 1, 3, and 5 over time",
    "plot_imbalance.png":      "Best-level order imbalance with 60-step rolling mean",
    "plot_trade_activity.png": "Traded volume aggregated into 60-second windows",
    "plot_returns.png":        "Log-return distribution compared to a normal fit",
    "plot_volatility.png":     "Rolling 60-step standard deviation of log-returns",
    "plot_agent_pnl.png":      "Mean realised PnL per archetype (error bars = +/-1 std)",
    "plot_trade_prices.png":   "Every trade price coloured by aggressor side, with mid-price overlay",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_scenario_outputs(scenario_dir: str, meta: Optional[dict] = None) -> None:
    """
    Save all diagnostic PNGs and report.md for *scenario_dir*.

    Parameters
    ----------
    scenario_dir : absolute path returned by OutputWriter.write()
    meta         : optional dict passed from the runner with keys:
                   scenario_name, scenario_idx, time_step, random_seed,
                   n_agents, start_time, end_time, tick_size, n_steps
    """
    mid_df    = _load(scenario_dir, "mid_price.parquet")
    lob_df    = _load(scenario_dir, "lob_snapshots.parquet")
    trade_df  = _load(scenario_dir, "trade_tape.parquet")
    pnl_df    = _load(scenario_dir, "agent_pnl.parquet")
    meta      = meta or {}

    with plt.rc_context(_RC):
        _save(scenario_dir, "plot_mid_price.png",      _fig_mid_price,      mid_df)
        _save(scenario_dir, "plot_spread.png",         _fig_spread,         mid_df)
        _save(scenario_dir, "plot_depth_profile.png",  _fig_depth_profile,  lob_df)
        _save(scenario_dir, "plot_level_prices.png",   _fig_level_prices,   lob_df)
        _save(scenario_dir, "plot_imbalance.png",      _fig_imbalance,      lob_df)
        _save(scenario_dir, "plot_trade_activity.png", _fig_trade_activity, trade_df, mid_df)
        _save(scenario_dir, "plot_returns.png",        _fig_returns,        mid_df)
        _save(scenario_dir, "plot_volatility.png",     _fig_volatility,     mid_df)
        _save(scenario_dir, "plot_agent_pnl.png",      _fig_agent_pnl,      pnl_df)
        _save(scenario_dir, "plot_trade_prices.png",   _fig_trade_prices,   trade_df, mid_df)

    _save_report(scenario_dir, mid_df, trade_df, pnl_df, meta)


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------

def _fig_mid_price(mid_df: pd.DataFrame) -> Optional[plt.Figure]:
    if mid_df.empty:
        return None
    # Drop rows where mid_price is undefined (empty LOB) to avoid empty x-axis tail
    valid = mid_df["mid_price"].notna()
    if not valid.any():
        return None
    df = mid_df[valid]
    t  = df["timestamp"].values
    mp = df["mid_price"].values
    fig, ax = plt.subplots(figsize=(10, 4))
    if "best_bid" in df.columns and "best_ask" in df.columns:
        band_ok = df["best_bid"].notna() & df["best_ask"].notna()
        if band_ok.any():
            bdf = df[band_ok]
            ax.fill_between(bdf["timestamp"].values,
                            bdf["best_bid"].values, bdf["best_ask"].values,
                            alpha=0.2, color=_TRADE, label="Bid-Ask band")
    ax.plot(t, mp, color=_MID, lw=1.2, label="Mid")
    ax.set_title("Mid-Price")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Price")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def _fig_spread(mid_df: pd.DataFrame) -> Optional[plt.Figure]:
    if mid_df.empty or "spread" not in mid_df.columns:
        return None
    s      = mid_df["spread"].dropna()
    t      = mid_df.loc[s.index, "timestamp"]
    median = float(s.median())
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t.values, s.values, color=_TRADE, alpha=0.7, lw=0.9)
    ax.axhline(median, color="k", lw=0.9, linestyle="--",
               label=f"Median = {median:.4f}")
    ax.set_title("Bid-Ask Spread")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spread")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def _fig_depth_profile(lob_df: pd.DataFrame) -> Optional[plt.Figure]:
    if lob_df.empty:
        return None
    max_level = int(lob_df["level"].max())
    levels    = np.arange(1, max_level + 1)
    bid_means = np.array([lob_df.loc[lob_df["level"] == l, "bid_qty"].mean()
                          for l in levels])
    ask_means = np.array([lob_df.loc[lob_df["level"] == l, "ask_qty"].mean()
                          for l in levels])
    fig, ax = plt.subplots(figsize=(7, 5))
    w = 0.35
    ax.bar(levels - w / 2, bid_means, width=w, color=_BID, alpha=0.85, label="Bid")
    ax.bar(levels + w / 2, ask_means, width=w, color=_ASK, alpha=0.85, label="Ask")
    ax.set_xticks(levels)
    ax.set_xticklabels([f"L{l}" for l in levels])
    ax.set_title("LOB Depth Profile (time-averaged)")
    ax.set_xlabel("Book level")
    ax.set_ylabel("Mean qty")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def _fig_level_prices(lob_df: pd.DataFrame) -> Optional[plt.Figure]:
    """Plot bid/ask prices at levels 1, 3, and 5 on one time axis."""
    if lob_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {
        "bid": {1: "#0b3c5d", 3: "#328cc1", 5: "#7fb3d5"},
        "ask": {1: "#7b241c", 3: "#c0392b", 5: "#f1948a"},
    }
    styles = {1: "-", 3: "--", 5: ":"}
    plotted = False
    for side in ("bid", "ask"):
        column = f"{side}_price"
        for level in (1, 3, 5):
            series = lob_df[lob_df["level"] == level][["timestamp", column]].dropna()
            if series.empty:
                continue
            ax.plot(
                series["timestamp"], series[column],
                color=colors[side][level], linestyle=styles[level],
                lw=1.1, label=f"L{level} {side.title()}",
            )
            plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_title("LOB Level Prices Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Price")
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    return fig


def _fig_imbalance(lob_df: pd.DataFrame) -> Optional[plt.Figure]:
    if lob_df.empty:
        return None
    lvl1  = lob_df[lob_df["level"] == 1].copy()
    bq    = lvl1["bid_qty"].fillna(0.0).values
    aq    = lvl1["ask_qty"].fillna(0.0).values
    total = bq + aq
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imb = np.where(total > 0, (bq - aq) / total, np.nan)
    t      = lvl1["timestamp"].values
    imb_rm = pd.Series(imb).rolling(60, min_periods=1, center=True).mean().values
    # Clip x-axis to the last step where both sides of the book were active
    two_sided = (bq > 0) & (aq > 0)
    active_idx = np.where(two_sided)[0]
    if len(active_idx) == 0:
        return None
    t_max = t[active_idx[-1]] * 1.05
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, imb,    color=_TRADE, alpha=0.2, lw=0.7)
    ax.plot(t, imb_rm, color=_MID,   lw=1.4,   label="60-step rolling mean")
    ax.axhline(0, color="k", lw=0.8, linestyle="--")
    ax.set_xlim(0, t_max)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Best-Level Imbalance")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("(Bid - Ask) / Total")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def _fig_trade_activity(
    trade_df: pd.DataFrame,
    mid_df: pd.DataFrame,
) -> Optional[plt.Figure]:
    if trade_df.empty:
        return None
    # Clip to 99.5th-percentile of trade timestamps; a handful of late stale-order
    # matches after liquidity dries up would otherwise create a long empty tail.
    end_time = float(np.nanpercentile(trade_df["timestamp"].values, 99.5)) * 1.05
    bins = np.arange(0, end_time + 120, 60)
    vol, edges = np.histogram(
        trade_df["timestamp"].values,
        bins=bins,
        weights=trade_df["quantity"].values,
    )
    centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(centers, vol, width=55, color=_TRADE, alpha=0.75, edgecolor="none")
    ax.set_title("Trade Activity - volume per 60-step window")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Traded volume")
    fig.tight_layout()
    return fig


def _fig_returns(mid_df: pd.DataFrame) -> Optional[plt.Figure]:
    if mid_df.empty or "mid_price" not in mid_df.columns:
        return None
    mp = mid_df["mid_price"].dropna().values.clip(min=1e-10)
    if len(mp) < 3:
        return None
    lr    = np.diff(np.log(mp))
    mu    = float(np.mean(lr))
    sigma = float(np.std(lr))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lr, bins=60, density=True, color=_TRADE, alpha=0.65,
            edgecolor="none", label="Observed")
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    ax.plot(x, scipy_stats.norm.pdf(x, mu, sigma), color=_ASK, lw=1.5,
            label="Normal fit")
    ax.set_title("Log-Return Distribution")
    ax.set_xlabel("Log-return")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def _fig_volatility(mid_df: pd.DataFrame) -> Optional[plt.Figure]:
    if mid_df.empty or "mid_price" not in mid_df.columns:
        return None
    mp = mid_df["mid_price"].dropna()
    if len(mp) < 5:
        return None
    lr = np.diff(np.log(mp.values.clip(min=1e-10)))
    t  = mid_df.loc[mp.index, "timestamp"].values[1:]
    rv = pd.Series(lr).rolling(60, min_periods=5).std().values
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(t, 0, rv, alpha=0.2, color=_TRADE)
    ax.plot(t, rv, color=_MID, lw=1.2)
    ax.set_title("Rolling 60-step Volatility (std of log-returns)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Volatility")
    fig.tight_layout()
    return fig


def _fig_agent_pnl(pnl_df: pd.DataFrame) -> Optional[plt.Figure]:
    if pnl_df.empty or "realized_pnl" not in pnl_df.columns:
        return None
    summary = (
        pnl_df.groupby("archetype", sort=False)["realized_pnl"]
        .agg(mean="mean", std="std", count="count")
        .reset_index()
    )
    if summary.empty:
        return None
    labels = [_shorten(a) for a in summary["archetype"]]
    means  = summary["mean"].values
    stds   = summary["std"].fillna(0).values
    colors = [_POS if m >= 0 else _NEG for m in means]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, width=0.55, capsize=4,
           color=colors, alpha=0.8, edgecolor="none")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title("Realised PnL by Archetype (mean +/- std)")
    ax.set_ylabel("Realised PnL")
    fig.tight_layout()
    return fig


def _fig_trade_prices(
    trade_df: pd.DataFrame,
    mid_df: pd.DataFrame,
) -> Optional[plt.Figure]:
    if trade_df.empty:
        return None
    df = trade_df
    if len(df) > 8_000:
        step = max(1, len(df) // 8_000)
        df   = df.iloc[::step]
    side_str  = df["aggressor_side"].astype(str).str.upper()
    buy_mask  = side_str.str.contains("BUY",  na=False)
    sell_mask = side_str.str.contains("SELL", na=False)
    # Percentile-based y-axis to suppress outlier price dots
    all_prices = trade_df["price"].values  # use full dataset for percentile
    p_lo = float(np.nanpercentile(all_prices, 0.5))
    p_hi = float(np.nanpercentile(all_prices, 99.5))
    margin = (p_hi - p_lo) * 0.08
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df.loc[buy_mask,  "timestamp"], df.loc[buy_mask,  "price"],
               s=3, alpha=0.35, color=_BID, linewidths=0, label="Buyer aggressor")
    ax.scatter(df.loc[sell_mask, "timestamp"], df.loc[sell_mask, "price"],
               s=3, alpha=0.35, color=_ASK, linewidths=0, label="Seller aggressor")
    x_max = None
    if not mid_df.empty and "mid_price" in mid_df.columns:
        # Filter NaN mid-prices so the overlay doesn't extend x-axis into empty space
        mid_valid = mid_df[mid_df["mid_price"].notna()]
        if not mid_valid.empty:
            ax.plot(mid_valid["timestamp"].values, mid_valid["mid_price"].values,
                    color=_MID, lw=1.2, zorder=3, label="Mid")
            x_max = float(mid_valid["timestamp"].max()) * 1.05
    ax.set_ylim(p_lo - margin, p_hi + margin)
    if x_max is not None:
        ax.set_xlim(0, x_max)
    ax.set_title("Trade Prices")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Price")
    ax.legend(fontsize=7, markerscale=3)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _save_report(
    scenario_dir: str,
    mid_df: pd.DataFrame,
    trade_df: pd.DataFrame,
    pnl_df: pd.DataFrame,
    meta: dict,
) -> None:
    lines: list[str] = []
    scenario_name = meta.get("scenario_name",
                             os.path.basename(os.path.dirname(scenario_dir)))
    scenario_idx  = meta.get("scenario_idx", "?")
    lines.append(f"# Simulation Report -- {scenario_name} / scenario {scenario_idx}\n")

    lines.append("## Parameters\n")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    start = meta.get("start_time", 0)
    end   = meta.get("end_time",   "?")
    steps = meta.get("n_steps",    "?")
    for k, v in [
        ("Random seed", meta.get("random_seed", "?")),
        ("Time step",   f"{meta.get('time_step', '?')} s"),
        ("Session",     f"{start:.0f} to {end} s ({steps} steps)"),
        ("Tick size",   meta.get("tick_size", "?")),
        ("Agents",      meta.get("n_agents",  "?")),
    ]:
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Market Statistics\n")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    n_trades = tot_vol = avg_size = 0.0
    if not trade_df.empty:
        n_trades = len(trade_df)
        tot_vol  = float(trade_df["quantity"].sum())
        avg_size = float(trade_df["quantity"].mean())
    p0 = p1 = pstd = pmin = pmax = float("nan")
    if not mid_df.empty and "mid_price" in mid_df.columns:
        mp   = mid_df["mid_price"].dropna()
        p0   = float(mp.iloc[0])  if len(mp) else float("nan")
        p1   = float(mp.iloc[-1]) if len(mp) else float("nan")
        pstd = float(mp.std())
        pmin = float(mp.min())
        pmax = float(mp.max())
    sp_avg = sp_med = float("nan")
    if not mid_df.empty and "spread" in mid_df.columns:
        sp     = mid_df["spread"].dropna()
        sp_avg = float(sp.mean())
        sp_med = float(sp.median())
    for k, v in [
        ("Total trades",    f"{int(n_trades):,}"),
        ("Total volume",    f"{tot_vol:,.2f}"),
        ("Avg trade size",  f"{avg_size:.3f}"),
        ("Avg spread",      f"{sp_avg:.4f}"),
        ("Median spread",   f"{sp_med:.4f}"),
        ("Mid-price start", f"{p0:.4f}"),
        ("Mid-price end",   f"{p1:.4f}"),
        ("Mid-price std",   f"{pstd:.4f}"),
        ("Price range",     f"[{pmin:.4f}, {pmax:.4f}]"),
    ]:
        lines.append(f"| {k} | {v} |")
    lines.append("")

    if not pnl_df.empty:
        lines.append("## Agent Summary\n")
        lines.append("| Archetype | Agents | Total Bought | Total Sold | PnL (sum) | Mean PnL |")
        lines.append("|-----------|-------:|-------------:|-----------:|----------:|---------:|")
        summary = pnl_df.groupby("archetype").agg(
            count=   ("agent_id",        "count"),
            bought=  ("total_bought_qty", "sum"),
            sold=    ("total_sold_qty",   "sum"),
            pnl_sum= ("realized_pnl",     "sum"),
            pnl_mean=("realized_pnl",     "mean"),
        ).reset_index()
        for row in summary.itertuples():
            lines.append(
                f"| {_shorten(row.archetype)} | {row.count} "
                f"| {row.bought:,.2f} | {row.sold:,.2f} "
                f"| {row.pnl_sum:+,.2f} | {row.pnl_mean:+,.2f} |"
            )
        lines.append("")

    lines.append("## Generated Plots\n")
    lines.append("| File | Description |")
    lines.append("|------|-------------|")
    for fname, desc in _PLOT_DESCRIPTIONS.items():
        if os.path.exists(os.path.join(scenario_dir, fname)):
            lines.append(f"| [{fname}]({fname}) | {desc} |")
    lines.append("")

    with open(os.path.join(scenario_dir, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(scenario_dir: str, filename: str, fig_fn, *args) -> None:
    try:
        fig = fig_fn(*args)
        if fig is not None:
            fig.savefig(
                os.path.join(scenario_dir, filename),
                dpi=130, bbox_inches="tight",
            )
            plt.close(fig)
    except Exception:
        pass


def _load(scenario_dir: str, filename: str) -> pd.DataFrame:
    path = os.path.join(scenario_dir, filename)
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    return pd.DataFrame()


def _shorten(archetype: str) -> str:
    return _ARCHETYPE_NAMES.get(archetype, archetype.replace("Agent", ""))
