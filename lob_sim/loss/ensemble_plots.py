from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str = "Density") -> None:
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _kde_band(
    ax: plt.Axes,
    entry: dict,
    color: str = "steelblue",
    label_prefix: str = "Sim",
) -> None:
    """Draw mean ± 1σ KDE band from a kde_data entry."""
    x = entry["x"]
    kdes = entry["kdes"]
    mean = kdes.mean(axis=0)
    std = kdes.std(axis=0)
    ax.plot(x, mean, color=color, lw=1.8, label=f"{label_prefix} mean KDE")
    ax.fill_between(x, mean - std, mean + std, alpha=0.25, color=color, label=f"{label_prefix} ±1σ")


def _finish(fig: plt.Figure, save_path: Optional[str]) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Log-return distribution
# ---------------------------------------------------------------------------

def plot_log_returns(
    kde_data: dict,
    n_scenarios: int,
    save_path: Optional[str] = None,
    real_kde: Optional[dict] = None,
) -> None:
    """
    KDE mean ± 1σ band of the log-return distribution across all scenarios.

    Parameters
    ----------
    real_kde : optional dict ``{"x": grid_array, "y": kde_values_array}``
        If provided, overlay a single line representing a real-data distribution.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    entry = kde_data.get("log_returns")
    if entry is not None:
        _kde_band(ax, entry)
        # Normal fit overlaid on the mean KDE
        x = entry["x"]
        mean_kde = entry["kdes"].mean(axis=0)
        mu = float(np.trapezoid(x * mean_kde, x))
        var = float(np.trapezoid((x - mu) ** 2 * mean_kde, x))
        if var > 0:
            ax.plot(x, stats.norm.pdf(x, mu, np.sqrt(var)), "r--", lw=1.5, label="Normal fit (mean KDE)")

    if real_kde is not None:
        ax.plot(real_kde["x"], real_kde["y"], color="darkorange", lw=1.8, ls="--", label="Real data")

    ax.legend(fontsize=9)
    _style(ax, f"Return Distribution — Ensemble ({n_scenarios} scenarios)", "Log-return")
    _finish(fig, save_path)


# ---------------------------------------------------------------------------
# 2. ACF of |returns| (volatility clustering)
# ---------------------------------------------------------------------------

def plot_acf(
    kde_data: dict,
    n_scenarios: int,
    n_obs: int = 3599,
    save_path: Optional[str] = None,
    real_acf: Optional[dict] = None,
) -> None:
    """
    Bar chart of mean ACF of |log-returns| ± 1σ across all scenarios.

    Parameters
    ----------
    n_obs : int
        Typical number of observations per scenario (used for the 95% white-noise CI band).
    real_acf : optional dict ``{"lags": array, "mean": array}``
        If provided, overlay real-data ACF as a line.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    entry = kde_data.get("acf_abs_returns")
    if entry is not None:
        lags = entry["lags"]
        mean_acf = entry["mean"]
        std_acf = entry["std"]
        ax.bar(lags, mean_acf, color="steelblue", alpha=0.7, width=0.8, label="Sim mean ACF")
        ax.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf,
                        alpha=0.3, color="steelblue", label="±1σ")
        ci = 1.96 / np.sqrt(max(n_obs, 1))
        ax.axhline(ci,  color="r", lw=0.9, linestyle="--", label="±95% CI (white noise)")
        ax.axhline(-ci, color="r", lw=0.9, linestyle="--")
        ax.axhline(0,   color="k", lw=0.5)

    if real_acf is not None:
        ax.plot(real_acf["lags"], real_acf["mean"], color="darkorange", lw=1.8, ls="--", label="Real data")

    ax.legend(fontsize=9)
    _style(
        ax,
        f"Volatility Clustering (ACF |Returns|) — Ensemble ({n_scenarios} scenarios)",
        "Lag",
        ylabel="Autocorrelation",
    )
    _finish(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Bid-ask spread distribution
# ---------------------------------------------------------------------------

def plot_spread(
    kde_data: dict,
    n_scenarios: int,
    save_path: Optional[str] = None,
    real_kde: Optional[dict] = None,
) -> None:
    """
    KDE mean ± 1σ band of the bid-ask spread distribution across all scenarios.

    real_kde : optional dict ``{"x": grid_array, "y": kde_values_array}``
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    entry = kde_data.get("spread")
    if entry is not None:
        _kde_band(ax, entry)

    if real_kde is not None:
        ax.plot(real_kde["x"], real_kde["y"], color="darkorange", lw=1.8, ls="--", label="Real data")

    ax.legend(fontsize=9)
    _style(ax, f"Bid-Ask Spread Distribution — Ensemble ({n_scenarios} scenarios)", "Spread")
    _finish(fig, save_path)


# ---------------------------------------------------------------------------
# 4. LOB depth profile
# ---------------------------------------------------------------------------

def plot_depth(
    kde_data: dict,
    n_scenarios: int,
    save_path: Optional[str] = None,
    real_depth: Optional[dict] = None,
) -> None:
    """
    Grouped bar chart of bid/ask mean depth per level with ±1σ error caps.

    real_depth : optional dict ``{"bid_mean": array, "ask_mean": array}``
        If provided, overlay real-data depth as markers.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    bid_entry = kde_data.get("depth_bid")
    ask_entry = kde_data.get("depth_ask")

    n_levels = 0
    if bid_entry is not None:
        n_levels = max(n_levels, len(bid_entry["mean"]))
    if ask_entry is not None:
        n_levels = max(n_levels, len(ask_entry["mean"]))

    levels = np.arange(1, n_levels + 1)

    if bid_entry is not None:
        bm, bs = bid_entry["mean"], bid_entry["std"]
        ax.bar(levels[: len(bm)] - 0.2, bm, width=0.35, yerr=bs, capsize=4,
               color="green", alpha=0.7, label="Bid mean ± 1σ")

    if ask_entry is not None:
        am, as_ = ask_entry["mean"], ask_entry["std"]
        ax.bar(levels[: len(am)] + 0.2, am, width=0.35, yerr=as_, capsize=4,
               color="red", alpha=0.7, label="Ask mean ± 1σ")

    if real_depth is not None:
        bm_r = real_depth.get("bid_mean")
        am_r = real_depth.get("ask_mean")
        if bm_r is not None:
            ax.plot(levels[: len(bm_r)] - 0.2, bm_r, "o--", color="darkgreen", lw=1.5, label="Real bid")
        if am_r is not None:
            ax.plot(levels[: len(am_r)] + 0.2, am_r, "s--", color="darkred", lw=1.5, label="Real ask")

    ax.set_xticks(levels)
    ax.legend(fontsize=9)
    _style(ax, f"LOB Depth Profile — Ensemble ({n_scenarios} scenarios)", "Level", ylabel="Avg qty")
    _finish(fig, save_path)


# ---------------------------------------------------------------------------
# 5. Best-level imbalance distribution
# ---------------------------------------------------------------------------

def plot_imbalance(
    kde_data: dict,
    n_scenarios: int,
    save_path: Optional[str] = None,
    real_kde: Optional[dict] = None,
) -> None:
    """
    KDE mean ± 1σ band of the best-level order-flow imbalance across all scenarios.

    real_kde : optional dict ``{"x": grid_array, "y": kde_values_array}``
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    entry = kde_data.get("imbalance")
    if entry is not None:
        _kde_band(ax, entry)
    ax.axvline(0, color="k", lw=0.9, linestyle="--")

    if real_kde is not None:
        ax.plot(real_kde["x"], real_kde["y"], color="darkorange", lw=1.8, ls="--", label="Real data")

    ax.legend(fontsize=9)
    _style(
        ax,
        f"Best-Level Imbalance Distribution — Ensemble ({n_scenarios} scenarios)",
        "(Bid qty − Ask qty) / Total",
    )
    _finish(fig, save_path)


# ---------------------------------------------------------------------------
# 6. Trade size distribution
# ---------------------------------------------------------------------------

def plot_trade_sizes(
    kde_data: dict,
    n_scenarios: int,
    save_path: Optional[str] = None,
    real_kde: Optional[dict] = None,
) -> None:
    """
    KDE mean ± 1σ band of the trade size distribution across all scenarios.

    real_kde : optional dict ``{"x": grid_array, "y": kde_values_array}``
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    entry = kde_data.get("trade_sizes")
    if entry is not None:
        _kde_band(ax, entry)

    if real_kde is not None:
        ax.plot(real_kde["x"], real_kde["y"], color="darkorange", lw=1.8, ls="--", label="Real data")

    ax.legend(fontsize=9)
    _style(ax, f"Trade Size Distribution — Ensemble ({n_scenarios} scenarios)", "Quantity")
    _finish(fig, save_path)


# ---------------------------------------------------------------------------
# 7. Inter-arrival time distribution
# ---------------------------------------------------------------------------

def plot_inter_arrival(
    kde_data: dict,
    n_scenarios: int,
    save_path: Optional[str] = None,
    real_kde: Optional[dict] = None,
) -> None:
    """
    KDE mean ± 1σ band of the inter-arrival time distribution across all scenarios.

    real_kde : optional dict ``{"x": grid_array, "y": kde_values_array}``
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    entry = kde_data.get("inter_arrival")
    if entry is not None:
        _kde_band(ax, entry)

    if real_kde is not None:
        ax.plot(real_kde["x"], real_kde["y"], color="darkorange", lw=1.8, ls="--", label="Real data")

    ax.legend(fontsize=9)
    _style(
        ax,
        f"Inter-Arrival Time Distribution — Ensemble ({n_scenarios} scenarios)",
        "Seconds between trades",
    )
    _finish(fig, save_path)
