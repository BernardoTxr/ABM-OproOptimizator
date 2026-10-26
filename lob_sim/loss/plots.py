from __future__ import annotations
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_hist(ax: plt.Axes, data: np.ndarray, **kw) -> None:
    """Plot a histogram only if data is non-empty."""
    if data.size > 1:
        ax.hist(data, bins=50, density=True, edgecolor="none", **kw)


def _normal_overlay(ax: plt.Axes, data: np.ndarray) -> None:
    """Overlay a fitted normal distribution on an Axes."""
    if data.size < 2:
        return
    mu, sigma = float(np.mean(data)), float(np.std(data))
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=1.5, label="Normal fit")


def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str = "Density") -> None:
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _make_figure(title: str) -> tuple[plt.Figure, list[plt.Axes]]:
    fig = plt.figure(figsize=(16, 20))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(4) for c in range(2)]
    return fig, axes


# ---------------------------------------------------------------------------
# Plot implementations
# ---------------------------------------------------------------------------

def _fill_axes(axes: list[plt.Axes], facts: dict, alpha: float = 0.7, color: str = "steelblue") -> None:
    """Populate all 7 subplot panels from a facts dict."""

    # 1. Return distribution
    ax = axes[0]
    _safe_hist(ax, facts["log_returns"], alpha=alpha, color=color)
    _normal_overlay(ax, facts["log_returns"])
    ax.legend(fontsize=7)
    _style(ax, "Return Distribution", "Log-return")

    # 2. ACF of |log-returns| (volatility clustering)
    ax = axes[1]
    acf = facts["acf_abs_returns"]
    if acf.size > 1:
        lags = np.arange(len(acf))
        ax.bar(lags, acf, color=color, alpha=alpha, width=0.8)
        ax.axhline(0, color="k", lw=0.5)
        # 95 % confidence band for white noise
        ci = 1.96 / np.sqrt(max(len(facts["log_returns"]), 1))
        ax.axhline(ci,  color="r", lw=0.8, linestyle="--", label="±95% CI")
        ax.axhline(-ci, color="r", lw=0.8, linestyle="--")
        ax.legend(fontsize=7)
    _style(ax, "Volatility Clustering (ACF |Returns|)", "Lag", ylabel="Autocorrelation")

    # 3. Spread distribution
    ax = axes[2]
    _safe_hist(ax, facts["spread"], alpha=alpha, color=color)
    _style(ax, "Bid-Ask Spread Distribution", "Spread")

    # 4. Order book depth profile
    ax = axes[3]
    bid = facts["depth_bid"]
    ask = facts["depth_ask"]
    levels = np.arange(1, max(len(bid), len(ask)) + 1)
    if bid.size:
        ax.bar(levels[:len(bid)] - 0.2, bid, width=0.35, label="Bid", color="green", alpha=alpha)
    if ask.size:
        ax.bar(levels[:len(ask)] + 0.2, ask, width=0.35, label="Ask", color="red",   alpha=alpha)
    ax.legend(fontsize=7)
    ax.set_xticks(levels)
    _style(ax, "Order Book Depth Profile", "Level", ylabel="Avg qty")

    # 5. Imbalance distribution
    ax = axes[4]
    _safe_hist(ax, facts["imbalance"], alpha=alpha, color=color)
    ax.axvline(0, color="k", lw=0.8, linestyle="--")
    _style(ax, "Best-Level Imbalance Distribution", "(Bid qty − Ask qty) / Total")

    # 6. Trade size distribution
    ax = axes[5]
    _safe_hist(ax, facts["trade_sizes"], alpha=alpha, color=color)
    _style(ax, "Trade Size Distribution", "Quantity")

    # 7. Inter-arrival time distribution
    ax = axes[6]
    _safe_hist(ax, facts["inter_arrival"], alpha=alpha, color=color)
    _style(ax, "Trade Inter-Arrival Time Distribution", "Seconds between trades")

    # Hide unused panel
    axes[7].set_visible(False)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_single(
    facts: dict,
    title: str = "Stylized Facts — Single Scenario",
    save_path: Optional[str] = None,
) -> None:
    """Plot all 7 stylized facts for one scenario."""
    fig, axes = _make_figure(title)
    _fill_axes(axes, facts)
    _finish(fig, save_path)


def plot_ensemble(
    all_facts: list[dict],
    title: str = "Stylized Facts — Ensemble (mean ± 1σ)",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot stylized facts aggregated across multiple scenario runs.
    For histogram-type panels: overlay mean ± 1σ band using kernel density.
    """
    if not all_facts:
        return

    fig, axes = _make_figure(title)

    # ---- 1. Return distribution: KDE mean ± std band ----
    ax = axes[0]
    _ensemble_kde_band(ax, [f["log_returns"] for f in all_facts], "Log-return")
    _style(ax, "Return Distribution (ensemble)", "Log-return")

    # ---- 2. ACF: mean ± std bars ----
    ax = axes[1]
    acfs = [f["acf_abs_returns"] for f in all_facts if f["acf_abs_returns"].size > 1]
    if acfs:
        min_len = min(len(a) for a in acfs)
        acf_mat = np.array([a[:min_len] for a in acfs])
        mean_acf = acf_mat.mean(axis=0)
        std_acf  = acf_mat.std(axis=0)
        lags = np.arange(min_len)
        ax.bar(lags, mean_acf, color="steelblue", alpha=0.7, width=0.8)
        ax.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf, alpha=0.3, color="steelblue")
        ax.axhline(0, color="k", lw=0.5)
        n = min(len(f["log_returns"]) for f in all_facts if f["log_returns"].size)
        ci = 1.96 / np.sqrt(max(n, 1))
        ax.axhline(ci,  color="r", lw=0.8, linestyle="--", label="±95% CI")
        ax.axhline(-ci, color="r", lw=0.8, linestyle="--")
        ax.legend(fontsize=7)
    _style(ax, "Volatility Clustering (ensemble)", "Lag", ylabel="Autocorrelation")

    # ---- 3. Spread KDE ----
    ax = axes[2]
    _ensemble_kde_band(ax, [f["spread"] for f in all_facts], "Spread")
    _style(ax, "Bid-Ask Spread (ensemble)", "Spread")

    # ---- 4. Depth profile: mean ± std bars ----
    ax = axes[3]
    bids = [f["depth_bid"] for f in all_facts if f["depth_bid"].size]
    asks = [f["depth_ask"] for f in all_facts if f["depth_ask"].size]
    if bids:
        mn = min(len(b) for b in bids)
        bmat = np.array([b[:mn] for b in bids])
        levels = np.arange(1, mn + 1)
        ax.bar(levels - 0.2, bmat.mean(0), yerr=bmat.std(0), width=0.35, label="Bid", color="green", alpha=0.7, capsize=3)
    if asks:
        mn = min(len(a) for a in asks)
        amat = np.array([a[:mn] for a in asks])
        levels = np.arange(1, mn + 1)
        ax.bar(levels + 0.2, amat.mean(0), yerr=amat.std(0), width=0.35, label="Ask", color="red",   alpha=0.7, capsize=3)
    ax.legend(fontsize=7)
    _style(ax, "Order Book Depth (ensemble)", "Level", ylabel="Avg qty")

    # ---- 5-7. KDE band plots ----
    keys_titles_xlabels = [
        ("imbalance",    "Imbalance (ensemble)",          "(Bid−Ask)/Total"),
        ("trade_sizes",  "Trade Sizes (ensemble)",        "Quantity"),
        ("inter_arrival","Inter-Arrival Times (ensemble)","Seconds"),
    ]
    for ax, (key, ttl, xl) in zip(axes[4:7], keys_titles_xlabels):
        _ensemble_kde_band(ax, [f[key] for f in all_facts], xl)
        if key == "imbalance":
            ax.axvline(0, color="k", lw=0.8, linestyle="--")
        _style(ax, ttl, xl)

    axes[7].set_visible(False)
    _finish(fig, save_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensemble_kde_band(ax: plt.Axes, data_list: list[np.ndarray], xlabel: str) -> None:
    """Plot a KDE mean ± std band across multiple arrays."""
    non_empty = [d for d in data_list if d.size > 1]
    if not non_empty:
        return

    all_data = np.concatenate(non_empty)
    x_min, x_max = np.percentile(all_data, 1), np.percentile(all_data, 99)
    x = np.linspace(x_min, x_max, 300)

    kde_vals = []
    for d in non_empty:
        try:
            kde = stats.gaussian_kde(d)
            kde_vals.append(kde(x))
        except Exception:
            pass

    if not kde_vals:
        return

    mat  = np.array(kde_vals)
    mean = mat.mean(axis=0)
    std  = mat.std(axis=0)

    ax.plot(x, mean, color="steelblue", lw=1.5, label="Mean KDE")
    ax.fill_between(x, mean - std, mean + std, alpha=0.3, color="steelblue", label="±1σ")
    ax.legend(fontsize=7)


def _finish(fig: plt.Figure, save_path: Optional[str]) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)
