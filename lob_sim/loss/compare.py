"""Loss computation and comparison between simulated and real LOB data.

The public API is:

* :func:`build_real_kde` — convert raw real-data fact arrays into the same
  ``kde_data`` dict structure produced by :func:`lob_sim.loss.aggregate.aggregate_all`.

* :func:`compute_loss` — compute robust distribution / normalized errors and a
  weighted total loss between the simulation ensemble and real-data KDEs.

* :func:`save_comparison_plots` — call the existing ``ensemble_plots`` functions
  with real-data overlays, writing PNGs to an output directory.

* :func:`save_loss_report` — write a Markdown table and JSON summary.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Grid points used to build real KDEs (matches aggregate.py GRID_SIZE)
GRID_SIZE = 200
_EPS = 1e-10

KDE_FACTS = ["log_returns", "spread", "imbalance", "trade_sizes", "inter_arrival"]
LOSS_COMPONENTS = frozenset([
    *KDE_FACTS, "acf_abs_returns", "depth", "book_coverage"
])
LOSS_VERSION = "pooled_v2"


# ---------------------------------------------------------------------------
# LossResult
# ---------------------------------------------------------------------------

@dataclass
class LossResult:
    """Weighted aggregate loss and per-fact breakdown."""
    total: float
    per_fact: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    specification: dict = field(default_factory=dict)


@dataclass(frozen=True)
class LossSpec:
    """Versioned choices that fully determine the scalar loss definition."""

    version: str = LOSS_VERSION
    winsor_quantiles: tuple[float, float] = (0.01, 0.99)
    missing_fact_penalty: float = 1.0
    weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: dict) -> "LossSpec":
        version = str(values.get("version", LOSS_VERSION))
        if version != LOSS_VERSION:
            raise ValueError(f"Unsupported loss specification version: {version}")
        quantiles = tuple(float(value) for value in values.get(
            "winsor_quantiles", (0.01, 0.99)
        ))
        if len(quantiles) != 2 or not 0 <= quantiles[0] < quantiles[1] <= 1:
            raise ValueError("winsor_quantiles must satisfy 0 <= low < high <= 1")
        penalty = float(values.get("missing_fact_penalty", 1.0))
        if not np.isfinite(penalty) or penalty < 0:
            raise ValueError("missing_fact_penalty must be finite and non-negative")
        weights = {str(key): float(value) for key, value in values.get("weights", {}).items()}
        unknown = sorted(set(weights) - LOSS_COMPONENTS)
        if unknown:
            raise ValueError(f"Unknown loss component weights: {unknown}")
        if any(not np.isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError("Loss weights must be finite and non-negative")
        if weights and not any(value > 0 for value in weights.values()):
            raise ValueError("At least one configured loss weight must be positive")
        return cls(
            version=version,
            winsor_quantiles=(quantiles[0], quantiles[1]),
            missing_fact_penalty=penalty,
            weights=weights,
        )

    def as_dict(self) -> dict:
        return {
            "version": self.version,
            "aggregation": "pooled_simulation_samples_vs_pooled_real_windows",
            "distribution_distance": "winsorized_scale_normalized_wasserstein_1",
            "winsor_quantiles": list(self.winsor_quantiles),
            "acf_distance": "rmse_nonzero_lags",
            "depth_distance": "mean_bid_ask_relative_rmse",
            "coverage_distance": "relative_absolute_error",
            "missing_fact_penalty": self.missing_fact_penalty,
            "weights": self.weights,
        }


# ---------------------------------------------------------------------------
# Build real-data KDE structure
# ---------------------------------------------------------------------------

def build_real_kde(real_facts: dict[str, np.ndarray]) -> dict:
    """Convert raw real-data fact arrays into a ``kde_data`` dict.

    The returned structure mirrors the output of
    :func:`lob_sim.loss.aggregate.aggregate_all` so it can be passed
    directly to the ``real_*`` keyword arguments of every
    ``ensemble_plots.plot_*`` function.

    Keys produced:

    * KDE facts → ``{fact: {"x": ndarray[200], "y": ndarray[200]}}``
      (single KDE line, no band — the ``y`` key maps to the ensemble
      ``real_kde={"x": ..., "y": ...}`` signature).
    * ACF → ``{"acf_abs_returns": {"lags": ndarray, "mean": ndarray}}``
    * Depth → ``{"depth_bid": {"bid_mean": ndarray}, "depth_ask": {"ask_mean": ndarray}}``
    """
    kde_data: dict = {}

    for fact in KDE_FACTS:
        arr = np.asarray(real_facts.get(fact, np.array([])), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            logger.warning("Insufficient real data for fact '%s' (%d samples)", fact, arr.size)
            continue
        kde_data[fact] = {"samples": arr}
        x_min = float(np.percentile(arr, 1))
        x_max = float(np.percentile(arr, 99))
        if x_min >= x_max:
            continue
        x = np.linspace(x_min, x_max, GRID_SIZE)
        try:
            y = stats.gaussian_kde(arr)(x)
        except Exception as exc:
            logger.warning("KDE failed for '%s': %s", fact, exc)
            continue
        kde_data[fact].update({"x": x, "y": y})

    # ACF
    acf = real_facts.get("acf_abs_returns", np.array([]))
    if acf.size > 1:
        kde_data["acf_abs_returns"] = {
            "lags": np.arange(len(acf), dtype=float),
            "mean": acf.copy(),
        }

    # Depth (structured to match ensemble_plots.plot_depth real_depth param)
    depth_bid = real_facts.get("depth_bid", np.array([]))
    depth_ask = real_facts.get("depth_ask", np.array([]))
    if depth_bid.size > 0 or depth_ask.size > 0:
        kde_data["depth"] = {
            "bid_mean": depth_bid if depth_bid.size > 0 else None,
            "ask_mean": depth_ask if depth_ask.size > 0 else None,
        }

    coverage = np.asarray(real_facts.get("book_coverage", np.array([])), dtype=float)
    if coverage.size:
        kde_data["book_coverage"] = {"mean": float(np.mean(coverage))}

    return kde_data


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_loss(
    sim_kde_data: dict,
    real_kde_data: dict,
    weights: Optional[dict[str, float]] = None,
    specification: LossSpec | None = None,
) -> LossResult:
    """Compute robust, normalized errors between simulated and real facts.

    Distribution facts use a winsorized, scale-normalized Wasserstein distance.
    This remains finite for discrete or constant samples, unlike KDE KL. ACF
    uses MSE over non-zero lags and depth uses per-level relative MSE. The total
    is the weighted mean, so it remains comparable when weights change.

    Parameters
    ----------
    sim_kde_data:
        Output of :func:`lob_sim.loss.aggregate.aggregate_all` (``kde_data`` key).
    real_kde_data:
        Output of :func:`build_real_kde`.
    weights:
        Per-fact weights.  Missing keys default to 1.0.

    Returns
    -------
    :class:`LossResult`
    """
    specification = specification or LossSpec()
    effective_weights = dict(specification.weights)
    if weights is not None:
        effective_weights.update(weights)

    per_fact: dict[str, float] = {}

    # ---- Distribution facts: robust normalized Wasserstein distance ----
    for fact in KDE_FACTS:
        sim_entry = sim_kde_data.get(fact)
        real_entry = real_kde_data.get(fact)
        if sim_entry is None or real_entry is None:
            per_fact[fact] = specification.missing_fact_penalty
            continue
        sim_samples = np.asarray(sim_entry.get("samples", np.array([])), dtype=float)
        real_samples = np.asarray(real_entry.get("samples", np.array([])), dtype=float)
        sim_samples = sim_samples[np.isfinite(sim_samples)]
        real_samples = real_samples[np.isfinite(real_samples)]
        if sim_samples.size == 0 or real_samples.size == 0:
            per_fact[fact] = specification.missing_fact_penalty
            continue
        low, high = np.quantile(real_samples, specification.winsor_quantiles)
        scale = float(high - low)
        if scale <= _EPS:
            scale = max(float(np.mean(np.abs(real_samples))), 1.0)
            low = float(np.min(real_samples)) - scale
            high = float(np.max(real_samples)) + scale
        sim_winsorized = np.clip(sim_samples, low, high)
        real_winsorized = np.clip(real_samples, low, high)
        distance = stats.wasserstein_distance(real_winsorized, sim_winsorized)
        per_fact[fact] = float(distance / scale)

    # ---- ACF: mean squared error ----
    sim_acf = sim_kde_data.get("acf_abs_returns")
    real_acf = real_kde_data.get("acf_abs_returns")
    if sim_acf is not None and real_acf is not None:
        min_len = min(len(sim_acf["mean"]), len(real_acf["mean"]))
        # Lag zero is identically one and contains no calibration information.
        sim_values = np.asarray(sim_acf["mean"][1:min_len], dtype=float)
        real_values = np.asarray(real_acf["mean"][1:min_len], dtype=float)
        valid = np.isfinite(sim_values) & np.isfinite(real_values)
        if np.any(valid):
            diff = sim_values[valid] - real_values[valid]
            squared_error = float(np.mean(diff ** 2))
            per_fact["acf_abs_returns"] = float(np.sqrt(squared_error))

    # ---- Depth: relative mean squared error (bid + ask averaged) ----
    sim_bid = sim_kde_data.get("depth_bid")
    sim_ask = sim_kde_data.get("depth_ask")
    real_depth = real_kde_data.get("depth")
    if sim_bid is not None and sim_ask is not None and real_depth is not None:
        depth_loss = 0.0
        n_components = 0
        for side, sim_entry, key in [
            ("bid", sim_bid, "bid_mean"),
            ("ask", sim_ask, "ask_mean"),
        ]:
            real_arr = real_depth.get(key)
            if real_arr is None:
                continue
            min_len = min(len(sim_entry["mean"]), len(real_arr))
            sim_values = np.asarray(sim_entry["mean"][:min_len], dtype=float)
            real_values = np.asarray(real_arr[:min_len], dtype=float)
            valid = np.isfinite(sim_values) & np.isfinite(real_values)
            if not np.any(valid):
                continue
            relative = (sim_values[valid] - real_values[valid]) / np.maximum(
                np.abs(real_values[valid]), _EPS
            )
            relative_mse = float(np.mean(relative ** 2))
            depth_loss += float(np.sqrt(relative_mse))
            n_components += 1
        if n_components:
            per_fact["depth"] = depth_loss / n_components

    # ---- Two-sided book availability ----
    sim_coverage = sim_kde_data.get("book_coverage")
    real_coverage = real_kde_data.get("book_coverage")
    if sim_coverage is not None and real_coverage is not None:
        target = max(float(real_coverage["mean"]), _EPS)
        per_fact["book_coverage"] = abs(float(sim_coverage["mean"]) - target) / target

    # A missing observable must not make a candidate look artificially better.
    for fact in [*KDE_FACTS, "acf_abs_returns", "depth", "book_coverage"]:
        per_fact.setdefault(fact, specification.missing_fact_penalty)

    # ---- Weighted total ----
    weighted_sum = sum(effective_weights.get(k, 1.0) * v for k, v in per_fact.items())
    weight_sum = sum(effective_weights.get(k, 1.0) for k in per_fact)
    total = weighted_sum / weight_sum if weight_sum else float("inf")

    return LossResult(
        total=total,
        per_fact=per_fact,
        weights={k: effective_weights.get(k, 1.0) for k in per_fact},
        specification=specification.as_dict(),
    )


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def save_comparison_plots(
    sim_kde_data: dict,
    real_kde_data: dict,
    output_dir: str,
    n_scenarios: int,
    sim_level_prices: Optional[pd.DataFrame] = None,
    real_level_prices: Optional[pd.DataFrame] = None,
    sim_n_obs: int | None = None,
    real_n_obs: int | None = None,
) -> None:
    """Generate side-by-side sim-vs-real figures and save them to *output_dir*.

    Each figure has two panels sharing the same x-axis scale:
    - Left  : simulation ensemble (mean ± 1σ band, steelblue)
    - Right : real BTG-OBD data (darkorange)
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    def _p(fname: str) -> str:
        return os.path.join(output_dir, fname)

    def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str = "Density") -> None:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _finish(fig: plt.Figure, path: str) -> None:
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close(fig)

    # ------------------------------------------------------------------
    # KDE facts — shared x range, density on both panels
    # ------------------------------------------------------------------
    kde_specs = [
        ("log_returns",   "Return Distribution",              "Log-return"),
        ("spread",        "Bid-Ask Spread Distribution",       "Spread"),
        ("imbalance",     "Order Book Imbalance Distribution", "(Bid qty − Ask qty) / Total"),
        ("trade_sizes",   "Trade Size Distribution",           "Quantity"),
        ("inter_arrival", "Inter-Arrival Time Distribution",   "Seconds between trades"),
    ]

    for fact, title, xlabel in kde_specs:
        sim_entry  = sim_kde_data.get(fact)
        real_entry = real_kde_data.get(fact)

        fig, (ax_sim, ax_real) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
        fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

        # Compute shared x range
        x_min = x_max = None
        if sim_entry is not None and "x" in sim_entry:
            x_min = float(sim_entry["x"][0])
            x_max = float(sim_entry["x"][-1])
        if real_entry is not None and "x" in real_entry:
            rx0, rx1 = float(real_entry["x"][0]), float(real_entry["x"][-1])
            x_min = min(x_min, rx0) if x_min is not None else rx0
            x_max = max(x_max, rx1) if x_max is not None else rx1

        # Left: simulation
        if sim_entry is not None and "x" in sim_entry and "kdes" in sim_entry:
            x = sim_entry["x"]
            kdes = sim_entry["kdes"]
            mean = kdes.mean(axis=0)
            std  = kdes.std(axis=0)
            ax_sim.plot(x, mean, color="steelblue", lw=1.8, label="Mean KDE")
            ax_sim.fill_between(x, mean - std, mean + std,
                                alpha=0.25, color="steelblue", label="±1σ")
            ax_sim.legend(fontsize=9)
        _style(ax_sim, f"Simulation — {n_scenarios} scenarios", xlabel)
        if x_min is not None:
            ax_sim.set_xlim(x_min, x_max)

        # Right: real data
        if real_entry is not None and "x" in real_entry and "y" in real_entry:
            ax_real.plot(real_entry["x"], real_entry["y"],
                        color="darkorange", lw=1.8, label="Real data KDE")
            ax_real.legend(fontsize=9)
        _style(ax_real, "Real Data — BTG-OBD", xlabel)
        if x_min is not None:
            ax_real.set_xlim(x_min, x_max)

        _finish(fig, _p(f"comparison_{fact}.png"))

    # ------------------------------------------------------------------
    # ACF — shared lag axis
    # ------------------------------------------------------------------
    sim_acf  = sim_kde_data.get("acf_abs_returns")
    real_acf = real_kde_data.get("acf_abs_returns")

    fig, (ax_sim, ax_real) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    fig.suptitle("Volatility Clustering — ACF of |Returns|",
                 fontsize=13, fontweight="bold", y=1.01)

    n_obs = sim_n_obs if sim_n_obs is not None else 3599
    if sim_acf is not None:
        lags = sim_acf["lags"]
        mean = sim_acf["mean"]
        std  = sim_acf["std"]
        ax_sim.bar(lags, mean, color="steelblue", alpha=0.7, width=0.8, label="Mean ACF")
        ax_sim.fill_between(lags, mean - std, mean + std,
                            alpha=0.3, color="steelblue", label="±1σ")
        ci = 1.96 / np.sqrt(max(n_obs, 1))
        ax_sim.axhline( ci, color="r", lw=0.9, ls="--", label="±95% CI")
        ax_sim.axhline(-ci, color="r", lw=0.9, ls="--")
        ax_sim.axhline(0,   color="k", lw=0.5)
        ax_sim.legend(fontsize=9)

    _style(ax_sim, f"Simulation — {n_scenarios} scenarios", "Lag", ylabel="Autocorrelation")

    if real_acf is not None:
        lags_r = real_acf["lags"]
        mean_r = real_acf["mean"]
        ax_real.bar(lags_r, mean_r, color="darkorange", alpha=0.8, width=0.8, label="ACF")
        n_obs_real = real_n_obs if real_n_obs is not None else n_obs
        ci_r = 1.96 / np.sqrt(max(n_obs_real, 1))
        ax_real.axhline( ci_r, color="r", lw=0.9, ls="--", label="±95% CI")
        ax_real.axhline(-ci_r, color="r", lw=0.9, ls="--")
        ax_real.axhline(0,    color="k", lw=0.5)
        ax_real.legend(fontsize=9)

    _style(ax_real, "Real Data — BTG-OBD", "Lag", ylabel="Autocorrelation")
    _finish(fig, _p("comparison_acf.png"))

    # ------------------------------------------------------------------
    # Depth — shared level axis
    # ------------------------------------------------------------------
    sim_bid   = sim_kde_data.get("depth_bid")
    sim_ask   = sim_kde_data.get("depth_ask")
    real_depth = real_kde_data.get("depth")

    fig, (ax_sim, ax_real) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    fig.suptitle("LOB Depth Profile", fontsize=13, fontweight="bold", y=1.01)

    n_levels = 0
    if sim_bid is not None:
        n_levels = max(n_levels, len(sim_bid["mean"]))
    if sim_ask is not None:
        n_levels = max(n_levels, len(sim_ask["mean"]))
    levels = np.arange(1, n_levels + 1) if n_levels else np.array([])

    if sim_bid is not None and levels.size:
        bm, bs = sim_bid["mean"], sim_bid["std"]
        ax_sim.bar(levels[: len(bm)] - 0.2, bm, width=0.35, yerr=bs, capsize=4,
                   color="green", alpha=0.7, label="Bid ±1σ")
    if sim_ask is not None and levels.size:
        am, as_ = sim_ask["mean"], sim_ask["std"]
        ax_sim.bar(levels[: len(am)] + 0.2, am, width=0.35, yerr=as_, capsize=4,
                   color="red", alpha=0.7, label="Ask ±1σ")
    if levels.size:
        ax_sim.set_xticks(levels)
    ax_sim.legend(fontsize=9)
    _style(ax_sim, f"Simulation — {n_scenarios} scenarios", "Level", ylabel="Avg qty")

    if real_depth is not None:
        bm_r = real_depth.get("bid_mean")
        am_r = real_depth.get("ask_mean")
        n_levels_r = max(
            len(bm_r) if bm_r is not None else 0,
            len(am_r) if am_r is not None else 0,
        )
        levels_r = np.arange(1, n_levels_r + 1) if n_levels_r else np.array([])
        if bm_r is not None and levels_r.size:
            ax_real.bar(levels_r[: len(bm_r)] - 0.2, bm_r, width=0.35,
                        color="green", alpha=0.7, label="Bid")
        if am_r is not None and levels_r.size:
            ax_real.bar(levels_r[: len(am_r)] + 0.2, am_r, width=0.35,
                        color="red", alpha=0.7, label="Ask")
        if levels_r.size:
            ax_real.set_xticks(levels_r)
        ax_real.legend(fontsize=9)
    _style(ax_real, "Real Data — BTG-OBD", "Level", ylabel="Avg qty")
    _finish(fig, _p("comparison_depth.png"))

    # ------------------------------------------------------------------
    # L1/L3/L5 bid and ask prices over elapsed session time
    # ------------------------------------------------------------------
    if (
        sim_level_prices is not None and not sim_level_prices.empty
        and real_level_prices is not None and not real_level_prices.empty
    ):
        fig, (ax_sim, ax_real) = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
        fig.suptitle("LOB Level Prices Over Time", fontsize=13, fontweight="bold", y=1.01)
        colors = {
            "bid": {1: "#0b3c5d", 3: "#328cc1", 5: "#7fb3d5"},
            "ask": {1: "#7b241c", 3: "#c0392b", 5: "#f1948a"},
        }
        styles = {1: "-", 3: "--", 5: ":"}

        def _plot_levels(ax: plt.Axes, frame: pd.DataFrame, title: str) -> None:
            for side in ("bid", "ask"):
                for level in (1, 3, 5):
                    column = f"{side}_l{level}"
                    if column not in frame:
                        continue
                    valid = frame[["elapsed_seconds", column]].dropna()
                    if valid.empty:
                        continue
                    ax.plot(
                        valid["elapsed_seconds"], valid[column],
                        color=colors[side][level], linestyle=styles[level],
                        lw=1.1, label=f"L{level} {side.title()}",
                    )
            _style(ax, title, "Elapsed time (s)", ylabel="Price")
            ax.legend(ncol=3, fontsize=8)

        _plot_levels(ax_sim, sim_level_prices, f"Simulation — mean of {n_scenarios}")
        _plot_levels(ax_real, real_level_prices, "Real Data — mean across dates")
        _finish(fig, _p("comparison_level_prices.png"))

    logger.info("Comparison plots written to: %s", output_dir)


def compute_sim_level_prices(
    scenario_root: str,
    levels: tuple[int, ...] = (1, 3, 5),
) -> pd.DataFrame:
    """Mean L1/L3/L5 price paths across simulation scenario directories."""
    scenario_dirs = sorted(
        path for path in Path(scenario_root).glob("scenario_*") if path.is_dir()
    )
    frames: list[pd.DataFrame] = []
    for scenario_dir in scenario_dirs:
        path = scenario_dir / "lob_snapshots.parquet"
        if not path.exists():
            continue
        lob = pd.read_parquet(path)
        if lob.empty:
            continue
        frame = pd.DataFrame({"elapsed_seconds": sorted(lob["timestamp"].unique())})
        for level in levels:
            rows = lob[lob["level"] == level][["timestamp", "bid_price", "ask_price"]]
            rows = rows.rename(columns={
                "timestamp": "elapsed_seconds",
                "bid_price": f"bid_l{level}",
                "ask_price": f"ask_l{level}",
            })
            frame = frame.merge(rows, on="elapsed_seconds", how="left")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return (
        pd.concat(frames, ignore_index=True)
        .groupby("elapsed_seconds", as_index=False)
        .mean(numeric_only=True)
    )


# ---------------------------------------------------------------------------
# Report I/O
# ---------------------------------------------------------------------------

def save_loss_report(loss: LossResult, output_dir: str) -> None:
    """Write ``loss_report.md`` and ``loss_summary.json`` to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)

    # Markdown table
    lines = [
        "# Sim vs Real — Loss Report\n",
        f"**Total weighted loss**: `{loss.total:.6f}`\n",
        "",
        "| Fact | Loss | Weight |",
        "|------|-----:|-------:|",
    ]
    for fact, value in sorted(loss.per_fact.items()):
        w = loss.weights.get(fact, 1.0)
        loss_type = (
            "normalized Wasserstein" if fact in KDE_FACTS
            else "relative MSE" if fact == "depth"
            else "MSE"
        )
        lines.append(f"| `{fact}` ({loss_type}) | {value:.6f} | {w:.2f} |")

    md_path = os.path.join(output_dir, "loss_report.md")
    Path(md_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # JSON summary
    summary = {
        "total": loss.total,
        "per_fact": loss.per_fact,
        "weights": loss.weights,
        "specification": loss.specification,
    }
    json_path = os.path.join(output_dir, "loss_summary.json")
    Path(json_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Loss report written to: %s", output_dir)


def comparison_statistics(
    sim_kde_data: dict,
    real_kde_data: dict,
) -> list[tuple[str, float, float]]:
    """Return directly comparable statistics from the pooled fact samples."""

    def _samples(data: dict, fact: str) -> np.ndarray:
        entry = data.get(fact, {})
        values = np.asarray(entry.get("samples", np.array([])), dtype=float)
        return values[np.isfinite(values)]

    rows: list[tuple[str, float, float]] = []

    def _add(label: str, sim_value: float, real_value: float) -> None:
        rows.append((label, float(sim_value), float(real_value)))

    sim_returns = _samples(sim_kde_data, "log_returns")
    real_returns = _samples(real_kde_data, "log_returns")
    if sim_returns.size and real_returns.size:
        _add("Return standard deviation", np.std(sim_returns), np.std(real_returns))
        _add("Non-zero return fraction", np.mean(sim_returns != 0), np.mean(real_returns != 0))

    sim_spread = _samples(sim_kde_data, "spread")
    real_spread = _samples(real_kde_data, "spread")
    if sim_spread.size and real_spread.size:
        _add("Mean spread", np.mean(sim_spread), np.mean(real_spread))
        _add("Spread standard deviation", np.std(sim_spread), np.std(real_spread))
        real_positive = real_spread[real_spread > 0]
        tick = float(np.min(real_positive)) if real_positive.size else 0.0
        if tick > 0:
            threshold = tick * (1.0 + 1e-6)
            _add(
                "Spread wider than one tick",
                np.mean(sim_spread > threshold),
                np.mean(real_spread > threshold),
            )

    sim_imbalance = _samples(sim_kde_data, "imbalance")
    real_imbalance = _samples(real_kde_data, "imbalance")
    if sim_imbalance.size and real_imbalance.size:
        _add("Mean imbalance", np.mean(sim_imbalance), np.mean(real_imbalance))
        _add("Imbalance standard deviation", np.std(sim_imbalance), np.std(real_imbalance))
        for percentile in (5, 50, 95):
            _add(
                f"Imbalance p{percentile:02d}",
                np.percentile(sim_imbalance, percentile),
                np.percentile(real_imbalance, percentile),
            )

    for fact, label in (
        ("trade_sizes", "Trade size"),
        ("inter_arrival", "Inter-arrival time"),
    ):
        sim_values = _samples(sim_kde_data, fact)
        real_values = _samples(real_kde_data, fact)
        if not sim_values.size or not real_values.size:
            continue
        _add(f"Mean {label.lower()}", np.mean(sim_values), np.mean(real_values))
        _add(f"Median {label.lower()}", np.median(sim_values), np.median(real_values))
        _add(f"{label} p95", np.percentile(sim_values, 95), np.percentile(real_values, 95))
        _add(f"{label} p99", np.percentile(sim_values, 99), np.percentile(real_values, 99))
        if fact == "inter_arrival":
            _add("Zero inter-arrival fraction", np.mean(sim_values == 0), np.mean(real_values == 0))

    sim_acf = sim_kde_data.get("acf_abs_returns", {}).get("mean")
    real_acf = real_kde_data.get("acf_abs_returns", {}).get("mean")
    if sim_acf is not None and real_acf is not None:
        limit = min(11, len(sim_acf), len(real_acf))
        if limit > 1:
            _add(
                "Mean absolute-return ACF, lags 1-10",
                np.nanmean(np.asarray(sim_acf)[1:limit]),
                np.nanmean(np.asarray(real_acf)[1:limit]),
            )

    real_depth = real_kde_data.get("depth", {})
    for side in ("bid", "ask"):
        sim_depth = sim_kde_data.get(f"depth_{side}", {}).get("mean")
        real_values = real_depth.get(f"{side}_mean")
        if sim_depth is None or real_values is None:
            continue
        for level, (sim_value, real_value) in enumerate(
            zip(sim_depth, real_values), start=1
        ):
            _add(f"{side.title()} depth L{level}", sim_value, real_value)

    sim_coverage = sim_kde_data.get("book_coverage", {}).get("mean")
    real_coverage = real_kde_data.get("book_coverage", {}).get("mean")
    if sim_coverage is not None and real_coverage is not None:
        _add("Two-sided book coverage", sim_coverage, real_coverage)

    return rows


def save_statistics_report(
    sim_kde_data: dict,
    real_kde_data: dict,
    output_dir: str,
) -> None:
    """Write pooled simulation-versus-real summary statistics to Markdown."""
    os.makedirs(output_dir, exist_ok=True)
    lines = [
        "# Sim vs Real — Pooled Statistics\n",
        "Statistics pool every simulated scenario and every configured real-data window.",
        "",
        "| Statistic | Simulation | Real data | Absolute difference | Relative difference |",
        "|-----------|-----------:|----------:|--------------------:|--------------------:|",
    ]
    for label, sim_value, real_value in comparison_statistics(sim_kde_data, real_kde_data):
        absolute = sim_value - real_value
        # A percentage is not interpretable when the values cross zero (most
        # notably mean/median imbalance); the absolute difference remains shown.
        relative = (
            f"{absolute / abs(real_value):+.2%}"
            if abs(real_value) > _EPS and sim_value * real_value >= 0 else "—"
        )
        lines.append(
            f"| {label} | {sim_value:.6g} | {real_value:.6g} | "
            f"{absolute:+.6g} | {relative} |"
        )
    path = Path(output_dir) / "statistics_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Statistics report written to: %s", path)
