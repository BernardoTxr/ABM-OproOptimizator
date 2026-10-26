from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scalar metric groups for the report sections
# ---------------------------------------------------------------------------

_SCALAR_GROUPS: dict[str, dict[str, str]] = {
    "Log Returns": {
        "mean_return":  "Mean log-return",
        "vol":          "Realized volatility (std dev)",
        "skew_returns": "Skewness",
        "kurt_returns": "Excess kurtosis",
    },
    "Volatility Clustering": {
        "mean_acf_1_10": "Mean ACF |returns|, lags 1–10",
    },
    "Bid-Ask Spread": {
        "mean_spread":   "Mean spread",
        "std_spread":    "Std dev of spread",
        "median_spread": "Median spread",
    },
    "LOB Depth (level 1)": {
        "mean_depth_bid_l1": "Mean bid depth (level 1)",
        "mean_depth_ask_l1": "Mean ask depth (level 1)",
    },
    "Imbalance": {
        "mean_imbalance": "Mean imbalance",
        "std_imbalance":  "Std dev of imbalance",
    },
    "Trade Sizes": {
        "mean_trade_size": "Mean trade size",
        "std_trade_size":  "Std dev of trade size",
    },
    "Inter-Arrival Times": {
        "mean_inter_arrival": "Mean inter-arrival (s)",
        "cv_inter_arrival":   "Coefficient of variation",
    },
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_report(
    scalars_df: pd.DataFrame,
    save_path: str,
    n_scenarios: int | None = None,
) -> None:
    """
    Write an ensemble statistical summary report in Markdown.

    For each stylized fact, the report contains a table of cross-scenario
    statistics (mean, std, and five percentiles) for the corresponding
    per-scenario scalar metric(s).

    Parameters
    ----------
    scalars_df : DataFrame
        One row per scenario; columns as produced by ``aggregate_all()``.
    save_path : str
        Output file path (e.g. ``results/large_scenario/ensemble_report.md``).
    n_scenarios : int | None
        Displayed scenario count; defaults to ``len(scalars_df)``.
    """
    n = n_scenarios if n_scenarios is not None else len(scalars_df)
    lines: list[str] = []

    lines += [
        "# Ensemble Stylized-Facts Report",
        "",
        f"**Scenarios analysed:** {n}  ",
        "",
        "For each stylized fact the table below reports cross-scenario statistics "
        "(mean, standard deviation, and percentiles) of the corresponding "
        "per-scenario summary metric.",
        "",
    ]

    for section, metrics in _SCALAR_GROUPS.items():
        lines += [f"## {section}", ""]
        lines += [
            "| Metric | Mean | Std | p5 | p25 | p50 | p75 | p95 |",
            "|--------|------|-----|----|-----|-----|-----|-----|",
        ]
        for col, label in metrics.items():
            if col not in scalars_df.columns:
                continue
            series = scalars_df[col].dropna()
            if series.empty:
                lines.append(f"| {label} | — | — | — | — | — | — | — |")
                continue
            p5, p25, p50, p75, p95 = np.percentile(series, [5, 25, 50, 75, 95])
            lines.append(
                f"| {label} "
                f"| {series.mean():.4g} "
                f"| {series.std():.4g} "
                f"| {p5:.4g} "
                f"| {p25:.4g} "
                f"| {p50:.4g} "
                f"| {p75:.4g} "
                f"| {p95:.4g} |"
            )
        lines.append("")

    with open(save_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"Report saved to: {save_path}")
