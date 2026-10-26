"""Grouped t-statistic simulated-method-of-moments objective.

The empirical target and weights are estimated once from date-clustered real
windows.  Candidate simulations are reduced to the same signed moment vector,
then scored with diagonal inverse bootstrap variance and equal group priority.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from lob_sim.loss.facts import compute_facts
from lob_sim.loss.real_facts import compute_real_window_fact_panel


SMM_VERSION = "smm_v4"
SUPPORTED_SMM_VERSIONS = frozenset({SMM_VERSION})
EXPECTED_GROUPS = (
    "returns",
    "spread",
    "imbalance",
    "trade_size",
    "inter_arrival",
    "depth",
)

AVAILABLE_MOMENTS = frozenset({
    "return_mean", "return_std", "return_q01", "return_q05",
    "return_q95", "return_q99",
    "spread_p1", "spread_p2", "spread_p3plus",
    "imbalance_mean", "imbalance_q10", "imbalance_q50", "imbalance_q90",
    "trade_size_q50", "trade_size_q90", "trade_size_q99",
    "inter_arrival_p0", "inter_arrival_cdf1", "inter_arrival_cdf5",
    "inter_arrival_q90", "inter_arrival_q99",
    "abs_return_acf_lag1", "abs_return_acf_lag5", "abs_return_acf_lag10",
    "abs_return_acf_lag20", "abs_return_acf_lag50",
    *{f"depth_bid_l{level}" for level in range(1, 6)},
    *{f"depth_ask_l{level}" for level in range(1, 6)},
})


@dataclass(frozen=True)
class SMMSpec:
    """Frozen choices defining one grouped SMM objective."""

    groups: dict[str, tuple[str, ...]]
    tick_size: float = 0.01
    bootstrap_draws: int = 5000
    bootstrap_seed: int = 20260907
    min_standard_error: float = 1e-12
    version: str = SMM_VERSION

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SMMSpec":
        version = str(values.get("version", ""))
        if version not in SUPPORTED_SMM_VERSIONS:
            raise ValueError(f"Unsupported SMM version: {version}")
        raw_groups = values.get("groups")
        if not isinstance(raw_groups, dict) or tuple(raw_groups) != EXPECTED_GROUPS:
            raise ValueError(
                f"SMM groups must be declared in this order: {EXPECTED_GROUPS}"
            )
        groups: dict[str, tuple[str, ...]] = {}
        used: list[str] = []
        for group, raw_moments in raw_groups.items():
            if not isinstance(raw_moments, list) or not raw_moments:
                raise ValueError(f"SMM group '{group}' must contain moments")
            moments = tuple(str(moment) for moment in raw_moments)
            unknown = sorted(set(moments) - AVAILABLE_MOMENTS)
            if unknown:
                raise ValueError(f"Unknown SMM moments in '{group}': {unknown}")
            groups[str(group)] = moments
            used.extend(moments)
        duplicates = sorted({name for name in used if used.count(name) > 1})
        if duplicates:
            raise ValueError(f"SMM moments cannot occur in multiple groups: {duplicates}")

        tick_size = float(values.get("tick_size", 0.01))
        bootstrap = values.get("standard_errors", {})
        draws = int(bootstrap.get("bootstrap_draws", 5000))
        seed = int(bootstrap.get("bootstrap_seed", 20260907))
        floor = float(bootstrap.get("min_standard_error", 1e-12))
        if tick_size <= 0 or not np.isfinite(tick_size):
            raise ValueError("tick_size must be positive and finite")
        if draws < 2:
            raise ValueError("bootstrap_draws must be at least two")
        if floor <= 0 or not np.isfinite(floor):
            raise ValueError("min_standard_error must be positive and finite")
        return cls(
            groups=groups,
            tick_size=tick_size,
            bootstrap_draws=draws,
            bootstrap_seed=seed,
            min_standard_error=floor,
            version=version,
        )

    @property
    def moment_names(self) -> tuple[str, ...]:
        return tuple(moment for group in self.groups.values() for moment in group)

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "objective": "equal_group_diagonal_t_statistic_smm",
            "target": "equal_date_mean_of_equal_clock_window_moments",
            "weighting": "inverse_date_cluster_bootstrap_variance_of_mean",
            "tick_size": self.tick_size,
            "standard_errors": {
                "bootstrap_draws": self.bootstrap_draws,
                "bootstrap_seed": self.bootstrap_seed,
                "min_standard_error": self.min_standard_error,
            },
            "groups": {name: list(moments) for name, moments in self.groups.items()},
        }


@dataclass(frozen=True)
class SMMTarget:
    """Empirical moments and frozen diagonal weighting information."""

    moment_names: tuple[str, ...]
    target: dict[str, float]
    standard_errors: dict[str, float]
    covariance: list[list[float]]
    date_moments: list[dict[str, Any]]
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "SMMTarget":
        return cls(
            moment_names=tuple(values["moment_names"]),
            target={key: float(value) for key, value in values["target"].items()},
            standard_errors={
                key: float(value) for key, value in values["standard_errors"].items()
            },
            covariance=[[float(value) for value in row] for row in values["covariance"]],
            date_moments=list(values["date_moments"]),
            metadata=dict(values["metadata"]),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "moment_names": list(self.moment_names),
            "target": self.target,
            "standard_errors": self.standard_errors,
            "covariance": self.covariance,
            "date_moments": self.date_moments,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SMMResult:
    """Scalar loss plus complete auditable moment decomposition."""

    total: float
    per_group: dict[str, float]
    simulated_moments: dict[str, float]
    target_moments: dict[str, float]
    raw_differences: dict[str, float]
    standardized_residuals: dict[str, float]
    scenario_moments: list[dict[str, Any]]
    specification: dict[str, Any]


def _finite(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _summary(values: Any, statistic: str, quantile: float | None = None) -> float:
    array = _finite(values)
    if not array.size:
        return float("nan")
    if statistic == "mean":
        return float(np.mean(array))
    if statistic == "std":
        return float(np.std(array, ddof=0))
    if statistic == "quantile" and quantile is not None:
        return float(np.quantile(array, quantile))
    raise ValueError(f"Unsupported summary statistic: {statistic}")


def compute_moment_vector(facts: dict[str, np.ndarray], spec: SMMSpec) -> dict[str, float]:
    """Reduce one empirical window or simulation scenario to signed moments."""
    returns = _finite(facts.get("log_returns", []))
    spread = _finite(facts.get("spread", []))
    imbalance = _finite(facts.get("imbalance", []))
    trade_size = _finite(facts.get("trade_sizes", []))
    inter_arrival = _finite(facts.get("inter_arrival", []))
    acf = np.asarray(facts.get("acf_abs_returns", []), dtype=float)
    depth_bid = np.asarray(facts.get("depth_bid", []), dtype=float)
    depth_ask = np.asarray(facts.get("depth_ask", []), dtype=float)
    spread_ticks = np.rint(spread / spec.tick_size).astype(int) if spread.size else np.array([])

    values: dict[str, float] = {
        "return_mean": _summary(returns, "mean"),
        "return_std": _summary(returns, "std"),
        "return_q01": _summary(returns, "quantile", 0.01),
        "return_q05": _summary(returns, "quantile", 0.05),
        "return_q95": _summary(returns, "quantile", 0.95),
        "return_q99": _summary(returns, "quantile", 0.99),
        "spread_p1": float(np.mean(spread_ticks == 1)) if spread_ticks.size else float("nan"),
        "spread_p2": float(np.mean(spread_ticks == 2)) if spread_ticks.size else float("nan"),
        "spread_p3plus": float(np.mean(spread_ticks >= 3)) if spread_ticks.size else float("nan"),
        "imbalance_mean": _summary(imbalance, "mean"),
        "imbalance_q10": _summary(imbalance, "quantile", 0.10),
        "imbalance_q50": _summary(imbalance, "quantile", 0.50),
        "imbalance_q90": _summary(imbalance, "quantile", 0.90),
        "trade_size_q50": _summary(trade_size, "quantile", 0.50),
        "trade_size_q90": _summary(trade_size, "quantile", 0.90),
        "trade_size_q99": _summary(trade_size, "quantile", 0.99),
        "inter_arrival_p0": float(np.mean(inter_arrival == 0)) if inter_arrival.size else float("nan"),
        "inter_arrival_cdf1": float(np.mean(inter_arrival <= 1)) if inter_arrival.size else float("nan"),
        "inter_arrival_cdf5": float(np.mean(inter_arrival <= 5)) if inter_arrival.size else float("nan"),
        "inter_arrival_q90": _summary(inter_arrival, "quantile", 0.90),
        "inter_arrival_q99": _summary(inter_arrival, "quantile", 0.99),
    }
    for lag in (1, 5, 10, 20, 50):
        values[f"abs_return_acf_lag{lag}"] = (
            float(acf[lag]) if acf.size > lag and np.isfinite(acf[lag]) else float("nan")
        )
    for level in range(1, 6):
        index = level - 1
        values[f"depth_bid_l{level}"] = (
            float(depth_bid[index])
            if depth_bid.size > index and np.isfinite(depth_bid[index]) else float("nan")
        )
        values[f"depth_ask_l{level}"] = (
            float(depth_ask[index])
            if depth_ask.size > index and np.isfinite(depth_ask[index]) else float("nan")
        )

    selected = {name: values[name] for name in spec.moment_names}
    missing = [name for name, value in selected.items() if not np.isfinite(value)]
    if missing:
        raise ValueError(f"Non-finite SMM moments: {missing}")
    return selected


def estimate_target_from_window_moments(
    window_moments: pd.DataFrame,
    spec: SMMSpec,
) -> tuple[dict[str, float], dict[str, float], np.ndarray, pd.DataFrame]:
    """Estimate equal-date targets and date-cluster bootstrap uncertainty."""
    target, date_moments = aggregate_target_from_window_moments(
        window_moments, spec
    )
    matrix = date_moments.to_numpy(dtype=float)
    rng = np.random.default_rng(spec.bootstrap_seed)
    indices = rng.integers(0, len(matrix), size=(spec.bootstrap_draws, len(matrix)))
    bootstrap_means = matrix[indices].mean(axis=1)
    covariance = np.atleast_2d(np.cov(bootstrap_means, rowvar=False, ddof=1))
    standard_errors = np.sqrt(np.diag(covariance))
    degenerate = [
        name for name, value in zip(spec.moment_names, standard_errors)
        if not np.isfinite(value) or value <= spec.min_standard_error
    ]
    if degenerate:
        raise ValueError(
            "Empirical SMM moments have degenerate standard errors; remove or "
            f"redefine them rather than silently flooring weights: {degenerate}"
        )
    return (
        target,
        dict(zip(spec.moment_names, standard_errors.astype(float))),
        covariance,
        date_moments,
    )


def aggregate_target_from_window_moments(
    window_moments: pd.DataFrame,
    spec: SMMSpec,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Aggregate empirical means without estimating a weighting matrix."""
    required = {"date", "window_start_utc", *spec.moment_names}
    missing = sorted(required - set(window_moments.columns))
    if missing:
        raise ValueError(f"Window-moment panel is missing columns: {missing}")
    numeric = window_moments.loc[:, spec.moment_names].apply(pd.to_numeric)
    if not np.all(np.isfinite(numeric.to_numpy(dtype=float))):
        raise ValueError("Window-moment panel contains non-finite values")
    date_moments = (
        pd.concat([window_moments[["date"]], numeric], axis=1)
        .groupby("date", sort=True)[list(spec.moment_names)]
        .mean()
    )
    if len(date_moments) < 2:
        raise ValueError("Date-cluster SMM requires at least two dates")
    matrix = date_moments.to_numpy(dtype=float)
    target_values = matrix.mean(axis=0)
    return dict(zip(spec.moment_names, target_values.astype(float))), date_moments


def _target_signature(
    symbol: str,
    dates: list[str],
    real_config: dict[str, Any],
    spec: SMMSpec,
    weight_target: SMMTarget | None = None,
) -> str:
    data_dir = Path(real_config["data_dir"]).resolve()
    depth = int(real_config.get("depth", 10))
    files = []
    for date in sorted(dates):
        path = data_dir / symbol / date / f"d{depth}" / "events.parquet"
        stat = path.stat()
        files.append((str(path), stat.st_size, stat.st_mtime_ns))
    payload = {
        "engine": "smm-target-v2-second-resolution-arrivals",
        "symbol": symbol,
        "dates": sorted(dates),
        "real_config": {
            "depth": depth,
            "lob_depth": int(real_config.get("lob_depth", 5)),
            "window_starts_utc": real_config.get("window_starts_utc"),
            "window_duration_hours": float(real_config["window_duration_hours"]),
        },
        "spec": spec.as_dict(),
        "files": files,
        "weight_target_sha256": (
            hashlib.sha256(json.dumps(
                weight_target.as_dict(), sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            if weight_target is not None else None
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_or_build_empirical_target(
    symbol: str,
    dates: list[str],
    real_config: dict[str, Any],
    spec: SMMSpec,
    cache_path: str | Path,
    window_moment_cache: str | Path | None = None,
    weight_target: SMMTarget | None = None,
) -> SMMTarget:
    """Load a matching empirical SMM artifact or build it atomically."""
    if weight_target is not None and weight_target.moment_names != spec.moment_names:
        raise ValueError("SMM weight target moment order does not match the specification")
    signature = _target_signature(symbol, dates, real_config, spec, weight_target)
    path = Path(cache_path)
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("metadata", {}).get("signature") == signature:
            return SMMTarget.from_dict(payload)

    source_panel_path = Path(window_moment_cache) if window_moment_cache else None
    source_panel_sha256 = None
    if source_panel_path is not None and source_panel_path.exists():
        frame = pd.read_parquet(source_panel_path)
        required = {"date", "window_start_utc", *spec.moment_names}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(
                f"Cached SMM window panel is missing required columns: {missing}"
            )
        frame = frame[["date", "window_start_utc", *spec.moment_names]].copy()
        frame["date"] = frame["date"].astype(str)
        observed_dates = sorted(frame["date"].unique())
        if observed_dates != sorted(str(date) for date in dates):
            raise ValueError(
                "Cached SMM window panel dates do not match the requested split"
            )
        source_panel_sha256 = hashlib.sha256(source_panel_path.read_bytes()).hexdigest()
    else:
        fact_panel = compute_real_window_fact_panel(symbol, dates, real_config)
        rows: list[dict[str, Any]] = []
        for record in fact_panel:
            rows.append({
                "date": record["date"],
                "window_start_utc": record["window_start_utc"],
                **compute_moment_vector(record["facts"], spec),
            })
        frame = pd.DataFrame(rows)
    if weight_target is None:
        target, standard_errors, covariance, date_moments = (
            estimate_target_from_window_moments(frame, spec)
        )
    else:
        target, date_moments = aggregate_target_from_window_moments(frame, spec)
        standard_errors = dict(weight_target.standard_errors)
        covariance = np.asarray(weight_target.covariance, dtype=float)
    artifact = SMMTarget(
        moment_names=spec.moment_names,
        target=target,
        standard_errors=standard_errors,
        covariance=covariance.tolist(),
        date_moments=[
            {"date": str(date), **{name: float(row[name]) for name in spec.moment_names}}
            for date, row in date_moments.iterrows()
        ],
        metadata={
            "version": spec.version,
            "signature": signature,
            "symbol": symbol,
            "dates": list(dates),
            "n_dates": len(date_moments),
            "n_windows": len(frame),
            "cluster_unit": "date",
            "within_date_aggregation": "equal_clock_window_mean",
            "specification": spec.as_dict(),
            "source_window_panel": (
                str(source_panel_path.resolve()) if source_panel_path is not None else None
            ),
            "source_window_panel_sha256": source_panel_sha256,
            "weight_source": (
                "self" if weight_target is None else "frozen_external_target"
            ),
            "weight_source_signature": (
                None if weight_target is None
                else weight_target.metadata.get("signature")
            ),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(artifact.as_dict(), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    frame.to_parquet(path.with_suffix(".windows.parquet"), index=False)
    return artifact


def _scenario_records(
    scenario_dirs: Iterable[str | Path], spec: SMMSpec
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in sorted(Path(path) for path in scenario_dirs):
        metadata_path = raw_path / "metadata.yaml"
        metadata = yaml.safe_load(metadata_path.read_text()) if metadata_path.exists() else {}
        rows.append({
            "scenario": raw_path.name,
            "date": str(metadata.get("real_date", raw_path.name)),
            "window_start_utc": metadata.get("window_start_utc"),
            **compute_moment_vector(compute_facts(str(raw_path)), spec),
        })
    if not rows:
        raise ValueError("SMM evaluation requires at least one scenario directory")
    return rows


def compute_smm_loss(
    scenario_dirs: Iterable[str | Path],
    target: SMMTarget,
    spec: SMMSpec,
) -> SMMResult:
    """Compute equal-group diagonal t-statistic SMM for an ensemble."""
    if target.moment_names != spec.moment_names:
        raise ValueError("SMM target moment order does not match the loss specification")
    rows = _scenario_records(scenario_dirs, spec)
    frame = pd.DataFrame(rows)
    # Give every sampled empirical date equal influence even when the balanced
    # schedule assigns one date two scenarios and another date one.
    date_means = frame.groupby("date", sort=True)[list(spec.moment_names)].mean()
    simulated = date_means.mean(axis=0)

    return score_smm_moments(
        {name: float(simulated[name]) for name in spec.moment_names},
        target,
        spec,
        scenario_moments=rows,
    )


def score_smm_moments(
    simulated_moments: dict[str, float],
    target: SMMTarget,
    spec: SMMSpec,
    scenario_moments: list[dict[str, Any]] | None = None,
) -> SMMResult:
    """Score an already aggregated vector; useful for audits and unit tests."""
    if target.moment_names != spec.moment_names:
        raise ValueError("SMM target moment order does not match the loss specification")
    missing = sorted(set(spec.moment_names) - set(simulated_moments))
    if missing:
        raise ValueError(f"Simulated SMM vector is missing moments: {missing}")

    raw_differences: dict[str, float] = {}
    residuals: dict[str, float] = {}
    for name in spec.moment_names:
        simulated_value = float(simulated_moments[name])
        if not np.isfinite(simulated_value):
            raise ValueError(f"Simulated SMM moment is non-finite: {name}")
        difference = float(simulated_value - target.target[name])
        raw_differences[name] = difference
        residuals[name] = difference / target.standard_errors[name]

    per_group = {
        group: float(np.mean([residuals[name] ** 2 for name in moments]))
        for group, moments in spec.groups.items()
    }
    total = float(np.mean(list(per_group.values())))
    return SMMResult(
        total=total,
        per_group=per_group,
        simulated_moments={name: float(simulated_moments[name]) for name in spec.moment_names},
        target_moments=dict(target.target),
        raw_differences=raw_differences,
        standardized_residuals=residuals,
        scenario_moments=scenario_moments or [],
        specification=spec.as_dict(),
    )
