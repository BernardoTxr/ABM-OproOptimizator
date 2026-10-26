"""Paired cluster bootstrap for matched simulation-ensemble losses."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from lob_sim.loss.aggregate import KDE_FACTS
from lob_sim.loss.compare import LossSpec, build_real_kde, compute_loss
from lob_sim.loss.facts import compute_facts


def aggregate_facts_for_loss(facts: list[dict[str, np.ndarray]]) -> dict:
    """Build the subset of aggregate data consumed by ``compute_loss``."""
    result: dict = {}
    for key in KDE_FACTS:
        arrays = [np.asarray(item[key]) for item in facts if item[key].size]
        if arrays:
            result[key] = {"samples": np.concatenate(arrays)}

    acfs = [item["acf_abs_returns"] for item in facts if item["acf_abs_returns"].size > 1]
    if acfs:
        length = min(map(len, acfs))
        matrix = np.asarray([values[:length] for values in acfs])
        result["acf_abs_returns"] = {
            "mean": np.nanmean(matrix, axis=0),
            "std": np.nanstd(matrix, axis=0),
        }
    for key in ("depth_bid", "depth_ask"):
        arrays = [item[key] for item in facts if item[key].size]
        if arrays:
            length = min(map(len, arrays))
            matrix = np.asarray([values[:length] for values in arrays])
            result[key] = {"mean": matrix.mean(axis=0), "std": matrix.std(axis=0)}
    coverage = [item["book_coverage"][0] for item in facts if item["book_coverage"].size]
    if coverage:
        result["book_coverage"] = {"mean": float(np.mean(coverage))}
    return result


def _scenario_dirs(root: str) -> list[Path]:
    return sorted(path for path in Path(root).glob("scenario_*") if path.is_dir())


def _metadata(path: Path) -> dict:
    metadata_path = path / "metadata.yaml"
    if not metadata_path.exists():
        raise ValueError(f"Missing matched-scenario metadata: {metadata_path}")
    return yaml.safe_load(metadata_path.read_text())


def paired_bootstrap_loss(
    baseline_dir: str,
    candidate_dir: str,
    real_facts: dict[str, np.ndarray],
    n_bootstrap: int = 500,
    seed: int = 2026,
    specification: LossSpec | None = None,
) -> dict:
    """Bootstrap candidate-minus-baseline pooled loss over matched scenarios."""
    baseline_paths = _scenario_dirs(baseline_dir)
    candidate_paths = _scenario_dirs(candidate_dir)
    if len(baseline_paths) != len(candidate_paths) or not baseline_paths:
        raise ValueError("Baseline and candidate require the same non-zero scenario count")

    match_keys = ("scenario_idx", "random_seed", "real_date", "window_start_utc")
    for baseline_path, candidate_path in zip(baseline_paths, candidate_paths):
        baseline_meta = _metadata(baseline_path)
        candidate_meta = _metadata(candidate_path)
        if any(baseline_meta.get(key) != candidate_meta.get(key) for key in match_keys):
            raise ValueError(
                f"Scenario pair is not matched: {baseline_path.name} / {candidate_path.name}"
            )

    baseline_facts = [compute_facts(str(path)) for path in baseline_paths]
    candidate_facts = [compute_facts(str(path)) for path in candidate_paths]
    real_aggregate = build_real_kde(real_facts)
    specification = specification or LossSpec()

    def score(items: list[dict[str, np.ndarray]]) -> tuple[float, dict[str, float]]:
        loss = compute_loss(
            aggregate_facts_for_loss(items),
            real_aggregate,
            specification=specification,
        )
        return loss.total, loss.per_fact

    baseline_point, baseline_components = score(baseline_facts)
    candidate_point, candidate_components = score(candidate_facts)
    rng = np.random.default_rng(seed)
    totals = np.empty((n_bootstrap, 3), dtype=float)
    component_names = sorted(baseline_components)
    component_deltas = {name: np.empty(n_bootstrap, dtype=float) for name in component_names}
    for bootstrap_idx in range(n_bootstrap):
        indices = rng.integers(0, len(baseline_facts), len(baseline_facts))
        baseline_total, baseline_per_fact = score([baseline_facts[index] for index in indices])
        candidate_total, candidate_per_fact = score([candidate_facts[index] for index in indices])
        totals[bootstrap_idx] = (
            baseline_total,
            candidate_total,
            candidate_total - baseline_total,
        )
        for name in component_names:
            component_deltas[name][bootstrap_idx] = (
                candidate_per_fact[name] - baseline_per_fact[name]
            )

    def interval(values: np.ndarray) -> list[float]:
        return [float(value) for value in np.percentile(values, [2.5, 97.5])]

    delta = totals[:, 2]
    relative = -delta / np.maximum(totals[:, 0], np.finfo(float).eps)
    return {
        "method": "paired scenario-cluster bootstrap",
        "n_scenarios": len(baseline_facts),
        "n_bootstrap": int(n_bootstrap),
        "bootstrap_seed": int(seed),
        "loss_specification": specification.as_dict(),
        "baseline_loss": float(baseline_point),
        "candidate_loss": float(candidate_point),
        "candidate_minus_baseline": float(candidate_point - baseline_point),
        "candidate_minus_baseline_ci95": interval(delta),
        "relative_improvement": float((baseline_point - candidate_point) / baseline_point),
        "relative_improvement_ci95": interval(relative),
        "probability_candidate_better": float(np.mean(delta < 0)),
        "component_deltas": {
            name: {
                "point": float(candidate_components[name] - baseline_components[name]),
                "ci95": interval(values),
            }
            for name, values in component_deltas.items()
        },
    }
