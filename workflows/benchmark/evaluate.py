#!/usr/bin/env python3
"""Confirm development incumbents and evaluate them on reused later dates."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from workflows import REPO_ROOT

from lob_sim.loss.smm import SMMSpec, SMMTarget, score_smm_moments
from lob_sim.optimization.evaluator import TrialEvaluator


PRIMARY_METHODS = ("random", "tpe", "opro_plain", "opro_heuristic")
EXTENSION_METHODS = ("sobol", "gp_bo", "cma_es", "bads")
ALL_METHODS = (*PRIMARY_METHODS, *EXTENSION_METHODS)
# Backward-compatible public name for the completed primary experiment.
METHODS = PRIMARY_METHODS


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _complete_incumbents(
    config_path: Path, study_dir: Path, methods: tuple[str, ...] = METHODS,
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    config = yaml.safe_load(config_path.read_text())
    repetitions = int(config["optimizer_repetitions"])
    budget = int(config["trial_budget"])
    output: dict[str, list[dict[str, Any]]] = {}
    fingerprints: set[str] = set()
    for method in methods:
        output[method] = []
        for repetition in range(repetitions):
            trial_dir = study_dir / "policies" / f"{method}_repeat{repetition:02d}" / "trials"
            paths = sorted(trial_dir.glob("trial_*.json"))
            expected = [f"trial_{index:04d}.json" for index in range(budget)]
            if [path.name for path in paths] != expected:
                raise RuntimeError(
                    f"{method} repetition {repetition} is not a contiguous {budget}-slot ledger"
                )
            records = [json.loads(path.read_text()) for path in paths]
            ranked = [
                (float(row["objective"]), int(row["policy_trial"]), row)
                for row in records
                if row.get("status") == "complete" and row.get("objective") is not None
            ]
            if not ranked:
                raise RuntimeError(f"{method} repetition {repetition} has no valid incumbent")
            for row in records:
                fingerprint = row.get("evaluation", {}).get("evaluator_fingerprint")
                if fingerprint:
                    fingerprints.add(str(fingerprint))
            objective, trial, record = min(ranked, key=lambda item: (item[0], item[1]))
            output[method].append({
                "repetition": repetition,
                "policy_trial": trial,
                "search_objective": objective,
                "candidate": record["candidate"],
                "source_trial": str(paths[trial].resolve()),
            })
    if len(fingerprints) != 1:
        raise RuntimeError(f"Expected one development evaluator fingerprint, found {fingerprints}")
    return output, next(iter(fingerprints))


def run_confirmation(
    config_path: Path,
    methods: tuple[str, ...] = METHODS,
    manifest_name: str = "confirmation_manifest.json",
    development_study_name: str | None = None,
    confirmation_study_name: str | None = None,
) -> dict[str, Any]:
    base_config = yaml.safe_load(config_path.read_text())
    development_name = str(
        development_study_name or base_config["study_name"]
    )
    development_overrides = (
        {"study_name": development_name}
        if development_name != str(base_config["study_name"])
        else None
    )
    development = TrialEvaluator(
        str(config_path), runtime_overrides=development_overrides
    )
    incumbents, ledger_fingerprint = _complete_incumbents(
        config_path, development.study_dir, methods
    )
    if development.evaluator_fingerprint != ledger_fingerprint:
        raise RuntimeError("Current development evaluator differs from completed ledgers")
    confirmation = TrialEvaluator(str(config_path), runtime_overrides={
        "study_name": confirmation_study_name or f"{development_name}_confirmation",
        "replications_per_trial": 60,
        "simulation_workers": 5,
        "evaluation_seed": 72700,
    })
    methods: dict[str, Any] = {}
    for method, rows in incumbents.items():
        confirmed = []
        for row in rows:
            result = confirmation.evaluate(
                row["candidate"], f"confirm-{method}-r{row['repetition']}"
            )
            if not result.get("valid") or result.get("objective") is None:
                raise RuntimeError(
                    f"Confirmation candidate is invalid: {method} repetition {row['repetition']}"
                )
            confirmed.append({
                **row,
                "confirmed_objective": float(result["objective"]),
                "confirmed_component_losses": result["component_losses"],
                "confirmation_candidate_hash": result["candidate_hash"],
            })
        selected = min(
            confirmed,
            key=lambda row: (
                row["confirmed_objective"],
                confirmation.space.canonical_key(row["candidate"]),
            ),
        )
        methods[method] = {"incumbents": confirmed, "selected": selected}
    report = {
        "stage": "fresh_common_development_confirmation",
        "selection_rule": (
            "best search incumbent per repetition, then lowest fresh common "
            "60-scenario confirmation loss; canonical candidate key breaks ties"
        ),
        "development_evaluator_fingerprint": ledger_fingerprint,
        "confirmation_evaluator_fingerprint": confirmation.evaluator_fingerprint,
        "confirmation_scenarios_per_candidate": confirmation.replications,
        "confirmation_evaluation_seed": confirmation.evaluation_seed,
        "methods": methods,
    }
    output = confirmation.study_dir / manifest_name
    if output.exists() and json.loads(output.read_text()) != report:
        raise RuntimeError("Existing confirmation manifest differs from current result")
    _atomic_json(output, report)
    return report


def _validate_later_compatibility(development_path: Path, later_path: Path) -> None:
    development = yaml.safe_load(development_path.read_text())
    later = yaml.safe_load(later_path.read_text())
    if development.get("split") != "development" or later.get("split") != "test":
        raise RuntimeError("Expected development and reused test split configurations")
    for key in ("symbol", "base_config", "real_config", "loss_config", "search_space", "validity_gates"):
        if development.get(key) != later.get(key):
            raise RuntimeError(f"Development and later-date configuration differ in {key}")
    if int(later.get("replications_per_trial", -1)) != 120:
        raise RuntimeError("Later-date evaluation requires 120 scenarios")
    if int(later.get("evaluation_seed", -1)) != 92700:
        raise RuntimeError("Later-date evaluation requires seed 92700")


def _date_means(rows: list[dict[str, Any]], names: tuple[str, ...]) -> dict[str, dict[str, float]]:
    dates = sorted({str(row["date"]) for row in rows})
    return {
        date: {
            name: float(np.mean([
                float(row[name]) for row in rows if str(row["date"]) == date
            ]))
            for name in names
        }
        for date in dates
    }


def _bootstrap_later(
    results: dict[str, dict[str, Any]], target: SMMTarget, spec: SMMSpec,
    draws: int = 5000, seed: int = 20260913,
    methods: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    methods = tuple(results) if methods is None else methods
    empirical = {
        str(row["date"]): {name: float(row[name]) for name in spec.moment_names}
        for row in target.date_moments
    }
    dates = sorted(empirical)
    simulated = {
        method: _date_means(result["smm"]["scenario_moments"], spec.moment_names)
        for method, result in results.items()
    }
    if any(sorted(values) != dates for values in simulated.values()):
        raise RuntimeError("Later-date methods do not share all empirical date clusters")
    rng = np.random.default_rng(seed)
    samples = {method: np.empty(draws) for method in methods}
    for draw in range(draws):
        selected = [dates[int(index)] for index in rng.integers(0, len(dates), len(dates))]
        target_values = {
            name: float(np.mean([empirical[date][name] for date in selected]))
            for name in spec.moment_names
        }
        draw_target = SMMTarget(
            moment_names=target.moment_names,
            target=target_values,
            standard_errors=target.standard_errors,
            covariance=target.covariance,
            date_moments=target.date_moments,
            metadata=target.metadata,
        )
        for method in methods:
            values = {
                name: float(np.mean([simulated[method][date][name] for date in selected]))
                for name in spec.moment_names
            }
            samples[method][draw] = score_smm_moments(values, draw_target, spec).total
    comparisons = {}
    preferred_pairs = [
        ("opro_heuristic", "opro_plain"),
        ("opro_heuristic", "tpe"),
        ("opro_plain", "tpe"),
        ("tpe", "random"),
    ]
    pairs = [
        pair for pair in preferred_pairs
        if pair[0] in methods and pair[1] in methods
    ]
    compared = {frozenset(pair) for pair in pairs}
    pairs.extend(
        pair for pair in itertools.combinations(methods, 2)
        if frozenset(pair) not in compared
    )
    for left, right in pairs:
        delta = samples[left] - samples[right]
        comparisons[f"{left}_minus_{right}"] = {
            "mean": float(np.mean(delta)),
            "ci95": [float(value) for value in np.quantile(delta, [0.025, 0.975])],
            "probability_left_better": float(np.mean(delta < 0)),
        }
    return {
        "method": "paired whole-date bootstrap with evaluator-frozen SMM standard errors",
        "dates": dates,
        "draws": draws,
        "seed": seed,
        "methods": {
            method: {
                "mean": float(np.mean(values)),
                "ci95": [float(value) for value in np.quantile(values, [0.025, 0.975])],
            }
            for method, values in samples.items()
        },
        "paired_comparisons": comparisons,
    }


def run_later(
    development_path: Path,
    later_path: Path,
    confirmation: dict[str, Any],
    methods: tuple[str, ...] = METHODS,
    summary_name: str = "later_date_summary.json",
    later_study_name: str | None = None,
) -> dict[str, Any]:
    _validate_later_compatibility(development_path, later_path)
    later_config = yaml.safe_load(later_path.read_text())
    configured_later_name = str(later_config["study_name"])
    selected_later_name = later_study_name or configured_later_name
    later_overrides = (
        {"study_name": selected_later_name}
        if selected_later_name != configured_later_name
        else None
    )
    evaluator = TrialEvaluator(str(later_path), runtime_overrides=later_overrides)
    if not isinstance(evaluator.loss_spec, SMMSpec) or evaluator.smm_target is None:
        raise RuntimeError("Later-date evaluator is not an SMM evaluator")
    results: dict[str, dict[str, Any]] = {}
    method_rows: dict[str, Any] = {}
    for method in methods:
        selection = confirmation["methods"][method]["selected"]
        result = evaluator.evaluate(selection["candidate"], f"later-{method}")
        if not result.get("valid") or result.get("objective") is None:
            raise RuntimeError(f"Later-date candidate is invalid: {method}")
        results[method] = result
        method_rows[method] = {
            "candidate": selection["candidate"],
            "confirmation_objective": selection["confirmed_objective"],
            "later_objective": float(result["objective"]),
            "later_component_losses": result["component_losses"],
            "later_candidate_hash": result["candidate_hash"],
            "later_runtime_seconds": result["runtime_seconds"],
        }
    report = {
        "stage": "reused_later_date_temporal_robustness",
        "interpretation": "non-pristine reused dates; selection bias explicitly acknowledged",
        "evaluator_fingerprint": evaluator.evaluator_fingerprint,
        "replications_per_method": evaluator.replications,
        "evaluation_seed": evaluator.evaluation_seed,
        "methods": method_rows,
        "whole_date_bootstrap": _bootstrap_later(
            results, evaluator.smm_target, evaluator.loss_spec, methods=methods
        ),
    }
    _atomic_json(evaluator.study_dir / summary_name, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-config", type=Path,
        default=Path("configs/paper/optimizer.yaml"),
    )
    parser.add_argument(
        "--later-config", type=Path,
        default=Path("configs/paper/later-evaluation.yaml"),
    )
    parser.add_argument("--stage", choices=("confirm", "later", "all"), default="all")
    parser.add_argument(
        "--method-set", choices=("primary", "new", "extended"), default="primary",
        help="evaluate the original four, the four new methods, or all eight",
    )
    parser.add_argument(
        "--study-name",
        help=(
            "development study to evaluate; defaults to the config study for "
            "primary and icaif_smm_v4_optimizer_comparison for extended"
        ),
    )
    args = parser.parse_args()
    development = args.development_config.resolve()
    methods = {
        "primary": METHODS,
        "new": EXTENSION_METHODS,
        "extended": ALL_METHODS,
    }[args.method_set]
    development_config = yaml.safe_load(development.read_text())
    development_study_name = args.study_name or (
        str(development_config["study_name"])
        if args.method_set == "primary"
        else "icaif_smm_v4_optimizer_comparison"
    )
    confirmation_study_name = f"{development_study_name}_confirmation"
    later_study_name = (
        str(yaml.safe_load(args.later_config.read_text())["study_name"])
        if args.method_set == "primary" and args.study_name is None
        else f"{development_study_name}_later_dates"
    )
    confirmation_name = {
        "primary": "confirmation_manifest.json",
        "new": "optimizer_extension_confirmation_manifest.json",
        "extended": "extended_confirmation_manifest.json",
    }[args.method_set]
    later_name = {
        "primary": "later_date_summary.json",
        "new": "optimizer_extension_later_date_summary.json",
        "extended": "extended_later_date_summary.json",
    }[args.method_set]
    confirmation_path = (
        REPO_ROOT / "results/optimization" / confirmation_study_name
        / confirmation_name
    )
    if args.stage in {"confirm", "all"}:
        confirmation = run_confirmation(
            development,
            methods=methods,
            manifest_name=confirmation_name,
            development_study_name=development_study_name,
            confirmation_study_name=confirmation_study_name,
        )
        print(f"Confirmation complete: {confirmation_path}")
    else:
        if not confirmation_path.exists():
            raise RuntimeError("Confirmation manifest is missing")
        confirmation = json.loads(confirmation_path.read_text())
    if args.stage in {"later", "all"}:
        report = run_later(
            development,
            args.later_config.resolve(),
            confirmation,
            methods=methods,
            summary_name=later_name,
            later_study_name=later_study_name,
        )
        print(json.dumps({
            "later_date_summary": str(
                REPO_ROOT / "results/optimization" / later_study_name
                / later_name
            ),
            "objectives": {
                method: row["later_objective"] for method, row in report["methods"].items()
            },
        }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
