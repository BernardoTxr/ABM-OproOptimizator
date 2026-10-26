#!/usr/bin/env python3
"""Run the frozen five-ticker, eight-way Calibration Books comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from workflows import REPO_ROOT
from lob_sim.optimization.evaluator import TrialEvaluator
from lob_sim.optimization.openrouter_runtime import load_openrouter_settings
from lob_sim.optimization.report import write_benchmark_report
import lob_sim.optimization.study as study_module
from lob_sim.optimization.study import run_study
from workflows.benchmark.audit_component_stage import (
    FEEDBACK_MODES,
    METHODS,
    audit_component_stage,
    audit_method_stage,
)
from workflows.benchmark.classical_optimizers import (
    BADSAskTellPolicy,
    ComponentAwareGPPolicy,
    MultiObjectiveTPEPolicy,
    OptunaGPPolicy,
    SobolPolicy,
)
from workflows.benchmark.preflight_openrouter import validate_openrouter_preflight
from workflows.benchmark.parallel_execution import apply_cpu_affinity, parse_cpu_set
from workflows.benchmark.ticker_books import (
    TICKERS,
    build_ticker_tasks,
    comparison_output_root,
    experiment_paths,
    provenance_artifacts,
    validate_ticker_book_contract,
)
from workflows.common.local_env import load_repository_local_env
from workflows.validation.environment import (
    validate_environment,
    validate_extension_environment,
)
from workflows.validation.heuristics import validate_registry
from workflows.validation.provenance import build_provenance, verify_provenance


LOCAL_METHODS = {"sobol", "bads", "gp_bo", "mo_tpe", "component_gp_bo"}
OPENROUTER_METHODS = {"opro_plain", "opro_heuristic"}


def method_stage_path(study_dir: Path, method: str) -> Path:
    return study_dir / "report" / f"component_aware_stage_audit_{method}.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def install_component_policies(
    comparison: dict[str, Any], comparison_sha256: str
) -> None:
    """Install workflow-local adapters without changing the shared controller."""
    original_policy = study_module._policy

    def policy_factory(name, evaluator, seed, run_dir, initial_design):
        if name not in LOCAL_METHODS:
            return original_policy(name, evaluator, seed, run_dir, initial_design)
        settings = comparison["method_settings"][name]
        if name == "sobol":
            repetition = int(seed) - int(evaluator.config["optimizer_seed"])
            adaptive_slots = int(evaluator.config["trial_budget"]) - len(initial_design)
            return SobolPolicy(
                evaluator.space,
                int(settings["sequence_seed"]),
                len(initial_design),
                len(initial_design) + repetition * adaptive_slots,
                comparison_sha256,
            )
        if name == "bads":
            return BADSAskTellPolicy(
                evaluator.space,
                seed,
                len(initial_design),
                comparison_sha256,
                invalid_penalty=float(
                    comparison["feedback_contract"]["invalid_penalty"]
                ),
            )
        if name == "gp_bo":
            return OptunaGPPolicy(
                evaluator.space,
                seed,
                run_dir / "optuna.sqlite3",
                run_dir.name,
                initial_design,
                comparison_sha256,
            )
        if name == "mo_tpe":
            return MultiObjectiveTPEPolicy(
                evaluator.space,
                seed,
                run_dir / "optuna.sqlite3",
                run_dir.name,
                initial_design,
                comparison_sha256,
            )
        return ComponentAwareGPPolicy(
            evaluator.space,
            seed,
            len(initial_design),
            comparison_sha256,
            candidate_pool_size=int(settings["candidate_pool_size"]),
            exploration=float(settings["exploration"]),
        )

    study_module._policy = policy_factory


def validate_component_contract(
    comparison: dict[str, Any],
    optimizer: dict[str, Any],
    test: dict[str, Any],
    real: dict[str, Any],
) -> dict[str, bool]:
    """Reject any drift from the frozen ticker-specific experiment."""
    checks = validate_ticker_book_contract(comparison, optimizer, test, real)
    common = comparison.get("common_contract", {})
    feedback = comparison.get("feedback_contract", {})
    optimizer_feedback = optimizer.get("optimizer_feedback", {})
    checks.update({
        "methods": tuple(comparison.get("methods", ())) == METHODS,
        "study": comparison.get("comparison_study_name")
        == "icaif_smm_v9_ticker_specific_10x30m",
        "settings": set(comparison.get("method_settings", {})) == set(METHODS),
        "feedback_modes": feedback.get("information_firewall") == FEEDBACK_MODES,
        "feedback_contract": all(
            feedback.get(key) == optimizer_feedback.get(key)
            for key in (
                "version", "component_order", "final_aggregation", "invalid_penalty"
            )
        ),
        "residuals_audit_only": feedback.get("signed_moment_residuals")
        == "audit_only",
        "budget": int(common.get("charged_slots_per_repetition", -1))
        == int(optimizer.get("trial_budget", -2)),
        "initial_design": int(common.get("common_initial_slots", -1))
        == len(optimizer.get("initial_design", [])),
        "deterministic_oracle": common.get("search_oracle")
        == "deterministic_common_random_numbers",
        "prospective_disclosure": (
            "prospective follow-up" in str(comparison.get("disclosure", ""))
            and "not part of the original preregistration"
            in str(comparison.get("disclosure", ""))
        ),
        "llm_model_key": bool(comparison.get("llm_model_key")),
    })
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError("Component-aware contract failed: " + ", ".join(failed))
    return checks


def component_plan(comparison_path: Path) -> dict[str, Any]:
    """Describe all five tasks without creating targets or running simulations."""
    (
        optimizer_path,
        test_path,
        real_path,
        comparison,
        optimizer,
        test,
        real,
    ) = experiment_paths(comparison_path)
    checks = validate_component_contract(comparison, optimizer, test, real)
    tasks = build_ticker_tasks(comparison, optimizer)
    repetitions = int(optimizer["optimizer_repetitions"])
    budget = int(optimizer["trial_budget"])
    ticker_rows: dict[str, Any] = {}
    for task in tasks:
        study_dir = (
            comparison_output_root(comparison, optimizer_path, optimizer)
            / task.symbol / "calibration"
        )
        methods: dict[str, list[dict[str, int]]] = {}
        for method in METHODS:
            methods[method] = []
            for repetition in range(repetitions):
                trial_dir = (
                    study_dir / "policies"
                    / f"{method}_repeat{repetition:02d}" / "trials"
                )
                present = len(list(trial_dir.glob("trial_*.json")))
                methods[method].append({
                    "repetition": repetition,
                    "slots_present": present,
                    "slots_remaining": max(0, budget - present),
                })
        ticker_rows[task.symbol] = {
            "parameter_scope": "ticker_specific",
            "calibration_variable": "CalibrationBooks",
            "test_variable": "TestBooks",
            "calibration_books": task.books_per_partition,
            "test_books": task.books_per_partition,
            "total_books": 2 * task.books_per_partition,
            "calibration_study_dir": str(study_dir.resolve()),
            "test_study_dir": str((study_dir.parent / "test").resolve()),
            "calibration_target": str(task.calibration_target),
            "test_target": str(task.test_target),
            "methods": methods,
        }
    model_settings = load_openrouter_settings(
        optimizer.get("llm", {}),
        optimizer_path.parent,
        model_key_override=comparison["llm_model_key"],
    )
    root = comparison_output_root(comparison, optimizer_path, optimizer)
    return {
        "comparison_config": str(comparison_path.resolve()),
        "comparison_config_sha256": hashlib.sha256(
            comparison_path.read_bytes()
        ).hexdigest(),
        "calibration_config": str(optimizer_path),
        "calibration_config_sha256": hashlib.sha256(
            optimizer_path.read_bytes()
        ).hexdigest(),
        "test_config": str(test_path),
        "test_config_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
        "book_config": str(real_path),
        "book_config_sha256": hashlib.sha256(real_path.read_bytes()).hexdigest(),
        "comparison_root": str(root.resolve()),
        "contract": checks,
        "feedback_modes": FEEDBACK_MODES,
        "split": {
            "unit": "whole_date",
            "calibration_dates": real["date_splits"]["calibration"],
            "test_dates": real["date_splits"]["test"],
            "windows": real["window_starts_utc"],
        },
        "llm": {
            "runtime": "openrouter_chat_completions_v1",
            "model_key": model_settings.model_key,
            "requested_model": model_settings.model_id,
            "provider_routing": model_settings.provider_preferences,
            "transport_reliability": {
                "serialized_across_tickers": bool(
                    model_settings.coordination_lock_path
                ),
                "max_attempts": model_settings.max_attempts,
                "retry_http_statuses": list(model_settings.retry_http_statuses),
            },
            "gateway_config_sha256": model_settings.gateway_sha256,
            "model_registry_sha256": model_settings.registry_sha256,
        },
        "tickers": ticker_rows,
        "disclosure": comparison["disclosure"].strip(),
    }


def _materialize_targets(
    tasks, optimizer_path: Path, test_path: Path, comparison: dict[str, Any]
) -> list[dict[str, Any]]:
    """Build empirical targets only; TrialEvaluator construction runs no simulator."""
    output = []
    for task in tasks:
        calibration = TrialEvaluator(
            str(optimizer_path),
            runtime_overrides=task.calibration_overrides(
                comparison["llm_model_key"]
            ),
        )
        test = TrialEvaluator(
            str(test_path), runtime_overrides=task.test_overrides()
        )
        output.append({
            "ticker": task.symbol,
            "calibration_evaluator_fingerprint": calibration.evaluator_fingerprint,
            "test_evaluator_fingerprint": test.evaluator_fingerprint,
            "calibration_target": str(task.calibration_target),
            "test_target": str(task.test_target),
        })
    return output


def _write_calibration_index(root: Path, tasks) -> None:
    rows = {}
    for task in tasks:
        summary_path = root / task.symbol / "calibration/report/benchmark_summary.json"
        rows[task.symbol] = {
            "summary": str(summary_path.resolve()),
            "present": summary_path.is_file(),
        }
    _atomic_json(root / "report/ticker_calibration_index.json", {
        "unit_of_analysis": "ticker_specific_optimizer_task",
        "parameter_pooling": False,
        "tickers": rows,
    })


def main() -> None:
    load_repository_local_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-config",
        type=Path,
        default=Path("configs/paper/component-aware-comparison.yaml"),
    )
    parser.add_argument("--ticker", action="append")
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--repetition", action="append", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-books", action="store_true")
    parser.add_argument("--freeze-provenance", action="store_true")
    parser.add_argument("--stage-first-adaptive-slot", action="store_true")
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--schema-preflight", type=Path)
    parser.add_argument(
        "--simulation-workers",
        type=int,
        help="Operational override for per-ticker scenario workers.",
    )
    parser.add_argument(
        "--cpu-set",
        help="Linux CPU-list affinity inherited by simulator child processes.",
    )
    args = parser.parse_args()
    if args.simulation_workers is not None and args.simulation_workers < 1:
        parser.error("--simulation-workers must be positive")
    if args.cpu_set:
        try:
            apply_cpu_affinity(parse_cpu_set(args.cpu_set))
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    modes = sum(bool(value) for value in (
        args.dry_run,
        args.prepare_books,
        args.freeze_provenance,
        args.stage_first_adaptive_slot,
    ))
    if modes > 1:
        parser.error(
            "--dry-run, --prepare-books, --freeze-provenance, and "
            "--stage-first-adaptive-slot are mutually exclusive"
        )

    comparison_path = args.comparison_config.resolve()
    (
        optimizer_path,
        test_path,
        _,
        comparison,
        optimizer,
        test,
        real,
    ) = experiment_paths(comparison_path)
    plan = component_plan(comparison_path)
    base_lock = REPO_ROOT / "configs/paper/environment-lock.yaml"
    add_on_lock = (comparison_path.parent / comparison["environment_lock"]).resolve()
    plan["environment"] = {
        "primary": validate_environment(yaml.safe_load(base_lock.read_text())),
        "component_aware": validate_extension_environment(
            yaml.safe_load(add_on_lock.read_text())
        ),
    }
    heuristic_path = (
        optimizer_path.parent / optimizer["llm"]["heuristic_config"]
    ).resolve()
    plan["heuristic_registry"] = validate_registry(
        yaml.safe_load(heuristic_path.read_text()), optimizer
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    validate_component_contract(comparison, optimizer, test, real)
    tasks = build_ticker_tasks(comparison, optimizer)
    selected_tickers = tuple(str(value).upper() for value in (args.ticker or TICKERS))
    unknown_tickers = sorted(set(selected_tickers) - set(TICKERS))
    if unknown_tickers:
        raise RuntimeError(f"Unknown ticker filters: {unknown_tickers}")
    if len(selected_tickers) != len(set(selected_tickers)):
        raise RuntimeError("Ticker filters contain duplicates")
    tasks = tuple(task for task in tasks if task.symbol in selected_tickers)
    root = comparison_output_root(comparison, optimizer_path, optimizer)
    provenance_path = (
        args.provenance.resolve()
        if args.provenance else root / "report/reconstruction_provenance.json"
    )
    preflight_path = (
        args.schema_preflight.resolve()
        if args.schema_preflight else root / "report/openrouter_schema_preflight.json"
    )

    if args.prepare_books or args.freeze_provenance:
        materialized = _materialize_targets(
            tasks, optimizer_path, test_path, comparison
        )
        if args.freeze_provenance:
            if set(selected_tickers) != set(TICKERS):
                raise RuntimeError("Provenance must freeze all five tickers together")
            if provenance_path.exists():
                raise RuntimeError(
                    f"Refusing to overwrite provenance lock: {provenance_path}"
                )
            payload = build_provenance(artifacts=provenance_artifacts(tasks))
            if payload["git"]["reconstruction_tree_clean"] is not True:
                raise RuntimeError(
                    "Refusing to freeze a changed real_lob_inference source tree"
                )
            _atomic_json(provenance_path, payload)
        print(json.dumps({
            "targets": materialized,
            "provenance": str(provenance_path) if args.freeze_provenance else None,
            "simulator_evaluations": 0,
        }, indent=2, sort_keys=True))
        return

    verify_provenance(
        provenance_path, artifacts=provenance_artifacts(
            build_ticker_tasks(comparison, optimizer)
        )
    )
    selected_methods = tuple(args.method or METHODS)
    if set(selected_methods) & OPENROUTER_METHODS:
        validate_openrouter_preflight(
            preflight_path,
            optimizer_path,
            optimizer,
            model_key_override=comparison["llm_model_key"],
        )
    install_component_policies(comparison, plan["comparison_config_sha256"])
    repetitions = tuple(
        args.repetition or range(int(optimizer["optimizer_repetitions"]))
    )
    if any(
        repetition < 0 or repetition >= int(optimizer["optimizer_repetitions"])
        for repetition in repetitions
    ):
        raise RuntimeError("Requested repetition is outside the frozen protocol")

    if args.stage_first_adaptive_slot:
        if args.repetition:
            raise RuntimeError(
                "The stage gate accepts ticker and method filters but not repetitions"
            )
        staged_execution_limit = len(optimizer["initial_design"]) + 1
        full_budget = int(optimizer["trial_budget"])
        for task in tasks:
            overrides = task.calibration_overrides(comparison["llm_model_key"])
            if (
                args.simulation_workers is not None
                and args.simulation_workers
                != int(optimizer.get("simulation_workers", 1))
            ):
                overrides["simulation_workers"] = args.simulation_workers
            evaluator = TrialEvaluator(
                str(optimizer_path), runtime_overrides=overrides
            )
            for method in selected_methods:
                run_study(
                    str(optimizer_path),
                    method,
                    0,
                    budget=full_budget,
                    execution_limit=staged_execution_limit,
                    evaluator_overrides=overrides,
                )
                report = audit_method_stage(
                    comparison_path, optimizer_path, evaluator.study_dir, method
                )
                report["ticker"] = task.symbol
                output = method_stage_path(evaluator.study_dir, method)
                _atomic_json(output, report)
                if not report["passed"]:
                    raise RuntimeError(
                        f"{task.symbol}/{method} stage audit failed: "
                        + ", ".join(report["failed_checks"])
                    )
            if set(selected_methods) == set(METHODS):
                report = audit_component_stage(
                    comparison_path, optimizer_path, evaluator.study_dir
                )
                report["ticker"] = task.symbol
                output = (
                    evaluator.study_dir
                    / "report/component_aware_stage_audit.json"
                )
                _atomic_json(output, report)
                if not report["passed"]:
                    raise RuntimeError(
                        f"{task.symbol} eight-way stage audit failed: "
                        + ", ".join(report["failed_checks"])
                    )
        print("Selected ticker-specific method stage gates passed; inspect every audit.")
        return

    for task in tasks:
        overrides = task.calibration_overrides(comparison["llm_model_key"])
        if (
            args.simulation_workers is not None
            and args.simulation_workers
            != int(optimizer.get("simulation_workers", 1))
        ):
            overrides["simulation_workers"] = args.simulation_workers
        evaluator = TrialEvaluator(str(optimizer_path), runtime_overrides=overrides)
        for method in selected_methods:
            stage_path = method_stage_path(evaluator.study_dir, method)
            if not stage_path.exists():
                raise RuntimeError(
                    "Run --stage-first-adaptive-slot "
                    f"--ticker {task.symbol} --method {method} first"
                )
            stage = json.loads(stage_path.read_text())
            if (
                stage.get("passed") is not True
                or stage.get("ticker") != task.symbol
                or stage.get("method") != method
                or stage.get("comparison_config_sha256")
                != plan["comparison_config_sha256"]
                or stage.get("optimizer_config_sha256")
                != plan["calibration_config_sha256"]
            ):
                raise RuntimeError(
                    f"{task.symbol}/{method} stage audit is failed or stale"
                )
            for repetition in repetitions:
                run_study(
                    str(optimizer_path),
                    method,
                    repetition,
                    evaluator_overrides=overrides,
                )
                write_benchmark_report(evaluator.study_dir)
    _write_calibration_index(root, build_ticker_tasks(comparison, optimizer))
    print("Selected ticker-specific calibration arms are complete.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
