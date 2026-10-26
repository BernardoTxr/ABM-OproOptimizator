#!/usr/bin/env python3
"""Two-arm OPRO follow-up: rerun only opro_plain/opro_heuristic with a new
proposal-model runtime, holding the v9 scientific pipeline byte-identical.

Deliberately separate from ``run_component_aware.py`` (the frozen, eight
-method v9 workflow): this module's unfiltered default is exactly
``(opro_plain, opro_heuristic)``, it never imports the frozen eight-method
``METHODS`` tuple as its own method universe, and it writes only into a new
study root under ``results/optimization/<comparison_study_name>/`` -- never
into either v9 result path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from lob_sim.optimization.anthropic_foundry_runtime import (
    PROVIDER_LABEL as FOUNDRY_PROVIDER_LABEL,
    RUNTIME_PROTOCOL as FOUNDRY_RUNTIME_PROTOCOL,
)
from lob_sim.optimization.evaluator import TrialEvaluator
from lob_sim.optimization.feedback import SMM_COMPONENTS
from lob_sim.optimization.runtime_factory import build_proposal_runtime
from lob_sim.optimization.scientific_identity import scientific_identity_sha256
from lob_sim.optimization.space import SearchSpace
from lob_sim.optimization.study import run_study
from lob_sim.paths import find_project_root
from workflows.benchmark.evaluate_component_aware import run_ticker_tests
from workflows.benchmark.preflight_proposal_runtime import validate_proposal_preflight
from workflows.benchmark.runtime_audit import model_identity_matches
from workflows.benchmark.ticker_books import (
    TICKERS,
    build_ticker_tasks,
    comparison_output_root,
    experiment_paths,
    resolved_proposal_runtime,
)
from workflows.common.local_env import load_repository_local_env


DECLARED_METHODS = ("opro_plain", "opro_heuristic")
BASELINE_METHOD_NAMES = (
    "sobol", "tpe", "bads", "gp_bo", "mo_tpe", "component_gp_bo",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def select_methods(requested: list[str] | None) -> tuple[str, ...]:
    """Resolve ``--method`` into the effective method tuple.

    The unfiltered default is exactly the two declared methods. Rejects
    duplicate, undeclared, empty, or baseline selections -- argparse's own
    ``choices=DECLARED_METHODS`` already blocks undeclared/baseline names at
    the CLI layer, but this is re-checked here as defense-in-depth for any
    other caller.
    """
    if requested is None:
        return DECLARED_METHODS
    selected = tuple(requested)
    if not selected:
        raise RuntimeError("Method selection must not be empty")
    if len(selected) != len(set(selected)):
        raise RuntimeError("Duplicate method selection")
    undeclared = set(selected) - set(DECLARED_METHODS)
    if undeclared:
        raise RuntimeError(
            f"Undeclared or baseline methods are not permitted here: {sorted(undeclared)}"
        )
    return selected


def validate_pair_contract(
    comparison: dict[str, Any],
    calibration: dict[str, Any],
    test: dict[str, Any],
) -> dict[str, bool]:
    """Validate the two-arm follow-up shape without reading data or ledgers."""
    checks = {
        "version": comparison.get("version") == "opro_pair_followup_v1",
        "methods_are_exactly_the_two_opro_arms": tuple(
            comparison.get("methods", ())
        ) == DECLARED_METHODS,
        "new_study_name": bool(comparison.get("comparison_study_name"))
        and comparison.get("comparison_study_name")
        != "icaif_smm_v9_ticker_specific_10x30m",
        "parent_study_recorded": bool(comparison.get("parent_study")),
        "proposal_runtime_declared": bool(
            comparison.get("proposal_runtime", {}).get("kind")
        ),
        "five_tickers": tuple(
            str(value).upper() for value in comparison.get("tickers", ())
        ) == TICKERS,
        "matched_evaluators": all(
            calibration.get(key) == test.get(key)
            for key in (
                "base_config", "real_config", "loss_config",
                "search_space", "validity_gates",
            )
        ),
        "twelve_shared_initial_slots": len(calibration.get("initial_design", [])) == 12,
        "five_repetitions": int(calibration.get("optimizer_repetitions", -1)) == 5,
        "fifty_charged_slots": int(calibration.get("trial_budget", -1)) == 50,
        "frozen_optimizer_seed": int(calibration.get("optimizer_seed", -1)) == 20260912,
        "frozen_calibration_seed": int(calibration.get("evaluation_seed", -1)) == 62700,
        "frozen_test_seed": int(test.get("evaluation_seed", -1)) == 92700,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError("OPRO pair follow-up contract failed: " + ", ".join(failed))
    return checks


def _frozen_configs_are_untouched(calibration_path: Path, test_path: Path) -> dict[str, bool]:
    """The calibration/test configs must be the exact frozen paper files, not copies."""
    paper_dir = calibration_path.parent
    return {
        "calibration_config_is_the_frozen_paper_file": (
            calibration_path.resolve() == (paper_dir / "optimizer.yaml").resolve()
        ),
        "test_config_is_the_frozen_paper_file": (
            test_path.resolve() == (paper_dir / "later-evaluation.yaml").resolve()
        ),
    }


def pair_plan(comparison_path: Path) -> dict[str, Any]:
    """Describe all five tasks and both OPRO arms without running anything."""
    (
        calibration_path, test_path, real_path,
        comparison, calibration, test, real,
    ) = experiment_paths(comparison_path)
    checks = validate_pair_contract(comparison, calibration, test)
    checks.update(_frozen_configs_are_untouched(calibration_path, test_path))
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError("OPRO pair follow-up contract failed: " + ", ".join(failed))

    tasks = build_ticker_tasks(comparison, calibration)
    proposal_runtime = resolved_proposal_runtime(comparison_path, comparison)
    # A dry plan must remain usable before endpoint/deployment credentials are
    # configured. Read only public config metadata here; live phases construct
    # the concrete runtime and require all referenced environment variables.
    gateway_path = Path(str(proposal_runtime["config"]))
    registry_path = Path(str(proposal_runtime["model_registry"]))
    gateway = yaml.safe_load(gateway_path.read_text())
    registry = yaml.safe_load(registry_path.read_text())
    model_key = str(comparison.get("llm_model_key"))
    model_profile = registry["models"][model_key]
    runtime_identity = {
        "protocol": FOUNDRY_RUNTIME_PROTOCOL,
        "model_key": model_key,
        "deployment_name_env": model_profile.get("deployment_name_env"),
        "base_url_env": gateway.get("base_url_env"),
        "api_key_env": gateway.get("api_key_env"),
        "gateway_config_sha256": hashlib.sha256(gateway_path.read_bytes()).hexdigest(),
        "model_registry_sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "pricing_status": gateway.get("pricing", {}).get("status"),
    }
    repetitions = int(calibration["optimizer_repetitions"])
    budget = int(calibration["trial_budget"])
    root = comparison_output_root(comparison, calibration_path, calibration)
    ticker_rows: dict[str, Any] = {}
    for task in tasks:
        study_dir = root / task.symbol / "calibration"
        methods: dict[str, list[dict[str, int]]] = {}
        for method in DECLARED_METHODS:
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
            "calibration_study_dir": str(study_dir.resolve()),
            "test_study_dir": str((study_dir.parent / "test").resolve()),
            "methods": methods,
        }
    return {
        "comparison_config": str(comparison_path.resolve()),
        "comparison_config_sha256": hashlib.sha256(comparison_path.read_bytes()).hexdigest(),
        "calibration_config": str(calibration_path),
        "test_config": str(test_path),
        "book_config": str(real_path),
        "comparison_root": str(root.resolve()),
        "parent_study": comparison.get("parent_study"),
        "methods": list(DECLARED_METHODS),
        "tickers": ticker_rows,
        "proposal_runtime": {
            "kind": proposal_runtime.get("kind"),
            "model_key": model_key,
            "runtime_identity": runtime_identity,
        },
        "expected_charged_slots_total": 5 * 2 * repetitions * budget,
        "simulator_evaluations": 0,
        "network_calls": 0,
    }


def audit_pair_stage(
    comparison_path: Path, calibration_path: Path, study_dir: Path
) -> dict[str, Any]:
    """Audit one ticker's first-adaptive-slot for both OPRO arms together.

    Provider-neutral sibling of ``audit_component_stage.audit_component_stage``,
    scoped to exactly the two declared methods and additionally asserting the
    scientific-identity hash so the audit proves "only the proposal runtime
    changed," not merely that the two arms agree with each other.
    """
    comparison = yaml.safe_load(comparison_path.read_text())
    calibration = yaml.safe_load(calibration_path.read_text())
    proposal_runtime = resolved_proposal_runtime(comparison_path, comparison)
    runtime = build_proposal_runtime(
        proposal_runtime, calibration_path.parent,
        model_key_override=comparison.get("llm_model_key"),
    )
    space = SearchSpace.from_config(calibration["search_space"])
    initial = [space.validate(row) for row in calibration["initial_design"]]
    staged_budget = len(initial) + 1
    records: dict[str, list[dict[str, Any]]] = {}
    for method in DECLARED_METHODS:
        trial_dir = study_dir / "policies" / f"{method}_repeat00" / "trials"
        paths = sorted(trial_dir.glob("trial_*.json"))
        expected = [f"trial_{index:04d}.json" for index in range(staged_budget)]
        if [path.name for path in paths] != expected:
            raise RuntimeError(
                f"{method} does not contain exactly the staged {staged_budget} slots"
            )
        records[method] = [json.loads(path.read_text()) for path in paths]

    checks: dict[str, bool] = {}
    anchor = records[DECLARED_METHODS[0]]
    checks["common_initial_candidates"] = all(
        records[method][index].get("candidate") == initial[index]
        for method in DECLARED_METHODS for index in range(len(initial))
    )
    checks["common_initial_objectives"] = all(
        records[method][index].get("objective") == anchor[index].get("objective")
        for method in DECLARED_METHODS for index in range(len(initial))
    )
    fingerprints = {
        row.get("evaluation", {}).get("evaluator_fingerprint")
        for method in DECLARED_METHODS for row in records[method]
    }
    checks["one_evaluator_fingerprint"] = len(fingerprints) == 1 and None not in fingerprints

    for method in DECLARED_METHODS:
        checks[f"{method}_feedback_mode"] = all(
            row.get("policy_feedback_mode") == "component" for row in records[method]
        )
        adaptive = records[method][-1]
        checks[f"{method}_adaptive_candidate_valid"] = bool(
            adaptive.get("candidate") and space.validate(adaptive["candidate"])
        )
        metadata = adaptive.get("proposal_metadata", {})
        # This follow-up's proposal_runtime is always the Anthropic Foundry
        # adapter, whose ModelCompletion.provider is always the fixed
        # FOUNDRY_PROVIDER_LABEL constant (no multi-provider routing/
        # failover exists for this adapter, unlike OpenRouter's allowlist).
        checks[f"{method}_model_identity"] = model_identity_matches(
            metadata, runtime.runtime_identity, {FOUNDRY_PROVIDER_LABEL},
        )

    prompts: dict[str, dict[str, Any]] = {}
    for method in DECLARED_METHODS:
        metadata = records[method][-1].get("proposal_metadata", {})
        raw_prompt = metadata.get("prompt")
        try:
            prompts[method] = json.loads(raw_prompt)
        except (TypeError, json.JSONDecodeError):
            prompts[method] = {}
        checks[f"{method}_prompt_information_firewall"] = bool(
            prompts[method]
            and "signed_diagnostics" not in raw_prompt
            and "diagnostic_keys" not in raw_prompt
            and "prediction_outcomes" not in raw_prompt
            and len(prompts[method].get("observations", [])) == len(initial)
            and all(
                set(observation.get("component_losses", {})) == set(SMM_COMPONENTS)
                for observation in prompts[method]["observations"]
            )
        )
    checks["opro_same_numerical_history"] = (
        prompts["opro_plain"].get("observations")
        == prompts["opro_heuristic"].get("observations")
    )
    expected_remaining = int(calibration["trial_budget"]) - len(initial)
    checks["opro_full_remaining_budget"] = all(
        prompts[method].get("remaining_proposal_budget") == expected_remaining
        for method in DECLARED_METHODS
    )
    checks["opro_heuristic_ablation"] = (
        prompts["opro_plain"].get("heuristic_condition") is False
        and prompts["opro_plain"].get("heuristics") is None
        and prompts["opro_heuristic"].get("heuristic_condition") is True
        and prompts["opro_heuristic"].get("heuristics", {}).get("version")
        == "component_mechanisms_smm_v1"
        and "rules" not in prompts["opro_heuristic"].get("heuristics", {})
    )

    repo_root = find_project_root(calibration_path)
    evaluator = TrialEvaluator(str(calibration_path))
    scientific_identity = scientific_identity_sha256(repo_root, evaluator)

    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "stage": "opro_pair_followup_first_adaptive_slot_gate",
        "comparison_config": str(comparison_path.resolve()),
        "comparison_config_sha256": hashlib.sha256(comparison_path.read_bytes()).hexdigest(),
        "calibration_config": str(calibration_path.resolve()),
        "calibration_config_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        "adaptive_policy_trial": len(initial),
        "methods": list(DECLARED_METHODS),
        "runtime_identity": dict(runtime.runtime_identity),
        "scientific_identity_sha256": scientific_identity,
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
    }


def assert_no_baseline_policy_dirs(root: Path) -> None:
    offenders = [
        path for path in root.glob("*/*/policies/*")
        if path.name.split("_repeat")[0] in BASELINE_METHOD_NAMES
    ]
    if offenders:
        raise RuntimeError(
            "Baseline policy directories present in follow-up root: "
            + ", ".join(str(path) for path in offenders)
        )


def main() -> None:
    load_repository_local_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-config",
        type=Path,
        default=Path("configs/experiments/followup_foundry_sonnet5/comparison.yaml"),
    )
    parser.add_argument("--ticker", action="append")
    parser.add_argument("--method", action="append", choices=DECLARED_METHODS)
    parser.add_argument("--repetition", action="append", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage-first-adaptive-slot", action="store_true")
    parser.add_argument("--evaluate-test-books", action="store_true")
    parser.add_argument(
        "--simulation-workers",
        type=int,
        help=(
            "Operational override for per-ticker scenario workers. The "
            "frozen optimizer.yaml value (3) is tuned for v9's five-"
            "ticker-processes-concurrently host; running fewer tickers at "
            "once on a given host can safely use more workers."
        ),
    )
    args = parser.parse_args()
    if args.simulation_workers is not None and args.simulation_workers < 1:
        parser.error("--simulation-workers must be positive")
    modes = sum(bool(value) for value in (
        args.dry_run, args.stage_first_adaptive_slot, args.evaluate_test_books,
    ))
    if modes > 1:
        parser.error(
            "--dry-run, --stage-first-adaptive-slot, and --evaluate-test-books "
            "are mutually exclusive"
        )

    comparison_path = args.comparison_config.resolve()
    plan = pair_plan(comparison_path)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    (
        calibration_path, test_path, _, comparison, calibration, test, _,
    ) = experiment_paths(comparison_path)
    selected_methods = select_methods(args.method)
    selected_tickers = tuple(str(value).upper() for value in (args.ticker or TICKERS))
    unknown_tickers = sorted(set(selected_tickers) - set(TICKERS))
    if unknown_tickers:
        raise RuntimeError(f"Unknown ticker filters: {unknown_tickers}")
    tasks = tuple(
        task for task in build_ticker_tasks(comparison, calibration)
        if task.symbol in selected_tickers
    )
    root = comparison_output_root(comparison, calibration_path, calibration)

    if args.evaluate_test_books:
        report = run_ticker_tests(
            comparison_path, selected_tickers, methods=selected_methods,
            calibration_simulation_workers=args.simulation_workers,
        )
        assert_no_baseline_policy_dirs(root)
        print(json.dumps({
            "tested_tickers": list(report["tickers"]),
            "cross_ticker_summary": report["cross_ticker_summary"],
        }, indent=2, sort_keys=True))
        return

    proposal_runtime = resolved_proposal_runtime(comparison_path, comparison)
    runtime = build_proposal_runtime(
        proposal_runtime, calibration_path.parent,
        model_key_override=comparison.get("llm_model_key"),
    )
    preflight_path = root / "report" / "anthropic_foundry_schema_preflight.json"
    validate_proposal_preflight(preflight_path, runtime, calibration_path, calibration)

    def _calibration_overrides(task) -> dict[str, Any]:
        overrides = task.calibration_overrides(
            comparison.get("llm_model_key"), proposal_runtime=proposal_runtime,
        )
        if (
            args.simulation_workers is not None
            and args.simulation_workers != int(calibration.get("simulation_workers", 1))
        ):
            overrides["simulation_workers"] = args.simulation_workers
        return overrides

    if args.stage_first_adaptive_slot:
        if args.repetition:
            raise RuntimeError(
                "The stage gate accepts ticker and method filters but not repetitions"
            )
        staged_execution_limit = len(calibration["initial_design"]) + 1
        full_budget = int(calibration["trial_budget"])
        for task in tasks:
            overrides = _calibration_overrides(task)
            evaluator = TrialEvaluator(str(calibration_path), runtime_overrides=overrides)
            for method in selected_methods:
                run_study(
                    str(calibration_path), method, 0,
                    budget=full_budget, execution_limit=staged_execution_limit,
                    evaluator_overrides=overrides,
                )
            report = audit_pair_stage(comparison_path, calibration_path, evaluator.study_dir)
            report["ticker"] = task.symbol
            _atomic_json(
                evaluator.study_dir / "report/opro_pair_stage_audit.json", report
            )
            if not report["passed"]:
                raise RuntimeError(
                    f"{task.symbol} pair stage audit failed: "
                    + ", ".join(report["failed_checks"])
                )
        assert_no_baseline_policy_dirs(root)
        print("Selected ticker-specific pair stage gates passed; inspect every audit.")
        return

    repetitions = tuple(args.repetition or range(int(calibration["optimizer_repetitions"])))
    for task in tasks:
        overrides = _calibration_overrides(task)
        evaluator = TrialEvaluator(str(calibration_path), runtime_overrides=overrides)
        stage_path = evaluator.study_dir / "report/opro_pair_stage_audit.json"
        if not stage_path.exists():
            raise RuntimeError(
                f"Run --stage-first-adaptive-slot --ticker {task.symbol} first"
            )
        stage = json.loads(stage_path.read_text())
        if stage.get("passed") is not True or stage.get("ticker") != task.symbol:
            raise RuntimeError(f"{task.symbol} pair stage audit is failed or stale")
        for method in selected_methods:
            for repetition in repetitions:
                run_study(
                    str(calibration_path), method, repetition,
                    evaluator_overrides=overrides,
                )
    assert_no_baseline_policy_dirs(root)
    print("Selected ticker-specific OPRO pair calibration arms are complete.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
