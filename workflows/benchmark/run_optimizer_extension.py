#!/usr/bin/env python3
"""Run the four added classical-optimizer arms of the post-hoc comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from workflows import REPO_ROOT
from lob_sim.optimization.evaluator import TrialEvaluator
from lob_sim.optimization.report import write_benchmark_report
import lob_sim.optimization.study as study_module
from lob_sim.optimization.study import run_study
from workflows.benchmark.classical_optimizers import (
    BADSAskTellPolicy,
    OptunaCMAESPolicy,
    OptunaGPPolicy,
    SobolPolicy,
)
from workflows.validation.environment import (
    validate_environment,
    validate_extension_environment,
)
from workflows.validation.provenance import verify_provenance


METHODS = ("sobol", "gp_bo", "cma_es", "bads")
PRIMARY_METHODS = ("random", "tpe", "opro_plain", "opro_heuristic")
NON_LLM_COMPARISON_METHODS = ("random", "tpe", *METHODS)


def install_extension_policies(
    extension: dict[str, Any], extension_config_sha256: str
) -> None:
    """Install the four workflow-local policy adapters for this process."""
    original_policy = study_module._policy

    def policy_factory(name, evaluator, seed, run_dir, initial_design):
        if name not in METHODS:
            return original_policy(name, evaluator, seed, run_dir, initial_design)
        if name == "sobol":
            settings = extension["method_settings"]["sobol"]
            repetition = int(seed) - int(evaluator.config["optimizer_seed"])
            adaptive_slots = int(evaluator.config["trial_budget"]) - len(initial_design)
            return SobolPolicy(
                evaluator.space,
                int(settings["sequence_seed"]),
                len(initial_design),
                len(initial_design) + repetition * adaptive_slots,
                extension_config_sha256,
            )
        if name == "gp_bo":
            return OptunaGPPolicy(
                evaluator.space, seed, run_dir / "optuna.sqlite3", run_dir.name,
                initial_design, extension_config_sha256,
            )
        if name == "cma_es":
            return OptunaCMAESPolicy(
                evaluator.space, seed, run_dir / "optuna.sqlite3", run_dir.name,
                initial_design, extension_config_sha256,
            )
        settings = extension["method_settings"]["bads"]
        return BADSAskTellPolicy(
            evaluator.space,
            seed,
            len(initial_design),
            extension_config_sha256,
            invalid_penalty=float(settings["invalid_penalty"]),
        )

    study_module._policy = policy_factory


def validate_extension_contract(
    extension: dict[str, Any], optimizer: dict[str, Any]
) -> dict[str, bool]:
    common = extension.get("common_contract", {})
    checks = {
        "version": extension.get("version") == "icaif_optimizer_extension_v1",
        "status": extension.get("status") == "ready_to_run",
        "methods": tuple(extension.get("methods", ())) == METHODS,
        "comparison_methods": tuple(extension.get("comparison_methods", ()))
        == (*PRIMARY_METHODS, *METHODS),
        "comparison_study": extension.get("comparison_study_name")
        == "icaif_smm_v4_optimizer_comparison",
        "settings": set(extension.get("method_settings", {})) == set(METHODS),
        "survey_basis": set(extension.get("survey_basis", {}))
        == {"simulation_optimization", "derivative_free_optimization"},
        "rationales": all(
            extension.get("method_settings", {}).get(method, {}).get("rationale")
            for method in METHODS
        ),
        "budget": int(common.get("charged_slots_per_repetition", -1))
        == int(optimizer.get("trial_budget", -2)),
        "repetitions": int(common.get("repetitions", -1))
        == int(optimizer.get("optimizer_repetitions", -2)),
        "initial_design": int(common.get("common_initial_slots", -1))
        == len(optimizer.get("initial_design", [])),
        "deterministic_oracle": common.get("search_oracle")
        == "deterministic_common_random_numbers",
        "posthoc_disclosure": "post-hoc" in str(extension.get("disclosure", "")),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            "Optimizer-extension contract failed: " + ", ".join(failed)
        )
    return checks


def extension_plan(
    extension_path: Path,
    optimizer_path: Path,
    evaluator: TrialEvaluator,
) -> dict[str, Any]:
    extension = yaml.safe_load(extension_path.read_text())
    optimizer = yaml.safe_load(optimizer_path.read_text())
    checks = validate_extension_contract(extension, optimizer)
    repetitions = int(optimizer["optimizer_repetitions"])
    budget = int(optimizer["trial_budget"])
    methods: dict[str, list[dict[str, int]]] = {}
    for method in METHODS:
        rows = []
        for repetition in range(repetitions):
            trial_dir = (
                evaluator.study_dir / "policies"
                / f"{method}_repeat{repetition:02d}" / "trials"
            )
            present = len(list(trial_dir.glob("trial_*.json")))
            rows.append({
                "repetition": repetition,
                "slots_present": present,
                "slots_remaining": max(0, budget - present),
            })
        methods[method] = rows
    return {
        "extension_config": str(extension_path.resolve()),
        "extension_config_sha256": hashlib.sha256(
            extension_path.read_bytes()
        ).hexdigest(),
        "optimizer_config": str(optimizer_path.resolve()),
        "study_dir": str(evaluator.study_dir.resolve()),
        "evaluator_fingerprint": evaluator.evaluator_fingerprint,
        "contract": checks,
        "methods": methods,
        "posthoc_disclosure": extension["disclosure"].strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extension-config", type=Path,
        default=Path("configs/paper/optimizer-extension.yaml"),
    )
    parser.add_argument(
        "--method", action="append", choices=NON_LLM_COMPARISON_METHODS,
        help="run only this method; repeat the option to select several",
    )
    parser.add_argument(
        "--repetition", action="append", type=int,
        help="run only this zero-based repetition; repeat to select several",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--provenance",
        type=Path,
        help="comparison-study reconstruction provenance lock",
    )
    args = parser.parse_args()

    extension_path = args.extension_config.resolve()
    extension = yaml.safe_load(extension_path.read_text())
    optimizer_path = (extension_path.parent / extension["base_optimizer_config"]).resolve()
    optimizer = yaml.safe_load(optimizer_path.read_text())
    base_lock_path = REPO_ROOT / "configs/paper/environment-lock.yaml"
    extension_lock_path = (
        extension_path.parent / extension["environment_lock"]
    ).resolve()
    base_environment = validate_environment(yaml.safe_load(base_lock_path.read_text()))
    extension_environment = validate_extension_environment(
        yaml.safe_load(extension_lock_path.read_text())
    )

    evaluator_overrides = {"study_name": extension["comparison_study_name"]}
    evaluator = TrialEvaluator(
        str(optimizer_path), runtime_overrides=evaluator_overrides
    )
    plan = extension_plan(extension_path, optimizer_path, evaluator)
    plan["environment"] = {
        "primary": base_environment,
        "extension": extension_environment,
    }
    provenance_path = (
        args.provenance.resolve()
        if args.provenance
        else evaluator.study_dir / "report" / "reconstruction_provenance.json"
    )
    plan["reconstruction_provenance"] = {
        "path": str(provenance_path.resolve()),
        "present": provenance_path.exists(),
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        return
    plan["reconstruction_provenance"].update(
        verify_provenance(provenance_path)
    )

    install_extension_policies(extension, plan["extension_config_sha256"])

    selected_methods = tuple(args.method or METHODS)
    repetitions = tuple(
        args.repetition or range(int(optimizer["optimizer_repetitions"]))
    )
    if any(
        repetition < 0 or repetition >= int(optimizer["optimizer_repetitions"])
        for repetition in repetitions
    ):
        raise RuntimeError("Requested repetition is outside the frozen protocol")

    for method in selected_methods:
        for repetition in repetitions:
            print(
                f"Starting/resuming {method} repetition {repetition}", flush=True
            )
            run_study(
                str(optimizer_path), method, repetition,
                evaluator_overrides=evaluator_overrides,
            )
            write_benchmark_report(evaluator.study_dir)
    print(
        "Selected optimizer-extension arms complete. Evaluate the four added "
        "methods with --method-set new; use --method-set extended only after "
        "all eight methods have been run under this study fingerprint."
    )


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
