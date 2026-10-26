"""Persistent equal-budget ask--evaluate--tell experiment controller."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from lob_sim.optimization.evaluator import TrialEvaluator
from lob_sim.optimization.feedback import (
    build_optimizer_feedback,
    feedback_contract,
    policy_feedback,
)
from lob_sim.optimization.openrouter_runtime import OpenRouterRuntime, load_openrouter_settings
from lob_sim.optimization.opro_policy import OPROPolicy
from lob_sim.optimization.runtime_factory import build_proposal_runtime
from lob_sim.optimization.samplers import OptunaTPEPolicy, RandomPolicy


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _policy(
    name: str,
    evaluator: TrialEvaluator,
    seed: int,
    run_dir: Path,
    initial_design: list[dict[str, Any]],
):
    if name == "random":
        return RandomPolicy(evaluator.space, seed)
    if name == "tpe":
        return OptunaTPEPolicy(
            evaluator.space,
            seed,
            run_dir / "optuna.sqlite3",
            run_dir.name,
            initial_design,
        )
    if name in {"opro_plain", "opro_heuristic"}:
        llm = evaluator.config.get("llm", {})
        proposal_runtime_config = evaluator.config.get("proposal_runtime")
        if proposal_runtime_config is not None:
            # New, opt-in provider-neutral path. v9 never sets this override
            # key, so the frozen path below is unreachable-by-construction
            # when this branch is taken.
            runtime = build_proposal_runtime(
                proposal_runtime_config,
                evaluator.config_dir,
                model_key_override=evaluator.config.get("llm_model_key"),
            )
        else:
            # Frozen v9 path: exact existing code, unchanged.
            settings = load_openrouter_settings(
                llm,
                evaluator.config_dir,
                model_key_override=evaluator.config.get("llm_model_key"),
            )
            runtime = OpenRouterRuntime(settings)
        heuristic_spec = None
        if name == "opro_heuristic":
            heuristic_path = llm.get("heuristic_config")
            if not heuristic_path:
                raise ValueError("opro_heuristic requires llm.heuristic_config")
            path = Path(str(heuristic_path))
            if not path.is_absolute():
                path = (evaluator.config_dir / path).resolve()
            heuristic_spec = yaml.safe_load(path.read_text())
        return OPROPolicy(evaluator.space, runtime, heuristic_spec)
    raise ValueError(f"Unknown proposal policy: {name}")


def run_study(
    config_path: str,
    sampler_name: str,
    repetition: int,
    budget: int | None = None,
    execution_limit: int | None = None,
    evaluator_overrides: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Run or resume one repetition, optionally stopping before its full budget.

    ``budget`` is the policy-visible proposal budget. ``execution_limit`` only
    limits how many ledger slots this invocation may materialize. Keeping these
    concepts separate lets a stage gate execute one adaptive proposal without
    falsely telling the policy that only one proposal remains.
    """
    evaluator = TrialEvaluator(config_path, runtime_overrides=evaluator_overrides)
    configured_budget = int(evaluator.config["trial_budget"])
    budget = configured_budget if budget is None else int(budget)
    if budget <= 0:
        raise ValueError("Trial budget must be positive")
    execution_limit = budget if execution_limit is None else int(execution_limit)
    if execution_limit <= 0 or execution_limit > budget:
        raise ValueError("Execution limit must be between one and the trial budget")
    initial_design = [
        evaluator.space.validate(candidate)
        for candidate in evaluator.config.get("initial_design", [])
    ]
    if len(initial_design) > budget:
        raise ValueError("Trial budget is smaller than the common initial design")

    seed = int(evaluator.config.get("optimizer_seed", 2026)) + int(repetition)
    run_dir = evaluator.study_dir / "policies" / f"{sampler_name}_repeat{repetition:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    policy = _policy(sampler_name, evaluator, seed, run_dir, initial_design)
    component_names, invalid_penalty = feedback_contract(evaluator.config)
    policy_feedback_mode = str(getattr(policy, "feedback_mode", "none"))
    trial_dir = run_dir / "trials"
    trial_dir.mkdir(exist_ok=True)

    existing: dict[int, dict[str, Any]] = {}
    for path in sorted(trial_dir.glob("trial_*.json")):
        record = json.loads(path.read_text())
        existing[int(record["policy_trial"])] = record
    if existing and set(existing) != set(range(max(existing) + 1)):
        raise RuntimeError("Policy trial ledger has a gap; refusing ambiguous resume")

    records: list[dict[str, Any]] = [existing[index] for index in sorted(existing)]
    if sampler_name in {"opro_plain", "opro_heuristic"}:
        expected_protocol = getattr(policy.runtime, "runtime_identity", {}).get("protocol")
        incompatible = [
            int(record["policy_trial"])
            for record in records
            if record.get("proposal_protocol") != expected_protocol
        ]
        if incompatible:
            raise RuntimeError(
                "The OPRO ledger uses a different model-runtime protocol; "
                "start a new study_name instead of resuming slots "
                f"{incompatible}"
            )
        model_records = [
            record for record in records
            if record.get("proposal_metadata", {}).get("model_runtime_identity")
        ]
        expected_identity = getattr(policy.runtime, "runtime_identity", {})
        if any(
            record["proposal_metadata"]["model_runtime_identity"] != expected_identity
            for record in model_records
        ):
            raise RuntimeError(
                "The configured model runtime differs from the OPRO ledger; "
                "start a new study_name"
            )
    existing_fingerprints = {
        record.get("evaluation", {}).get("evaluator_fingerprint")
        for record in records
        if record.get("evaluation", {}).get("evaluator_fingerprint")
    }
    if existing_fingerprints and existing_fingerprints != {
        evaluator.evaluator_fingerprint
    }:
        raise RuntimeError(
            "Evaluator inputs changed since this policy ledger was created; "
            "start a new study_name instead of mixing objectives"
        )
    for trial_index in range(len(records), execution_limit):
        fixed = initial_design[trial_index] if trial_index < len(initial_design) else None
        started = time.time()
        proposal = None
        proposal_metadata: dict[str, Any] = {}
        candidate_key = None
        duplicate = False
        result: dict[str, Any] = {"valid": False, "objective": None}
        optimizer_feedback = None
        try:
            proposal = policy.ask(
                trial_index,
                fixed=fixed,
                history=records,
                remaining_budget=budget - trial_index,
            )
            proposal_metadata = proposal.metadata or {}
            candidate_key = evaluator.space.canonical_key(proposal.candidate)
            duplicate = any(
                record.get("candidate_key") == candidate_key for record in records
            )
            result = evaluator.evaluate(
                proposal.candidate,
                f"{sampler_name}-r{repetition}-t{trial_index}",
            )
            optimizer_feedback = build_optimizer_feedback(
                result,
                component_names=component_names,
                invalid_penalty=invalid_penalty,
            )
            policy.tell(
                proposal,
                policy_feedback(optimizer_feedback, policy_feedback_mode),
            )
            status = "complete" if result.get("valid") else "invalid"
            error = None
        except Exception as exc:
            proposal_metadata = getattr(exc, "proposal_metadata", proposal_metadata)
            if proposal is not None and optimizer_feedback is None:
                # A submitted vector that fails before returning structured
                # losses is a hidden-feasibility failure for proposal-policy
                # purposes. Feeding the common finite penalty also releases
                # synchronous policies such as BADS instead of deadlocking.
                result = {"valid": False, "objective": None}
                optimizer_feedback = build_optimizer_feedback(
                    result,
                    component_names=component_names,
                    invalid_penalty=invalid_penalty,
                )
                try:
                    policy.tell(
                        proposal,
                        policy_feedback(optimizer_feedback, policy_feedback_mode),
                    )
                except Exception as tell_exc:
                    exc = RuntimeError(f"{exc}; penalty feedback failed: {tell_exc}")
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        record = {
            "policy_trial": trial_index,
            "sampler": sampler_name,
            "proposal_protocol": (
                policy.runtime.runtime_identity["protocol"]
                if sampler_name in {"opro_plain", "opro_heuristic"}
                else "classical_v1"
            ),
            "repetition": int(repetition),
            "proposal_seed": seed,
            "policy_feedback_mode": policy_feedback_mode,
            "candidate": proposal.candidate if proposal is not None else None,
            "candidate_key": candidate_key,
            "duplicate": duplicate,
            "proposal_metadata": proposal_metadata,
            "status": status,
            "error": error,
            "objective": result.get("objective"),
            "optimizer_feedback": optimizer_feedback,
            "evaluation": result,
            "controller_runtime_seconds": time.time() - started,
        }
        _atomic_write(trial_dir / f"trial_{trial_index:04d}.json", record)
        records.append(record)
        _write_summary(run_dir, records, budget)
    _write_summary(run_dir, records, budget)
    return records, str(run_dir)


def _write_summary(run_dir: Path, records: list[dict[str, Any]], budget: int) -> None:
    best = float("inf")
    trajectory: list[float | None] = []
    for record in records:
        objective = record.get("objective")
        if objective is not None:
            best = min(best, float(objective))
        trajectory.append(best if best < float("inf") else None)
    valid = sum(record["status"] == "complete" for record in records)
    ranked = [record for record in records if record.get("objective") is not None]
    best_record = min(ranked, key=lambda record: float(record["objective"])) if ranked else None
    usage: dict[str, int | float] = {}
    model_calls = 0
    for record in records:
        model_usage = record.get("proposal_metadata", {}).get("model_usage", {})
        if model_usage:
            model_calls += 1
        for key in (
            "input_tokens", "output_tokens", "reasoning_tokens",
            "cache_read_tokens", "cache_write_tokens", "total_tokens",
            "reported_cost_usd",
        ):
            value = model_usage.get(key)
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
    summary = {
        "budget": int(budget),
        "completed_proposals": len(records),
        "valid_proposals": valid,
        "invalid_or_error_proposals": len(records) - valid,
        "duplicate_proposals": sum(bool(record["duplicate"]) for record in records),
        "best_objective": trajectory[-1] if trajectory else None,
        "best_policy_trial": best_record.get("policy_trial") if best_record else None,
        "best_candidate": best_record.get("candidate") if best_record else None,
        "best_so_far": trajectory,
        "controller_runtime_seconds": sum(
            float(record.get("controller_runtime_seconds", 0.0)) for record in records
        ),
        "model_calls": model_calls,
        "model_usage": usage,
        "evaluator_fingerprints": sorted({
            fingerprint
            for record in records
            if (fingerprint := record.get("evaluation", {}).get(
                "evaluator_fingerprint"
            ))
        }),
    }
    _atomic_write(run_dir / "summary.json", summary)
