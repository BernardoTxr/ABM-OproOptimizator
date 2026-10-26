#!/usr/bin/env python3
"""Audit the common design and first adaptive slot of new-study arms."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lob_sim.optimization.feedback import SMM_COMPONENTS
from lob_sim.optimization.openrouter_runtime import RUNTIME_PROTOCOL, load_openrouter_settings
from lob_sim.optimization.space import SearchSpace
from workflows.benchmark.runtime_audit import model_identity_matches


METHODS = (
    "sobol", "tpe", "bads", "gp_bo", "mo_tpe", "component_gp_bo",
    "opro_plain", "opro_heuristic",
)
FEEDBACK_MODES = {
    "sobol": "none",
    "tpe": "scalar",
    "bads": "scalar",
    "gp_bo": "scalar",
    "mo_tpe": "component",
    "component_gp_bo": "component",
    "opro_plain": "component",
    "opro_heuristic": "component",
}


def audit_component_stage(
    comparison_path: Path, optimizer_path: Path, study_dir: Path
) -> dict[str, Any]:
    comparison = yaml.safe_load(comparison_path.read_text())
    optimizer = yaml.safe_load(optimizer_path.read_text())
    model_settings = load_openrouter_settings(
        optimizer["llm"], optimizer_path.parent,
        model_key_override=comparison["llm_model_key"],
    )
    space = SearchSpace.from_config(optimizer["search_space"])
    initial = [space.validate(row) for row in optimizer["initial_design"]]
    staged_budget = len(initial) + 1
    records: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        trial_dir = study_dir / "policies" / f"{method}_repeat00" / "trials"
        paths = sorted(trial_dir.glob("trial_*.json"))
        expected = [f"trial_{index:04d}.json" for index in range(staged_budget)]
        if [path.name for path in paths] != expected:
            raise RuntimeError(
                f"{method} does not contain exactly the staged {staged_budget} slots"
            )
        records[method] = [json.loads(path.read_text()) for path in paths]

    checks: dict[str, bool] = {}
    anchor = records[METHODS[0]]
    checks["common_initial_candidates"] = all(
        records[method][index].get("candidate") == initial[index]
        for method in METHODS for index in range(len(initial))
    )
    checks["common_initial_objectives"] = all(
        records[method][index].get("objective") == anchor[index].get("objective")
        for method in METHODS for index in range(len(initial))
    )
    fingerprints = {
        row.get("evaluation", {}).get("evaluator_fingerprint")
        for method in METHODS for row in records[method]
    }
    checks["one_evaluator_fingerprint"] = len(fingerprints) == 1 and None not in fingerprints

    for method in METHODS:
        checks[f"{method}_feedback_mode"] = all(
            row.get("policy_feedback_mode") == FEEDBACK_MODES[method]
            for row in records[method]
        )
        for index, row in enumerate(records[method]):
            feedback = row.get("optimizer_feedback")
            valid_feedback = isinstance(feedback, dict)
            if valid_feedback:
                components = feedback.get("component_losses", {})
                valid_feedback = (
                    tuple(feedback.get("component_order", ())) == SMM_COMPONENTS
                    and set(components) == set(SMM_COMPONENTS)
                    and np.isclose(
                        float(feedback["final_loss"]),
                        np.mean([float(components[name]) for name in SMM_COMPONENTS]),
                        rtol=1e-10,
                        atol=1e-12,
                    )
                )
            checks[f"{method}_slot_{index}_feedback"] = bool(valid_feedback)
        adaptive = records[method][-1]
        checks[f"{method}_adaptive_candidate_valid"] = bool(
            adaptive.get("candidate") and space.validate(adaptive["candidate"])
        )

    prompts: dict[str, dict[str, Any]] = {}
    for method in ("opro_plain", "opro_heuristic"):
        metadata = records[method][-1].get("proposal_metadata", {})
        raw_prompt = metadata.get("prompt")
        try:
            prompts[method] = json.loads(raw_prompt)
        except (TypeError, json.JSONDecodeError):
            prompts[method] = {}
        checks[f"{method}_model_identity"] = model_identity_matches(
            metadata,
            {
                "protocol": RUNTIME_PROTOCOL,
                "model_key": model_settings.model_key,
                "requested_model": model_settings.model_id,
                "gateway_config_sha256": model_settings.gateway_sha256,
                "model_registry_sha256": model_settings.registry_sha256,
            },
            set(model_settings.provider_preferences["only"]),
            expected_resolved_model=model_settings.model_id,
        )
        checks[f"{method}_prompt_information_firewall"] = bool(
            prompts[method]
            and "signed_diagnostics" not in raw_prompt
            and "diagnostic_keys" not in raw_prompt
            and "prediction_outcomes" not in raw_prompt
            and len(prompts[method].get("observations", [])) == len(initial)
                and all(
                    set(observation.get("component_losses", {}))
                    == set(SMM_COMPONENTS)
                    for observation in prompts[method]["observations"]
                )
        )
    checks["opro_same_numerical_history"] = (
        prompts["opro_plain"].get("observations")
        == prompts["opro_heuristic"].get("observations")
    )
    expected_remaining = int(optimizer["trial_budget"]) - len(initial)
    checks["opro_full_remaining_budget"] = all(
        prompts[method].get("remaining_proposal_budget") == expected_remaining
        for method in ("opro_plain", "opro_heuristic")
    )
    checks["opro_heuristic_ablation"] = (
        prompts["opro_plain"].get("heuristic_condition") is False
        and prompts["opro_plain"].get("heuristics") is None
        and prompts["opro_heuristic"].get("heuristic_condition") is True
        and prompts["opro_heuristic"].get("heuristics", {}).get("version")
        == "component_mechanisms_smm_v1"
        and "rules" not in prompts["opro_heuristic"].get("heuristics", {})
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "stage": "component_aware_first_adaptive_slot_gate",
        "comparison_config": str(comparison_path.resolve()),
        "comparison_config_sha256": hashlib.sha256(
            comparison_path.read_bytes()
        ).hexdigest(),
        "optimizer_config": str(optimizer_path.resolve()),
        "optimizer_config_sha256": hashlib.sha256(optimizer_path.read_bytes()).hexdigest(),
        "adaptive_policy_trial": len(initial),
        "methods": list(METHODS),
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
    }


def audit_method_stage(
    comparison_path: Path,
    optimizer_path: Path,
    study_dir: Path,
    method: str,
) -> dict[str, Any]:
    """Audit one arm so methods can be executed independently and resumed."""
    if method not in METHODS:
        raise RuntimeError(f"Unknown staged method: {method}")
    comparison = yaml.safe_load(comparison_path.read_text())
    optimizer = yaml.safe_load(optimizer_path.read_text())
    space = SearchSpace.from_config(optimizer["search_space"])
    initial = [space.validate(row) for row in optimizer["initial_design"]]
    staged_budget = len(initial) + 1

    def staged_records(candidate_method: str, *, required: bool) -> list[dict[str, Any]]:
        trial_dir = (
            study_dir / "policies" / f"{candidate_method}_repeat00" / "trials"
        )
        paths = sorted(trial_dir.glob("trial_*.json"))
        expected = [f"trial_{index:04d}.json" for index in range(staged_budget)]
        if required and [path.name for path in paths[:staged_budget]] != expected:
            raise RuntimeError(
                f"{candidate_method} does not contain the staged {staged_budget} slots"
            )
        if [path.name for path in paths[:staged_budget]] != expected:
            return []
        return [json.loads(path.read_text()) for path in paths[:staged_budget]]

    records = staged_records(method, required=True)
    anchor_method = method
    anchor = records
    for candidate_method in METHODS:
        if candidate_method == method:
            continue
        candidate_records = staged_records(candidate_method, required=False)
        if candidate_records:
            anchor_method = candidate_method
            anchor = candidate_records
            break

    checks: dict[str, bool] = {
        "initial_candidates": all(
            records[index].get("candidate") == initial[index]
            for index in range(len(initial))
        ),
        "common_initial_objectives": all(
            records[index].get("objective") == anchor[index].get("objective")
            for index in range(len(initial))
        ),
        "one_evaluator_fingerprint": len({
            row.get("evaluation", {}).get("evaluator_fingerprint")
            for row in records
        }) == 1 and all(
            row.get("evaluation", {}).get("evaluator_fingerprint") is not None
            for row in records
        ),
        "feedback_mode": all(
            row.get("policy_feedback_mode") == FEEDBACK_MODES[method]
            for row in records
        ),
        "adaptive_candidate_valid": bool(
            records[-1].get("candidate")
            and space.validate(records[-1]["candidate"])
        ),
    }
    for index, row in enumerate(records):
        feedback = row.get("optimizer_feedback")
        valid_feedback = isinstance(feedback, dict)
        if valid_feedback:
            components = feedback.get("component_losses", {})
            valid_feedback = (
                tuple(feedback.get("component_order", ())) == SMM_COMPONENTS
                and set(components) == set(SMM_COMPONENTS)
                and np.isclose(
                    float(feedback["final_loss"]),
                    np.mean([float(components[name]) for name in SMM_COMPONENTS]),
                    rtol=1e-10,
                    atol=1e-12,
                )
            )
        checks[f"slot_{index}_feedback"] = bool(valid_feedback)

    if method in {"opro_plain", "opro_heuristic"}:
        model_settings = load_openrouter_settings(
            optimizer["llm"],
            optimizer_path.parent,
            model_key_override=comparison["llm_model_key"],
        )
        metadata = records[-1].get("proposal_metadata", {})
        raw_prompt = metadata.get("prompt")
        try:
            prompt = json.loads(raw_prompt)
        except (TypeError, json.JSONDecodeError):
            prompt = {}
        checks["model_identity"] = model_identity_matches(
            metadata,
            {
                "protocol": RUNTIME_PROTOCOL,
                "model_key": model_settings.model_key,
                "requested_model": model_settings.model_id,
                "gateway_config_sha256": model_settings.gateway_sha256,
                "model_registry_sha256": model_settings.registry_sha256,
            },
            set(model_settings.provider_preferences["only"]),
            expected_resolved_model=model_settings.model_id,
        )
        checks["prompt_information_firewall"] = bool(
            prompt
            and "signed_diagnostics" not in raw_prompt
            and "diagnostic_keys" not in raw_prompt
            and "prediction_outcomes" not in raw_prompt
            and len(prompt.get("observations", [])) == len(initial)
            and all(
                set(observation.get("component_losses", {}))
                == set(SMM_COMPONENTS)
                for observation in prompt["observations"]
            )
        )
        if method == "opro_plain":
            checks["heuristic_condition"] = (
                prompt.get("heuristic_condition") is False
                and prompt.get("heuristics") is None
            )
        else:
            checks["heuristic_condition"] = (
                prompt.get("heuristic_condition") is True
                and prompt.get("heuristics", {}).get("version")
                == "component_mechanisms_smm_v1"
                and "rules" not in prompt.get("heuristics", {})
            )
        checks["full_remaining_budget"] = (
            prompt.get("remaining_proposal_budget")
            == int(optimizer["trial_budget"]) - len(initial)
        )

    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "stage": "component_aware_single_method_first_adaptive_slot_gate",
        "comparison_config": str(comparison_path.resolve()),
        "comparison_config_sha256": hashlib.sha256(
            comparison_path.read_bytes()
        ).hexdigest(),
        "optimizer_config": str(optimizer_path.resolve()),
        "optimizer_config_sha256": hashlib.sha256(
            optimizer_path.read_bytes()
        ).hexdigest(),
        "adaptive_policy_trial": len(initial),
        "method": method,
        "feedback_mode": FEEDBACK_MODES[method],
        "objective_anchor_method": anchor_method,
        "checks": checks,
        "failed_checks": failed,
        "passed": not failed,
    }
