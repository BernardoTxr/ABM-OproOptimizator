#!/usr/bin/env python3
"""Test the OPRO contract through OpenRouter without consuming a trial slot."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from lob_sim.optimization.feedback import SMM_COMPONENTS
from lob_sim.optimization.openrouter_runtime import (
    OpenRouterRuntime,
    RUNTIME_PROTOCOL,
    load_openrouter_settings,
)
from lob_sim.optimization.opro import parse_opro_response
from lob_sim.optimization.opro_policy import (
    build_transport_prompt,
    normalize_transport_response,
    transport_response_json_schema,
)
from lob_sim.optimization.space import SearchSpace
from workflows.common.local_env import load_repository_local_env


def representative_preflight_history(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Create the maximum proposal history without empirical outcomes."""
    initial_design = config.get("initial_design", [])
    if not isinstance(initial_design, list) or not initial_design:
        raise ValueError("OpenRouter preflight requires the common initial design")
    space = SearchSpace.from_config(config.get("search_space", []))
    history_size = int(config.get("trial_budget", len(initial_design))) - 1
    if history_size < len(initial_design):
        raise ValueError("Trial budget cannot be smaller than the initial design")
    rows = []
    for index in range(history_size):
        if index < len(initial_design):
            candidate = copy.deepcopy(initial_design[index])
        else:
            candidate = {}
            for offset, dimension in enumerate(space.dimensions):
                if dimension.kind == "categorical":
                    candidate[dimension.name] = copy.deepcopy(
                        dimension.choices[(index + offset) % len(dimension.choices)]
                    )
                elif dimension.kind == "int":
                    width = int(dimension.high) - int(dimension.low) + 1
                    candidate[dimension.name] = int(dimension.low) + (
                        index * 7919 + offset * 104729
                    ) % width
                else:
                    unit = ((index + 1) * 0.61803398875 + offset * 0.137) % 1.0
                    candidate[dimension.name] = round(
                        float(dimension.low)
                        + (float(dimension.high) - float(dimension.low)) * unit,
                        6,
                    )
            candidate = space.validate(candidate)
        # Positive synthetic values exercise the full prompt shape and
        # approximate its size without reading a book or simulator result.
        components = {
            name: float(1.0 + index / 10.0 + component_index / 100.0)
            for component_index, name in enumerate(SMM_COMPONENTS)
        }
        rows.append({
            "policy_trial": index,
            "candidate": candidate,
            "status": "complete",
            "error": None,
            "duplicate": False,
            "optimizer_feedback": {
                "version": "grouped_smm_feedback_v1",
                "feasible": True,
                "component_order": list(SMM_COMPONENTS),
                "component_losses": components,
                "final_loss": float(sum(components.values()) / len(components)),
                "invalid_penalty_applied": False,
            },
        })
    return rows


def proposal_preflight_contract(
    config: dict[str, Any],
    heuristic_spec: dict[str, Any] | None = None,
) -> tuple[SearchSpace, str, str, dict[str, Any]]:
    space = SearchSpace.from_config(config.get("search_space", []))
    history = representative_preflight_history(config)
    prompt, prompt_hash, _ = build_transport_prompt(
        space,
        history,
        int(config.get("trial_budget", len(history))) - len(history),
        heuristic_spec,
    )
    return space, prompt, prompt_hash, transport_response_json_schema(space)


def _heuristic_spec(config: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    value = config.get("llm", {}).get("heuristic_config")
    if not value:
        raise ValueError("OpenRouter dual preflight requires llm.heuristic_config")
    path = Path(str(value))
    if not path.is_absolute():
        path = (config_dir / path).resolve()
    spec = yaml.safe_load(path.read_text())
    if not isinstance(spec, dict):
        raise ValueError("OPRO heuristic registry must be an object")
    return spec


def _sum_usage(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    fields = (
        "input_tokens", "output_tokens", "reasoning_tokens", "total_tokens",
        "cache_read_tokens", "cache_write_tokens", "reported_cost_usd",
    )
    return {
        field: sum(row.get(field, 0) for row in rows)
        for field in fields
    }


def parse_proposal_preflight_response(
    text: str,
    space: SearchSpace,
    prompt_hash: str,
    available_trial_ids: set[int],
) -> dict[str, Any]:
    normalized = normalize_transport_response(text, space)
    parsed = parse_opro_response(
        normalized, space, prompt_hash, available_trial_ids
    )
    return {
        "candidate": parsed.candidate,
        "hypothesis": parsed.hypothesis,
        "evidence_trial_ids": list(parsed.evidence_trial_ids),
        "predictions": list(parsed.predictions),
        "normalized_response": normalized,
    }


def _digest_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def validate_openrouter_preflight(
    artifact_path: Path,
    config_path: Path,
    config: dict[str, Any],
    *,
    model_key_override: str | None = None,
) -> dict[str, bool]:
    """Reject absent, stale, mismatched, or non-zero-slot preflight records."""
    if not artifact_path.exists():
        raise RuntimeError(
            "Live OPRO requires the OpenRouter preflight artifact; run "
            "python -m workflows.benchmark.preflight_openrouter first"
        )
    record = json.loads(artifact_path.read_text())
    settings = load_openrouter_settings(
        config.get("llm", {}), config_path.parent, model_key_override=model_key_override
    )
    heuristic_spec = _heuristic_spec(config, config_path.parent)
    specs = {"plain": None, "heuristic": heuristic_spec}
    _, _, _, schema = proposal_preflight_contract(config)
    checks = {
        "runtime": record.get("runtime") == RUNTIME_PROTOCOL,
        "config_hash": record.get("config_sha256")
        == hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "gateway_hash": record.get("gateway_config_sha256") == settings.gateway_sha256,
        "registry_hash": record.get("model_registry_sha256") == settings.registry_sha256,
        "model_key": record.get("model_key") == settings.model_key,
        "schema_hash": record.get("schema_sha256") == _digest_json(schema)
        and record.get("schema") == schema,
        "conditions": set(record.get("conditions", {})) == set(specs),
        "representative_history": (
            record.get("preflight_history") == "synthetic_maximum_adaptive_history"
            and record.get("preflight_observations")
            == int(config.get("trial_budget", 0)) - 1
        ),
        "zero_slots": record.get("optimizer_slots_consumed") == 0,
        "zero_simulations": record.get("simulator_evaluations") == 0,
        "no_test_books": record.get("test_books_accessed") is False,
    }
    conditions = record.get("conditions", {})
    for name, spec in specs.items():
        _, prompt, prompt_hash, _ = proposal_preflight_contract(config, spec)
        condition = conditions.get(name, {}) if isinstance(conditions, dict) else {}
        attempts = condition.get("transport_attempts", [])
        checks[f"{name}_contract"] = bool(
            condition.get("heuristic_condition") is (spec is not None)
            and condition.get("requested_model") == settings.model_id
            and condition.get("resolved_model") == settings.model_id
            and str(condition.get("provider", "")).lower()
            in settings.provider_preferences["only"]
            and condition.get("model_request_id")
            and condition.get("model_request_sha256")
            and condition.get("model_response_sha256")
            and condition.get("finish_reason") == "stop"
            and condition.get("prompt_sha256") == prompt_hash
            and condition.get("prompt") == prompt
            and isinstance(condition.get("parsed_proposal"), dict)
            and isinstance(attempts, list)
            and attempts
            and [row.get("attempt") for row in attempts]
            == list(range(1, len(attempts) + 1))
            and 200 <= int(attempts[-1].get("status", 0)) < 300
        )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError("OpenRouter preflight is stale or invalid: " + ", ".join(failed))
    return checks


def main() -> None:
    load_repository_local_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paper/optimizer.yaml"))
    parser.add_argument("--model-key")
    parser.add_argument(
        "--output", type=Path,
        default=Path(
            "results/optimization/icaif_smm_v9_ticker_specific_10x30m/"
            "report/openrouter_schema_preflight.json"
        ),
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    settings = load_openrouter_settings(
        config.get("llm", {}), config_path.parent, model_key_override=args.model_key
    )
    heuristic_spec = _heuristic_spec(config, config_path.parent)
    runtime = OpenRouterRuntime(settings)
    conditions: dict[str, Any] = {}
    usages = []
    schema = None
    for name, spec in (("plain", None), ("heuristic", heuristic_spec)):
        space, prompt, prompt_hash, current_schema = proposal_preflight_contract(
            config, spec
        )
        schema = current_schema
        completion = runtime.complete(prompt, current_schema)
        parsed = parse_proposal_preflight_response(
            completion.text,
            space,
            prompt_hash,
            set(range(int(config.get("trial_budget", 0)) - 1)),
        )
        usages.append(completion.usage)
        conditions[name] = {
            "heuristic_condition": spec is not None,
            "requested_model": completion.requested_model,
            "resolved_model": completion.resolved_model,
            "provider": completion.provider,
            "finish_reason": completion.finish_reason,
            "model_request_id": completion.request_id,
            "model_usage": completion.usage,
            "model_request_sha256": completion.request_sha256,
            "model_response_sha256": completion.response_sha256,
            "transport_attempts": list(completion.transport_attempts),
            "prompt": prompt,
            "prompt_sha256": prompt_hash,
            "raw_response": completion.text,
            "parsed_proposal": parsed,
        }
    assert schema is not None
    record = {
        "purpose": (
            "OpenRouter maximum-history capacity and response-contract test "
            "for both OPRO conditions; uses synthetic losses and no optimizer "
            "slot or simulator evaluation"
        ),
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "gateway_config_sha256": settings.gateway_sha256,
        "model_registry_sha256": settings.registry_sha256,
        "runtime": RUNTIME_PROTOCOL,
        "model_key": settings.model_key,
        "conditions": conditions,
        "model_usage": _sum_usage(usages),
        "providers": sorted({
            str(condition["provider"]) for condition in conditions.values()
        }),
        "schema": schema,
        "schema_sha256": _digest_json(schema),
        "preflight_history": "synthetic_maximum_adaptive_history",
        "preflight_observations": int(config.get("trial_budget", 0)) - 1,
        "optimizer_slots_consumed": 0,
        "simulator_evaluations": 0,
        "test_books_accessed": False,
    }
    output = args.output.resolve()
    _atomic_json(output, record)
    validate_openrouter_preflight(
        output,
        config_path,
        config,
        model_key_override=args.model_key,
    )
    print(f"OpenRouter schema preflight passed: {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"ERROR: {type(exc).__name__}: {exc}") from exc
