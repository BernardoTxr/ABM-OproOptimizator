#!/usr/bin/env python3
"""Provider-neutral live preflight for any ``ProposalRuntime`` adapter.

Sends both maximum-history Plain and Heuristic OPRO prompts through a
runtime built by ``runtime_factory.build_proposal_runtime``, validates
strict JSON parsing, and asserts zero optimizer slots, simulator
evaluations, and Test Book access. Generalizes
``preflight_openrouter.py``'s OpenRouter-only ``validate_openrouter_preflight``
/``main`` without modifying that historical v9 preflight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from lob_sim.optimization.model_runtime import ProposalRuntime
from lob_sim.optimization.runtime_factory import build_proposal_runtime
from workflows.benchmark.preflight_openrouter import (
    _digest_json,
    _heuristic_spec,
    _sum_usage,
    parse_proposal_preflight_response,
    proposal_preflight_contract,
)
from workflows.common.local_env import load_repository_local_env


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run_preflight(
    runtime: ProposalRuntime, config_path: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Exercise both OPRO conditions through ``runtime`` and return the record."""
    heuristic_spec = _heuristic_spec(config, config_path.parent)
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
    return {
        "purpose": (
            "Provider-neutral maximum-history capacity and response-contract "
            "test for both OPRO conditions; uses synthetic losses and no "
            "optimizer slot or simulator evaluation"
        ),
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "runtime_identity": dict(runtime.runtime_identity),
        "runtime": runtime.runtime_identity.get("protocol"),
        "model_key": runtime.model_key,
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


def validate_proposal_preflight(
    artifact_path: Path,
    runtime: ProposalRuntime,
    config_path: Path,
    config: dict[str, Any],
) -> dict[str, bool]:
    """Reject absent, stale, mismatched, or non-zero-slot preflight records."""
    if not artifact_path.exists():
        raise RuntimeError(
            "Live OPRO requires a provider-neutral preflight artifact; run "
            "python -m workflows.benchmark.preflight_proposal_runtime first"
        )
    record = json.loads(artifact_path.read_text())
    heuristic_spec = _heuristic_spec(config, config_path.parent)
    specs = {"plain": None, "heuristic": heuristic_spec}
    _, _, _, schema = proposal_preflight_contract(config)
    checks = {
        "runtime_identity": record.get("runtime_identity") == dict(runtime.runtime_identity),
        "config_hash": record.get("config_sha256")
        == hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model_key": record.get("model_key") == runtime.model_key,
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
    requested_model = runtime.runtime_identity.get("requested_model")
    for name, spec in specs.items():
        _, prompt, prompt_hash, _ = proposal_preflight_contract(config, spec)
        condition = conditions.get(name, {}) if isinstance(conditions, dict) else {}
        attempts = condition.get("transport_attempts", [])
        checks[f"{name}_contract"] = bool(
            condition.get("heuristic_condition") is (spec is not None)
            and condition.get("requested_model") == requested_model
            and condition.get("resolved_model")
            and condition.get("provider")
            and condition.get("model_request_id")
            and condition.get("model_request_sha256")
            and condition.get("model_response_sha256")
            and condition.get("finish_reason")
            and condition.get("prompt_sha256") == prompt_hash
            and condition.get("prompt") == prompt
            and isinstance(condition.get("parsed_proposal"), dict)
            and isinstance(attempts, list)
            and attempts
            and [row.get("attempt") for row in attempts]
            == list(range(1, len(attempts) + 1))
        )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            "Proposal-runtime preflight is stale or invalid: " + ", ".join(failed)
        )
    return checks


def main() -> None:
    load_repository_local_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paper/optimizer.yaml"))
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--model-registry", type=Path, required=True)
    parser.add_argument("--runtime-kind", default="anthropic_foundry")
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    runtime_config = {
        "kind": args.runtime_kind,
        "config": str(args.runtime_config.resolve()),
        "model_registry": str(args.model_registry.resolve()),
        "model_key": args.model_key,
    }
    runtime = build_proposal_runtime(runtime_config, config_path.parent)

    record = run_preflight(runtime, config_path, config)
    output = args.output.resolve()
    _atomic_json(output, record)
    validate_proposal_preflight(output, runtime, config_path, config)
    print(f"Provider-neutral proposal-runtime preflight passed: {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"ERROR: {type(exc).__name__}: {exc}") from exc
