"""Deterministic prompt and response boundary for OPRO-style policies.

The proposal prompt exposes the same six-loss decomposition used by the
component-aware numerical baselines. The 29 signed moment residuals remain in
the evaluator ledger for scientific auditing but never enter this interface.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from lob_sim.optimization.feedback import SMM_COMPONENTS
from lob_sim.optimization.space import SearchSpace


_DIRECTIONS = {"increase", "decrease", "no_change"}


@dataclass(frozen=True)
class OPROProposal:
    hypothesis: str
    evidence_trial_ids: tuple[int, ...]
    candidate: dict[str, Any]
    predictions: tuple[dict[str, str], ...]
    prompt_hash: str


def _history_view(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose every submitted vector under the frozen component contract."""
    output = []
    for item in history:
        feedback = item.get("optimizer_feedback")
        output.append({
            "trial_id": int(item["policy_trial"]),
            "candidate": item.get("candidate"),
            "feasible": feedback.get("feasible") if isinstance(feedback, dict) else None,
            "component_losses": (
                feedback.get("component_losses") if isinstance(feedback, dict) else None
            ),
            "final_loss": (
                feedback.get("final_loss") if isinstance(feedback, dict) else None
            ),
            "status": item.get("status"),
            "error": item.get("error"),
            "duplicate": bool(item.get("duplicate", False)),
        })
    return output


def build_opro_prompt(
    space: SearchSpace,
    history: list[dict[str, Any]],
    remaining_budget: int,
    heuristic_spec: dict[str, Any] | None,
) -> tuple[str, str]:
    """Build the identical numerical task with an optional heuristic block."""
    payload = {
        "task": (
            "Propose one population-level parameter configuration that is likely "
            "to reduce the next final grouped SMM loss of an agent-based limit "
            "order-book simulator. The final loss is the equal-weight mean of "
            "the six component losses; lower values are better."
        ),
        "constraints": [
            "Return one JSON object only; do not use markdown.",
            "Use exactly the declared candidate keys and legal domain values.",
            "Do not propose source-code, metric, data-split, seed, or bound changes.",
            "Do not claim causality from a persuasive explanation.",
            "Every prediction is tested by the evaluator and may be wrong.",
        ],
        "remaining_proposal_budget": int(remaining_budget),
        "search_space": space.prompt_schema(),
        "observations": _history_view(history),
        "heuristic_condition": heuristic_spec is not None,
        "heuristics": heuristic_spec,
        "response_schema": {
            "hypothesis": "non-empty string",
            "evidence_trial_ids": ["integer trial ids from observations"],
            "candidate": {name: "value inside declared domain" for name in space.names},
            "predictions": [{
                "metric": (
                    "one of returns | spread | imbalance | trade_size | "
                    "inter_arrival | depth"
                ),
                "direction": "increase | decrease | no_change",
                "rationale": "short falsifiable reason",
            }],
        },
    }
    prompt = json.dumps(payload, indent=2, sort_keys=True)
    return prompt, hashlib.sha256(prompt.encode()).hexdigest()


def opro_response_json_schema(space: SearchSpace) -> dict[str, Any]:
    """Strict provider-side schema; ``SearchSpace.validate`` remains authoritative."""
    candidate_properties: dict[str, Any] = {}
    for dimension in space.dimensions:
        if dimension.kind == "categorical":
            candidate_properties[dimension.name] = {
                "enum": copy.deepcopy(list(dimension.choices))
            }
        else:
            candidate_properties[dimension.name] = {
                "type": "integer" if dimension.kind == "int" else "number",
                "minimum": dimension.low,
                "maximum": dimension.high,
            }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["hypothesis", "evidence_trial_ids", "candidate", "predictions"],
        "properties": {
            "hypothesis": {"type": "string"},
            "evidence_trial_ids": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "candidate": {
                "type": "object",
                "additionalProperties": False,
                "required": list(space.names),
                "properties": candidate_properties,
            },
            "predictions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["metric", "direction", "rationale"],
                    "properties": {
                        "metric": {"type": "string", "enum": list(SMM_COMPONENTS)},
                        "direction": {"type": "string", "enum": sorted(_DIRECTIONS)},
                        "rationale": {"type": "string"},
                    },
                },
            },
        },
    }


def parse_opro_response(
    response_text: str,
    space: SearchSpace,
    prompt_hash: str,
    available_trial_ids: set[int],
) -> OPROProposal:
    """Strictly validate model output before it can reach the simulator."""
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError("OPRO response is not a single JSON object") from exc
    if not isinstance(payload, dict):
        raise ValueError("OPRO response must be a JSON object")
    required = {"hypothesis", "evidence_trial_ids", "candidate", "predictions"}
    if set(payload) != required:
        raise ValueError(
            f"OPRO response schema mismatch: expected {sorted(required)}"
        )
    hypothesis = payload["hypothesis"]
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("OPRO hypothesis must be a non-empty string")
    evidence = payload["evidence_trial_ids"]
    if not isinstance(evidence, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in evidence
    ):
        raise ValueError("evidence_trial_ids must be a list of integers")
    if available_trial_ids and not evidence:
        raise ValueError("OPRO response must cite at least one available trial")
    unknown = sorted(set(evidence) - available_trial_ids)
    if unknown:
        raise ValueError(f"OPRO response cites unavailable trials: {unknown}")
    if not isinstance(payload["candidate"], dict):
        raise ValueError("OPRO candidate must be an object")
    candidate = space.validate(payload["candidate"])

    predictions = payload["predictions"]
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("OPRO response requires at least one prediction")
    normalized_predictions: list[dict[str, str]] = []
    for prediction in predictions:
        if not isinstance(prediction, dict) or set(prediction) != {
            "metric", "direction", "rationale"
        }:
            raise ValueError("Every prediction must match the strict schema")
        if prediction["direction"] not in _DIRECTIONS:
            raise ValueError("Prediction direction is invalid")
        if prediction["metric"] not in SMM_COMPONENTS:
            raise ValueError("Prediction metric is not a grouped SMM component")
        if any(not isinstance(prediction[key], str) or not prediction[key].strip()
               for key in ("metric", "rationale")):
            raise ValueError("Prediction metric and rationale must be non-empty strings")
        normalized_predictions.append({key: prediction[key] for key in (
            "metric", "direction", "rationale"
        )})
    return OPROProposal(
        hypothesis=hypothesis.strip(),
        evidence_trial_ids=tuple(evidence),
        candidate=candidate,
        predictions=tuple(normalized_predictions),
        prompt_hash=prompt_hash,
    )
