"""Model-independent constrained OPRO proposal policy."""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from lob_sim.optimization.feedback import SMM_COMPONENTS
from lob_sim.optimization.model_runtime import ProposalRuntime
from lob_sim.optimization.opro import build_opro_prompt, parse_opro_response
from lob_sim.optimization.samplers import OPROProposalError, Proposal
from lob_sim.optimization.space import SearchDimension, SearchSpace


def _requires_token_encoding(dimension: SearchDimension) -> bool:
    return dimension.kind == "categorical" and any(
        isinstance(choice, (list, dict)) for choice in dimension.choices
    )


def categorical_choice_token(index: int) -> str:
    return f"choice_{int(index)}"


def transport_candidate_property(dimension: SearchDimension) -> dict[str, Any]:
    if dimension.kind == "categorical":
        if _requires_token_encoding(dimension):
            return {
                "type": "string",
                "enum": [
                    categorical_choice_token(index)
                    for index in range(len(dimension.choices))
                ],
            }
        return {"enum": copy.deepcopy(list(dimension.choices))}
    return {
        "type": "integer" if dimension.kind == "int" else "number",
        "minimum": dimension.low,
        "maximum": dimension.high,
    }


def transport_response_json_schema(space: SearchSpace) -> dict[str, Any]:
    candidate_properties = {
        dimension.name: transport_candidate_property(dimension)
        for dimension in space.dimensions
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["hypothesis", "evidence_trial_ids", "candidate", "predictions"],
        "properties": {
            "hypothesis": {"type": "string"},
            "evidence_trial_ids": {"type": "array", "items": {"type": "integer"}},
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
                        "direction": {
                            "type": "string",
                            "enum": ["decrease", "increase", "no_change"],
                        },
                        "rationale": {"type": "string"},
                    },
                },
            },
        },
    }


def _transport_map(space: SearchSpace) -> dict[str, list[dict[str, Any]]]:
    return {
        dimension.name: [
            {"token": categorical_choice_token(index), "value": copy.deepcopy(choice)}
            for index, choice in enumerate(dimension.choices)
        ]
        for dimension in space.dimensions
        if _requires_token_encoding(dimension)
    }


def build_transport_prompt(
    space: SearchSpace,
    history: list[dict[str, Any]],
    remaining_budget: int,
    heuristic_spec: dict[str, Any] | None,
) -> tuple[str, str, dict[str, list[dict[str, Any]]]]:
    """Add an explicit reversible token map to the ordinary OPRO prompt."""
    original, _ = build_opro_prompt(space, history, remaining_budget, heuristic_spec)
    payload = json.loads(original)
    mapping = _transport_map(space)
    if mapping:
        payload["constraints"].append(
            "For token-encoded categorical parameters, return the declared "
            "token; it is decoded to the mapped frozen value before evaluation."
        )
        payload["transport_categorical_encoding"] = mapping
        for item in payload["search_space"]:
            if item["name"] in mapping:
                item["choices"] = [entry["token"] for entry in mapping[item["name"]]]
        for name, entries in mapping.items():
            payload["response_schema"]["candidate"][name] = {
                "tokens": [entry["token"] for entry in entries],
                "meaning": "selects the corresponding complete frozen value",
            }
    prompt = json.dumps(payload, indent=2, sort_keys=True)
    return prompt, hashlib.sha256(prompt.encode()).hexdigest(), mapping


def normalize_transport_response(response_text: str, space: SearchSpace) -> str:
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise ValueError("OPRO response is not a single JSON object") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("candidate"), dict):
        raise ValueError("OPRO response must contain a candidate object")
    candidate = payload["candidate"]
    for dimension in space.dimensions:
        if not _requires_token_encoding(dimension):
            continue
        raw = candidate.get(dimension.name)
        tokens = [categorical_choice_token(i) for i in range(len(dimension.choices))]
        if raw not in tokens:
            raise ValueError(f"{dimension.name} must be one of the declared transport tokens")
        candidate[dimension.name] = copy.deepcopy(dimension.choices[tokens.index(raw)])
    return json.dumps(payload, sort_keys=True)


class OPROPolicy:
    """History-conditioned OPRO policy using any ProposalRuntime."""

    feedback_mode = "component"

    def __init__(
        self,
        space: SearchSpace,
        runtime: ProposalRuntime,
        heuristic_spec: dict[str, Any] | None,
    ) -> None:
        self.space = space
        self.runtime = runtime
        self.heuristic_spec = heuristic_spec
        self.name = "opro_heuristic" if heuristic_spec is not None else "opro_plain"

    def ask(
        self,
        trial_index: int,
        fixed: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        remaining_budget: int | None = None,
    ) -> Proposal:
        if fixed is not None:
            return Proposal(
                self.space.validate(fixed),
                metadata={
                    "source": "initial_design",
                    "model_key": self.runtime.model_key,
                    "model_runtime_identity": copy.deepcopy(
                        self.runtime.runtime_identity
                    ),
                },
            )
        history = history or []
        prompt, prompt_hash, mapping = build_transport_prompt(
            self.space, history, int(remaining_budget or 0), self.heuristic_spec
        )
        completion = self.runtime.complete(
            prompt, transport_response_json_schema(self.space)
        )
        runtime_metadata = {
            "source": self.name,
            "prompt_hash": prompt_hash,
            "prompt": prompt,
            "model_runtime_identity": copy.deepcopy(self.runtime.runtime_identity),
            "model_key": self.runtime.model_key,
            "model_request_id": completion.request_id,
            "model_requested": completion.requested_model,
            "model_resolved": completion.resolved_model,
            "model_provider": completion.provider,
            "model_finish_reason": completion.finish_reason,
            "model_usage": completion.usage,
            "model_request_sha256": completion.request_sha256,
            "model_response_sha256": completion.response_sha256,
            "transport_attempts": list(completion.transport_attempts),
            "model_raw_response": completion.raw_response,
            "raw_response": completion.text,
            "transport_categorical_encoding": mapping,
        }
        try:
            normalized_response = normalize_transport_response(
                completion.text, self.space
            )
            parsed = parse_opro_response(
                normalized_response,
                self.space,
                prompt_hash,
                {int(record["policy_trial"]) for record in history},
            )
        except ValueError as exc:
            raise OPROProposalError(str(exc), runtime_metadata) from exc
        return Proposal(
            parsed.candidate,
            metadata={
                **runtime_metadata,
                "normalized_response": normalized_response,
                "hypothesis": parsed.hypothesis,
                "evidence_trial_ids": list(parsed.evidence_trial_ids),
                "predictions": list(parsed.predictions),
            },
        )

    def tell(self, proposal: Proposal, result: dict[str, Any]) -> None:
        return None
