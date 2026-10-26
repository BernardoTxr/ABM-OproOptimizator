"""Provider-neutral model-identity predicate shared by stage audits.

Factored out of ``audit_component_stage.py``'s two duplicated OpenRouter-only
model-identity checks so the same logic can validate a persisted OPRO trial
against *any* ``ProposalRuntime``'s ``runtime_identity`` -- OpenRouter,
Anthropic Foundry, or a future adapter -- by comparing whole identity dicts
and an explicit acceptable-provider set rather than reaching into
OpenRouter-specific settings fields.
"""
from __future__ import annotations

from typing import Any, Mapping


def _last_attempt_succeeded(attempt: Mapping[str, Any]) -> bool:
    """True iff a transport-attempt row reports success, across adapters.

    OpenRouter rows use a numeric HTTP status directly (``"status": 200``).
    Anthropic Foundry rows use a semantic string (``"status": "ok"``) plus an
    optional numeric ``"status_code"`` (``None`` for connection errors, which
    have no HTTP status at all). Neither convention should be assumed
    universal, so both are accepted explicitly.
    """
    status = attempt.get("status")
    if isinstance(status, bool):
        return False
    if isinstance(status, int):
        return 200 <= status < 300
    if isinstance(status, str):
        if status == "ok":
            return True
        status_code = attempt.get("status_code")
        return isinstance(status_code, int) and 200 <= status_code < 300
    return False


def model_identity_matches(
    metadata: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
    acceptable_providers: set[str],
    *,
    expected_resolved_model: str | None = None,
) -> bool:
    """True iff a persisted OPRO trial's stamped runtime identity matches the
    currently configured runtime, transport succeeded, and every required
    audit field is present.
    """
    identity = metadata.get("model_runtime_identity", {})
    transport_attempts = metadata.get("transport_attempts", [])
    provider = str(metadata.get("model_provider", "")).lower()
    resolved_ok = (
        expected_resolved_model is None
        or metadata.get("model_resolved") == expected_resolved_model
    )
    return bool(
        identity == dict(expected_identity)
        and metadata.get("model_key") == expected_identity.get("model_key")
        and metadata.get("model_requested") == expected_identity.get("requested_model")
        and resolved_ok
        and provider in acceptable_providers
        and metadata.get("model_request_id")
        and metadata.get("model_request_sha256")
        and metadata.get("model_response_sha256")
        and isinstance(transport_attempts, list)
        and transport_attempts
        and _last_attempt_succeeded(transport_attempts[-1])
    )
