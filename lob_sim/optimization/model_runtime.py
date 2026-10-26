"""Transport-neutral interface between OPRO and a configured language model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelCompletion:
    text: str
    request_id: str
    requested_model: str
    resolved_model: str
    provider: str | None
    finish_reason: str
    usage: dict[str, Any]
    request_sha256: str
    response_sha256: str
    raw_response: dict[str, Any]
    transport_attempts: tuple[dict[str, Any], ...] = ()


class ProposalRuntimeError(RuntimeError):
    """Provider-neutral failure that preserves auditable request metadata."""

    def __init__(self, message: str, proposal_metadata: dict[str, Any]) -> None:
        super().__init__(message)
        self.proposal_metadata = proposal_metadata


class ProposalRuntime(Protocol):
    """The only model capability required by the OPRO proposal policy."""

    model_key: str
    runtime_identity: dict[str, Any]

    def complete(
        self, prompt: str, response_schema: dict[str, Any]
    ) -> ModelCompletion:
        ...
