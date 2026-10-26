"""Azure AI Foundry (Anthropic Messages API) transport for OPRO proposals.

Structural sibling of ``openrouter_runtime.py``: same injectable-transport
seam so this module is fully unit-testable without the ``anthropic`` package
installed or any network access. The real transport imports ``anthropic``
lazily, only inside ``_default_transport``, so ``make smoke`` and
``python -m pytest -q`` never require it.

Structured output uses Claude's native ``output_config.format`` JSON-schema
mode (confirmed against the installed ``anthropic`` SDK: ``messages.create``
accepts ``output_config: {"format": {"type": "json_schema", "schema": ...}}``)
rather than forced tool-use. That schema subset does not enforce numeric
``minimum``/``maximum`` constraints; final candidate bounds and shape are
still enforced downstream by ``SearchSpace.validate()``, exactly as for the
OpenRouter adapter.

Request/response hashes cover the canonicalized logical payload (the JSON
request parameters, and ``response.to_dict()``), not raw wire bytes -- the
SDK does not expose those as directly as ``urllib`` does. This is an
intentional, disclosed difference from ``openrouter_runtime.py``'s raw-byte
hashes.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

from lob_sim.optimization.model_runtime import ModelCompletion, ProposalRuntimeError


RUNTIME_PROTOCOL = "anthropic_foundry_messages_v1"
PROVIDER_LABEL = "azure_ai_foundry"


@dataclass(frozen=True)
class FoundryPricing:
    """A cost rate card that is always present and always status-tagged.

    ``status`` is ``"unconfirmed"`` until a human verifies the rates against
    their own Azure invoice/portal; it is never silently treated as zero
    cost. Missing individual rates (e.g. cache read/write) stay ``None`` and
    are reported as ``"unknown_no_rate_card"``, never coerced to ``0.0``.
    """

    status: str
    currency: str
    input_usd_per_token: float | None
    output_usd_per_token: float | None
    cache_read_usd_per_token: float | None
    cache_write_usd_per_token: float | None
    source: str

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> "FoundryPricing":
        status = str(raw.get("status", "unconfirmed"))
        if status not in {"unconfirmed", "confirmed"}:
            raise ValueError("pricing.status must be 'unconfirmed' or 'confirmed'")
        return FoundryPricing(
            status=status,
            currency=str(raw.get("currency", "USD")),
            input_usd_per_token=_optional_float(raw.get("input_usd_per_token")),
            output_usd_per_token=_optional_float(raw.get("output_usd_per_token")),
            cache_read_usd_per_token=_optional_float(raw.get("cache_read_usd_per_token")),
            cache_write_usd_per_token=_optional_float(raw.get("cache_write_usd_per_token")),
            source=str(raw.get("source", "")),
        )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class AnthropicFoundrySettings:
    model_key: str
    deployment_name: str
    base_url: str
    api_key_env: str
    timeout_seconds: int
    system_prompt: str
    request_parameters: dict[str, Any]
    output_effort: str | None
    gateway_path: Path
    registry_path: Path
    gateway_sha256: str
    registry_sha256: str
    pricing: FoundryPricing
    retry_statuses: tuple[str, ...] = ("rate_limited", "server_error", "connection_error")
    max_attempts: int = 1
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0


@dataclass(frozen=True)
class FoundryHTTPResult:
    """Normalized outcome of one ``client.messages.create`` attempt."""

    status: str  # "ok" | "rate_limited" | "server_error" | "client_error" | "connection_error"
    status_code: int | None
    message: dict[str, Any] | None
    error_detail: str | None
    retry_after_seconds: float | None


FoundryTransport = Callable[[dict[str, Any], int], FoundryHTTPResult]
Sleeper = Callable[[float], None]


class AnthropicFoundryTransportError(ProposalRuntimeError):
    """A terminal transport failure retaining request-attempt metadata."""


class AnthropicFoundryResponseError(ProposalRuntimeError):
    """A successful response without a usable structured-output text block."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_anthropic_foundry_settings(
    llm_config: Mapping[str, Any],
    config_dir: Path,
    *,
    model_key_override: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> AnthropicFoundrySettings:
    environment = os.environ if environment is None else environment
    if str(llm_config.get("kind", "")) != "anthropic_foundry":
        raise ValueError("This loader requires proposal_runtime.kind: anthropic_foundry")
    gateway_path = Path(str(llm_config.get("config", "anthropic_foundry.yaml")))
    registry_path = Path(str(llm_config.get("model_registry", "models.yaml")))
    if not gateway_path.is_absolute():
        gateway_path = (config_dir / gateway_path).resolve()
    if not registry_path.is_absolute():
        registry_path = (config_dir / registry_path).resolve()
    gateway = yaml.safe_load(gateway_path.read_text())
    registry = yaml.safe_load(registry_path.read_text())
    if gateway.get("version") != "anthropic_foundry_gateway_v1":
        raise ValueError("Unsupported Anthropic Foundry gateway configuration")
    if registry.get("version") != "anthropic_foundry_model_registry_v1":
        raise ValueError("Unsupported Anthropic Foundry model registry")
    model_key = str(model_key_override or llm_config.get("model_key", ""))
    profiles = registry.get("models", {})
    if model_key not in profiles:
        raise ValueError(f"Unknown Anthropic Foundry model key: {model_key}")
    profile = profiles[model_key]
    deployment_name = str(profile.get("deployment_name", ""))
    deployment_name_env = str(profile.get("deployment_name_env", ""))
    if not deployment_name and deployment_name_env:
        deployment_name = str(environment.get(deployment_name_env, ""))
    if not deployment_name:
        if deployment_name_env:
            raise ValueError(
                "Anthropic Foundry deployment name is missing: "
                f"set {deployment_name_env}"
            )
        raise ValueError(
            "Anthropic Foundry model profile requires deployment_name "
            "or deployment_name_env"
        )
    request_parameters = dict(profile.get("request", {}))
    # output_effort is a request.output_effort registry key, not a raw
    # messages.create() kwarg -- it belongs inside output_config.effort,
    # merged there in _request_params, not spread at the top level.
    output_effort = request_parameters.pop("output_effort", None)
    if output_effort is not None:
        allowed_efforts = {"low", "medium", "high", "xhigh", "max"}
        if output_effort not in allowed_efforts:
            raise ValueError(f"output_effort must be one of {sorted(allowed_efforts)}")
    forbidden = {"model", "messages", "system", "output_config", "stream"}
    overlap = forbidden & set(request_parameters)
    if overlap:
        raise ValueError(
            "Model request profile overrides structural fields: "
            + ", ".join(sorted(overlap))
        )
    base_url = str(gateway.get("base_url", ""))
    base_url_env = str(gateway.get("base_url_env", ""))
    if not base_url and base_url_env:
        base_url = str(environment.get(base_url_env, ""))
    if not base_url:
        if base_url_env:
            raise ValueError(
                "Anthropic Foundry base URL is missing: "
                f"set {base_url_env}"
            )
        raise ValueError(
            "Anthropic Foundry gateway requires base_url or base_url_env"
        )
    timeout = int(gateway.get("timeout_seconds", 0))
    if timeout <= 0:
        raise ValueError("Anthropic Foundry timeout_seconds must be positive")
    reliability = gateway.get("transport_reliability", {})
    if not isinstance(reliability, dict):
        raise ValueError("Anthropic Foundry transport_reliability must be an object")
    retry_statuses = tuple(str(s) for s in reliability.get("retry_statuses", ()))
    max_attempts = int(reliability.get("max_attempts", 1))
    initial_backoff = float(reliability.get("initial_backoff_seconds", 1.0))
    max_backoff = float(reliability.get("max_backoff_seconds", 60.0))
    if max_attempts < 1:
        raise ValueError("Anthropic Foundry max_attempts must be positive")
    if initial_backoff < 0 or max_backoff < initial_backoff:
        raise ValueError("Invalid Anthropic Foundry retry backoff")
    pricing_raw = gateway.get("pricing")
    if not isinstance(pricing_raw, dict):
        raise ValueError("Anthropic Foundry gateway must declare a pricing block")
    return AnthropicFoundrySettings(
        model_key=model_key,
        deployment_name=deployment_name,
        base_url=base_url,
        api_key_env=str(gateway.get("api_key_env", "ANTHROPIC_FOUNDRY_API_KEY")),
        timeout_seconds=timeout,
        system_prompt=str(gateway.get("system_prompt", "")),
        request_parameters=request_parameters,
        output_effort=output_effort,
        gateway_path=gateway_path,
        registry_path=registry_path,
        gateway_sha256=_sha256(gateway_path),
        registry_sha256=_sha256(registry_path),
        pricing=FoundryPricing.from_dict(pricing_raw),
        retry_statuses=retry_statuses,
        max_attempts=max_attempts,
        initial_backoff_seconds=initial_backoff,
        max_backoff_seconds=max_backoff,
    )


def _retry_after_seconds(response: Any) -> float | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


_UNSUPPORTED_SCHEMA_KEYWORDS = (
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
)


def _sanitize_schema_for_foundry(schema: Any) -> Any:
    """Strip JSON-Schema keywords Anthropic's structured-output subset
    rejects outright (HTTP 400 ``invalid_request_error``, not a silent
    ignore) -- confirmed live: ``output_config.format.schema: For 'number'
    type, properties maximum, minimum are not supported``. Candidate bounds
    and shape remain enforced downstream by ``SearchSpace.validate()``, same
    as for the OpenRouter adapter; only the wire schema sent to this
    provider differs.
    """
    if isinstance(schema, dict):
        return {
            key: _sanitize_schema_for_foundry(value)
            for key, value in schema.items()
            if key not in _UNSUPPORTED_SCHEMA_KEYWORDS
        }
    if isinstance(schema, list):
        return [_sanitize_schema_for_foundry(item) for item in schema]
    return schema


def _default_transport(
    settings: AnthropicFoundrySettings, environment: Mapping[str, str]
) -> FoundryTransport:
    """Build the real SDK-backed transport. Imported lazily so the
    ``anthropic`` package is never required to import this module.

    The credential check happens before the ``anthropic`` import so a
    missing-credential error is raised even when the package is not
    installed (e.g. under the offline test/smoke environment).
    """
    api_key = environment.get(settings.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Anthropic Foundry credential is missing: set {settings.api_key_env}"
        )
    import anthropic  # local import: keep optional for offline tests/smoke

    client = anthropic.AnthropicFoundry(api_key=api_key, base_url=settings.base_url, max_retries=0)

    def _call(request_params: dict[str, Any], timeout_seconds: int) -> FoundryHTTPResult:
        try:
            response = client.with_options(timeout=timeout_seconds).messages.create(**request_params)
            return FoundryHTTPResult("ok", 200, response.to_dict(), None, None)
        except anthropic.RateLimitError as exc:
            return FoundryHTTPResult(
                "rate_limited", exc.status_code, None, str(exc), _retry_after_seconds(exc.response)
            )
        except anthropic.APIStatusError as exc:
            status = "server_error" if exc.status_code >= 500 else "client_error"
            return FoundryHTTPResult(status, exc.status_code, None, str(exc), None)
        except anthropic.APIConnectionError as exc:
            return FoundryHTTPResult("connection_error", None, None, str(exc), None)

    return _call


def _normalized_usage(usage: Mapping[str, Any] | None, pricing: FoundryPricing) -> dict[str, Any]:
    usage = usage or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    output_details = usage.get("output_tokens_details") or {}
    reasoning_tokens = int(output_details.get("thinking_tokens", 0) or 0)
    cache_read_tokens = int(usage.get("cache_read_input_tokens", 0) or 0)
    cache_write_tokens = int(usage.get("cache_creation_input_tokens", 0) or 0)
    total_tokens = input_tokens + output_tokens

    cost, cost_status = _compute_cost(
        input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, pricing
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
        "total_tokens": total_tokens,
        "reported_cost_usd": cost,
        "cost_status": cost_status,
        "cost_currency": pricing.currency,
        "cost_source": pricing.source or "anthropic_foundry_rate_card",
        "pricing_status": pricing.status,
    }


def _compute_cost(
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_write_tokens: int,
    pricing: FoundryPricing,
) -> tuple[float | None, str]:
    rates = {
        "input": (input_tokens, pricing.input_usd_per_token),
        "output": (output_tokens, pricing.output_usd_per_token),
        "cache_read": (cache_read_tokens, pricing.cache_read_usd_per_token),
        "cache_write": (cache_write_tokens, pricing.cache_write_usd_per_token),
    }
    missing = [name for name, (tokens, rate) in rates.items() if tokens and rate is None]
    if missing:
        return None, "unknown_no_rate_card"
    total = sum(tokens * rate for tokens, rate in rates.values() if rate is not None)
    status = "calculated" if pricing.status == "confirmed" else "calculated_unconfirmed_rate_card"
    return total, status


class AnthropicFoundryRuntime:
    """Call one configured Foundry deployment through the Messages API."""

    def __init__(
        self,
        settings: AnthropicFoundrySettings,
        *,
        transport: FoundryTransport | None = None,
        environment: Mapping[str, str] | None = None,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        self.settings = settings
        self.model_key = settings.model_key
        self.runtime_identity = {
            "protocol": RUNTIME_PROTOCOL,
            "model_key": settings.model_key,
            "requested_model": settings.deployment_name,
            "base_url": settings.base_url,
            "gateway_config_sha256": settings.gateway_sha256,
            "model_registry_sha256": settings.registry_sha256,
            "pricing_status": settings.pricing.status,
        }
        self._environment = os.environ if environment is None else environment
        self._explicit_transport = transport
        self.sleeper = sleeper

    def _transport(self) -> FoundryTransport:
        if self._explicit_transport is not None:
            return self._explicit_transport
        return _default_transport(self.settings, self._environment)

    def _request_params(self, prompt: str, response_schema: dict[str, Any]) -> dict[str, Any]:
        schema = _sanitize_schema_for_foundry(response_schema)
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if self.settings.output_effort is not None:
            output_config["effort"] = self.settings.output_effort
        return {
            "model": self.settings.deployment_name,
            "system": self.settings.system_prompt,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": output_config,
            **self.settings.request_parameters,
        }

    def complete(self, prompt: str, response_schema: dict[str, Any]) -> ModelCompletion:
        request_params = self._request_params(prompt, response_schema)
        request_body = json.dumps(request_params, sort_keys=True, separators=(",", ":")).encode()
        request_hash = hashlib.sha256(request_body).hexdigest()
        transport = self._transport()
        attempts: list[dict[str, Any]] = []
        result: FoundryHTTPResult | None = None
        for attempt in range(1, self.settings.max_attempts + 1):
            started = time.time()
            result = transport(request_params, self.settings.timeout_seconds)
            elapsed = time.time() - started
            retryable = result.status in self.settings.retry_statuses
            row: dict[str, Any] = {
                "attempt": attempt,
                "status": result.status,
                "status_code": result.status_code,
                "duration_seconds": elapsed,
                "retryable": retryable and attempt < self.settings.max_attempts,
            }
            if result.status != "ok":
                row["error_detail"] = result.error_detail
            attempts.append(row)
            if result.status == "ok":
                break
            if retryable and attempt < self.settings.max_attempts:
                backoff = min(
                    self.settings.max_backoff_seconds,
                    self.settings.initial_backoff_seconds * 2 ** (attempt - 1),
                )
                delay = max(backoff, result.retry_after_seconds or 0.0)
                row["backoff_seconds"] = delay
                self.sleeper(delay)
                continue
            raise AnthropicFoundryTransportError(
                f"Anthropic Foundry request failed after {attempt} attempts: "
                f"{result.status} ({result.error_detail})",
                self._transport_error_metadata(request_hash, attempts),
            )
        assert result is not None
        message = result.message or {}
        content = message.get("content") or []
        text = next(
            (block.get("text") for block in content if isinstance(block, dict) and block.get("type") == "text"),
            None,
        )
        usage = _normalized_usage(message.get("usage"), self.settings.pricing)
        if not isinstance(text, str) or not text.strip():
            raise AnthropicFoundryResponseError(
                "Anthropic Foundry response has no text content block "
                f"(stop_reason={message.get('stop_reason', 'missing')})",
                {
                    "source": "anthropic_foundry_response_error",
                    "model_runtime_identity": dict(self.runtime_identity),
                    "model_key": self.model_key,
                    "model_request_id": message.get("id"),
                    "model_requested": self.settings.deployment_name,
                    "model_resolved": message.get("model"),
                    "model_provider": PROVIDER_LABEL,
                    "model_finish_reason": message.get("stop_reason"),
                    "model_usage": usage,
                    "model_request_sha256": request_hash,
                    "model_response_sha256": hashlib.sha256(
                        json.dumps(message, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "model_raw_response": message,
                    "transport_attempts": list(attempts),
                },
            )
        request_id = message.get("id")
        resolved_model = message.get("model")
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError("Anthropic Foundry response has no request ID")
        if not isinstance(resolved_model, str) or not resolved_model:
            raise RuntimeError("Anthropic Foundry response has no resolved model ID")
        response_hash = hashlib.sha256(
            json.dumps(message, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return ModelCompletion(
            text=text.strip(),
            request_id=request_id,
            requested_model=self.settings.deployment_name,
            resolved_model=resolved_model,
            provider=PROVIDER_LABEL,
            finish_reason=str(message.get("stop_reason", "")),
            usage=usage,
            request_sha256=request_hash,
            response_sha256=response_hash,
            raw_response=message,
            transport_attempts=tuple(attempts),
        )

    def _transport_error_metadata(
        self, request_hash: str, attempts: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "source": "anthropic_foundry_transport_error",
            "model_runtime_identity": dict(self.runtime_identity),
            "model_key": self.model_key,
            "model_requested": self.settings.deployment_name,
            "model_request_sha256": request_hash,
            "transport_attempts": list(attempts),
        }
