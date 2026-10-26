"""Direct, model-independent OpenRouter transport for OPRO proposals."""
from __future__ import annotations

from contextlib import contextmanager
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

from lob_sim.optimization.model_runtime import ModelCompletion, ProposalRuntimeError


RUNTIME_PROTOCOL = "openrouter_chat_completions_v1"


@dataclass(frozen=True)
class OpenRouterSettings:
    model_key: str
    model_id: str
    endpoint: str
    api_key_env: str
    timeout_seconds: int
    application_title: str
    system_prompt: str
    request_parameters: dict[str, Any]
    provider_preferences: dict[str, Any]
    gateway_path: Path
    registry_path: Path
    gateway_sha256: str
    registry_sha256: str
    retry_http_statuses: tuple[int, ...] = ()
    max_attempts: int = 1
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0
    coordination_lock_path: Path | None = None


@dataclass(frozen=True)
class HTTPResponse:
    status: int
    body: bytes
    headers: dict[str, str]


HTTPTransport = Callable[[str, dict[str, str], bytes, int], HTTPResponse]
Sleeper = Callable[[float], None]


class OpenRouterTransportError(ProposalRuntimeError):
    """A terminal transport failure retaining request-attempt metadata."""

    def __init__(self, message: str, proposal_metadata: dict[str, Any]) -> None:
        super().__init__(message, proposal_metadata)


class OpenRouterResponseError(ProposalRuntimeError):
    """A successful HTTP response without a usable assistant message."""

    def __init__(self, message: str, proposal_metadata: dict[str, Any]) -> None:
        super().__init__(message, proposal_metadata)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_openrouter_settings(
    llm_config: Mapping[str, Any],
    config_dir: Path,
    *,
    model_key_override: str | None = None,
) -> OpenRouterSettings:
    if str(llm_config.get("runtime", "")).lower() != "openrouter":
        raise ValueError("OPRO requires llm.runtime: openrouter")
    gateway_path = Path(str(llm_config.get("gateway_config", "openrouter.yaml")))
    registry_path = Path(str(llm_config.get("model_registry", "llm-models.yaml")))
    if not gateway_path.is_absolute():
        gateway_path = (config_dir / gateway_path).resolve()
    if not registry_path.is_absolute():
        registry_path = (config_dir / registry_path).resolve()
    gateway = yaml.safe_load(gateway_path.read_text())
    registry = yaml.safe_load(registry_path.read_text())
    if gateway.get("version") != "openrouter_gateway_v1":
        raise ValueError("Unsupported OpenRouter gateway configuration")
    if registry.get("version") != "opro_model_registry_v1":
        raise ValueError("Unsupported OPRO model registry")
    model_key = str(model_key_override or llm_config.get("model_key", ""))
    profiles = registry.get("models", {})
    if model_key not in profiles:
        raise ValueError(f"Unknown OPRO model key: {model_key}")
    profile = profiles[model_key]
    model_id = str(profile.get("model_id", ""))
    if "/" not in model_id:
        raise ValueError("OpenRouter model_id must use provider/model form")
    request_parameters = dict(profile.get("request", {}))
    forbidden = {
        "model", "models", "messages", "plugins", "provider", "response_format",
        "stream", "tool_choice", "tools",
    }
    overlap = forbidden & set(request_parameters)
    if overlap:
        raise ValueError(
            "Model request profile overrides structural fields: "
            + ", ".join(sorted(overlap))
        )
    provider = dict(profile.get("provider", {}))
    if provider.get("require_parameters") is not True:
        raise ValueError("OpenRouter profiles must require requested parameters")
    allowed_providers = provider.get("only")
    if not isinstance(allowed_providers, list) or not allowed_providers:
        raise ValueError("Paper model profiles must declare allowed providers")
    allow_fallbacks = provider.get("allow_fallbacks")
    if not isinstance(allow_fallbacks, bool):
        raise ValueError("OpenRouter profiles must declare allow_fallbacks")
    if allow_fallbacks and len(allowed_providers) < 2:
        raise ValueError("Provider fallback requires at least two allowed providers")
    order = provider.get("order", allowed_providers)
    if (
        not isinstance(order, list)
        or not order
        or not set(order).issubset(set(allowed_providers))
    ):
        raise ValueError("Provider order must use the declared allowed providers")
    if provider.get("data_collection") != "deny":
        raise ValueError("Paper model profiles must deny provider data collection")
    endpoint = str(gateway.get("endpoint", ""))
    if endpoint != "https://openrouter.ai/api/v1/chat/completions":
        raise ValueError("Unexpected OpenRouter endpoint")
    timeout = int(gateway.get("timeout_seconds", 0))
    if timeout <= 0:
        raise ValueError("OpenRouter timeout_seconds must be positive")
    reliability = gateway.get("transport_reliability", {})
    if not isinstance(reliability, dict):
        raise ValueError("OpenRouter transport_reliability must be an object")
    retry_http_statuses = tuple(
        int(status) for status in reliability.get("retry_http_statuses", ())
    )
    max_attempts = int(reliability.get("max_attempts", 1))
    initial_backoff = float(reliability.get("initial_backoff_seconds", 1.0))
    max_backoff = float(reliability.get("max_backoff_seconds", 60.0))
    if max_attempts < 1:
        raise ValueError("OpenRouter max_attempts must be positive")
    if initial_backoff < 0 or max_backoff < initial_backoff:
        raise ValueError("Invalid OpenRouter retry backoff")
    if any(status < 400 or status > 599 for status in retry_http_statuses):
        raise ValueError("Invalid OpenRouter retry HTTP status")
    lock_value = reliability.get("coordination_lock")
    coordination_lock = None
    if lock_value:
        coordination_lock = Path(str(lock_value))
        if not coordination_lock.is_absolute():
            coordination_lock = (gateway_path.parent / coordination_lock).resolve()
    return OpenRouterSettings(
        model_key=model_key,
        model_id=model_id,
        endpoint=endpoint,
        api_key_env=str(gateway.get("api_key_env", "OPENROUTER_API_KEY")),
        timeout_seconds=timeout,
        application_title=str(gateway.get("application_title", "")),
        system_prompt=str(gateway.get("system_prompt", "")),
        request_parameters=request_parameters,
        provider_preferences=provider,
        gateway_path=gateway_path,
        registry_path=registry_path,
        gateway_sha256=_sha256(gateway_path),
        registry_sha256=_sha256(registry_path),
        retry_http_statuses=retry_http_statuses,
        max_attempts=max_attempts,
        initial_backoff_seconds=initial_backoff,
        max_backoff_seconds=max_backoff,
        coordination_lock_path=coordination_lock,
    )


@contextmanager
def _exclusive_request_lock(path: Path | None):
    """Serialize OpenRouter requests across the five ticker processes."""
    if path is None:
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    raw = next(
        (value for key, value in headers.items() if key.lower() == "retry-after"),
        None,
    )
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        try:
            target = parsedate_to_datetime(raw)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            return max(0.0, (target - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _post_json(
    url: str, headers: dict[str, str], body: bytes, timeout_seconds: int
) -> HTTPResponse:
    request = Request(url, data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return HTTPResponse(
                status=int(response.status),
                body=response.read(),
                headers={str(key): str(value) for key, value in response.headers.items()},
            )
    except HTTPError as exc:
        return HTTPResponse(
            status=int(exc.code),
            body=exc.read(),
            headers={str(key): str(value) for key, value in exc.headers.items()},
        )
    except URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc.reason}") from exc


def _error_detail(body: bytes) -> str:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "non-JSON error response"
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    if isinstance(error, dict):
        message = str(error.get("message", "OpenRouter request failed"))
        code = error.get("code")
        parts = [f"{code}: {message}" if code is not None else message]
        metadata = error.get("metadata", {})
        if isinstance(metadata, dict):
            provider = metadata.get("provider_name") or metadata.get("provider")
            if provider:
                parts.append(f"provider={provider}")
            raw = metadata.get("raw")
            if raw:
                if not isinstance(raw, str):
                    raw = json.dumps(raw, sort_keys=True)
                parts.append(f"upstream={raw[:2000]}")
        return "; ".join(parts)
    return "OpenRouter request failed"


def _normalized_usage(raw: Any) -> dict[str, int | float]:
    usage = raw if isinstance(raw, dict) else {}
    completion_details = usage.get("completion_tokens_details", {})
    prompt_details = usage.get("prompt_tokens_details", {})
    if not isinstance(completion_details, dict):
        completion_details = {}
    if not isinstance(prompt_details, dict):
        prompt_details = {}
    return {
        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens", 0) or 0),
        "cache_read_tokens": int(prompt_details.get("cached_tokens", 0) or 0),
        "cache_write_tokens": int(prompt_details.get("cache_write_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "reported_cost_usd": float(usage.get("cost", 0.0) or 0.0),
    }


class OpenRouterRuntime:
    """Call one configured model through OpenRouter for each proposal."""

    def __init__(
        self,
        settings: OpenRouterSettings,
        *,
        transport: HTTPTransport = _post_json,
        environment: Mapping[str, str] | None = None,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        self.settings = settings
        self.model_key = settings.model_key
        self.runtime_identity = {
            "protocol": RUNTIME_PROTOCOL,
            "model_key": settings.model_key,
            "requested_model": settings.model_id,
            "gateway_config_sha256": settings.gateway_sha256,
            "model_registry_sha256": settings.registry_sha256,
        }
        self.transport = transport
        self.environment = os.environ if environment is None else environment
        self.sleeper = sleeper

    def _request_payload(
        self, prompt: str, response_schema: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "model": self.settings.model_id,
            "messages": [
                {"role": "system", "content": self.settings.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "opro_proposal",
                    "strict": True,
                    "schema": response_schema,
                },
            },
            "provider": dict(self.settings.provider_preferences),
            "stream": False,
            **self.settings.request_parameters,
        }

    def complete(
        self, prompt: str, response_schema: dict[str, Any]
    ) -> ModelCompletion:
        api_key = self.environment.get(self.settings.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"OpenRouter credential is missing: set {self.settings.api_key_env}"
            )
        payload = self._request_payload(prompt, response_schema)
        body = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.settings.application_title:
            headers["X-OpenRouter-Title"] = self.settings.application_title
        request_hash = hashlib.sha256(body).hexdigest()
        attempts: list[dict[str, Any]] = []
        response: HTTPResponse | None = None
        with _exclusive_request_lock(self.settings.coordination_lock_path):
            for attempt in range(1, self.settings.max_attempts + 1):
                try:
                    response = self.transport(
                        self.settings.endpoint,
                        headers,
                        body,
                        self.settings.timeout_seconds,
                    )
                except RuntimeError as exc:
                    row = {
                        "attempt": attempt,
                        "status": None,
                        "error": type(exc).__name__,
                        "retryable": attempt < self.settings.max_attempts,
                    }
                    attempts.append(row)
                    if attempt >= self.settings.max_attempts:
                        raise OpenRouterTransportError(
                            f"OpenRouter transport failed after {attempt} attempts: {exc}",
                            self._transport_error_metadata(request_hash, attempts),
                        ) from exc
                    delay = min(
                        self.settings.max_backoff_seconds,
                        self.settings.initial_backoff_seconds * 2 ** (attempt - 1),
                    )
                    row["backoff_seconds"] = delay
                    self.sleeper(delay)
                    continue
                retryable = response.status in self.settings.retry_http_statuses
                detail = _error_detail(response.body)
                row = {
                    "attempt": attempt,
                    "status": response.status,
                    "response_sha256": hashlib.sha256(response.body).hexdigest(),
                    "retryable": retryable,
                }
                if not 200 <= response.status < 300:
                    row["error_detail"] = detail
                attempts.append(row)
                if 200 <= response.status < 300:
                    break
                if retryable and attempt < self.settings.max_attempts:
                    backoff = min(
                        self.settings.max_backoff_seconds,
                        self.settings.initial_backoff_seconds * 2 ** (attempt - 1),
                    )
                    delay = max(
                        backoff,
                        _retry_after_seconds(response.headers) or 0.0,
                    )
                    row["backoff_seconds"] = delay
                    self.sleeper(delay)
                    continue
                raise OpenRouterTransportError(
                    f"OpenRouter request returned HTTP {response.status} "
                    f"after {attempt} attempts: {detail}",
                    self._transport_error_metadata(request_hash, attempts),
                )
        if response is None:
            raise RuntimeError("OpenRouter transport produced no response")
        try:
            raw = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("OpenRouter returned invalid JSON") from exc
        if not isinstance(raw, dict):
            raise RuntimeError("OpenRouter response is not an object")
        choices = raw.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise RuntimeError("OpenRouter response must contain exactly one choice")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise RuntimeError("OpenRouter response choice is not an object")
        message = choice.get("message", {})
        text = message.get("content") if isinstance(message, dict) else None
        if not isinstance(text, str) or not text.strip():
            usage = _normalized_usage(raw.get("usage"))
            finish_reason = str(choice.get("finish_reason", ""))
            raise OpenRouterResponseError(
                "OpenRouter response has no assistant text "
                f"(finish_reason={finish_reason or 'missing'}, "
                f"output_tokens={usage['output_tokens']}, "
                f"reasoning_tokens={usage['reasoning_tokens']})",
                {
                    "source": "openrouter_response_error",
                    "model_runtime_identity": dict(self.runtime_identity),
                    "model_key": self.model_key,
                    "model_request_id": raw.get("id"),
                    "model_requested": self.settings.model_id,
                    "model_resolved": raw.get("model"),
                    "model_provider": raw.get("provider"),
                    "model_finish_reason": finish_reason,
                    "model_usage": usage,
                    "model_request_sha256": request_hash,
                    "model_response_sha256": hashlib.sha256(
                        response.body
                    ).hexdigest(),
                    "model_raw_response": raw,
                    "transport_attempts": list(attempts),
                },
            )
        metadata = raw.get("openrouter_metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        provider = raw.get("provider") or metadata.get("provider_name")
        request_id = raw.get("id")
        resolved_model = raw.get("model")
        if not isinstance(request_id, str) or not request_id:
            raise RuntimeError("OpenRouter response has no request ID")
        if not isinstance(resolved_model, str) or not resolved_model:
            raise RuntimeError("OpenRouter response has no resolved model ID")
        return ModelCompletion(
            text=text.strip(),
            request_id=request_id,
            requested_model=self.settings.model_id,
            resolved_model=resolved_model,
            provider=str(provider) if provider else None,
            finish_reason=str(choice.get("finish_reason", "")),
            usage=_normalized_usage(raw.get("usage")),
            request_sha256=request_hash,
            response_sha256=hashlib.sha256(response.body).hexdigest(),
            raw_response=raw,
            transport_attempts=tuple(attempts),
        )

    def _transport_error_metadata(
        self, request_hash: str, attempts: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "source": "openrouter_transport_error",
            "model_runtime_identity": dict(self.runtime_identity),
            "model_key": self.model_key,
            "model_requested": self.settings.model_id,
            "model_request_sha256": request_hash,
            "transport_attempts": list(attempts),
        }
