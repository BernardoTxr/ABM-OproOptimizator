"""Dispatch a ``proposal_runtime`` config block to a concrete adapter.

Only the new opt-in ``proposal_runtime`` override path (see
``evaluator.py``'s ``allowed_overrides`` and ``study.py::_policy``) calls
this factory. The frozen v9 OpenRouter construction in ``study.py`` does not
go through it, so v9's code path and identity remain unreachable-by
-construction from this module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from lob_sim.optimization.anthropic_foundry_runtime import (
    AnthropicFoundryRuntime,
    load_anthropic_foundry_settings,
)
from lob_sim.optimization.model_runtime import ProposalRuntime
from lob_sim.optimization.openrouter_runtime import OpenRouterRuntime, load_openrouter_settings


def build_proposal_runtime(
    runtime_config: Mapping[str, Any],
    config_dir: Path,
    *,
    model_key_override: str | None = None,
) -> ProposalRuntime:
    kind = str(runtime_config.get("kind", ""))
    if kind == "openrouter":
        # Translate the uniform {kind, config, model_registry, model_key}
        # shape into load_openrouter_settings' own {runtime, gateway_config,
        # model_registry, model_key} shape.
        llm_config = {
            "runtime": "openrouter",
            "gateway_config": runtime_config.get("config", "openrouter.yaml"),
            "model_registry": runtime_config.get("model_registry", "llm-models.yaml"),
            "model_key": runtime_config.get("model_key"),
        }
        settings = load_openrouter_settings(
            llm_config, config_dir, model_key_override=model_key_override
        )
        return OpenRouterRuntime(settings)
    if kind == "anthropic_foundry":
        settings = load_anthropic_foundry_settings(
            runtime_config, config_dir, model_key_override=model_key_override
        )
        return AnthropicFoundryRuntime(settings)
    raise ValueError(f"Unknown proposal_runtime.kind: {kind!r}")
