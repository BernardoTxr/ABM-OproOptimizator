import tempfile
import unittest
from pathlib import Path

import yaml

from lob_sim.optimization.anthropic_foundry_runtime import AnthropicFoundryRuntime
from lob_sim.optimization.openrouter_runtime import OpenRouterRuntime
from lob_sim.optimization.runtime_factory import build_proposal_runtime


class RuntimeFactoryTests(unittest.TestCase):
    def test_dispatches_to_anthropic_foundry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gateway.yaml").write_text(yaml.safe_dump({
                "version": "anthropic_foundry_gateway_v1",
                "base_url": "https://example.invalid/anthropic",
                "api_key_env": "ANTHROPIC_FOUNDRY_API_KEY",
                "timeout_seconds": 60,
                "system_prompt": "JSON only",
                "pricing": {
                    "status": "unconfirmed", "currency": "USD",
                    "input_usd_per_token": 1e-6, "output_usd_per_token": 1e-6,
                    "cache_read_usd_per_token": None, "cache_write_usd_per_token": None,
                    "source": "test",
                },
            }))
            (root / "models.yaml").write_text(yaml.safe_dump({
                "version": "anthropic_foundry_model_registry_v1",
                "models": {"model_a": {"deployment_name": "dep-a", "request": {"max_tokens": 100}}},
            }))
            runtime = build_proposal_runtime(
                {
                    "kind": "anthropic_foundry", "config": "gateway.yaml",
                    "model_registry": "models.yaml", "model_key": "model_a",
                },
                root,
            )
        self.assertIsInstance(runtime, AnthropicFoundryRuntime)

    def test_dispatches_to_openrouter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gateway.yaml").write_text(yaml.safe_dump({
                "version": "openrouter_gateway_v1",
                "endpoint": "https://openrouter.ai/api/v1/chat/completions",
                "api_key_env": "OPENROUTER_API_KEY",
                "timeout_seconds": 60,
                "application_title": "test",
                "system_prompt": "JSON only",
            }))
            (root / "models.yaml").write_text(yaml.safe_dump({
                "version": "opro_model_registry_v1",
                "models": {
                    "model_a": {
                        "model_id": "vendor/model-a",
                        "request": {"temperature": 0.1},
                        "provider": {
                            "only": ["vendor"], "allow_fallbacks": False,
                            "require_parameters": True, "data_collection": "deny",
                        },
                    },
                },
            }))
            runtime = build_proposal_runtime(
                {
                    "kind": "openrouter", "config": "gateway.yaml",
                    "model_registry": "models.yaml", "model_key": "model_a",
                },
                root,
            )
        self.assertIsInstance(runtime, OpenRouterRuntime)

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            build_proposal_runtime({"kind": "made_up"}, Path("."))


if __name__ == "__main__":
    unittest.main()
