import tempfile
import unittest
from pathlib import Path

import yaml

from lob_sim.optimization.anthropic_foundry_runtime import (
    AnthropicFoundryResponseError,
    AnthropicFoundryRuntime,
    AnthropicFoundrySettings,
    AnthropicFoundryTransportError,
    FoundryHTTPResult,
    FoundryPricing,
    load_anthropic_foundry_settings,
)


def _settings(root: Path, **overrides) -> AnthropicFoundrySettings:
    defaults = dict(
        model_key="claude_sonnet5_foundry_v1",
        deployment_name="example-claude-deployment",
        base_url="https://example.invalid/anthropic",
        api_key_env="ANTHROPIC_FOUNDRY_API_KEY",
        timeout_seconds=300,
        system_prompt="JSON only",
        request_parameters={"max_tokens": 4096},
        output_effort=None,
        gateway_path=root / "gateway.yaml",
        registry_path=root / "models.yaml",
        gateway_sha256="gateway-hash",
        registry_sha256="registry-hash",
        pricing=FoundryPricing(
            status="unconfirmed", currency="USD",
            input_usd_per_token=2e-6, output_usd_per_token=1e-5,
            cache_read_usd_per_token=None, cache_write_usd_per_token=None,
            source="test rate card",
        ),
        retry_statuses=("rate_limited", "server_error", "connection_error"),
        max_attempts=1, initial_backoff_seconds=1.0, max_backoff_seconds=5.0,
    )
    defaults.update(overrides)
    return AnthropicFoundrySettings(**defaults)


def _message(text='{"ok":true}', **overrides) -> dict:
    payload = {
        "id": "msg_1",
        "model": "example-claude-deployment",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }
    payload.update(overrides)
    return payload


class AnthropicFoundryRuntimeTests(unittest.TestCase):
    def test_happy_path_builds_request_and_normalizes_usage(self):
        calls = []

        def fake_transport(request_params, timeout_seconds):
            calls.append((request_params, timeout_seconds))
            return FoundryHTTPResult("ok", 200, _message(), None, None)

        with tempfile.TemporaryDirectory() as directory:
            settings = _settings(Path(directory))
            runtime = AnthropicFoundryRuntime(settings, transport=fake_transport)
            result = runtime.complete("proposal", {"type": "object"})

        self.assertEqual(result.request_id, "msg_1")
        self.assertEqual(result.requested_model, "example-claude-deployment")
        self.assertEqual(result.resolved_model, "example-claude-deployment")
        self.assertEqual(result.provider, "azure_ai_foundry")
        self.assertEqual(result.text, '{"ok":true}')
        request_params, timeout = calls[0]
        self.assertEqual(timeout, 300)
        self.assertEqual(request_params["model"], "example-claude-deployment")
        self.assertEqual(
            request_params["output_config"]["format"],
            {"type": "json_schema", "schema": {"type": "object"}},
        )
        self.assertNotIn("tools", request_params)
        self.assertEqual(result.usage["input_tokens"], 100)
        self.assertEqual(result.usage["output_tokens"], 20)
        self.assertEqual(result.usage["reasoning_tokens"], 0)
        self.assertEqual(result.usage["total_tokens"], 120)
        self.assertAlmostEqual(result.usage["reported_cost_usd"], 100 * 2e-6 + 20 * 1e-5)
        self.assertEqual(result.usage["cost_status"], "calculated_unconfirmed_rate_card")

    def test_numeric_range_keywords_are_stripped_from_the_sent_schema(self):
        # Anthropic's structured-output API hard-rejects (HTTP 400) any
        # 'number'/'integer' schema carrying minimum/maximum -- confirmed
        # live, not merely undocumented. SearchSpace.validate() still
        # enforces the real bounds on the parsed candidate afterward.
        calls = []

        def fake_transport(request_params, timeout_seconds):
            calls.append(request_params)
            return FoundryHTTPResult("ok", 200, _message(), None, None)

        schema = {
            "type": "object",
            "properties": {
                "candidate": {
                    "type": "object",
                    "properties": {
                        "liquidity_activity": {
                            "type": "number", "minimum": 0.0, "maximum": 1.0,
                        },
                        "maker_base_depth": {
                            "type": "integer", "minimum": 1, "maximum": 100000,
                        },
                    },
                },
                "predictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"metric": {"type": "string", "enum": ["a", "b"]}},
                    },
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            settings = _settings(Path(directory))
            runtime = AnthropicFoundryRuntime(settings, transport=fake_transport)
            runtime.complete("proposal", schema)

        sent_schema = calls[0]["output_config"]["format"]["schema"]
        candidate_props = sent_schema["properties"]["candidate"]["properties"]
        self.assertNotIn("minimum", candidate_props["liquidity_activity"])
        self.assertNotIn("maximum", candidate_props["liquidity_activity"])
        self.assertNotIn("minimum", candidate_props["maker_base_depth"])
        self.assertNotIn("maximum", candidate_props["maker_base_depth"])
        # Type and enum information (structural, not a range constraint)
        # must survive sanitization.
        self.assertEqual(candidate_props["liquidity_activity"]["type"], "number")
        self.assertEqual(candidate_props["maker_base_depth"]["type"], "integer")
        self.assertEqual(
            sent_schema["properties"]["predictions"]["items"]["properties"]["metric"]["enum"],
            ["a", "b"],
        )

    def test_output_effort_is_merged_into_output_config_not_sent_top_level(self):
        # Confirmed live: this Foundry deployment rejects thinking.type
        # "enabled"+budget_tokens and requires thinking.type "adaptive" paired
        # with output_config.effort instead. output_effort must land inside
        # output_config, not as a bogus top-level messages.create() kwarg.
        calls = []

        def fake_transport(request_params, timeout_seconds):
            calls.append(request_params)
            return FoundryHTTPResult("ok", 200, _message(), None, None)

        with tempfile.TemporaryDirectory() as directory:
            settings = _settings(
                Path(directory),
                request_parameters={"max_tokens": 8192, "thinking": {"type": "adaptive"}},
                output_effort="medium",
            )
            runtime = AnthropicFoundryRuntime(settings, transport=fake_transport)
            runtime.complete("proposal", {"type": "object"})

        sent = calls[0]
        self.assertEqual(sent["output_config"]["effort"], "medium")
        self.assertEqual(sent["thinking"], {"type": "adaptive"})
        self.assertNotIn("effort", sent)
        self.assertNotIn("output_effort", sent)

    def test_reasoning_and_cache_tokens_are_normalized(self):
        message = _message(usage={
            "input_tokens": 50,
            "output_tokens": 10,
            "output_tokens_details": {"thinking_tokens": 7},
            "cache_read_input_tokens": 4,
            "cache_creation_input_tokens": 6,
        })

        def fake_transport(request_params, timeout_seconds):
            return FoundryHTTPResult("ok", 200, message, None, None)

        with tempfile.TemporaryDirectory() as directory:
            settings = _settings(Path(directory))
            runtime = AnthropicFoundryRuntime(settings, transport=fake_transport)
            result = runtime.complete("proposal", {"type": "object"})

        self.assertEqual(result.usage["reasoning_tokens"], 7)
        self.assertEqual(result.usage["cache_read_tokens"], 4)
        self.assertEqual(result.usage["cache_write_tokens"], 6)

    def test_missing_rate_reports_unknown_cost_not_zero(self):
        def fake_transport(request_params, timeout_seconds):
            return FoundryHTTPResult("ok", 200, _message(), None, None)

        with tempfile.TemporaryDirectory() as directory:
            pricing = FoundryPricing(
                status="unconfirmed", currency="USD",
                input_usd_per_token=None, output_usd_per_token=None,
                cache_read_usd_per_token=None, cache_write_usd_per_token=None,
                source="no rate card",
            )
            settings = _settings(Path(directory), pricing=pricing)
            runtime = AnthropicFoundryRuntime(settings, transport=fake_transport)
            result = runtime.complete("proposal", {"type": "object"})

        self.assertIsNone(result.usage["reported_cost_usd"])
        self.assertEqual(result.usage["cost_status"], "unknown_no_rate_card")

    def test_retries_rate_limit_and_retains_every_transport_attempt(self):
        sleeps = []
        responses = [
            FoundryHTTPResult("rate_limited", 429, None, "throttled", 3.0),
            FoundryHTTPResult("ok", 200, _message(), None, None),
        ]

        def fake_transport(request_params, timeout_seconds):
            return responses.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            settings = _settings(
                Path(directory), max_attempts=3, initial_backoff_seconds=1,
                max_backoff_seconds=5,
            )
            runtime = AnthropicFoundryRuntime(
                settings, transport=fake_transport, sleeper=sleeps.append,
            )
            result = runtime.complete("proposal", {"type": "object"})

        self.assertEqual(sleeps, [3.0])
        self.assertEqual(
            [attempt["status"] for attempt in result.transport_attempts],
            ["rate_limited", "ok"],
        )

    def test_terminal_transport_failure_raises_with_full_metadata(self):
        def fake_transport(request_params, timeout_seconds):
            return FoundryHTTPResult("server_error", 500, None, "upstream down", None)

        with tempfile.TemporaryDirectory() as directory:
            settings = _settings(Path(directory), max_attempts=2, initial_backoff_seconds=0)
            runtime = AnthropicFoundryRuntime(
                settings, transport=fake_transport, sleeper=lambda _seconds: None,
            )
            with self.assertRaises(AnthropicFoundryTransportError) as context:
                runtime.complete("proposal", {"type": "object"})

        metadata = context.exception.proposal_metadata
        self.assertEqual(metadata["model_runtime_identity"]["protocol"], "anthropic_foundry_messages_v1")
        self.assertEqual(len(metadata["transport_attempts"]), 2)
        self.assertIn("upstream down", str(context.exception))

    def test_empty_response_raises_response_error_with_usage_and_finish_reason(self):
        message = _message(content=[], stop_reason="max_tokens")

        def fake_transport(request_params, timeout_seconds):
            return FoundryHTTPResult("ok", 200, message, None, None)

        with tempfile.TemporaryDirectory() as directory:
            settings = _settings(Path(directory))
            runtime = AnthropicFoundryRuntime(settings, transport=fake_transport)
            with self.assertRaises(AnthropicFoundryResponseError) as context:
                runtime.complete("proposal", {"type": "object"})

        metadata = context.exception.proposal_metadata
        self.assertEqual(metadata["model_finish_reason"], "max_tokens")
        self.assertEqual(metadata["model_usage"]["input_tokens"], 100)

    def test_runtime_identity_is_credentials_free_and_hash_sensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings_a = _settings(root, gateway_sha256="hash-a")
            settings_b = _settings(root, gateway_sha256="hash-b")
            runtime_a = AnthropicFoundryRuntime(settings_a, transport=lambda *_a: None)
            runtime_b = AnthropicFoundryRuntime(settings_b, transport=lambda *_a: None)

        identity_a = runtime_a.runtime_identity
        identity_b = runtime_b.runtime_identity
        self.assertNotIn("api_key", identity_a)
        self.assertNotIn("ANTHROPIC_FOUNDRY_API_KEY", str(identity_a))
        self.assertNotEqual(identity_a["gateway_config_sha256"], identity_b["gateway_config_sha256"])

    def test_missing_credential_raises_before_any_transport_call(self):
        calls = []

        def fake_transport(request_params, timeout_seconds):
            calls.append(1)
            return FoundryHTTPResult("ok", 200, _message(), None, None)

        with tempfile.TemporaryDirectory() as directory:
            settings = _settings(Path(directory))
            # No explicit transport injected: forces the lazy real-transport
            # path, which must check the credential before importing/using
            # the anthropic SDK.
            runtime = AnthropicFoundryRuntime(settings, environment={})
            with self.assertRaises(RuntimeError) as context:
                runtime.complete("proposal", {"type": "object"})

        self.assertIn("ANTHROPIC_FOUNDRY_API_KEY", str(context.exception))
        self.assertEqual(calls, [])


class LoadAnthropicFoundrySettingsTests(unittest.TestCase):
    def _write_configs(self, root: Path, *, pricing_status: str = "unconfirmed") -> None:
        gateway = {
            "version": "anthropic_foundry_gateway_v1",
            "base_url": "https://example.invalid/anthropic",
            "api_key_env": "ANTHROPIC_FOUNDRY_API_KEY",
            "timeout_seconds": 300,
            "system_prompt": "JSON only",
            "pricing": {
                "status": pricing_status,
                "currency": "USD",
                "input_usd_per_token": 2e-6,
                "output_usd_per_token": 1e-5,
                "cache_read_usd_per_token": None,
                "cache_write_usd_per_token": None,
                "source": "test",
            },
        }
        registry = {
            "version": "anthropic_foundry_model_registry_v1",
            "models": {
                "model_a": {"deployment_name": "deployment-a", "request": {"max_tokens": 100}},
                "model_b": {"deployment_name": "deployment-b", "request": {"max_tokens": 200}},
            },
        }
        (root / "gateway.yaml").write_text(yaml.safe_dump(gateway))
        (root / "models.yaml").write_text(yaml.safe_dump(registry))

    def test_model_key_can_be_overridden_without_changing_gateway_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_configs(root)
            runtime_config = {
                "kind": "anthropic_foundry", "config": "gateway.yaml",
                "model_registry": "models.yaml", "model_key": "model_a",
            }
            first = load_anthropic_foundry_settings(runtime_config, root)
            second = load_anthropic_foundry_settings(
                runtime_config, root, model_key_override="model_b"
            )
        self.assertEqual(first.deployment_name, "deployment-a")
        self.assertEqual(second.deployment_name, "deployment-b")
        self.assertEqual(first.gateway_sha256, second.gateway_sha256)

    def test_endpoint_and_deployment_can_be_loaded_from_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_configs(root)
            gateway_path = root / "gateway.yaml"
            gateway = yaml.safe_load(gateway_path.read_text())
            gateway.pop("base_url")
            gateway["base_url_env"] = "TEST_FOUNDRY_BASE_URL"
            gateway_path.write_text(yaml.safe_dump(gateway))
            registry_path = root / "models.yaml"
            registry = yaml.safe_load(registry_path.read_text())
            registry["models"]["model_a"].pop("deployment_name")
            registry["models"]["model_a"]["deployment_name_env"] = (
                "TEST_FOUNDRY_DEPLOYMENT"
            )
            registry_path.write_text(yaml.safe_dump(registry))
            runtime_config = {
                "kind": "anthropic_foundry", "config": "gateway.yaml",
                "model_registry": "models.yaml", "model_key": "model_a",
            }
            settings = load_anthropic_foundry_settings(
                runtime_config,
                root,
                environment={
                    "TEST_FOUNDRY_BASE_URL": "https://public-example.invalid/anthropic",
                    "TEST_FOUNDRY_DEPLOYMENT": "public-example-deployment",
                },
            )
        self.assertEqual(settings.base_url, "https://public-example.invalid/anthropic")
        self.assertEqual(settings.deployment_name, "public-example-deployment")

    def test_missing_environment_value_names_the_required_variable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_configs(root)
            gateway_path = root / "gateway.yaml"
            gateway = yaml.safe_load(gateway_path.read_text())
            gateway.pop("base_url")
            gateway["base_url_env"] = "TEST_FOUNDRY_BASE_URL"
            gateway_path.write_text(yaml.safe_dump(gateway))
            runtime_config = {
                "kind": "anthropic_foundry", "config": "gateway.yaml",
                "model_registry": "models.yaml", "model_key": "model_a",
            }
            with self.assertRaisesRegex(ValueError, "TEST_FOUNDRY_BASE_URL"):
                load_anthropic_foundry_settings(
                    runtime_config, root, environment={}
                )

    def test_unconfirmed_pricing_status_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_configs(root, pricing_status="unconfirmed")
            runtime_config = {
                "kind": "anthropic_foundry", "config": "gateway.yaml",
                "model_registry": "models.yaml", "model_key": "model_a",
            }
            settings = load_anthropic_foundry_settings(runtime_config, root)
        self.assertEqual(settings.pricing.status, "unconfirmed")

    def test_output_effort_is_extracted_from_the_request_block(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_configs(root)
            registry_path = root / "models.yaml"
            registry = yaml.safe_load(registry_path.read_text())
            registry["models"]["model_a"]["request"]["output_effort"] = "medium"
            registry_path.write_text(yaml.safe_dump(registry))
            runtime_config = {
                "kind": "anthropic_foundry", "config": "gateway.yaml",
                "model_registry": "models.yaml", "model_key": "model_a",
            }
            settings = load_anthropic_foundry_settings(runtime_config, root)
        self.assertEqual(settings.output_effort, "medium")
        self.assertNotIn("output_effort", settings.request_parameters)

    def test_invalid_output_effort_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_configs(root)
            registry_path = root / "models.yaml"
            registry = yaml.safe_load(registry_path.read_text())
            registry["models"]["model_a"]["request"]["output_effort"] = "extreme"
            registry_path.write_text(yaml.safe_dump(registry))
            runtime_config = {
                "kind": "anthropic_foundry", "config": "gateway.yaml",
                "model_registry": "models.yaml", "model_key": "model_a",
            }
            with self.assertRaisesRegex(ValueError, "output_effort must be one of"):
                load_anthropic_foundry_settings(runtime_config, root)

    def test_wrong_kind_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_configs(root)
            runtime_config = {
                "kind": "openrouter", "config": "gateway.yaml",
                "model_registry": "models.yaml", "model_key": "model_a",
            }
            with self.assertRaises(ValueError):
                load_anthropic_foundry_settings(runtime_config, root)


if __name__ == "__main__":
    unittest.main()
