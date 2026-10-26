import json
import tempfile
import unittest
from pathlib import Path

import yaml

from lob_sim.optimization.opro import (
    build_opro_prompt,
    opro_response_json_schema,
    parse_opro_response,
)
from lob_sim.optimization.model_runtime import ModelCompletion
from lob_sim.optimization.openrouter_runtime import (
    HTTPResponse,
    OpenRouterResponseError,
    OpenRouterRuntime,
    OpenRouterSettings,
    OpenRouterTransportError,
    load_openrouter_settings,
)
from lob_sim.optimization.samplers import OPROProposalError
from lob_sim.optimization.space import SearchSpace
from lob_sim.optimization.opro_policy import (
    OPROPolicy,
    build_transport_prompt,
    normalize_transport_response,
    transport_response_json_schema,
)


def completion(text: str, request_id: str = "gen_test") -> ModelCompletion:
    return ModelCompletion(
        text=text,
        request_id=request_id,
        requested_model="openai/gpt-5.4",
        resolved_model="openai/gpt-5.4",
        provider="OpenAI",
        finish_reason="stop",
        usage={"total_tokens": 7},
        request_sha256="request-hash",
        response_sha256="response-hash",
        raw_response={"id": request_id},
    )


class FakeRuntime:
    model_key = "openai_gpt_5_4"
    runtime_identity = {
        "protocol": "openrouter_chat_completions_v1",
        "model_key": model_key,
        "requested_model": "openai/gpt-5.4",
        "gateway_config_sha256": "gateway-hash",
        "model_registry_sha256": "registry-hash",
    }


class OPROProtocolTests(unittest.TestCase):
    def setUp(self):
        self.space = SearchSpace.from_config([{
            "name": "x", "type": "float", "low": 0.0, "high": 1.0, "step": 0.1,
            "targets": [{"agent": "a", "field": "parameters.x"}],
        }])

    @staticmethod
    def _feedback(value):
        names = (
            "returns", "spread", "imbalance", "trade_size",
            "inter_arrival", "depth",
        )
        return {
            "feasible": True,
            "component_order": list(names),
            "component_losses": {name: value for name in names},
            "final_loss": value,
        }

    def test_prompt_ablation_changes_only_heuristic_condition(self):
        history = [{
            "policy_trial": 0, "candidate": {"x": 0.5}, "objective": 0.2,
            "status": "complete", "optimizer_feedback": self._feedback(0.2),
        }]
        plain, plain_hash = build_opro_prompt(self.space, history, 4, None)
        informed, informed_hash = build_opro_prompt(
            self.space, history, 4, {"version": "rules_v1", "rules": []}
        )
        self.assertNotEqual(plain_hash, informed_hash)
        plain_payload, informed_payload = json.loads(plain), json.loads(informed)
        self.assertEqual(plain_payload["observations"], informed_payload["observations"])
        self.assertFalse(plain_payload["heuristic_condition"])
        self.assertTrue(informed_payload["heuristic_condition"])

    def test_response_must_be_bounded_and_cite_available_trials(self):
        response = json.dumps({
            "hypothesis": "Reduce x to lower spread.",
            "evidence_trial_ids": [0],
            "candidate": {"x": 0.4},
            "predictions": [{
                "metric": "spread", "direction": "decrease", "rationale": "test"
            }],
        })
        proposal = parse_opro_response(response, self.space, "abc", {0})
        self.assertEqual(proposal.candidate, {"x": 0.4})
        with self.assertRaises(ValueError):
            parse_opro_response(response.replace('0.4', '1.4'), self.space, "abc", {0})

    def test_json_schema_is_closed_and_matches_candidate_keys(self):
        schema = opro_response_json_schema(self.space)
        candidate = schema["properties"]["candidate"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(candidate["required"], ["x"])
        self.assertEqual(candidate["properties"]["x"]["minimum"], 0.0)

    def test_non_scalar_category_uses_reversible_provider_token(self):
        space = SearchSpace.from_config([{
            "name": "ladder", "type": "categorical",
            "choices": [[100.0, 200.0], [300.0, 400.0]],
            "targets": [{"agent": "a", "field": "parameters.ladder"}],
        }])
        schema = transport_response_json_schema(space)
        self.assertEqual(
            schema["properties"]["candidate"]["properties"]["ladder"],
            {"type": "string", "enum": ["choice_0", "choice_1"]},
        )
        prompt, _, mapping = build_transport_prompt(space, [], 3, None)
        self.assertEqual(mapping["ladder"][1]["value"], [300.0, 400.0])
        self.assertIn("transport_categorical_encoding", json.loads(prompt))
        raw = json.dumps({
            "hypothesis": "Use the deeper ladder.",
            "evidence_trial_ids": [],
            "candidate": {"ladder": "choice_1"},
            "predictions": [{
                "metric": "depth", "direction": "increase", "rationale": "test"
            }],
        })
        normalized = normalize_transport_response(raw, space)
        self.assertEqual(
            json.loads(normalized)["candidate"]["ladder"], [300.0, 400.0]
        )

        class FakeClient(FakeRuntime):
            def complete(self, prompt, response_schema):
                return completion(raw, "gen-token")

        proposal = OPROPolicy(space, FakeClient(), None).ask(
            0, history=[], remaining_budget=3
        )
        self.assertEqual(proposal.candidate["ladder"], [300.0, 400.0])
        self.assertEqual(proposal.metadata["raw_response"], raw)

    def test_opro_policy_uses_history_and_persists_model_metadata(self):
        response = json.dumps({
            "hypothesis": "Reduce x after trial zero.",
            "evidence_trial_ids": [0],
            "candidate": {"x": 0.4},
            "predictions": [{
                "metric": "spread", "direction": "decrease", "rationale": "test"
            }],
        })

        class FakeClient(FakeRuntime):
            def complete(self, prompt, response_schema):
                self.prompt = json.loads(prompt)
                self.schema = response_schema
                return completion(response)

        client = FakeClient()
        policy = OPROPolicy(self.space, client, {"version": "rules_v1"})
        history = [{
            "policy_trial": 0, "candidate": {"x": 0.5}, "objective": 0.2,
            "status": "complete", "optimizer_feedback": self._feedback(0.2),
        }]
        proposal = policy.ask(1, history=history, remaining_budget=3)
        self.assertEqual(proposal.candidate, {"x": 0.4})
        self.assertEqual(client.prompt["remaining_proposal_budget"], 3)
        self.assertTrue(client.prompt["heuristic_condition"])
        self.assertEqual(proposal.metadata["model_request_id"], "gen_test")
        self.assertEqual(proposal.metadata["model_key"], "openai_gpt_5_4")

    def test_prompt_exposes_full_numeric_history_but_not_residuals_or_reasoning(self):
        history = [
            {
                "policy_trial": 0, "candidate": {"x": 0.5}, "objective": 0.4,
                "status": "complete", "optimizer_feedback": self._feedback(0.4),
                "evaluation": {"signed_diagnostics": {"return_mean": -2.0}},
            },
            {
                "policy_trial": 1, "candidate": {"x": 0.4}, "objective": 0.2,
                "status": "complete", "duplicate": False,
                "proposal_metadata": {
                    "hypothesis": "Lower x lowers spread.",
                    "evidence_trial_ids": [0],
                    "predictions": [{
                        "metric": "spread", "direction": "decrease",
                        "rationale": "mechanism",
                    }],
                },
                "optimizer_feedback": self._feedback(0.2),
                "evaluation": {"signed_diagnostics": {"return_mean": 1.0}},
            },
        ]
        prompt, _ = build_opro_prompt(self.space, history, 2, None)
        payload = json.loads(prompt)
        self.assertEqual(len(payload["observations"]), 2)
        observation = payload["observations"][1]
        self.assertEqual(observation["final_loss"], 0.2)
        self.assertEqual(len(observation["component_losses"]), 6)
        self.assertNotIn("hypothesis", observation)
        self.assertNotIn("prediction_outcomes", observation)
        self.assertNotIn("signed_diagnostics", prompt)

    def test_openrouter_runtime_uses_strict_schema_and_frozen_model_profile(self):
        calls = []
        raw_response = {
            "id": "gen_1",
            "model": "openai/gpt-5.4",
            "provider": "OpenAI",
            "choices": [{
                "message": {"content": '{"ok":true}'}, "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13,
                "completion_tokens_details": {"reasoning_tokens": 2},
                "prompt_tokens_details": {"cached_tokens": 4}, "cost": 0.02,
            },
        }

        def fake_transport(url, headers, body, timeout):
            calls.append((url, headers, json.loads(body), timeout))
            return HTTPResponse(200, json.dumps(raw_response).encode(), {})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = OpenRouterSettings(
                model_key="openai_gpt_5_4", model_id="openai/gpt-5.4",
                endpoint="https://openrouter.ai/api/v1/chat/completions",
                api_key_env="OPENROUTER_API_KEY", timeout_seconds=300,
                application_title="test", system_prompt="JSON only",
                request_parameters={"temperature": 0.2},
                provider_preferences={
                    "only": ["openai"], "allow_fallbacks": False,
                    "require_parameters": True, "data_collection": "deny",
                },
                gateway_path=root / "gateway.yaml", registry_path=root / "models.yaml",
                gateway_sha256="gateway-hash", registry_sha256="registry-hash",
            )
            runtime = OpenRouterRuntime(
                settings, transport=fake_transport,
                environment={"OPENROUTER_API_KEY": "secret"},
            )
            result = runtime.complete("proposal", {"type": "object"})
        self.assertEqual(result.request_id, "gen_1")
        self.assertEqual(result.usage["total_tokens"], 13)
        _, headers, payload, timeout = calls[0]
        self.assertEqual(headers["Authorization"], "Bearer secret")
        self.assertEqual(timeout, 300)
        self.assertEqual(payload["model"], "openai/gpt-5.4")
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertFalse(payload["provider"]["allow_fallbacks"])

    def test_openrouter_retries_429_and_retains_every_transport_attempt(self):
        sleeps = []
        raw_response = {
            "id": "gen_retry",
            "model": "qwen/qwen3.8-flash",
            "provider": "Alibaba",
            "choices": [{
                "message": {"content": '{"ok":true}'}, "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13,
            },
        }
        responses = [
            HTTPResponse(
                429,
                b'{"error":{"code":429,"message":"provider throttled"}}',
                {"Retry-After": "3"},
            ),
            HTTPResponse(200, json.dumps(raw_response).encode(), {}),
        ]

        def fake_transport(url, headers, body, timeout):
            return responses.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = OpenRouterSettings(
                model_key="qwen_routed", model_id="qwen/qwen3.8-flash",
                endpoint="https://openrouter.ai/api/v1/chat/completions",
                api_key_env="OPENROUTER_API_KEY", timeout_seconds=300,
                application_title="test", system_prompt="JSON only",
                request_parameters={"temperature": 0.2},
                provider_preferences={
                    "order": ["makora", "alibaba"],
                    "only": ["makora", "alibaba"],
                    "allow_fallbacks": True,
                    "require_parameters": True,
                    "data_collection": "deny",
                },
                gateway_path=root / "gateway.yaml",
                registry_path=root / "models.yaml",
                gateway_sha256="gateway-hash", registry_sha256="registry-hash",
                retry_http_statuses=(429, 500, 502, 503, 504),
                max_attempts=3, initial_backoff_seconds=1,
                max_backoff_seconds=5,
            )
            runtime = OpenRouterRuntime(
                settings,
                transport=fake_transport,
                environment={"OPENROUTER_API_KEY": "secret"},
                sleeper=sleeps.append,
            )
            result = runtime.complete("proposal", {"type": "object"})

        self.assertEqual(sleeps, [3.0])
        self.assertEqual(result.provider, "Alibaba")
        self.assertEqual(
            [attempt["status"] for attempt in result.transport_attempts],
            [429, 200],
        )
        self.assertEqual(result.transport_attempts[0]["backoff_seconds"], 3.0)

    def test_empty_completion_retains_finish_reason_usage_and_request_identity(self):
        raw_response = {
            "id": "gen_empty",
            "model": "qwen/qwen3.8-flash",
            "provider": "Makora",
            "choices": [{
                "message": {"content": "", "reasoning": "thinking"},
                "finish_reason": "length",
            }],
            "usage": {
                "prompt_tokens": 9400,
                "completion_tokens": 4096,
                "total_tokens": 13496,
                "completion_tokens_details": {"reasoning_tokens": 4096},
            },
        }

        def fake_transport(url, headers, body, timeout):
            return HTTPResponse(200, json.dumps(raw_response).encode(), {})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = OpenRouterSettings(
                model_key="qwen_routed", model_id="qwen/qwen3.8-flash",
                endpoint="https://openrouter.ai/api/v1/chat/completions",
                api_key_env="OPENROUTER_API_KEY", timeout_seconds=300,
                application_title="test", system_prompt="JSON only",
                request_parameters={"max_tokens": 8192},
                provider_preferences={
                    "order": ["makora", "alibaba"],
                    "only": ["makora", "alibaba"],
                    "allow_fallbacks": True,
                    "require_parameters": True,
                    "data_collection": "deny",
                },
                gateway_path=root / "gateway.yaml",
                registry_path=root / "models.yaml",
                gateway_sha256="gateway-hash", registry_sha256="registry-hash",
            )
            runtime = OpenRouterRuntime(
                settings,
                transport=fake_transport,
                environment={"OPENROUTER_API_KEY": "secret"},
            )
            with self.assertRaises(OpenRouterResponseError) as context:
                runtime.complete("proposal", {"type": "object"})

        metadata = context.exception.proposal_metadata
        self.assertEqual(metadata["model_request_id"], "gen_empty")
        self.assertEqual(metadata["model_finish_reason"], "length")
        self.assertEqual(metadata["model_usage"]["reasoning_tokens"], 4096)
        self.assertEqual(metadata["transport_attempts"][0]["status"], 200)

    def test_terminal_provider_error_exposes_upstream_diagnostic(self):
        error = {
            "error": {
                "code": 400,
                "message": "Provider returned error",
                "metadata": {
                    "provider_name": "Alibaba",
                    "raw": {"message": "unsupported JSON schema keyword"},
                },
            }
        }

        def fake_transport(url, headers, body, timeout):
            return HTTPResponse(400, json.dumps(error).encode(), {})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = OpenRouterSettings(
                model_key="qwen_routed", model_id="qwen/qwen3.8-flash",
                endpoint="https://openrouter.ai/api/v1/chat/completions",
                api_key_env="OPENROUTER_API_KEY", timeout_seconds=300,
                application_title="test", system_prompt="JSON only",
                request_parameters={"max_tokens": 8192},
                provider_preferences={
                    "order": ["makora", "alibaba"],
                    "only": ["makora", "alibaba"],
                    "allow_fallbacks": True,
                    "require_parameters": True,
                    "data_collection": "deny",
                },
                gateway_path=root / "gateway.yaml",
                registry_path=root / "models.yaml",
                gateway_sha256="gateway-hash", registry_sha256="registry-hash",
            )
            runtime = OpenRouterRuntime(
                settings,
                transport=fake_transport,
                environment={"OPENROUTER_API_KEY": "secret"},
            )
            with self.assertRaises(OpenRouterTransportError) as context:
                runtime.complete("proposal", {"type": "object"})

        message = str(context.exception)
        self.assertIn("provider=Alibaba", message)
        self.assertIn("unsupported JSON schema keyword", message)
        attempts = context.exception.proposal_metadata["transport_attempts"]
        self.assertIn("unsupported JSON schema keyword", attempts[0]["error_detail"])

    def test_model_profile_can_change_without_changing_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gateway = {
                "version": "openrouter_gateway_v1",
                "endpoint": "https://openrouter.ai/api/v1/chat/completions",
                "api_key_env": "OPENROUTER_API_KEY",
                "timeout_seconds": 30,
                "application_title": "test",
                "system_prompt": "JSON only",
            }
            registry = {
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
                    "model_b": {
                        "model_id": "other/model-b",
                        "request": {"temperature": 0.3},
                        "provider": {
                            "only": ["other"], "allow_fallbacks": False,
                            "require_parameters": True, "data_collection": "deny",
                        },
                    },
                },
            }
            (root / "gateway.yaml").write_text(yaml.safe_dump(gateway))
            (root / "models.yaml").write_text(yaml.safe_dump(registry))
            llm = {
                "runtime": "openrouter", "gateway_config": "gateway.yaml",
                "model_registry": "models.yaml", "model_key": "model_a",
            }
            first = load_openrouter_settings(llm, root)
            second = load_openrouter_settings(llm, root, model_key_override="model_b")
        self.assertEqual(first.model_id, "vendor/model-a")
        self.assertEqual(second.model_id, "other/model-b")
        self.assertEqual(first.gateway_sha256, second.gateway_sha256)

    def test_malformed_model_output_retains_audit_metadata(self):
        class FakeClient(FakeRuntime):
            def complete(self, prompt, response_schema):
                return completion("{}", "gen_bad")

        policy = OPROPolicy(self.space, FakeClient(), None)
        with self.assertRaises(OPROProposalError) as context:
            policy.ask(1, history=[{
                "policy_trial": 0, "candidate": {"x": 0.5}, "objective": 0.2,
                "status": "complete", "optimizer_feedback": self._feedback(0.2),
            }], remaining_budget=2)
        self.assertEqual(
            context.exception.proposal_metadata["model_request_id"],
            "gen_bad",
        )


if __name__ == "__main__":
    unittest.main()
