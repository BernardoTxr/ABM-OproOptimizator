import unittest

from workflows.benchmark.runtime_audit import model_identity_matches


def _metadata(**overrides):
    payload = {
        "model_runtime_identity": {
            "protocol": "openrouter_chat_completions_v1",
            "model_key": "qwen_qwen3_8_flash_8192_routed",
            "requested_model": "qwen/qwen3.8-flash",
            "gateway_config_sha256": "gateway-hash",
            "model_registry_sha256": "registry-hash",
        },
        "model_key": "qwen_qwen3_8_flash_8192_routed",
        "model_requested": "qwen/qwen3.8-flash",
        "model_resolved": "qwen/qwen3.8-flash",
        "model_provider": "Alibaba",
        "model_request_id": "gen_1",
        "model_request_sha256": "req-hash",
        "model_response_sha256": "resp-hash",
        "transport_attempts": [{"status": 200}],
    }
    payload.update(overrides)
    return payload


_EXPECTED_IDENTITY = {
    "protocol": "openrouter_chat_completions_v1",
    "model_key": "qwen_qwen3_8_flash_8192_routed",
    "requested_model": "qwen/qwen3.8-flash",
    "gateway_config_sha256": "gateway-hash",
    "model_registry_sha256": "registry-hash",
}


class ModelIdentityMatchesTests(unittest.TestCase):
    def test_matching_record_passes(self):
        self.assertTrue(
            model_identity_matches(
                _metadata(), _EXPECTED_IDENTITY, {"makora", "alibaba"},
                expected_resolved_model="qwen/qwen3.8-flash",
            )
        )

    def test_gateway_hash_drift_fails(self):
        metadata = _metadata()
        metadata["model_runtime_identity"] = {
            **metadata["model_runtime_identity"], "gateway_config_sha256": "different",
        }
        self.assertFalse(
            model_identity_matches(metadata, _EXPECTED_IDENTITY, {"alibaba"})
        )

    def test_provider_not_in_acceptable_set_fails(self):
        metadata = _metadata(model_provider="Unknown")
        self.assertFalse(
            model_identity_matches(metadata, _EXPECTED_IDENTITY, {"alibaba", "makora"})
        )

    def test_resolved_model_mismatch_fails(self):
        metadata = _metadata(model_resolved="other/model")
        self.assertFalse(
            model_identity_matches(
                metadata, _EXPECTED_IDENTITY, {"alibaba"},
                expected_resolved_model="qwen/qwen3.8-flash",
            )
        )

    def test_missing_transport_attempts_fails(self):
        metadata = _metadata(transport_attempts=[])
        self.assertFalse(
            model_identity_matches(metadata, _EXPECTED_IDENTITY, {"alibaba"})
        )

    def test_non_2xx_last_attempt_fails(self):
        metadata = _metadata(transport_attempts=[{"status": 200}, {"status": 500}])
        self.assertFalse(
            model_identity_matches(metadata, _EXPECTED_IDENTITY, {"alibaba"})
        )

    def test_works_for_a_non_openrouter_shaped_identity(self):
        expected = {
            "protocol": "anthropic_foundry_messages_v1",
            "model_key": "claude_sonnet5_foundry_v1",
            "requested_model": "example-claude-deployment",
            "base_url": "https://example.invalid/anthropic",
            "gateway_config_sha256": "gw-hash",
            "model_registry_sha256": "reg-hash",
            "pricing_status": "unconfirmed",
        }
        metadata = {
            "model_runtime_identity": dict(expected),
            "model_key": "claude_sonnet5_foundry_v1",
            "model_requested": "example-claude-deployment",
            "model_resolved": "example-claude-deployment",
            "model_provider": "azure_ai_foundry",
            "model_request_id": "msg_1",
            "model_request_sha256": "req-hash",
            "model_response_sha256": "resp-hash",
            # Real Foundry transport_attempts rows use a semantic string
            # status ("ok"), not an HTTP int like OpenRouter's -- this must
            # not be coerced with int(status).
            "transport_attempts": [
                {"attempt": 1, "status": "rate_limited", "status_code": 429},
                {"attempt": 2, "status": "ok", "status_code": 200},
            ],
        }
        self.assertTrue(
            model_identity_matches(metadata, expected, {"azure_ai_foundry"})
        )

    def test_foundry_shaped_failed_last_attempt_fails(self):
        metadata = _metadata(
            transport_attempts=[{"attempt": 1, "status": "server_error", "status_code": 500}]
        )
        self.assertFalse(
            model_identity_matches(metadata, _EXPECTED_IDENTITY, {"alibaba"})
        )


if __name__ == "__main__":
    unittest.main()
