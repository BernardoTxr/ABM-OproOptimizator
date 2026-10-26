import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from workflows.benchmark.preflight_openrouter import (
    proposal_preflight_contract,
    validate_openrouter_preflight,
)


ROOT = Path(__file__).resolve().parents[1]


class OpenRouterPreflightTests(unittest.TestCase):
    def test_checked_in_contract_is_strict(self):
        config = yaml.safe_load((ROOT / "configs/paper/optimizer.yaml").read_text())
        _, prompt, _, schema = proposal_preflight_contract(config)
        prompt_payload = json.loads(prompt)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            schema["properties"]["candidate"]["required"],
            [row["name"] for row in config["search_space"]],
        )
        self.assertEqual(len(prompt_payload["observations"]), 49)
        self.assertEqual(prompt_payload["remaining_proposal_budget"], 1)
        self.assertTrue(all(
            set(row["component_losses"]) == {
                "returns", "spread", "imbalance", "trade_size",
                "inter_arrival", "depth",
            }
            for row in prompt_payload["observations"]
        ))
        heuristic = yaml.safe_load(
            (ROOT / "configs/paper/heuristics.yaml").read_text()
        )
        _, heuristic_prompt, _, heuristic_schema = proposal_preflight_contract(
            config, heuristic
        )
        heuristic_payload = json.loads(heuristic_prompt)
        self.assertTrue(heuristic_payload["heuristic_condition"])
        self.assertIsNotNone(heuristic_payload["heuristics"])
        self.assertGreater(len(heuristic_prompt), len(prompt))
        self.assertEqual(heuristic_schema, schema)

    def test_validated_preflight_locks_model_and_config_files(self):
        config_path = ROOT / "configs/paper/optimizer.yaml"
        config = yaml.safe_load(config_path.read_text())
        llm = config["llm"]
        gateway = config_path.parent / llm["gateway_config"]
        registry = config_path.parent / llm["model_registry"]
        heuristic_path = config_path.parent / llm["heuristic_config"]
        heuristic = yaml.safe_load(heuristic_path.read_text())
        _, _, _, schema = proposal_preflight_contract(config)
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        schema_hash = hashlib.sha256(
            json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        record = {
            "runtime": "openrouter_chat_completions_v1",
            "config_sha256": digest(config_path),
            "gateway_config_sha256": digest(gateway),
            "model_registry_sha256": digest(registry),
            "model_key": "openai_gpt_5_4",
            "conditions": {},
            "schema": schema,
            "schema_sha256": schema_hash,
            "preflight_history": "synthetic_maximum_adaptive_history",
            "preflight_observations": 49,
            "optimizer_slots_consumed": 0,
            "simulator_evaluations": 0,
            "test_books_accessed": False,
        }
        for name, spec in (("plain", None), ("heuristic", heuristic)):
            _, prompt, prompt_hash, _ = proposal_preflight_contract(config, spec)
            record["conditions"][name] = {
                "heuristic_condition": spec is not None,
                "requested_model": "openai/gpt-5.4",
                "resolved_model": "openai/gpt-5.4",
                "provider": "OpenAI",
                "finish_reason": "stop",
                "model_request_id": f"gen_{name}",
                "model_request_sha256": f"request-hash-{name}",
                "model_response_sha256": f"response-hash-{name}",
                "transport_attempts": [{"attempt": 1, "status": 200}],
                "prompt": prompt,
                "prompt_sha256": prompt_hash,
                "parsed_proposal": {},
            }
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "preflight.json"
            artifact.write_text(json.dumps(record))
            checks = validate_openrouter_preflight(
                artifact, config_path, config,
                model_key_override="openai_gpt_5_4",
            )
            self.assertTrue(all(checks.values()))
            record["model_key"] = "different"
            artifact.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError, "model_key"):
                validate_openrouter_preflight(artifact, config_path, config)


if __name__ == "__main__":
    unittest.main()
