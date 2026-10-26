import json
import unittest
from pathlib import Path

import yaml

from lob_sim.optimization.anthropic_foundry_runtime import (
    AnthropicFoundryRuntime,
    FoundryHTTPResult,
    load_anthropic_foundry_settings,
)
from lob_sim.optimization.feedback import SMM_COMPONENTS
from workflows.benchmark.preflight_proposal_runtime import (
    run_preflight,
    validate_proposal_preflight,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_CONFIG_PATH = REPO_ROOT / "configs" / "paper" / "optimizer.yaml"
FOLLOWUP_CONFIG_DIR = REPO_ROOT / "configs" / "experiments" / "followup_foundry_sonnet5"


def _fake_message(candidate: dict) -> dict:
    payload = {
        "hypothesis": "Preflight smoke response",
        "evidence_trial_ids": [0],
        "candidate": candidate,
        "predictions": [{
            "metric": SMM_COMPONENTS[0], "direction": "decrease",
            "rationale": "Exercises the strict preflight parser.",
        }],
    }
    return {
        "id": "msg_preflight",
        "model": "example-claude-deployment",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": json.dumps(payload)}],
        "usage": {"input_tokens": 500, "output_tokens": 50},
    }


class PreflightProposalRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(PAPER_CONFIG_PATH.read_text())
        self.candidate = self.config["initial_design"][0]
        self.settings = load_anthropic_foundry_settings(
            {
                "kind": "anthropic_foundry", "config": "anthropic_foundry.yaml",
                "model_registry": "models.yaml", "model_key": "claude_sonnet5_foundry_v1",
            },
            FOLLOWUP_CONFIG_DIR,
            environment={
                "ANTHROPIC_FOUNDRY_BASE_URL": "https://example.invalid/anthropic",
                "ANTHROPIC_FOUNDRY_DEPLOYMENT": "example-claude-deployment",
            },
        )

    def _runtime(self):
        return AnthropicFoundryRuntime(
            self.settings,
            transport=lambda request_params, timeout: FoundryHTTPResult(
                "ok", 200, _fake_message(self.candidate), None, None
            ),
        )

    def test_run_preflight_then_validate_round_trips(self):
        runtime = self._runtime()
        record = run_preflight(runtime, PAPER_CONFIG_PATH, self.config)
        self.assertEqual(record["optimizer_slots_consumed"], 0)
        self.assertEqual(record["simulator_evaluations"], 0)
        self.assertIs(record["test_books_accessed"], False)
        self.assertEqual(set(record["conditions"]), {"plain", "heuristic"})
        self.assertEqual(
            record["preflight_observations"], int(self.config["trial_budget"]) - 1
        )

        checks = validate_proposal_preflight(
            _write_record(record), runtime, PAPER_CONFIG_PATH, self.config
        )
        self.assertTrue(all(checks.values()), checks)

    def test_missing_artifact_raises(self):
        runtime = self._runtime()
        with self.assertRaisesRegex(RuntimeError, "preflight_proposal_runtime"):
            validate_proposal_preflight(
                Path("/nonexistent/preflight.json"), runtime, PAPER_CONFIG_PATH, self.config
            )

    def test_stale_runtime_identity_is_rejected(self):
        runtime = self._runtime()
        record = run_preflight(runtime, PAPER_CONFIG_PATH, self.config)
        record["runtime_identity"] = {**record["runtime_identity"], "model_key": "different"}
        with self.assertRaisesRegex(RuntimeError, "stale or invalid"):
            validate_proposal_preflight(
                _write_record(record), runtime, PAPER_CONFIG_PATH, self.config
            )


def _write_record(record: dict) -> Path:
    import tempfile
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    json.dump(record, handle)
    handle.close()
    return Path(handle.name)


if __name__ == "__main__":
    unittest.main()
