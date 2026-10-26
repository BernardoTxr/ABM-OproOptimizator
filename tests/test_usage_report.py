import json
import tempfile
import unittest
from pathlib import Path

from workflows.benchmark.usage_report import summarize


def _write_trial(
    study_dir: Path, ticker: str, method: str, repetition: int, index: int,
    *, protocol: str, model_key: str, requested: str, resolved: str,
    provider: str, cost_source: str | None,
) -> None:
    trial_dir = (
        study_dir / ticker / "calibration" / "policies"
        / f"{method}_repeat{repetition:02d}" / "trials"
    )
    trial_dir.mkdir(parents=True, exist_ok=True)
    usage = {
        "input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 0, "total_tokens": 15,
        "reported_cost_usd": 0.001,
    }
    if cost_source is not None:
        usage["cost_source"] = cost_source
    record = {
        "policy_trial": index,
        "proposal_metadata": {
            "model_runtime_identity": {"protocol": protocol},
            "model_key": model_key,
            "model_requested": requested,
            "model_resolved": resolved,
            "model_provider": provider,
            "model_request_id": f"{method}-{repetition}-{index}",
            "model_usage": usage,
        },
    }
    (trial_dir / f"trial_{index:04d}.json").write_text(json.dumps(record))


class UsageReportTests(unittest.TestCase):
    def test_legacy_openrouter_shaped_records_report_without_cost_source(self):
        with tempfile.TemporaryDirectory() as directory:
            study_dir = Path(directory)
            for method in ("opro_plain", "opro_heuristic"):
                _write_trial(
                    study_dir, "ASAI3", method, 0, 0,
                    protocol="openrouter_chat_completions_v1",
                    model_key="qwen_qwen3_8_flash_8192_routed",
                    requested="qwen/qwen3.8-flash", resolved="qwen/qwen3.8-flash",
                    provider="alibaba", cost_source=None,
                )
            report = summarize(study_dir)
        self.assertEqual(report["runtime"], "openrouter_chat_completions_v1")
        self.assertIn("not recorded per-completion", report["cost_source"])

    def test_anthropic_foundry_shaped_records_report_their_own_protocol_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            study_dir = Path(directory)
            for method in ("opro_plain", "opro_heuristic"):
                _write_trial(
                    study_dir, "ASAI3", method, 0, 0,
                    protocol="anthropic_foundry_messages_v1",
                    model_key="claude_sonnet5_foundry_v1",
                    requested="example-claude-deployment",
                    resolved="example-claude-deployment",
                    provider="azure_ai_foundry",
                    cost_source="Anthropic's public Claude Sonnet 5 rate card",
                )
            report = summarize(study_dir)
        self.assertEqual(report["runtime"], "anthropic_foundry_messages_v1")
        self.assertIn("Anthropic's public Claude Sonnet 5 rate card", report["cost_source"])
        self.assertNotIn("OpenRouter", report["runtime"])
        self.assertNotIn("OpenRouter", report["cost_source"])

    def test_mixed_protocols_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            study_dir = Path(directory)
            _write_trial(
                study_dir, "ASAI3", "opro_plain", 0, 0,
                protocol="openrouter_chat_completions_v1",
                model_key="k", requested="r", resolved="r",
                provider="p", cost_source=None,
            )
            _write_trial(
                study_dir, "ASAI3", "opro_heuristic", 0, 0,
                protocol="anthropic_foundry_messages_v1",
                model_key="k", requested="r", resolved="r",
                provider="p", cost_source=None,
            )
            with self.assertRaisesRegex(RuntimeError, "one proposal-runtime protocol"):
                summarize(study_dir)


if __name__ == "__main__":
    unittest.main()
