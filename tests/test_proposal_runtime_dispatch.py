"""study.py::_policy dispatch between the frozen v9 path and the new
provider-neutral proposal_runtime override, using the real committed config
files (no BTG data or network access required for either path)."""
import unittest
from unittest.mock import patch
from pathlib import Path

from lob_sim.optimization import study
from lob_sim.optimization.anthropic_foundry_runtime import AnthropicFoundryRuntime
from lob_sim.optimization.openrouter_runtime import OpenRouterRuntime

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPER_CONFIG_DIR = REPO_ROOT / "configs" / "paper"
FOLLOWUP_CONFIG_DIR = REPO_ROOT / "configs" / "experiments" / "followup_foundry_sonnet5"


class _Space:
    pass


class _Evaluator:
    def __init__(self, config, config_dir):
        self.config = config
        self.config_dir = config_dir
        self.space = _Space()


class ProposalRuntimeDispatchTests(unittest.TestCase):
    def test_v9_shaped_config_still_builds_openrouter_runtime(self):
        evaluator = _Evaluator(
            {
                "llm": {
                    "runtime": "openrouter",
                    "gateway_config": "openrouter.yaml",
                    "model_registry": "llm-models.yaml",
                    "model_key": "qwen_qwen3_8_flash_8192_routed",
                    "heuristic_config": "heuristics.yaml",
                },
            },
            PAPER_CONFIG_DIR,
        )
        policy = study._policy("opro_plain", evaluator, seed=1, run_dir=Path("."), initial_design=[])
        self.assertIsInstance(policy.runtime, OpenRouterRuntime)
        self.assertIsNone(policy.heuristic_spec)

    def test_v9_shaped_config_heuristic_arm_loads_frozen_registry(self):
        evaluator = _Evaluator(
            {
                "llm": {
                    "runtime": "openrouter",
                    "gateway_config": "openrouter.yaml",
                    "model_registry": "llm-models.yaml",
                    "model_key": "qwen_qwen3_8_flash_8192_routed",
                    "heuristic_config": "heuristics.yaml",
                },
            },
            PAPER_CONFIG_DIR,
        )
        policy = study._policy("opro_heuristic", evaluator, seed=1, run_dir=Path("."), initial_design=[])
        self.assertIsInstance(policy.runtime, OpenRouterRuntime)
        self.assertIsNotNone(policy.heuristic_spec)
        self.assertEqual(policy.heuristic_spec.get("version"), "component_mechanisms_smm_v1")

    def test_proposal_runtime_override_builds_anthropic_foundry_runtime(self):
        evaluator = _Evaluator(
            {
                # heuristic_config still comes from the frozen llm block --
                # only the runtime/model changes.
                "llm": {"heuristic_config": str(PAPER_CONFIG_DIR / "heuristics.yaml")},
                "proposal_runtime": {
                    "kind": "anthropic_foundry",
                    "config": "anthropic_foundry.yaml",
                    "model_registry": "models.yaml",
                    "model_key": "claude_sonnet5_foundry_v1",
                },
            },
            FOLLOWUP_CONFIG_DIR,
        )
        with patch.dict(
            "os.environ",
            {
                "ANTHROPIC_FOUNDRY_BASE_URL": "https://example.invalid/anthropic",
                "ANTHROPIC_FOUNDRY_DEPLOYMENT": "example-claude-deployment",
            },
        ):
            policy = study._policy(
                "opro_plain", evaluator, seed=1, run_dir=Path("."), initial_design=[]
            )
        self.assertIsInstance(policy.runtime, AnthropicFoundryRuntime)
        self.assertEqual(
            policy.runtime.runtime_identity["protocol"], "anthropic_foundry_messages_v1"
        )

    def test_proposal_runtime_absent_never_touches_the_new_factory_path(self):
        # Sanity check for the "unreachable-by-construction" claim: a
        # v9-shaped config with no proposal_runtime key never returns
        # anything but the frozen OpenRouterRuntime, regardless of what
        # llm_model_key override is present.
        evaluator = _Evaluator(
            {
                "llm": {
                    "runtime": "openrouter",
                    "gateway_config": "openrouter.yaml",
                    "model_registry": "llm-models.yaml",
                    "model_key": "qwen_qwen3_8_flash_8192_routed",
                    "heuristic_config": "heuristics.yaml",
                },
                "llm_model_key": "qwen_qwen3_8_flash_8192_routed",
            },
            PAPER_CONFIG_DIR,
        )
        policy = study._policy("opro_plain", evaluator, seed=1, run_dir=Path("."), initial_design=[])
        self.assertIsInstance(policy.runtime, OpenRouterRuntime)


if __name__ == "__main__":
    unittest.main()
