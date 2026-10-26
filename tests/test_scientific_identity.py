import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from lob_sim.optimization.scientific_identity import scientific_identity_sha256


def _build_fixture(root: Path) -> SimpleNamespace:
    (root / "lob_sim" / "core").mkdir(parents=True)
    (root / "lob_sim" / "archetypes").mkdir(parents=True)
    (root / "lob_sim" / "agents").mkdir(parents=True)
    (root / "lob_sim" / "loss").mkdir(parents=True)
    (root / "lob_sim" / "optimization").mkdir(parents=True)
    (root / "lob_sim" / "core" / "lob.py").write_text("LOB = 1\n")
    (root / "lob_sim" / "environment.py").write_text("ENV = 1\n")
    (root / "lob_sim" / "archetypes" / "noise_trader.py").write_text("NOISE = 1\n")
    (root / "lob_sim" / "agents" / "factory.py").write_text("FACTORY = 1\n")
    (root / "lob_sim" / "loss" / "smm.py").write_text("SMM = 1\n")
    (root / "lob_sim" / "optimization" / "evaluator.py").write_text("EVAL = 1\n")
    (root / "lob_sim" / "optimization" / "space.py").write_text("SPACE = 1\n")
    (root / "lob_sim" / "optimization" / "feedback.py").write_text("FEEDBACK = 1\n")
    # Adapter/runtime files must never affect the scientific identity, even
    # though they live under lob_sim/optimization/.
    (root / "lob_sim" / "optimization" / "openrouter_runtime.py").write_text("OR = 1\n")
    (root / "lob_sim" / "optimization" / "anthropic_foundry_runtime.py").write_text("AF = 1\n")

    (root / "agents.yaml").write_text("agents: []\n")
    (root / "environment.yaml").write_text("tick_size: 0.01\n")
    (root / "scenario.yaml").write_text("n_scenarios: 20\n")
    (root / "loss.yaml").write_text("version: smm_v4\n")

    return SimpleNamespace(
        base_agents_path=root / "agents.yaml",
        environment_path=root / "environment.yaml",
        base_scenario_path=root / "scenario.yaml",
        loss_config_path=root / "loss.yaml",
        config={
            "search_space": [{"name": "x", "type": "float", "low": 0.0, "high": 1.0}],
            "validity_gates": {"min_book_coverage": 0.99},
            "optimizer_seed": 20260912,
            "evaluation_seed": 62700,
            "initial_design": [{"x": 0.5}],
            "trial_budget": 50,
            "optimizer_repetitions": 5,
            "llm": {"runtime": "openrouter", "model_key": "qwen_qwen3_8_flash_8192_routed"},
        },
    )


class ScientificIdentityTests(unittest.TestCase):
    def test_hash_is_stable_across_repeated_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = _build_fixture(root)
            first = scientific_identity_sha256(root, evaluator)
            second = scientific_identity_sha256(root, evaluator)
        self.assertEqual(first, second)

    def test_hash_changes_when_search_space_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = _build_fixture(root)
            baseline = scientific_identity_sha256(root, evaluator)
            evaluator.config["search_space"][0]["high"] = 2.0
            changed = scientific_identity_sha256(root, evaluator)
        self.assertNotEqual(baseline, changed)

    def test_hash_changes_when_simulator_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = _build_fixture(root)
            baseline = scientific_identity_sha256(root, evaluator)
            (root / "lob_sim" / "core" / "lob.py").write_text("LOB = 2\n")
            changed = scientific_identity_sha256(root, evaluator)
        self.assertNotEqual(baseline, changed)

    def test_hash_is_unaffected_by_llm_or_proposal_runtime_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = _build_fixture(root)
            baseline = scientific_identity_sha256(root, evaluator)
            evaluator.config["llm"] = {"runtime": "openrouter", "model_key": "different_model"}
            evaluator.config["proposal_runtime"] = {
                "kind": "anthropic_foundry", "model_key": "claude_sonnet5_foundry_v1",
            }
            changed = scientific_identity_sha256(root, evaluator)
        self.assertEqual(baseline, changed)

    def test_hash_is_unaffected_by_adapter_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator = _build_fixture(root)
            baseline = scientific_identity_sha256(root, evaluator)
            (root / "lob_sim" / "optimization" / "anthropic_foundry_runtime.py").write_text(
                "AF = 2  # changed\n"
            )
            (root / "lob_sim" / "optimization" / "openrouter_runtime.py").write_text(
                "OR = 2  # changed\n"
            )
            changed = scientific_identity_sha256(root, evaluator)
        self.assertEqual(baseline, changed)


if __name__ == "__main__":
    unittest.main()
