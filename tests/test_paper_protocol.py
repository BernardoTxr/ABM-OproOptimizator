import unittest
from pathlib import Path

import yaml

from lob_sim.optimization.space import SearchSpace


ROOT = Path(__file__).resolve().parents[1]


class PaperProtocolTests(unittest.TestCase):
    def setUp(self):
        self.optimizer = yaml.safe_load(
            (ROOT / "configs/paper/optimizer.yaml").read_text()
        )
        self.space = SearchSpace.from_config(self.optimizer["search_space"])

    def test_proposed_common_design_is_reproducible_and_valid(self):
        artifact = yaml.safe_load(
            (ROOT / "configs/paper/initial-design.yaml").read_text()
        )
        candidates = artifact["candidates"]
        self.assertEqual(artifact["design"], "scrambled_sobol")
        self.assertEqual(len(candidates), 12)
        self.assertEqual(candidates, self.space.sobol(12, artifact["seed"]))
        for candidate in candidates:
            self.assertEqual(candidate, self.space.validate(candidate))

    def test_protocol_uses_approved_budget_and_reused_dates(self):
        protocol = yaml.safe_load(
            (ROOT / "configs/paper/protocol.yaml").read_text()
        )
        development = protocol["development_evaluation"]
        experiment = protocol["optimizer_experiment"]
        later = protocol["later_date_evaluation"]

        self.assertEqual(protocol["status"], "ready_for_benchmark")
        self.assertEqual(development["scenarios_per_candidate"], 60)
        self.assertEqual(development["simulation_output_retention"], "summary_only")
        self.assertEqual(experiment["charged_slots_per_repetition"], 50)
        self.assertEqual(experiment["independent_repetitions"], 3)
        self.assertEqual(experiment["common_initial_slots"], 12)
        self.assertEqual(later["status"], "accepted_non_pristine_evaluation")
        self.assertEqual(later["dates_from"], "data.yaml#date_splits.test")
        self.assertEqual(later["scenarios_per_candidate"], 120)
        extension = protocol["posthoc_optimizer_extension"]
        self.assertEqual(extension["status"], "historical_only")
        self.assertEqual(
            extension["methods"], ["sobol", "gp_bo", "cma_es", "bads"]
        )
        self.assertIn("post-hoc", extension["disclosure"])
        self.assertEqual(
            extension["problem_classification"]["oracle"],
            "deterministic_sample_average_with_common_random_numbers",
        )
        component = protocol["prospective_component_aware_comparison"]
        self.assertEqual(component["status"], "ready_to_run")
        self.assertEqual(len(component["methods"]), 8)
        self.assertEqual(component["signed_moment_residuals"], "audit_only")
        self.assertEqual(
            component["llm_model_key"],
            "qwen_qwen3_8_flash_8192_routed",
        )
        self.assertEqual(
            component["tickers"], ["ASAI3", "CSAN3", "ITSA4", "LREN3", "MGLU3"]
        )
        self.assertEqual(component["parameter_scope"], "independent_vector_per_ticker")
        self.assertEqual(component["calibration_books_per_ticker"], 90)
        self.assertEqual(component["test_books_per_ticker"], 90)
        self.assertEqual(component["windows_per_date"], 10)
        self.assertEqual(component["optimizer_repetitions_per_ticker"], 5)
        self.assertFalse(component["confirmation_stage"])
        self.assertIn("not part of the original preregistration", component["disclosure"])
        archive = component["archival_contract"]
        self.assertEqual(
            archive["calibration_raw_outputs"],
            "compact_after_validated_smm_archive",
        )
        self.assertEqual(
            archive["test_raw_outputs"],
            "retain_all_forty_selected_candidates",
        )

    def test_executable_config_matches_ticker_calibration_protocol(self):
        config = yaml.safe_load(
            (ROOT / "configs/paper/optimizer.yaml").read_text()
        )
        design = yaml.safe_load(
            (ROOT / "configs/paper/initial-design.yaml").read_text()
        )
        self.assertEqual(config["split"], "calibration")
        self.assertEqual(config["real_config"], "ticker-books.yaml")
        self.assertEqual(config["replications_per_trial"], 90)
        self.assertEqual(config["trial_budget"], 50)
        self.assertEqual(config["optimizer_repetitions"], 5)
        self.assertEqual(config["evaluation_seed"], 62700)
        self.assertEqual(config["simulation_output_retention"], "summary_only")
        scenario = yaml.safe_load(
            (ROOT / "configs/paper/scenario.yaml").read_text()
        )
        self.assertFalse(scenario["write_diagnostics"])
        self.assertEqual(config["initial_design"], design["candidates"])
        self.assertEqual(config["llm"]["heuristic_config"], "heuristics.yaml")
        self.assertEqual(config["llm"]["runtime"], "openrouter")
        self.assertEqual(
            config["llm"]["model_key"],
            "qwen_qwen3_8_flash_8192_routed",
        )
        protocol = yaml.safe_load(
            (ROOT / "configs/paper/protocol.yaml").read_text()
        )
        self.assertEqual(
            protocol["llm"]["heuristic_registry_version"],
            "component_mechanisms_smm_v1",
        )

    def test_test_books_evaluator_matches_calibration_contract(self):
        calibration = yaml.safe_load(
            (ROOT / "configs/paper/optimizer.yaml").read_text()
        )
        test = yaml.safe_load(
            (ROOT / "configs/paper/later-evaluation.yaml").read_text()
        )
        for key in ("symbol", "base_config", "real_config", "loss_config", "search_space", "validity_gates"):
            self.assertEqual(calibration[key], test[key])
        self.assertEqual(test["split"], "test")
        self.assertEqual(test["replications_per_trial"], 90)
        self.assertEqual(test["evaluation_seed"], 92700)
        self.assertEqual(test["smm_weight_source"], "ticker_calibration_target")
        self.assertEqual(test["simulation_output_retention"], "full")


if __name__ == "__main__":
    unittest.main()
