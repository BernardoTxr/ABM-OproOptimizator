import unittest
from pathlib import Path

from workflows.benchmark.opro_pair_followup import (
    DECLARED_METHODS,
    assert_no_baseline_policy_dirs,
    pair_plan,
    select_methods,
    validate_pair_contract,
)
from workflows.benchmark.ticker_books import TICKERS, experiment_paths


ROOT = Path(__file__).resolve().parents[1]
COMPARISON_PATH = ROOT / "configs/experiments/followup_foundry_sonnet5/comparison.yaml"


class SelectMethodsTests(unittest.TestCase):
    def test_defaults_to_exactly_the_two_declared_methods(self):
        self.assertEqual(select_methods(None), DECLARED_METHODS)

    def test_empty_list_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "must not be empty"):
            select_methods([])

    def test_duplicates_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            select_methods(["opro_plain", "opro_plain"])

    def test_baseline_method_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Undeclared or baseline"):
            select_methods(["opro_plain", "sobol"])

    def test_single_declared_method_is_accepted(self):
        self.assertEqual(select_methods(["opro_heuristic"]), ("opro_heuristic",))


class PairContractTests(unittest.TestCase):
    def setUp(self):
        (
            self.calibration_path, self.test_path, self.real_path,
            self.comparison, self.calibration, self.test, self.real,
        ) = experiment_paths(COMPARISON_PATH)

    def test_checked_in_followup_config_satisfies_the_contract(self):
        checks = validate_pair_contract(self.comparison, self.calibration, self.test)
        self.assertTrue(all(checks.values()), checks)

    def test_reusing_the_v9_study_name_is_rejected(self):
        comparison = dict(self.comparison)
        comparison["comparison_study_name"] = "icaif_smm_v9_ticker_specific_10x30m"
        with self.assertRaisesRegex(RuntimeError, "new_study_name"):
            validate_pair_contract(comparison, self.calibration, self.test)

    def test_adding_a_baseline_method_is_rejected(self):
        comparison = dict(self.comparison)
        comparison["methods"] = ["opro_plain", "opro_heuristic", "sobol"]
        with self.assertRaisesRegex(RuntimeError, "methods_are_exactly_the_two_opro_arms"):
            validate_pair_contract(comparison, self.calibration, self.test)


class PairPlanTests(unittest.TestCase):
    def test_plan_covers_five_tickers_and_exactly_two_methods(self):
        plan = pair_plan(COMPARISON_PATH)
        self.assertTrue(
            plan["comparison_root"].endswith("icaif_smm_v9_followup_foundry_sonnet5")
        )
        self.assertEqual(tuple(plan["tickers"]), TICKERS)
        self.assertEqual(plan["methods"], list(DECLARED_METHODS))
        self.assertEqual(plan["simulator_evaluations"], 0)
        self.assertEqual(plan["network_calls"], 0)
        self.assertEqual(plan["expected_charged_slots_total"], 5 * 2 * 5 * 50)
        for ticker, task in plan["tickers"].items():
            self.assertEqual(set(task["methods"]), set(DECLARED_METHODS))
            self.assertIn(f"/{ticker}/calibration", task["calibration_study_dir"])
        self.assertEqual(
            plan["proposal_runtime"]["runtime_identity"]["protocol"],
            "anthropic_foundry_messages_v1",
        )

    def test_plan_never_points_at_a_v9_result_path(self):
        plan = pair_plan(COMPARISON_PATH)
        self.assertNotIn("icaif_smm_v9_ticker_specific_10x30m", plan["comparison_root"])
        for task in plan["tickers"].values():
            self.assertNotIn(
                "icaif_smm_v9_ticker_specific_10x30m", task["calibration_study_dir"]
            )


class AssertNoBaselinePolicyDirsTests(unittest.TestCase):
    def test_passes_on_an_empty_root(self):
        assert_no_baseline_policy_dirs(Path("/nonexistent-root-for-test"))

    def test_raises_when_a_baseline_directory_exists(self, tmp_path=None):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ASAI3" / "calibration" / "policies" / "sobol_repeat00").mkdir(
                parents=True
            )
            with self.assertRaisesRegex(RuntimeError, "Baseline policy directories"):
                assert_no_baseline_policy_dirs(root)

    def test_passes_when_only_opro_directories_exist(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                root / "ASAI3" / "calibration" / "policies" / "opro_plain_repeat00"
            ).mkdir(parents=True)
            assert_no_baseline_policy_dirs(root)


if __name__ == "__main__":
    unittest.main()
