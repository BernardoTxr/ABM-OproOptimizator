import unittest
from pathlib import Path

import yaml

from lob_sim.loss.real_facts import (
    resolve_measurement_windows,
    select_window_assignments,
)
from workflows.benchmark.audit_component_stage import FEEDBACK_MODES, METHODS
from workflows.benchmark.run_component_aware import (
    component_plan,
    method_stage_path,
    validate_component_contract,
)
from workflows.benchmark.evaluate_component_aware import choose_calibration_incumbents
from workflows.benchmark.ticker_books import (
    TICKERS,
    build_ticker_tasks,
    experiment_paths,
)


ROOT = Path(__file__).resolve().parents[1]


class ComponentAwareWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.comparison_path = ROOT / "configs/paper/component-aware-comparison.yaml"
        (
            self.optimizer_path,
            self.test_path,
            self.real_path,
            self.comparison,
            self.optimizer,
            self.test,
            self.real,
        ) = experiment_paths(self.comparison_path)

    def test_checked_in_protocol_is_frozen_and_disclosed(self):
        checks = validate_component_contract(
            self.comparison, self.optimizer, self.test, self.real
        )
        self.assertTrue(all(checks.values()))
        self.assertEqual(tuple(self.comparison["methods"]), METHODS)
        self.assertEqual(
            self.comparison["feedback_contract"]["information_firewall"],
            FEEDBACK_MODES,
        )
        self.assertEqual(
            self.comparison["feedback_contract"]["signed_moment_residuals"],
            "audit_only",
        )

    def test_plan_has_five_isolated_ticker_tasks(self):
        plan = component_plan(self.comparison_path)
        self.assertTrue(plan["comparison_root"].endswith(
            "icaif_smm_v9_ticker_specific_10x30m"
        ))
        self.assertEqual(tuple(plan["tickers"]), TICKERS)
        for ticker, task in plan["tickers"].items():
            self.assertEqual(task["parameter_scope"], "ticker_specific")
            self.assertEqual(task["calibration_books"], 90)
            self.assertEqual(task["test_books"], 90)
            self.assertEqual(task["total_books"], 180)
            self.assertEqual(set(task["methods"]), set(METHODS))
            self.assertIn(f"/{ticker}/calibration", task["calibration_study_dir"])
        self.assertEqual(
            plan["llm"]["model_key"],
            "qwen_qwen3_8_flash_8192_routed",
        )
        self.assertEqual(plan["llm"]["requested_model"], "qwen/qwen3.8-flash")
        self.assertTrue(plan["llm"]["provider_routing"]["allow_fallbacks"])
        self.assertEqual(
            plan["llm"]["provider_routing"]["order"],
            ["makora", "alibaba"],
        )
        self.assertTrue(
            plan["llm"]["transport_reliability"]["serialized_across_tickers"]
        )
        self.assertEqual(plan["llm"]["transport_reliability"]["max_attempts"], 6)

    def test_book_split_is_whole_date_and_exactly_ninety_per_partition(self):
        calibration = self.real["date_splits"]["calibration"]
        test = self.real["date_splits"]["test"]
        self.assertEqual(len(calibration), 9)
        self.assertEqual(len(test), 9)
        self.assertFalse(set(calibration) & set(test))
        self.assertEqual(len(self.real["window_starts_utc"]), 10)
        self.assertEqual(self.real["window_duration_hours"], 0.5)
        tasks = build_ticker_tasks(self.comparison, self.optimizer)
        self.assertEqual(tuple(task.symbol for task in tasks), TICKERS)
        self.assertTrue(all(task.books_per_partition == 90 for task in tasks))
        self.assertEqual(len({task.calibration_study_name for task in tasks}), 5)
        windows = resolve_measurement_windows(self.real)
        assignments = select_window_assignments(calibration, windows, 90, 62700)
        self.assertEqual(len(assignments), 90)
        self.assertEqual(len(set(assignments)), 90)
        self.assertEqual(
            set(assignments),
            {(date, start) for date in calibration for start, _ in windows},
        )

    def test_protocol_cannot_be_relabelled_as_original_preregistration(self):
        self.comparison["disclosure"] = "original preregistration"
        with self.assertRaisesRegex(RuntimeError, "prospective_disclosure"):
            validate_component_contract(
                self.comparison, self.optimizer, self.test, self.real
            )

    def test_direct_selection_has_no_confirmation_stage(self):
        selected = choose_calibration_incumbents({
            "tpe": [
                {"search_objective": 2.0, "candidate": {"x": 1}, "repetition": 0},
                {"search_objective": 1.0, "candidate": {"x": 2}, "repetition": 1},
            ]
        })
        self.assertEqual(selected["tpe"]["repetition"], 1)

    def test_method_stage_audits_are_isolated(self):
        study_dir = ROOT / "results/optimization/example/TICKER/calibration"
        self.assertEqual(
            method_stage_path(study_dir, "sobol"),
            study_dir / "report/component_aware_stage_audit_sobol.json",
        )


if __name__ == "__main__":
    unittest.main()
