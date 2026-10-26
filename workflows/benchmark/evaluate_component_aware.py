#!/usr/bin/env python3
"""Select ticker-specific incumbents and score them on the Test Books."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from lob_sim.loss.smm import SMMSpec
from lob_sim.optimization.evaluator import TrialEvaluator
from workflows.benchmark.audit_component_stage import METHODS
from workflows.benchmark.evaluate import (
    _atomic_json,
    _bootstrap_later,
    _complete_incumbents,
)
from workflows.benchmark.run_component_aware import validate_component_contract
from workflows.benchmark.ticker_books import (
    TICKERS,
    build_ticker_tasks,
    comparison_output_root,
    experiment_paths,
    resolved_proposal_runtime,
)


def choose_calibration_incumbents(
    candidates: dict[str, list[dict[str, Any]]]
) -> dict[str, dict[str, Any]]:
    """Select the lowest-loss incumbent across five runs, with no confirmation."""
    return {
        method: min(
            rows,
            key=lambda row: (
                float(row["search_objective"]),
                json.dumps(row["candidate"], sort_keys=True, separators=(",", ":")),
            ),
        )
        for method, rows in candidates.items()
    }


def _average_ranks(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and np.isclose(
            ordered[end][1], ordered[index][1], rtol=1e-12, atol=1e-12
        ):
            end += 1
        rank = float(np.mean(np.arange(index + 1, end + 1)))
        for method, _ in ordered[index:end]:
            ranks[method] = rank
        index = end
    return ranks


def run_ticker_tests(
    comparison_path: Path,
    selected_tickers: tuple[str, ...] = TICKERS,
    *,
    methods: tuple[str, ...] = METHODS,
    calibration_simulation_workers: int | None = None,
) -> dict[str, Any]:
    """Freeze one best-of-five vector per ticker/method and test it once.

    ``methods`` defaults to the frozen eight-way ``METHODS`` tuple, in which
    case the full v6/v9 ``validate_component_contract`` still runs unchanged.
    A caller that passes a narrower ``methods`` (e.g. a two-arm OPRO
    follow-up against a differently-shaped comparison.yaml) skips that
    v9-specific contract check -- it does not apply to a study that isn't
    the frozen eight-method comparison -- and every method loop below is
    scoped to the given subset instead.

    ``calibration_simulation_workers`` must match whatever
    ``--simulation-workers`` override (if any) was used to *produce* the
    calibration ledger being read here. That value is folded into
    ``runtime_overrides`` and therefore into ``evaluator_fingerprint`` --
    reconstructing the calibration evaluator with a different value (even
    the config's own default) produces a technically-different but
    equally-valid fingerprint that cannot find the existing ledger. This
    parameter exists only to let a caller reproduce the exact evaluator an
    operational override already used; it does not change any scientific
    input.
    """
    (
        calibration_path,
        test_path,
        _,
        comparison,
        calibration_config,
        test_config,
        real,
    ) = experiment_paths(comparison_path)
    if methods == METHODS:
        validate_component_contract(
            comparison, calibration_config, test_config, real
        )
    unknown = sorted(set(selected_tickers) - set(TICKERS))
    if unknown:
        raise RuntimeError(f"Unknown ticker filters: {unknown}")
    if len(selected_tickers) != len(set(selected_tickers)):
        raise RuntimeError("Ticker filters contain duplicates")
    tasks = tuple(
        task for task in build_ticker_tasks(comparison, calibration_config)
        if task.symbol in selected_tickers
    )
    ticker_reports: dict[str, Any] = {}
    test_losses: dict[str, dict[str, float]] = {}
    for task in tasks:
        calibration_overrides = task.calibration_overrides(
            comparison["llm_model_key"],
            proposal_runtime=resolved_proposal_runtime(comparison_path, comparison),
        )
        if calibration_simulation_workers is not None:
            calibration_overrides["simulation_workers"] = calibration_simulation_workers
        calibration = TrialEvaluator(
            str(calibration_path), runtime_overrides=calibration_overrides
        )
        incumbents, ledger_fingerprint = _complete_incumbents(
            calibration_path, calibration.study_dir, methods=methods
        )
        if ledger_fingerprint != calibration.evaluator_fingerprint:
            raise RuntimeError(
                f"{task.symbol} evaluator differs from its calibration ledgers"
            )
        selected = choose_calibration_incumbents(incumbents)

        test = TrialEvaluator(str(test_path), runtime_overrides=task.test_overrides())
        if not isinstance(test.loss_spec, SMMSpec) or test.smm_target is None:
            raise RuntimeError(f"{task.symbol} Test Books evaluator is not SMM")
        if test.smm_target.metadata.get("weight_source") != "calibration_target":
            raise RuntimeError(
                f"{task.symbol} Test Books evaluator did not freeze calibration weights"
            )
        results: dict[str, dict[str, Any]] = {}
        rows: dict[str, Any] = {}
        for method in methods:
            selection = selected[method]
            result = test.evaluate(
                selection["candidate"], f"test-{task.symbol}-{method}"
            )
            if not result.get("valid") or result.get("objective") is None:
                raise RuntimeError(
                    f"Test Books candidate is invalid: {task.symbol}/{method}"
                )
            results[method] = result
            rows[method] = {
                "candidate": selection["candidate"],
                "selected_repetition": selection["repetition"],
                "selected_policy_trial": selection["policy_trial"],
                "calibration_objective": selection["search_objective"],
                "selection_rule": (
                    "lowest_calibration_loss_across_five_repetition_incumbents"
                ),
                "test_objective": float(result["objective"]),
                "test_component_losses": result["component_losses"],
                "test_candidate_hash": result["candidate_hash"],
                "test_runtime_seconds": result["runtime_seconds"],
            }
        ranks = _average_ranks({
            method: row["test_objective"] for method, row in rows.items()
        })
        for method, rank in ranks.items():
            rows[method]["test_rank"] = rank
        ticker_report = {
            "ticker": task.symbol,
            "parameter_scope": "ticker_specific",
            "calibration_variable": "CalibrationBooks",
            "test_variable": "TestBooks",
            "calibration_books": task.books_per_partition,
            "test_books": task.books_per_partition,
            "confirmation_stage": False,
            "calibration_evaluator_fingerprint": ledger_fingerprint,
            "test_evaluator_fingerprint": test.evaluator_fingerprint,
            "test_evaluation_seed": test.evaluation_seed,
            "test_standardization": "frozen_ticker_calibration_standard_errors",
            "methods": rows,
            "whole_date_bootstrap": _bootstrap_later(
                results, test.smm_target, test.loss_spec, methods=methods
            ),
        }
        _atomic_json(
            test.study_dir / "ticker_test_summary.json", ticker_report
        )
        ticker_reports[task.symbol] = ticker_report
        test_losses[task.symbol] = {
            method: row["test_objective"] for method, row in rows.items()
        }

    method_summary: dict[str, Any] = {}
    for method in methods:
        values = np.asarray([
            test_losses[ticker][method] for ticker in selected_tickers
        ], dtype=float)
        ranks = np.asarray([
            ticker_reports[ticker]["methods"][method]["test_rank"]
            for ticker in selected_tickers
        ], dtype=float)
        method_summary[method] = {
            "ticker_count": len(selected_tickers),
            "mean_test_loss": float(values.mean()),
            "median_test_loss": float(np.median(values)),
            "mean_test_rank": float(ranks.mean()),
            "ticker_wins": int(np.sum(ranks == 1.0)),
        }
    report = {
        "stage": "ticker_specific_best_of_five_then_test_books",
        "parameter_pooling": False,
        "confirmation_stage": False,
        "selection_rule": (
            "independently_for_each_ticker_and_method_select_the_lowest_"
            "calibration_loss_across_five_repetition_incumbents"
        ),
        "test_interpretation": (
            "regime-balanced whole-date holdout from the observed sample; "
            "not a future-regime forecast"
        ),
        "tickers": ticker_reports,
        "cross_ticker_summary": method_summary,
    }
    root = comparison_output_root(
        comparison, calibration_path, calibration_config
    )
    _atomic_json(root / "report/ticker_test_summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-config",
        type=Path,
        default=Path("configs/paper/component-aware-comparison.yaml"),
    )
    parser.add_argument("--ticker", action="append")
    args = parser.parse_args()
    tickers = tuple(str(value).upper() for value in (args.ticker or TICKERS))
    report = run_ticker_tests(args.comparison_config.resolve(), tickers)
    print(json.dumps({
        "confirmation_stage": False,
        "parameter_pooling": False,
        "tested_tickers": list(report["tickers"]),
        "cross_ticker_summary": report["cross_ticker_summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
