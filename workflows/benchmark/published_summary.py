#!/usr/bin/env python3
"""Rebuild the published eight-method summary from completed study artifacts.

The six numerical methods come from the frozen v9 study.  The two OPRO
methods come from the later Foundry/Sonnet follow-up.  No reference values are
stored here: every number is derived from policy ledgers and Test Book reports.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from lob_sim.optimization.report import load_policy_runs
from workflows.benchmark.ticker_books import TICKERS


BASELINE_METHODS = (
    "sobol", "tpe", "bads", "gp_bo", "mo_tpe", "component_gp_bo",
)
OPRO_METHODS = ("opro_plain", "opro_heuristic")
METHODS = BASELINE_METHODS + OPRO_METHODS
EXPECTED_REPETITIONS = tuple(range(5))
EXPECTED_SLOTS = 50
EXPECTED_INITIAL_SLOTS = 12
DEFAULT_BASELINE_ROOT = Path(
    "results/optimization/icaif_smm_v9_ticker_specific_10x30m"
)
DEFAULT_OPRO_ROOT = Path(
    "results/optimization/icaif_smm_v9_followup_foundry_sonnet5"
)
DEFAULT_OUTPUT_DIR = Path("results/reproduction_summary")


def _average_ranks(values: dict[str, float]) -> dict[str, float]:
    """Return one-based average ranks, treating numerically equal values as ties."""
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    ranks: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and math.isclose(
            ordered[end][1], ordered[index][1], rel_tol=1e-12, abs_tol=1e-12
        ):
            end += 1
        rank = fmean(range(index + 1, end + 1))
        for method, _ in ordered[index:end]:
            ranks[method] = rank
        index = end
    return ranks


def _load_test_report(root: Path, methods: tuple[str, ...]) -> dict[str, Any]:
    path = root / "report" / "ticker_test_summary.json"
    if not path.is_file():
        raise RuntimeError(f"Missing Test Book report: {path}")
    report = json.loads(path.read_text())
    tickers = report.get("tickers")
    if not isinstance(tickers, dict) or set(tickers) != set(TICKERS):
        found = sorted(tickers) if isinstance(tickers, dict) else type(tickers).__name__
        raise RuntimeError(
            f"{path} must contain exactly the five published tickers; found {found}"
        )
    for ticker in TICKERS:
        rows = tickers[ticker].get("methods", {})
        missing = sorted(set(methods) - set(rows))
        if missing:
            raise RuntimeError(f"{path}: {ticker} is missing methods {missing}")
        for method in methods:
            value = rows[method].get("test_objective")
            if value is None or not math.isfinite(float(value)):
                raise RuntimeError(
                    f"{path}: {ticker}/{method} has no finite test_objective"
                )
    return report


def _same_curve(left: list[float | None], right: list[float | None]) -> bool:
    return len(left) == len(right) and all(
        (a is None and b is None)
        or (
            a is not None
            and b is not None
            and math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
        )
        for a, b in zip(left, right)
    )


def _calibration_values(
    root: Path,
    ticker: str,
    methods: tuple[str, ...],
    *,
    expected_initial_curve: list[float | None] | None = None,
) -> tuple[dict[str, list[float]], dict[str, dict[str, Any]], list[float | None]]:
    study_dir = root / ticker / "calibration"
    runs = load_policy_runs(study_dir)
    output: dict[str, list[float]] = {}
    selected_incumbents: dict[str, dict[str, Any]] = {}
    initial_curve = expected_initial_curve
    for method in methods:
        selected = [run for run in runs if run.get("sampler") == method]
        repetitions = tuple(sorted(int(run["repetition"]) for run in selected))
        if repetitions != EXPECTED_REPETITIONS:
            raise RuntimeError(
                f"{study_dir}: {method} repetitions are {repetitions}; "
                f"expected {EXPECTED_REPETITIONS}"
            )
        values: list[float] = []
        for run in selected:
            if run.get("configured_budget") != EXPECTED_SLOTS:
                raise RuntimeError(
                    f"{study_dir}: {method} repetition {run['repetition']} has "
                    f"configured budget {run.get('configured_budget')!r}, expected "
                    f"{EXPECTED_SLOTS}"
                )
            if run.get("proposal_slots") != EXPECTED_SLOTS:
                raise RuntimeError(
                    f"{study_dir}: {method} repetition {run['repetition']} has "
                    f"{run.get('proposal_slots')!r} slots, expected {EXPECTED_SLOTS}"
                )
            curve = run.get("best_so_far")
            if not isinstance(curve, list) or len(curve) != EXPECTED_SLOTS:
                raise RuntimeError(
                    f"{study_dir}: {method} repetition {run['repetition']} has "
                    "an incomplete best-so-far trajectory"
                )
            if any(value is None or not math.isfinite(float(value)) for value in curve):
                raise RuntimeError(
                    f"{study_dir}: {method} repetition {run['repetition']} has "
                    "a non-finite best-so-far trajectory"
                )
            if any(
                float(right) > float(left) + 1e-12
                for left, right in zip(curve, curve[1:])
            ):
                raise RuntimeError(
                    f"{study_dir}: {method} repetition {run['repetition']} has "
                    "a non-monotone best-so-far trajectory"
                )
            prefix = curve[:EXPECTED_INITIAL_SLOTS]
            if initial_curve is None:
                initial_curve = prefix
            elif not _same_curve(prefix, initial_curve):
                raise RuntimeError(
                    f"{study_dir}: {method} repetition {run['repetition']} "
                    "does not share the frozen 12-slot initial trajectory"
                )
            if run.get("symbol") != ticker:
                raise RuntimeError(
                    f"{study_dir}: {method} ledger symbol is {run.get('symbol')!r}"
                )
            if run.get("split") != "calibration":
                raise RuntimeError(
                    f"{study_dir}: {method} ledger split is {run.get('split')!r}"
                )
            objective = run.get("best_objective")
            if objective is None or not math.isfinite(float(objective)):
                raise RuntimeError(
                    f"{study_dir}: {method} repetition {run['repetition']} "
                    "has no finite calibration incumbent"
                )
            if not math.isclose(
                float(curve[-1]), float(objective), rel_tol=0.0, abs_tol=1e-12
            ):
                raise RuntimeError(
                    f"{study_dir}: {method} repetition {run['repetition']} has "
                    "a final trajectory value different from its incumbent"
                )
            if method in OPRO_METHODS and (
                run.get("valid_proposals") != EXPECTED_SLOTS
                or run.get("invalid_or_error_proposals") != 0
                or run.get("model_calls") != EXPECTED_SLOTS - EXPECTED_INITIAL_SLOTS
            ):
                raise RuntimeError(
                    f"{study_dir}: {method} repetition {run['repetition']} does "
                    "not match the completed OPRO call/status contract"
                )
            values.append(float(objective))
        output[method] = values
        selected_incumbents[method] = min(
            selected, key=lambda run: float(run["best_objective"])
        )
    if initial_curve is None:
        raise RuntimeError(f"{study_dir}: no calibration trajectories found")
    return output, selected_incumbents, initial_curve


def _validate_test_selection(
    report: dict[str, Any],
    ticker: str,
    incumbents: dict[str, dict[str, Any]],
) -> None:
    rows = report["tickers"][ticker]["methods"]
    for method, incumbent in incumbents.items():
        row = rows[method]
        calibration_objective = row.get("calibration_objective")
        if calibration_objective is None or not math.isclose(
            float(calibration_objective),
            float(incumbent["best_objective"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"{ticker}/{method}: Test Book report selected a different "
                "calibration objective"
            )
        if int(row.get("selected_repetition", -1)) != int(incumbent["repetition"]):
            raise RuntimeError(
                f"{ticker}/{method}: Test Book report selected a different repetition"
            )
        if row.get("selected_policy_trial") != incumbent.get("best_policy_trial"):
            raise RuntimeError(
                f"{ticker}/{method}: Test Book report selected a different policy trial"
            )
        if row.get("candidate") != incumbent.get("best_candidate"):
            raise RuntimeError(
                f"{ticker}/{method}: Test Book report selected a different candidate"
            )


def build_summary(
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    opro_root: Path = DEFAULT_OPRO_ROOT,
) -> dict[str, Any]:
    """Validate the two study roots and compute the combined paper statistics."""
    baseline_report = _load_test_report(baseline_root, BASELINE_METHODS)
    opro_report = _load_test_report(opro_root, OPRO_METHODS)
    ticker_rows: dict[str, Any] = {}
    for ticker in TICKERS:
        baseline_values, baseline_incumbents, initial_curve = _calibration_values(
            baseline_root, ticker, BASELINE_METHODS
        )
        opro_values, opro_incumbents, _ = _calibration_values(
            opro_root,
            ticker,
            OPRO_METHODS,
            expected_initial_curve=initial_curve,
        )
        calibration = {**baseline_values, **opro_values}
        _validate_test_selection(baseline_report, ticker, baseline_incumbents)
        _validate_test_selection(opro_report, ticker, opro_incumbents)
        test_values = {
            **{
                method: float(
                    baseline_report["tickers"][ticker]["methods"][method][
                        "test_objective"
                    ]
                )
                for method in BASELINE_METHODS
            },
            **{
                method: float(
                    opro_report["tickers"][ticker]["methods"][method][
                        "test_objective"
                    ]
                )
                for method in OPRO_METHODS
            },
        }
        calibration_means = {
            method: fmean(calibration[method]) for method in METHODS
        }
        calibration_ranks = _average_ranks(calibration_means)
        test_ranks = _average_ranks(test_values)
        ticker_rows[ticker] = {
            "methods": {
                method: {
                    "calibration_best_loss_mean": calibration_means[method],
                    "calibration_best_loss_std": stdev(calibration[method]),
                    "calibration_rank": calibration_ranks[method],
                    "test_objective": test_values[method],
                    "test_rank": test_ranks[method],
                }
                for method in METHODS
            }
        }

    method_rows: dict[str, Any] = {}
    for method in METHODS:
        rows = [ticker_rows[ticker]["methods"][method] for ticker in TICKERS]
        method_rows[method] = {
            "ticker_count": len(TICKERS),
            "mean_calibration_best_loss": fmean(
                row["calibration_best_loss_mean"] for row in rows
            ),
            "mean_calibration_best_loss_std": fmean(
                row["calibration_best_loss_std"] for row in rows
            ),
            "mean_calibration_rank": fmean(row["calibration_rank"] for row in rows),
            "mean_test_objective": fmean(row["test_objective"] for row in rows),
            "mean_test_rank": fmean(row["test_rank"] for row in rows),
        }
    return {
        "version": "published_eight_method_summary_v1",
        "unit_of_replication": "optimizer repetition within ticker",
        "calibration_statistic": "mean and sample standard deviation of five run incumbents",
        "test_statistic": "one selected incumbent per ticker and method on Test Books",
        "rank_direction": "lower loss is better; ties receive average ranks",
        "expected_repetitions": len(EXPECTED_REPETITIONS),
        "expected_slots_per_repetition": EXPECTED_SLOTS,
        "tickers": ticker_rows,
        "methods": method_rows,
    }


def write_summary(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write structured JSON plus flat ticker and aggregate CSV rows."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    fieldnames = (
        "scope", "ticker", "method", "calibration_best_loss_mean",
        "calibration_best_loss_std", "calibration_rank", "test_objective",
        "test_rank", "mean_calibration_best_loss",
        "mean_calibration_best_loss_std", "mean_calibration_rank",
        "mean_test_objective", "mean_test_rank",
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ticker in TICKERS:
            for method in METHODS:
                writer.writerow({
                    "scope": "ticker",
                    "ticker": ticker,
                    "method": method,
                    **summary["tickers"][ticker]["methods"][method],
                })
        for method in METHODS:
            aggregate = summary["methods"][method]
            writer.writerow({
                "scope": "aggregate",
                "ticker": "",
                "method": method,
                **{
                    key: aggregate[key]
                    for key in (
                        "mean_calibration_best_loss",
                        "mean_calibration_best_loss_std",
                        "mean_calibration_rank",
                        "mean_test_objective",
                        "mean_test_rank",
                    )
                },
            })
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--opro-root", type=Path, default=DEFAULT_OPRO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    paths = write_summary(
        build_summary(args.baseline_root, args.opro_root), args.output_dir
    )
    print(f"Published summary written to {paths[0]} and {paths[1]}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
