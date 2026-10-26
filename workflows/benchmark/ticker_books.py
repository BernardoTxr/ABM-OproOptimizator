"""Frozen ticker-specific Calibration Books and Test Books experiment design."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import yaml

from workflows import REPO_ROOT


TICKERS = ("ASAI3", "CSAN3", "ITSA4", "LREN3", "MGLU3")
CALIBRATION_SPLIT = "calibration"
TEST_SPLIT = "test"


def _resolve(value: str, base: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


@dataclass(frozen=True)
class TickerTask:
    """Paths and runtime overrides for one independent ticker calibration."""

    symbol: str
    calibration_study_name: str
    test_study_name: str
    calibration_target: Path
    calibration_windows: Path
    test_target: Path
    test_windows: Path
    books_per_partition: int

    def calibration_overrides(
        self, llm_model_key: str, *, proposal_runtime: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        overrides = {
            "study_name": self.calibration_study_name,
            "symbol": self.symbol,
            "split": CALIBRATION_SPLIT,
            "replications_per_trial": self.books_per_partition,
            "llm_model_key": llm_model_key,
            "smm_target_cache": str(self.calibration_target),
            "smm_window_moment_cache": str(self.calibration_windows),
        }
        if proposal_runtime is not None:
            overrides["proposal_runtime"] = proposal_runtime
        return overrides

    def test_overrides(self) -> dict[str, Any]:
        return {
            "study_name": self.test_study_name,
            "symbol": self.symbol,
            "split": TEST_SPLIT,
            "replications_per_trial": self.books_per_partition,
            "smm_target_cache": str(self.test_target),
            "smm_window_moment_cache": str(self.test_windows),
            "smm_weight_target_cache": str(self.calibration_target),
        }


def resolved_proposal_runtime(
    comparison_path: Path, comparison: dict[str, Any]
) -> dict[str, Any] | None:
    """Resolve a comparison-level ``proposal_runtime`` block's ``config``/
    ``model_registry`` paths relative to ``comparison_path``'s directory.

    Those files generally live beside a new study's ``comparison.yaml``, not
    beside the frozen ``calibration_optimizer_config`` (``configs/paper/``)
    that the resulting ``TrialEvaluator.config_dir`` would otherwise resolve
    relative paths against -- so callers must not pass the raw block
    through unresolved.
    """
    proposal_runtime = comparison.get("proposal_runtime")
    if proposal_runtime is None:
        return None
    config_dir = comparison_path.resolve().parent
    resolved = dict(proposal_runtime)
    for key in ("config", "model_registry"):
        if key in resolved:
            resolved[key] = str(_resolve(str(resolved[key]), config_dir))
    return resolved


def experiment_paths(
    comparison_path: Path,
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve all four frozen configuration artifacts."""
    comparison_path = comparison_path.resolve()
    comparison = yaml.safe_load(comparison_path.read_text())
    config_dir = comparison_path.parent
    calibration_path = _resolve(comparison["calibration_optimizer_config"], config_dir)
    test_path = _resolve(comparison["test_evaluator_config"], config_dir)
    calibration = yaml.safe_load(calibration_path.read_text())
    test = yaml.safe_load(test_path.read_text())
    real_path = _resolve(calibration["real_config"], calibration_path.parent)
    real = yaml.safe_load(real_path.read_text())
    if _resolve(test["real_config"], test_path.parent) != real_path:
        raise RuntimeError("Calibration and test configurations use different book data")
    return (
        calibration_path,
        test_path,
        real_path,
        comparison,
        calibration,
        test,
        real,
    )


def validate_ticker_book_contract(
    comparison: dict[str, Any],
    calibration: dict[str, Any],
    test: dict[str, Any],
    real: dict[str, Any],
) -> dict[str, bool]:
    """Validate the complete no-pooling ticker/book design without reading data."""
    tickers = tuple(str(value).upper() for value in comparison.get("tickers", ()))
    real_tickers = tuple(str(value).upper() for value in real.get("symbols", ()))
    date_splits = real.get("date_splits", {})
    calibration_dates = tuple(str(value) for value in date_splits.get(
        CALIBRATION_SPLIT, ()
    ))
    test_dates = tuple(str(value) for value in date_splits.get(TEST_SPLIT, ()))
    all_dates = tuple(str(value) for value in real.get("all_dates", ()))
    windows = tuple(str(value) for value in real.get("window_starts_utc", ()))
    pairs = [all_dates[index:index + 2] for index in range(0, len(all_dates), 2)]
    one_date_per_pair = bool(pairs) and all(
        len(pair) == 2
        and sum(date in calibration_dates for date in pair) == 1
        and sum(date in test_dates for date in pair) == 1
        for pair in pairs
    )
    split_design = real.get("split_design", {})
    rng = random.Random(int(split_design.get("random_seed", -1)))
    reproduced_calibration = []
    reproduced_test = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        if rng.randrange(2) == 0:
            reproduced_calibration.append(pair[0])
            reproduced_test.append(pair[1])
        else:
            reproduced_calibration.append(pair[1])
            reproduced_test.append(pair[0])
    books_per_ticker = len(calibration_dates) * len(windows)
    book_contract = comparison.get("book_sets", {})
    common = comparison.get("common_contract", {})
    checks = {
        "version": comparison.get("version")
        == "icaif_ticker_specific_component_comparison_v6",
        "status": comparison.get("status") == "ready_to_run",
        "tickers": tickers == TICKERS and real_tickers == TICKERS,
        "ticker_parameters_not_pooled": comparison.get("parameter_scope")
        == "independent_vector_per_ticker",
        "ten_windows": windows == (
            "14:00:00", "14:30:00", "15:00:00", "15:30:00", "16:00:00",
            "16:30:00", "17:00:00", "17:30:00", "18:00:00", "18:30:00",
        ),
        "half_hour_full_session_tiling": (
            float(real.get("window_duration_hours", -1)) == 0.5
            and real.get("session_start_utc") == "14:00:00"
            and real.get("session_end_utc") == "19:00:00"
        ),
        "eighteen_dates": len(all_dates) == 18 and len(set(all_dates)) == 18,
        "nine_calibration_dates": len(calibration_dates) == 9,
        "nine_test_dates": len(test_dates) == 9,
        "date_disjoint": not set(calibration_dates) & set(test_dates),
        "date_complete": set(calibration_dates) | set(test_dates) == set(all_dates),
        "paired_randomized_split": one_date_per_pair
        and split_design.get("unit") == "whole_date"
        and calibration_dates == tuple(reproduced_calibration)
        and test_dates == tuple(reproduced_test)
        and split_design.get("outcome_blind") is True,
        "ninety_books_per_partition": books_per_ticker == 90
        and int(book_contract.get("calibration_books_per_ticker", -1)) == 90
        and int(book_contract.get("test_books_per_ticker", -1)) == 90,
        "one_hundred_eighty_books_per_ticker": int(
            book_contract.get("total_books_per_ticker", -1)
        ) == 180,
        "nine_independent_date_clusters_per_partition": (
            book_contract.get("statistical_cluster_unit") == "whole_date"
            and int(book_contract.get("independent_clusters_per_partition", -1))
            == 9
        ),
        "calibration_template": calibration.get("split") == CALIBRATION_SPLIT
        and int(calibration.get("replications_per_trial", -1)) == 90,
        "test_template": test.get("split") == TEST_SPLIT
        and int(test.get("replications_per_trial", -1)) == 90,
        "five_optimizer_repetitions": int(
            calibration.get("optimizer_repetitions", -1)
        ) == 5 and int(common.get("repetitions", -2)) == 5,
        "matched_evaluators": all(
            calibration.get(key) == test.get(key)
            for key in ("base_config", "real_config", "loss_config", "search_space", "validity_gates")
        ),
        "calibration_weights_on_test": test.get("smm_weight_source")
        == "ticker_calibration_target",
        "best_of_five_selection": comparison.get("selection", {}).get("rule")
        == "lowest_calibration_loss_across_five_repetition_incumbents",
        "no_confirmation": comparison.get("selection", {}).get("confirmation_stage")
        is False,
        "common_date_split": book_contract.get("same_date_split_for_all_tickers")
        is True,
        "analysis_ready_calibration_archive": (
            calibration.get("simulation_output_retention") == "summary_only"
            and comparison.get("archival_contract", {}).get(
                "calibration_raw_outputs"
            ) == "compact_after_validated_smm_archive"
        ),
        "full_selected_test_archive": (
            test.get("simulation_output_retention") == "full"
            and comparison.get("archival_contract", {}).get("test_raw_outputs")
            == "retain_all_forty_selected_candidates"
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError("Ticker-book contract failed: " + ", ".join(failed))
    return checks


def build_ticker_tasks(
    comparison: dict[str, Any], calibration: dict[str, Any]
) -> tuple[TickerTask, ...]:
    """Build the five isolated study directories and empirical artifact paths."""
    output_root = _resolve(
        str(calibration["output_directory"]),
        REPO_ROOT / "configs/paper",
    )
    comparison_name = str(comparison["comparison_study_name"])
    artifact_root = REPO_ROOT / "results/real"
    books = int(comparison["book_sets"]["calibration_books_per_ticker"])
    tasks = []
    for raw_symbol in comparison["tickers"]:
        symbol = str(raw_symbol).upper()
        target_dir = artifact_root / symbol
        tasks.append(TickerTask(
            symbol=symbol,
            calibration_study_name=f"{comparison_name}/{symbol}/calibration",
            test_study_name=f"{comparison_name}/{symbol}/test",
            calibration_target=(
                target_dir / "smm_target_calibration_10x30m_smm_v4.json"
            ).resolve(),
            calibration_windows=(
                target_dir / "smm_target_calibration_10x30m_smm_v4.windows.parquet"
            ).resolve(),
            test_target=(
                target_dir / "smm_target_test_10x30m_smm_v4.json"
            ).resolve(),
            test_windows=(
                target_dir / "smm_target_test_10x30m_smm_v4.windows.parquet"
            ).resolve(),
            books_per_partition=books,
        ))
    expected_root = output_root / comparison_name
    if any(
        Path(task.calibration_study_name).parts[0] != comparison_name
        for task in tasks
    ):
        raise RuntimeError(f"Ticker studies do not belong to {expected_root}")
    return tuple(tasks)


def comparison_output_root(
    comparison: dict[str, Any], calibration_path: Path, calibration: dict[str, Any]
) -> Path:
    output_root = _resolve(str(calibration["output_directory"]), calibration_path.parent)
    return output_root / str(comparison["comparison_study_name"])


def provenance_artifacts(tasks: tuple[TickerTask, ...]) -> dict[str, Path]:
    """Return all calibration/test targets and window panels that must be frozen."""
    artifacts: dict[str, Path] = {}
    for task in tasks:
        prefix = task.symbol.lower()
        artifacts[f"{prefix}_calibration_target"] = task.calibration_target
        artifacts[f"{prefix}_calibration_windows"] = task.calibration_windows
        artifacts[f"{prefix}_test_target"] = task.test_target
        artifacts[f"{prefix}_test_windows"] = task.test_windows
    return artifacts
