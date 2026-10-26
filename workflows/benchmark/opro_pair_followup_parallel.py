#!/usr/bin/env python3
"""Run the two-arm OPRO follow-up's ticker tasks concurrently as subprocesses.

Mirrors ``run_component_aware_parallel.py``'s subprocess-per-ticker pattern
(each ticker is a fully independent ``TrialEvaluator``/study_dir, so there is
no shared mutable state to race on), but does not reuse
``parallel_execution.load_execution_config`` -- that loader is hard-locked to
the frozen v9 ``xeon_e2146g_parallel_v1`` machine contract via its `version`
check, which does not describe this host. CPU affinity pinning is
deliberately skipped here too: it existed in the v9 launcher for benchmark
-timing reproducibility on a dedicated paper host, not as a functional
requirement, and this host is shared with other work.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from workflows.benchmark.opro_pair_followup import DECLARED_METHODS
from workflows.benchmark.ticker_books import (
    TICKERS,
    comparison_output_root,
    experiment_paths,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_ticker_command(
    *,
    comparison_path: Path,
    ticker: str,
    methods: tuple[str, ...],
    repetitions: tuple[int, ...],
    stage_first_adaptive_slot: bool,
    simulation_workers: int,
) -> list[str]:
    """Construct one isolated child command without shell interpolation."""
    command = [
        sys.executable, "-m", "workflows.benchmark.opro_pair_followup",
        "--comparison-config", str(comparison_path),
        "--ticker", ticker,
        "--simulation-workers", str(simulation_workers),
    ]
    if stage_first_adaptive_slot:
        command.append("--stage-first-adaptive-slot")
    for method in methods:
        command.extend(("--method", method))
    for repetition in repetitions:
        command.extend(("--repetition", str(repetition)))
    return command


def _run_ticker(command: list[str], ticker: str, log_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    })
    with log_path.open("a", buffering=1) as log:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "ticker": ticker,
        "returncode": completed.returncode,
        "wall_seconds": time.monotonic() - started,
        "log": str(log_path.resolve()),
        "command": command,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-config",
        type=Path,
        default=Path("configs/experiments/followup_foundry_sonnet5/comparison.yaml"),
    )
    parser.add_argument("--ticker", action="append")
    parser.add_argument("--method", action="append", choices=DECLARED_METHODS)
    parser.add_argument("--repetition", action="append", type=int)
    parser.add_argument("--stage-first-adaptive-slot", action="store_true")
    parser.add_argument(
        "--ticker-workers", type=int, default=3,
        help="Number of tickers to run as concurrent subprocesses.",
    )
    parser.add_argument(
        "--simulation-workers", type=int, default=2,
        help="Per-ticker scenario worker count, passed through to each subprocess.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    comparison_path = args.comparison_config.resolve()
    selected_tickers = tuple(str(ticker).upper() for ticker in (args.ticker or TICKERS))
    unknown = sorted(set(selected_tickers) - set(TICKERS))
    if unknown or len(selected_tickers) != len(set(selected_tickers)):
        parser.error("Ticker filters must be unique members of the fixed ticker set")
    if args.ticker_workers < 1:
        parser.error("--ticker-workers must be positive")
    if args.simulation_workers < 1:
        parser.error("--simulation-workers must be positive")
    if args.stage_first_adaptive_slot and args.repetition:
        parser.error("The first-adaptive-slot stage does not accept repetitions")

    (calibration_path, _, _, comparison, calibration, _, _) = experiment_paths(comparison_path)
    root = comparison_output_root(comparison, calibration_path, calibration)
    methods = tuple(args.method or ())
    repetitions = tuple(args.repetition or ())
    ticker_workers = min(args.ticker_workers, len(selected_tickers))
    mode = "stage" if args.stage_first_adaptive_slot else "calibration"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    commands: dict[str, list[str]] = {}
    logs: dict[str, Path] = {}
    for ticker in selected_tickers:
        commands[ticker] = build_ticker_command(
            comparison_path=comparison_path,
            ticker=ticker,
            methods=methods,
            repetitions=repetitions,
            stage_first_adaptive_slot=args.stage_first_adaptive_slot,
            simulation_workers=args.simulation_workers,
        )
        logs[ticker] = (
            root / ticker / "calibration" / "report" / "execution"
            / f"{timestamp}_{mode}.log"
        )

    if args.dry_run:
        print(json.dumps({
            "ticker_workers": ticker_workers,
            "simulation_workers_per_ticker": args.simulation_workers,
            "commands": commands,
            "simulator_evaluations": 0,
        }, indent=2, sort_keys=True))
        return

    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    print(
        f"Launching {len(selected_tickers)} ticker tasks with {ticker_workers} "
        f"concurrent ticker workers x {args.simulation_workers} simulation "
        "workers each."
    )
    rows: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=ticker_workers) as executor:
        futures = {
            executor.submit(_run_ticker, commands[ticker], ticker, logs[ticker]): ticker
            for ticker in selected_tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            row = future.result()
            rows[ticker] = row
            state = "complete" if row["returncode"] == 0 else "failed"
            print(
                f"{ticker}: {state} in {row['wall_seconds'] / 60:.1f} min; "
                f"log={row['log']}"
            )

    finished_at = datetime.now(timezone.utc)
    failed = sorted(ticker for ticker, row in rows.items() if row["returncode"] != 0)
    record = {
        "version": "opro_pair_followup_parallel_execution_v1",
        "status": "failed" if failed else "complete",
        "mode": mode,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "wall_seconds": time.monotonic() - started,
        "comparison_config": str(comparison_path),
        "comparison_config_sha256": hashlib.sha256(comparison_path.read_bytes()).hexdigest(),
        "ticker_workers": ticker_workers,
        "simulation_workers_per_ticker": args.simulation_workers,
        "methods": list(methods or DECLARED_METHODS),
        "repetitions": list(repetitions or range(int(calibration["optimizer_repetitions"]))),
        "tickers": {ticker: rows[ticker] for ticker in sorted(rows)},
        "failed_tickers": failed,
    }
    record_path = root / "report" / "execution" / f"{timestamp}_{mode}.json"
    _atomic_json(record_path, record)
    print(f"Recorded execution summary at {record_path.resolve()}")
    if failed:
        raise SystemExit("Ticker tasks failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
