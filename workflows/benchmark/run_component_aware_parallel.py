#!/usr/bin/env python3
"""Run ticker-isolated paper calibrations concurrently on the Xeon host."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

from workflows.benchmark.audit_component_stage import METHODS
from workflows.benchmark.parallel_execution import (
    apply_cpu_affinity,
    format_cpu_set,
    load_execution_config,
)
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


def _write_execution_summary(directory: Path) -> Path:
    """Summarize measured launcher time across resumable invocations."""
    records = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "summary.json":
            continue
        value = json.loads(path.read_text())
        # The complete-workflow journal intentionally shares this directory
        # with launcher records. Summarize only records owned by this launcher;
        # other JSON artifacts have different schemas and no ``mode`` field.
        if value.get("version") != "component_aware_parallel_execution_v1":
            continue
        records.append({
            "path": str(path.resolve()),
            "mode": value["mode"],
            "status": value["status"],
            "started_at_utc": value["started_at_utc"],
            "finished_at_utc": value["finished_at_utc"],
            "wall_seconds": float(value["wall_seconds"]),
            "methods": value.get("methods", []),
        })
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in sorted({record["mode"] for record in records}):
        selected = [record for record in records if record["mode"] == mode]
        by_mode[mode] = {
            "invocations": len(selected),
            "completed_invocations": sum(
                record["status"] == "complete" for record in selected
            ),
            "cumulative_wall_seconds": sum(
                record["wall_seconds"] for record in selected
            ),
        }
    output = directory / "summary.json"
    _atomic_json(output, {
        "version": "component_aware_execution_summary_v1",
        "note": (
            "Cumulative wall time sums resumable launcher invocations; the "
            "individual records remain authoritative."
        ),
        "by_mode": by_mode,
        "records": records,
    })
    return output


def build_ticker_command(
    *,
    comparison_path: Path,
    ticker: str,
    methods: tuple[str, ...],
    repetitions: tuple[int, ...],
    stage_first_adaptive_slot: bool,
    provenance: Path | None,
    schema_preflight: Path | None,
    simulation_workers: int,
    cpu_affinity: tuple[int, ...],
) -> list[str]:
    """Construct one isolated child command without shell interpolation."""
    command = [
        sys.executable,
        "-m",
        "workflows.benchmark.run_component_aware",
        "--comparison-config",
        str(comparison_path),
        "--ticker",
        ticker,
        "--simulation-workers",
        str(simulation_workers),
        "--cpu-set",
        format_cpu_set(cpu_affinity),
    ]
    if stage_first_adaptive_slot:
        command.append("--stage-first-adaptive-slot")
    for method in methods:
        command.extend(("--method", method))
    for repetition in repetitions:
        command.extend(("--repetition", str(repetition)))
    if provenance is not None:
        command.extend(("--provenance", str(provenance)))
    if schema_preflight is not None:
        command.extend(("--schema-preflight", str(schema_preflight)))
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
        default=Path("configs/paper/component-aware-comparison.yaml"),
    )
    parser.add_argument(
        "--execution-config",
        type=Path,
        default=Path("configs/paper/xeon-e2146g-execution.yaml"),
    )
    parser.add_argument("--ticker", action="append")
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--repetition", action="append", type=int)
    parser.add_argument("--stage-first-adaptive-slot", action="store_true")
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--schema-preflight", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    comparison_path = args.comparison_config.resolve()
    execution_path = args.execution_config.resolve()
    execution_config = load_execution_config(execution_path)
    execution = execution_config["execution"]
    selected_tickers = tuple(
        str(ticker).upper() for ticker in (args.ticker or TICKERS)
    )
    unknown = sorted(set(selected_tickers) - set(TICKERS))
    if unknown or len(selected_tickers) != len(set(selected_tickers)):
        parser.error("Ticker filters must be unique members of the paper ticker set")
    if args.stage_first_adaptive_slot and args.repetition:
        parser.error("The first-adaptive-slot stage does not accept repetitions")

    optimizer_path, _, _, comparison, optimizer, _, _ = experiment_paths(
        comparison_path
    )
    root = comparison_output_root(comparison, optimizer_path, optimizer)
    methods = tuple(args.method or ())
    repetitions = tuple(args.repetition or ())
    affinity = tuple(int(cpu) for cpu in execution["cpu_affinity"])
    simulation_workers = int(execution["simulation_workers_per_ticker"])
    ticker_workers = min(int(execution["ticker_workers"]), len(selected_tickers))
    mode = "stage" if args.stage_first_adaptive_slot else "calibration"
    method_label = "all" if not methods else "-".join(methods)
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
            provenance=args.provenance.resolve() if args.provenance else None,
            schema_preflight=(
                args.schema_preflight.resolve() if args.schema_preflight else None
            ),
            simulation_workers=simulation_workers,
            cpu_affinity=affinity,
        )
        logs[ticker] = (
            root / ticker / "calibration" / "report" / "execution"
            / f"{timestamp}_{mode}_{method_label}.log"
        )

    if args.dry_run:
        print(json.dumps({
            "execution_config": str(execution_path),
            "ticker_workers": ticker_workers,
            "simulation_workers_per_ticker": simulation_workers,
            "cpu_affinity": list(affinity),
            "reserved_logical_cpus": execution["reserved_logical_cpus"],
            "commands": commands,
            "simulator_evaluations": 0,
        }, indent=2, sort_keys=True))
        return

    try:
        apply_cpu_affinity(affinity)
    except RuntimeError as exc:
        parser.error(str(exc))

    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    print(
        f"Launching {len(selected_tickers)} ticker tasks with {ticker_workers} "
        f"concurrent ticker workers on CPUs {format_cpu_set(affinity)}; "
        f"CPU {format_cpu_set(tuple(execution['reserved_logical_cpus']))} is reserved."
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
                f"{ticker}: {state} in {row['wall_seconds'] / 3600:.2f} h; "
                f"log={row['log']}"
            )

    finished_at = datetime.now(timezone.utc)
    failed = sorted(
        ticker for ticker, row in rows.items() if row["returncode"] != 0
    )
    record = {
        "version": "component_aware_parallel_execution_v1",
        "status": "failed" if failed else "complete",
        "mode": mode,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "wall_seconds": time.monotonic() - started,
        "comparison_config": str(comparison_path),
        "execution_config": str(execution_path),
        "execution_config_sha256": hashlib.sha256(
            execution_path.read_bytes()
        ).hexdigest(),
        "declared_machine": execution_config["machine"],
        "observed_host": {
            "node": platform.node(),
            "machine": platform.machine(),
            "logical_cpus_visible": os.cpu_count(),
        },
        "ticker_workers": ticker_workers,
        "simulation_workers_per_ticker": simulation_workers,
        "cpu_affinity": list(affinity),
        "reserved_logical_cpus": execution["reserved_logical_cpus"],
        "numerical_library_threads": execution["numerical_library_threads"],
        "methods": list(methods or METHODS),
        "repetitions": list(
            repetitions or range(int(optimizer["optimizer_repetitions"]))
        ),
        "tickers": {ticker: rows[ticker] for ticker in sorted(rows)},
        "failed_tickers": failed,
    }
    record_path = (
        root / "report" / "execution"
        / f"{timestamp}_{mode}_{method_label}.json"
    )
    _atomic_json(record_path, record)
    summary_path = _write_execution_summary(record_path.parent)
    print(
        f"Recorded measured wall time ({record['wall_seconds'] / 3600:.2f} h) "
        f"in {record_path.resolve()}"
    )
    print(f"Updated cumulative execution summary at {summary_path.resolve()}")
    if failed:
        raise SystemExit("Ticker tasks failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
