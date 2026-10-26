#!/usr/bin/env python3
"""Validate, prepare, run, test, and report the complete frozen experiment."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml

from workflows import REPO_ROOT
from workflows.benchmark.audit_component_stage import METHODS
from workflows.benchmark.preflight_openrouter import validate_openrouter_preflight
from workflows.benchmark.parallel_execution import (
    apply_cpu_affinity,
    load_execution_config,
)
from workflows.benchmark.run_component_aware import component_plan, method_stage_path
from workflows.benchmark.ticker_books import (
    TICKERS,
    build_ticker_tasks,
    comparison_output_root,
    experiment_paths,
    provenance_artifacts,
)
from workflows.validation.provenance import verify_provenance


DEFAULT_COMPARISON = REPO_ROOT / "configs/paper/component-aware-comparison.yaml"
DEFAULT_EXECUTION = REPO_ROOT / "configs/paper/xeon-e2146g-execution.yaml"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def assert_clean_tracked_tree(repo_root: Path = REPO_ROOT) -> None:
    """Refuse a scientific run when tracked source/config files are modified."""
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Could not inspect Git state: {completed.stderr.strip()}")
    changed = [line for line in completed.stdout.splitlines() if line.strip()]
    if changed:
        raise RuntimeError(
            "Tracked files are modified; commit or restore them before the paper "
            "experiment:\n" + "\n".join(changed)
        )


def stage_audit_is_current(
    path: Path,
    *,
    ticker: str,
    method: str,
    comparison_sha256: str,
    optimizer_sha256: str,
) -> bool:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        value.get("passed") is True
        and value.get("ticker") == ticker
        and value.get("method") == method
        and value.get("comparison_config_sha256") == comparison_sha256
        and value.get("optimizer_config_sha256") == optimizer_sha256
    )


def pending_stage_methods(
    plan: dict[str, Any],
) -> tuple[str, ...]:
    pending = []
    for method in METHODS:
        current_for_every_ticker = all(
            stage_audit_is_current(
                method_stage_path(
                    Path(plan["tickers"][ticker]["calibration_study_dir"]), method
                ),
                ticker=ticker,
                method=method,
                comparison_sha256=plan["comparison_config_sha256"],
                optimizer_sha256=plan["calibration_config_sha256"],
            )
            for ticker in TICKERS
        )
        if not current_for_every_ticker:
            pending.append(method)
    return tuple(pending)


class FullExperiment:
    def __init__(self, comparison_path: Path, execution_path: Path) -> None:
        self.comparison_path = comparison_path.resolve()
        self.execution_path = execution_path.resolve()
        self.execution = load_execution_config(self.execution_path)
        (
            self.optimizer_path,
            _,
            _,
            self.comparison,
            self.optimizer,
            _,
            _,
        ) = experiment_paths(self.comparison_path)
        self.root = comparison_output_root(
            self.comparison, self.optimizer_path, self.optimizer
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.journal_path = (
            self.root / "report/execution" / f"{timestamp}_full_experiment.json"
        )
        self.journal: dict[str, Any] = {
            "version": "full_component_experiment_v1",
            "status": "running",
            "started_at_utc": _utc_now(),
            "finished_at_utc": None,
            "comparison_config": str(self.comparison_path),
            "comparison_config_sha256": hashlib.sha256(
                self.comparison_path.read_bytes()
            ).hexdigest(),
            "study": self.comparison["comparison_study_name"],
            "llm_model_key": self.comparison["llm_model_key"],
            "execution_config": str(self.execution_path),
            "cpu_affinity": self.execution["execution"]["cpu_affinity"],
            "reserved_logical_cpus": self.execution["execution"][
                "reserved_logical_cpus"
            ],
            "steps": [],
        }

    def write_journal(self) -> None:
        _atomic_json(self.journal_path, self.journal)

    def run_module(self, label: str, module: str, *arguments: str) -> None:
        command = [sys.executable, "-m", module, *arguments]
        step = {
            "label": label,
            "command": command,
            "started_at_utc": _utc_now(),
            "finished_at_utc": None,
            "runtime_seconds": None,
            "returncode": None,
        }
        self.journal["steps"].append(step)
        self.write_journal()
        print(f"\n=== {label} ===", flush=True)
        started = time.monotonic()
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        step["runtime_seconds"] = time.monotonic() - started
        step["finished_at_utc"] = _utc_now()
        step["returncode"] = completed.returncode
        self.write_journal()
        if completed.returncode != 0:
            raise RuntimeError(
                f"Step failed ({label}) with exit code {completed.returncode}"
            )

    def ensure_preflight(self) -> None:
        preflight_path = self.root / "report/openrouter_schema_preflight.json"
        try:
            validate_openrouter_preflight(
                preflight_path,
                self.optimizer_path,
                self.optimizer,
                model_key_override=self.comparison["llm_model_key"],
            )
            print("\n=== OpenRouter preflight already valid; reusing it ===")
            return
        except RuntimeError:
            pass
        self.run_module(
            "OpenRouter Qwen schema preflight",
            "workflows.benchmark.preflight_openrouter",
            "--config", str(self.optimizer_path),
            "--model-key", str(self.comparison["llm_model_key"]),
            "--output", str(preflight_path),
        )
        validate_openrouter_preflight(
            preflight_path,
            self.optimizer_path,
            self.optimizer,
            model_key_override=self.comparison["llm_model_key"],
        )

    def ensure_provenance(self) -> None:
        provenance_path = self.root / "report/reconstruction_provenance.json"
        tasks = build_ticker_tasks(self.comparison, self.optimizer)
        if provenance_path.exists():
            verify_provenance(
                provenance_path,
                artifacts=provenance_artifacts(tasks),
            )
            print("\n=== Reconstruction provenance already valid; reusing it ===")
            return
        self.run_module(
            "freeze empirical provenance",
            "workflows.benchmark.run_component_aware",
            "--comparison-config", str(self.comparison_path),
            "--freeze-provenance",
        )

    def run(self) -> None:
        assert_clean_tracked_tree()
        apply_cpu_affinity(tuple(
            int(cpu) for cpu in self.execution["execution"]["cpu_affinity"]
        ))
        self.write_journal()
        try:
            self.run_module("repository tests", "pytest", "-q")
            self.run_module(
                "frozen protocol and environment validation",
                "workflows.benchmark.run_component_aware",
                "--comparison-config", str(self.comparison_path),
                "--dry-run",
            )
            self.ensure_preflight()
            self.run_module(
                "materialize Calibration and Test Books",
                "workflows.benchmark.run_component_aware",
                "--comparison-config", str(self.comparison_path),
                "--prepare-books",
            )
            self.ensure_provenance()

            plan = component_plan(self.comparison_path)
            for method in pending_stage_methods(plan):
                self.run_module(
                    f"first-adaptive-slot stage gate: {method}",
                    "workflows.benchmark.run_component_aware_parallel",
                    "--comparison-config", str(self.comparison_path),
                    "--stage-first-adaptive-slot",
                    "--method", method,
                )

            self.run_module(
                "complete all five-ticker, eight-method calibrations",
                "workflows.benchmark.run_component_aware_parallel",
                "--comparison-config", str(self.comparison_path),
            )
            self.run_module(
                "evaluate 40 selected candidates on Test Books",
                "workflows.benchmark.evaluate_component_aware",
                "--comparison-config", str(self.comparison_path),
            )
            self.run_module(
                "aggregate OpenRouter usage and cost",
                "workflows.benchmark.usage_report",
                "--study-dir", str(self.root),
            )
            for ticker in TICKERS:
                self.run_module(
                    f"score OPRO predictions: {ticker}",
                    "workflows.benchmark.score_predictions",
                    "--study-dir", str(self.root / ticker / "calibration"),
                )
        except Exception as exc:
            self.journal["status"] = "failed"
            self.journal["finished_at_utc"] = _utc_now()
            self.journal["error"] = f"{type(exc).__name__}: {exc}"
            self.write_journal()
            raise
        self.journal["status"] = "complete"
        self.journal["finished_at_utc"] = _utc_now()
        self.journal["error"] = None
        self.write_journal()
        print(
            "\nFULL EXPERIMENT COMPLETE\n"
            f"Study: {self.comparison['comparison_study_name']}\n"
            f"Model: {self.comparison['llm_model_key']}\n"
            f"Journal: {self.journal_path}",
            flush=True,
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-config", type=Path, default=DEFAULT_COMPARISON
    )
    parser.add_argument(
        "--execution-config", type=Path, default=DEFAULT_EXECUTION
    )
    args = parser.parse_args()
    try:
        FullExperiment(args.comparison_config, args.execution_config).run()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
