#!/usr/bin/env python3
"""Plan or run the reproduction workflow for the published comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from workflows import REPO_ROOT
from workflows.benchmark.published_summary import (
    BASELINE_METHODS,
    DEFAULT_BASELINE_ROOT,
    DEFAULT_OPRO_ROOT,
    DEFAULT_OUTPUT_DIR,
)
from workflows.benchmark.ticker_books import TICKERS


BASELINE_CONFIG = Path("configs/paper/component-aware-comparison.yaml")
REFERENCE_EXECUTION_CONFIG = Path("configs/paper/xeon-e2146g-execution.yaml")
OPRO_CONFIG = Path("configs/experiments/followup_foundry_sonnet5/comparison.yaml")
OPRO_RUNTIME_CONFIG = Path(
    "configs/experiments/followup_foundry_sonnet5/anthropic_foundry.yaml"
)
OPRO_MODEL_REGISTRY = Path(
    "configs/experiments/followup_foundry_sonnet5/models.yaml"
)
OPRO_PREFLIGHT = (
    DEFAULT_OPRO_ROOT / "report" / "anthropic_foundry_schema_preflight.json"
)
PROVENANCE = DEFAULT_BASELINE_ROOT / "report" / "reconstruction_provenance.json"
CALIBRATION_CONFIG = Path("configs/paper/optimizer.yaml")
PHASES = ("plan", "check", "prepare", "baselines", "opro", "evaluate", "summarize", "all")


def _module(name: str, *arguments: str | Path) -> list[str]:
    return [sys.executable, "-m", name, *(str(value) for value in arguments)]


def _methods(arguments: list[str], methods: tuple[str, ...]) -> list[str]:
    for method in methods:
        arguments.extend(("--method", method))
    return arguments


def build_plan(
    *,
    mode: str = "portable",
    baseline_simulation_workers: int = 3,
    opro_simulation_workers: int = 2,
    opro_ticker_workers: int = 3,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, list[dict[str, Any]]]:
    """Construct every command without reading data, creating files, or calling APIs."""
    if mode not in {"portable", "reference"}:
        raise ValueError("mode must be 'portable' or 'reference'")
    if baseline_simulation_workers < 1:
        raise ValueError("baseline_simulation_workers must be positive")
    if opro_simulation_workers < 1:
        raise ValueError("opro_simulation_workers must be positive")
    if opro_ticker_workers < 1:
        raise ValueError("opro_ticker_workers must be positive")

    baseline_dry_run = _module(
        "workflows.benchmark.run_component_aware",
        "--comparison-config", BASELINE_CONFIG, "--dry-run",
    )
    opro_dry_run = _module(
        "workflows.benchmark.opro_pair_followup",
        "--comparison-config", OPRO_CONFIG, "--dry-run",
    )
    plan: dict[str, list[dict[str, Any]]] = {
        "check": [
            {
                "name": "validate primary environment",
                "command": _module("workflows.validation.environment"),
            },
            {
                "name": "validate component-aware environment",
                "command": _module(
                    "workflows.validation.environment",
                    "--lock", "configs/paper/component-aware-environment-lock.yaml",
                ),
            },
            {
                "name": "validate heuristic registry",
                "command": _module("workflows.validation.heuristics"),
            },
            {
                "name": "offline simulator and OPRO smoke",
                "command": _module("workflows.validation.smoke"),
            },
            {"name": "test suite", "command": _module("pytest", "-q")},
            {"name": "baseline dry run", "command": baseline_dry_run},
            {"name": "OPRO follow-up dry run", "command": opro_dry_run},
        ],
        "prepare": [
            {
                "name": "materialize Calibration and Test Books",
                "command": _module(
                    "workflows.benchmark.run_component_aware",
                    "--comparison-config", BASELINE_CONFIG, "--prepare-books",
                ),
            },
            {
                "name": "freeze reconstruction provenance",
                "condition": f"run only when {PROVENANCE} is absent",
                "command": _module(
                    "workflows.benchmark.run_component_aware",
                    "--comparison-config", BASELINE_CONFIG, "--freeze-provenance",
                ),
            },
        ],
        "baselines": [],
        "opro": [],
        "evaluate": [],
        "summarize": [
            {
                "name": "combine all eight methods",
                "command": _module(
                    "workflows.benchmark.published_summary",
                    "--baseline-root", DEFAULT_BASELINE_ROOT,
                    "--opro-root", DEFAULT_OPRO_ROOT,
                    "--output-dir", output_dir,
                ),
            }
        ],
    }

    if mode == "reference":
        launcher = "workflows.benchmark.run_component_aware_parallel"
        common = [
            "--comparison-config", BASELINE_CONFIG,
            "--execution-config", REFERENCE_EXECUTION_CONFIG,
        ]
        for method in BASELINE_METHODS:
            plan["baselines"].append({
                "name": f"stage {method}",
                "condition": "run unless current stage audits exist for all five tickers",
                "command": _module(
                    launcher, *common, "--method", method,
                    "--stage-first-adaptive-slot",
                ),
            })
        plan["baselines"].append({
            "name": "complete six numerical baselines",
            "command": _methods(_module(launcher, *common), BASELINE_METHODS),
        })
    else:
        common = [
            "--comparison-config", BASELINE_CONFIG,
            "--simulation-workers", str(baseline_simulation_workers),
        ]
        for method in BASELINE_METHODS:
            plan["baselines"].append({
                "name": f"stage {method}",
                "condition": "run unless current stage audits exist for all five tickers",
                "command": _module(
                    "workflows.benchmark.run_component_aware", *common,
                    "--method", method, "--stage-first-adaptive-slot",
                ),
            })
        plan["baselines"].append({
            "name": "complete six numerical baselines",
            "command": _methods(
                _module("workflows.benchmark.run_component_aware", *common),
                BASELINE_METHODS,
            ),
        })

    plan["opro"].append({
        "name": "live maximum-history Foundry preflight",
        "condition": f"run only when {OPRO_PREFLIGHT} is absent",
        "command": _module(
            "workflows.benchmark.preflight_proposal_runtime",
            "--config", "configs/paper/optimizer.yaml",
            "--runtime-config", OPRO_RUNTIME_CONFIG,
            "--model-registry", OPRO_MODEL_REGISTRY,
            "--runtime-kind", "anthropic_foundry",
            "--model-key", "claude_sonnet5_foundry_v1",
            "--output", OPRO_PREFLIGHT,
        ),
    })
    opro_launcher = "workflows.benchmark.opro_pair_followup_parallel"
    opro_common = [
        "--comparison-config", OPRO_CONFIG,
        "--ticker-workers", str(opro_ticker_workers),
        "--simulation-workers", str(opro_simulation_workers),
    ]
    plan["opro"].extend((
        {
            "name": "stage OPRO pair",
            "condition": "run unless current pair-stage audits exist for all five tickers",
            "command": _module(
                opro_launcher, *opro_common, "--stage-first-adaptive-slot",
            ),
        },
        {
            "name": "complete OPRO pair calibration",
            "command": _module(opro_launcher, *opro_common),
        },
        {
            "name": "write OPRO calibration reports",
            "command": _module(
                "workflows.benchmark.reproduce_published_study",
                "--internal-report-opro",
            ),
        },
    ))
    plan["evaluate"] = [
        {
            "name": "evaluate numerical incumbents on Test Books",
            "command": _module(
                "workflows.benchmark.reproduce_published_study",
                "--internal-evaluate-baselines",
                "--baseline-simulation-workers", str(baseline_simulation_workers),
            ),
        },
        {
            "name": "evaluate OPRO incumbents on Test Books",
            "command": _module(
                "workflows.benchmark.opro_pair_followup",
                "--comparison-config", OPRO_CONFIG,
                "--evaluate-test-books",
                "--simulation-workers", str(opro_simulation_workers),
            ),
        },
        {
            "name": "summarize OPRO usage and recorded cost",
            "command": _module(
                "workflows.benchmark.usage_report", "--study-dir", DEFAULT_OPRO_ROOT,
            ),
        },
    ]
    return plan


def _display_plan(plan: dict[str, list[dict[str, Any]]]) -> None:
    payload = {
        phase: [
            {
                **({"condition": step["condition"]} if "condition" in step else {}),
                "name": step["name"],
                "command": shlex.join(step["command"]),
            }
            for step in steps
        ]
        for phase, steps in plan.items()
    }
    print(json.dumps(payload, indent=2))


def _sha256(path: Path) -> str:
    return hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()


def _json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _baseline_stage_is_current(method: str) -> bool:
    comparison_sha256 = _sha256(BASELINE_CONFIG)
    optimizer_sha256 = _sha256(CALIBRATION_CONFIG)
    for ticker in TICKERS:
        path = (
            REPO_ROOT / DEFAULT_BASELINE_ROOT / ticker / "calibration" / "report"
            / f"component_aware_stage_audit_{method}.json"
        )
        value = _json_object(path)
        if not value or not (
            value.get("passed") is True
            and value.get("ticker") == ticker
            and value.get("method") == method
            and value.get("comparison_config_sha256") == comparison_sha256
            and value.get("optimizer_config_sha256") == optimizer_sha256
        ):
            return False
    return True


def _opro_stage_is_current() -> bool:
    comparison_sha256 = _sha256(OPRO_CONFIG)
    calibration_sha256 = _sha256(CALIBRATION_CONFIG)
    for ticker in TICKERS:
        path = (
            REPO_ROOT / DEFAULT_OPRO_ROOT / ticker / "calibration" / "report"
            / "opro_pair_stage_audit.json"
        )
        value = _json_object(path)
        if not value or not (
            value.get("passed") is True
            and not value.get("failed_checks")
            and value.get("ticker") == ticker
            and tuple(value.get("methods", ())) == ("opro_plain", "opro_heuristic")
            and value.get("comparison_config_sha256") == comparison_sha256
            and value.get("calibration_config_sha256") == calibration_sha256
        ):
            return False
    return True


def _run_steps(steps: list[dict[str, Any]]) -> None:
    for step in steps:
        if (
            step["name"] == "freeze reconstruction provenance"
            and (REPO_ROOT / PROVENANCE).exists()
        ):
            print(f"Skipping existing provenance lock: {PROVENANCE}")
            continue
        if (
            step["name"] == "live maximum-history Foundry preflight"
            and (REPO_ROOT / OPRO_PREFLIGHT).exists()
        ):
            print(f"Skipping existing Foundry preflight: {OPRO_PREFLIGHT}")
            continue
        if step["name"].startswith("stage ") and step["name"] != "stage OPRO pair":
            method = step["name"].removeprefix("stage ")
            if _baseline_stage_is_current(method):
                print(f"Skipping current baseline stage audits: {method}")
                continue
        if step["name"] == "stage OPRO pair" and _opro_stage_is_current():
            print("Skipping current OPRO pair-stage audits")
            continue
        print(f"Running: {shlex.join(step['command'])}", flush=True)
        subprocess.check_call(step["command"], cwd=REPO_ROOT)


def _evaluate_baselines(simulation_workers: int) -> None:
    from workflows.benchmark.evaluate_component_aware import run_ticker_tests

    run_ticker_tests(
        (REPO_ROOT / BASELINE_CONFIG).resolve(),
        methods=BASELINE_METHODS,
        calibration_simulation_workers=simulation_workers,
    )


def _report_opro() -> None:
    from lob_sim.optimization.report import write_benchmark_report
    from workflows.benchmark.ticker_books import TICKERS

    for ticker in TICKERS:
        write_benchmark_report(
            REPO_ROOT / DEFAULT_OPRO_ROOT / ticker / "calibration"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, default="plan")
    parser.add_argument("--mode", choices=("portable", "reference"), default="portable")
    parser.add_argument("--baseline-simulation-workers", type=int, default=3)
    parser.add_argument("--opro-simulation-workers", type=int, default=2)
    parser.add_argument("--opro-ticker-workers", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--internal-evaluate-baselines", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--internal-report-opro", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.internal_evaluate_baselines:
        _evaluate_baselines(args.baseline_simulation_workers)
        return
    if args.internal_report_opro:
        _report_opro()
        return
    plan = build_plan(
        mode=args.mode,
        baseline_simulation_workers=args.baseline_simulation_workers,
        opro_simulation_workers=args.opro_simulation_workers,
        opro_ticker_workers=args.opro_ticker_workers,
        output_dir=args.output_dir,
    )
    if args.phase == "plan":
        _display_plan(plan)
        return
    phases = (
        ("check", "prepare", "baselines", "opro", "evaluate", "summarize")
        if args.phase == "all" else (args.phase,)
    )
    for phase in phases:
        _run_steps(plan[phase])


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
