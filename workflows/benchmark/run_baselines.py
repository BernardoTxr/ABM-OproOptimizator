#!/usr/bin/env python3
"""Run or resume the frozen SMM-v4 Random and TPE reference arms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from workflows import REPO_ROOT

from lob_sim.optimization.evaluator import TrialEvaluator
from lob_sim.optimization.report import write_benchmark_report
from lob_sim.optimization.study import run_study


def validate_contract(config: dict) -> dict:
    checks = {
        "study_name": config.get("study_name") == "icaif_smm_v4_development",
        "split": config.get("split") == "development",
        "loss": config.get("loss_config") == "loss.yaml",
        "replications": int(config.get("replications_per_trial", -1)) == 60,
        "evaluation_seed": int(config.get("evaluation_seed", -1)) == 62700,
        "budget": int(config.get("trial_budget", -1)) == 50,
        "repetitions": int(config.get("optimizer_repetitions", -1)) == 3,
        "initial_design": len(config.get("initial_design", [])) == 12,
        "summary_only": config.get("simulation_output_retention") == "summary_only",
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError("Frozen reference contract failed: " + ", ".join(failed))
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/paper/optimizer.yaml"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    checks = validate_contract(config)
    evaluator = TrialEvaluator(str(config_path))
    plan = {
        "config": str(config_path),
        "study_dir": str(evaluator.study_dir),
        "contract": checks,
        "evaluator_fingerprint": evaluator.evaluator_fingerprint,
        "methods": ["random", "tpe"],
        "repetitions": int(config["optimizer_repetitions"]),
        "budget_per_repetition": int(config["trial_budget"]),
    }
    print(json.dumps(plan, indent=2, sort_keys=True))
    if args.dry_run:
        return
    for method in ("random", "tpe"):
        for repetition in range(int(config["optimizer_repetitions"])):
            print(f"Starting/resuming {method} repetition {repetition}", flush=True)
            run_study(str(config_path), method, repetition)
    write_benchmark_report(evaluator.study_dir)
    print("Random and TPE reference benchmark complete.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
