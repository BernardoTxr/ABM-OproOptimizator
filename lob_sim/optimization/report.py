"""Machine-readable tables and best-so-far plots for optimizer benchmarks."""
from __future__ import annotations

import json
import itertools
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _method_sort_key(name: str) -> tuple[int, str]:
    """Keep both OPRO variants at the end of comparative outputs."""
    return (1 if name in {"opro_plain", "opro_heuristic"} else 0, name)


def load_policy_runs(study_dir: str | Path) -> list[dict[str, Any]]:
    """Load completed policy ledgers into one record per repetition."""
    root = Path(study_dir)
    runs: list[dict[str, Any]] = []
    for run_dir in sorted((root / "policies").glob("*_repeat*")):
        run_summary = (
            json.loads((run_dir / "summary.json").read_text())
            if (run_dir / "summary.json").exists() else {}
        )
        trials = [
            json.loads(path.read_text())
            for path in sorted((run_dir / "trials").glob("trial_*.json"))
        ]
        if not trials:
            continue
        objectives = [trial.get("objective") for trial in trials]
        best = float("inf")
        trajectory: list[float | None] = []
        for objective in objectives:
            if objective is not None:
                best = min(best, float(objective))
            trajectory.append(best if np.isfinite(best) else None)
        ranked = [trial for trial in trials if trial.get("objective") is not None]
        best_trial = min(ranked, key=lambda trial: float(trial["objective"])) if ranked else None
        sampler = str(trials[0]["sampler"])
        evaluator_fingerprints = sorted({
            fingerprint
            for trial in trials
            if (fingerprint := trial.get("evaluation", {}).get(
                "evaluator_fingerprint"
            ))
        })
        feedback_modes = sorted({
            str(trial["policy_feedback_mode"])
            for trial in trials if trial.get("policy_feedback_mode") is not None
        })
        if len(feedback_modes) > 1:
            raise ValueError(f"Run {run_dir.name} mixes optimizer feedback modes")
        runs.append({
            "sampler": sampler,
            "symbol": trials[0].get("evaluation", {}).get("symbol"),
            "split": trials[0].get("evaluation", {}).get("split"),
            "policy_feedback_mode": feedback_modes[0] if feedback_modes else None,
            "repetition": int(trials[0]["repetition"]),
            "proposal_slots": len(trials),
            "configured_budget": run_summary.get("budget"),
            "valid_proposals": sum(trial.get("status") == "complete" for trial in trials),
            "invalid_or_error_proposals": sum(
                trial.get("status") != "complete" for trial in trials
            ),
            "duplicate_proposals": sum(bool(trial.get("duplicate")) for trial in trials),
            "best_objective": best_trial.get("objective") if best_trial else None,
            "best_policy_trial": best_trial.get("policy_trial") if best_trial else None,
            "best_candidate": best_trial.get("candidate") if best_trial else None,
            "best_component_losses": (
                best_trial.get("evaluation", {}).get("component_losses", {})
                if best_trial else {}
            ),
            "mean_best_so_far": (
                float(np.mean([value for value in trajectory if value is not None]))
                if any(value is not None for value in trajectory) else None
            ),
            "best_so_far": trajectory,
            "controller_runtime_seconds": float(sum(
                trial.get("controller_runtime_seconds", 0.0) for trial in trials
            )),
            "logical_simulator_runtime_seconds": float(sum(
                trial.get("evaluation", {}).get("runtime_seconds", 0.0)
                for trial in trials
            )),
            "model_calls": sum(
                bool(trial.get("proposal_metadata", {}).get("model_request_id"))
                for trial in trials
            ),
            "evaluator_fingerprints": evaluator_fingerprints,
        })
    return sorted(runs, key=lambda run: (
        _method_sort_key(run["sampler"]), run["repetition"]
    ))


def summarize_policy_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate repeated runs without treating proposal steps as replicates."""
    output: dict[str, Any] = {}
    for sampler in sorted({run["sampler"] for run in runs}, key=_method_sort_key):
        group = [run for run in runs if run["sampler"] == sampler]
        best = np.asarray([
            run["best_objective"] for run in group
            if run["best_objective"] is not None
        ], dtype=float)
        efficiency = np.asarray([
            run["mean_best_so_far"] for run in group
            if run["mean_best_so_far"] is not None
        ], dtype=float)
        output[sampler] = {
            "policy_feedback_modes": sorted({
                run["policy_feedback_mode"] for run in group
                if run.get("policy_feedback_mode") is not None
            }),
            "repetitions": len(group),
            "complete_repetitions": sum(
                run["configured_budget"] is not None
                and run["proposal_slots"] == run["configured_budget"]
                for run in group
            ),
            "proposal_slots": [run["proposal_slots"] for run in group],
            "best_objective_mean": float(best.mean()) if best.size else None,
            "best_objective_std": float(best.std(ddof=1)) if best.size > 1 else None,
            "mean_best_so_far_mean": float(efficiency.mean()) if efficiency.size else None,
            "mean_best_so_far_std": (
                float(efficiency.std(ddof=1)) if efficiency.size > 1 else None
            ),
            "invalid_or_error_proposals": sum(
                run["invalid_or_error_proposals"] for run in group
            ),
            "duplicate_proposals": sum(run["duplicate_proposals"] for run in group),
            "model_calls": sum(run["model_calls"] for run in group),
        }
    return output


def paired_policy_comparisons(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Descriptively compare methods on matching optimizer repetition ids."""
    methods = sorted({run["sampler"] for run in runs}, key=_method_sort_key)
    output: list[dict[str, Any]] = []
    for method_a, method_b in itertools.combinations(methods, 2):
        left = {
            run["repetition"]: run for run in runs if run["sampler"] == method_a
        }
        right = {
            run["repetition"]: run for run in runs if run["sampler"] == method_b
        }
        repetition_ids = sorted(set(left) & set(right))
        final_differences = []
        efficiency_differences = []
        for repetition in repetition_ids:
            if (
                left[repetition]["best_objective"] is not None
                and right[repetition]["best_objective"] is not None
            ):
                final_differences.append(
                    float(left[repetition]["best_objective"])
                    - float(right[repetition]["best_objective"])
                )
            if (
                left[repetition]["mean_best_so_far"] is not None
                and right[repetition]["mean_best_so_far"] is not None
            ):
                efficiency_differences.append(
                    float(left[repetition]["mean_best_so_far"])
                    - float(right[repetition]["mean_best_so_far"])
                )
        final = np.asarray(final_differences, dtype=float)
        efficiency = np.asarray(efficiency_differences, dtype=float)
        output.append({
            "method_a": method_a,
            "method_b": method_b,
            "matched_repetitions": repetition_ids,
            "final_loss_a_minus_b_mean": float(final.mean()) if final.size else None,
            "final_loss_a_minus_b_std": (
                float(final.std(ddof=1)) if final.size > 1 else None
            ),
            "method_a_final_wins": int(np.sum(final < 0)),
            "method_b_final_wins": int(np.sum(final > 0)),
            "mean_best_so_far_a_minus_b_mean": (
                float(efficiency.mean()) if efficiency.size else None
            ),
        })
    return output


def write_benchmark_report(study_dir: str | Path) -> tuple[dict[str, Any], Path]:
    """Write JSON/CSV summaries and a Calibration Book search figure."""
    root = Path(study_dir)
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    runs = load_policy_runs(root)
    fingerprints = sorted({
        fingerprint
        for run in runs
        for fingerprint in run.get("evaluator_fingerprints", [])
    })
    if len(fingerprints) > 1:
        raise ValueError(
            "Policy runs contain different evaluator fingerprints and cannot "
            "be compared in one benchmark"
        )
    symbols = sorted({str(run["symbol"]) for run in runs if run.get("symbol")})
    splits = sorted({str(run["split"]) for run in runs if run.get("split")})
    if len(symbols) > 1 or len(splits) > 1:
        raise ValueError("One benchmark report cannot pool ticker or book partitions")
    summary = {
        "study_directory": str(root.resolve()),
        "ticker": symbols[0] if len(symbols) == 1 else None,
        "book_partition": splits[0] if len(splits) == 1 else None,
        "unit_of_replication": "optimizer repetition",
        "efficiency_metric": "mean discrete best-so-far loss over proposal slots",
        "methods": summarize_policy_runs(runs),
        "paired_method_comparisons": paired_policy_comparisons(runs),
        "evaluator_fingerprints": fingerprints,
        "runs": runs,
    }
    (report_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    table_rows = [{
        key: value for key, value in run.items()
        if key not in {
            "best_candidate", "best_component_losses", "best_so_far",
            "evaluator_fingerprints",
        }
    } for run in runs]
    pd.DataFrame(table_rows).to_csv(report_dir / "benchmark_runs.csv", index=False)
    component_rows = [
        {
            "sampler": run["sampler"],
            "repetition": run["repetition"],
            "component": component,
            "loss": loss,
        }
        for run in runs
        for component, loss in run["best_component_losses"].items()
    ]
    pd.DataFrame(component_rows).to_csv(
        report_dir / "best_group_losses.csv", index=False
    )

    if runs:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        colors = {
            "random": "#7f7f7f", "tpe": "#1f77b4",
            "opro_plain": "#ff7f0e", "opro_heuristic": "#2ca02c",
            "sobol": "#cc79a7", "gp_bo": "#d55e00",
            "cma_es": "#56b4e9", "bads": "#332288",
            "mo_tpe": "#009e73", "component_gp_bo": "#aa4499",
        }
        for sampler in sorted(
            {run["sampler"] for run in runs}, key=_method_sort_key
        ):
            group = [run for run in runs if run["sampler"] == sampler]
            length = max(len(run["best_so_far"]) for run in group)
            matrix = np.full((len(group), length), np.nan)
            for row_index, run in enumerate(group):
                values = [
                    np.nan if value is None else value
                    for value in run["best_so_far"]
                ]
                matrix[row_index, :len(values)] = values
            x = np.arange(1, length + 1)
            color = colors.get(sampler)
            for row_index, row in enumerate(matrix):
                run_length = len(group[row_index]["best_so_far"])
                ax.step(
                    x[:run_length], row[:run_length], where="post",
                    color=color, alpha=0.18, linewidth=1,
                )
            mean = np.nanmean(matrix, axis=0)
            ax.step(x, mean, where="post", color=color, linewidth=2, label=sampler)
        # Infer the forced common-design prefix from ledger metadata instead
        # of hard-coding a study-specific slot count.
        prefixes = []
        for run_dir in sorted((root / "policies").glob("*_repeat*")):
            prefix = 0
            for path in sorted((run_dir / "trials").glob("trial_*.json")):
                trial = json.loads(path.read_text())
                if trial.get("proposal_metadata", {}).get("source") != "initial_design":
                    break
                prefix += 1
            if prefix:
                prefixes.append(prefix)
        if prefixes and len(set(prefixes)) == 1:
            common_slots = prefixes[0]
            boundary = common_slots + 0.5
            ax.axvspan(0.5, boundary, color="#d9d9d9", alpha=0.22, zorder=-1)
            ax.axvline(boundary, color="#666666", linestyle="--", linewidth=0.9)
            ax.text(
                (common_slots + 1) / 2,
                0.985,
                "common design",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
                color="#555555",
            )
        ax.set_xlabel("Proposal slot")
        ax.set_ylabel("Best Calibration Book loss")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(report_dir / "best_so_far.pdf", bbox_inches="tight")
        fig.savefig(report_dir / "best_so_far.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

        if component_rows:
            component_frame = pd.DataFrame(component_rows)
            components = list(dict.fromkeys(component_frame["component"]))
            samplers = sorted(
                component_frame["sampler"].unique(), key=_method_sort_key
            )
            positions = np.arange(len(components), dtype=float)
            width = 0.8 / len(samplers)
            fig, ax = plt.subplots(figsize=(7.2, 3.9))
            for index, sampler in enumerate(samplers):
                subset = component_frame[component_frame["sampler"] == sampler]
                means = [
                    subset.loc[subset["component"] == component, "loss"].mean()
                    for component in components
                ]
                offset = (index - (len(samplers) - 1) / 2) * width
                ax.bar(
                    positions + offset, means, width=width,
                    color=colors.get(sampler), label=sampler,
                )
            ax.set_xticks(positions, [name.replace("_", "\n") for name in components])
            ax.set_ylabel("Loss-group score of incumbent")
            ax.grid(axis="y", alpha=0.2)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(report_dir / "best_group_losses.pdf", bbox_inches="tight")
            fig.savefig(
                report_dir / "best_group_losses.png", dpi=180, bbox_inches="tight"
            )
            plt.close(fig)
    return summary, report_dir
