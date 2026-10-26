#!/usr/bin/env python3
"""Score falsifiable OPRO metric-direction predictions from trial ledgers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


OPRO_METHODS = ("opro_plain", "opro_heuristic")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _metric(record: dict[str, Any], name: str) -> tuple[float | None, str | None]:
    evaluation = record.get("evaluation", {})
    for source_key, source_label in (
        ("component_losses", "component_loss"),
        ("signed_diagnostics", "signed_diagnostic"),
    ):
        source = evaluation.get(source_key, {})
        value = source.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), source_label
    return None, None


def _reference_record(
    records: dict[int, dict[str, Any]], evidence_ids: list[int]
) -> dict[str, Any] | None:
    eligible = [
        records[trial_id]
        for trial_id in evidence_ids
        if trial_id in records and records[trial_id].get("objective") is not None
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda record: (
            float(record["objective"]), int(record["policy_trial"])
        ),
    )


def _visible_prompt_trials(metadata: dict[str, Any]) -> set[int] | None:
    """Return trial IDs actually supplied to the model, or None if unauditable."""
    prompt = metadata.get("prompt")
    if not isinstance(prompt, str):
        return None
    try:
        payload = json.loads(prompt)
    except json.JSONDecodeError:
        return None
    observations = payload.get("observations")
    if not isinstance(observations, list):
        return None
    visible: set[int] = set()
    for observation in observations:
        if not isinstance(observation, dict):
            return None
        trial_id = observation.get("trial_id")
        if isinstance(trial_id, bool) or not isinstance(trial_id, int):
            return None
        visible.add(trial_id)
    return visible


def score_run(
    trial_paths: list[Path], no_change_tolerance: float = 1e-3
) -> list[dict[str, Any]]:
    """Return one scored row per persisted prediction in a policy run."""
    records = {
        int(record["policy_trial"]): record
        for path in sorted(trial_paths)
        for record in [json.loads(path.read_text())]
    }
    rows: list[dict[str, Any]] = []
    for trial_id, record in sorted(records.items()):
        metadata = record.get("proposal_metadata", {})
        predictions = metadata.get("predictions", [])
        evidence_ids = [int(value) for value in metadata.get("evidence_trial_ids", [])]
        visible_trials = _visible_prompt_trials(metadata)
        hidden_evidence = (
            sorted(set(evidence_ids) - visible_trials)
            if visible_trials is not None else []
        )
        reference = _reference_record(records, evidence_ids)
        for prediction_index, prediction in enumerate(predictions):
            metric_name = str(prediction.get("metric", ""))
            direction = str(prediction.get("direction", ""))
            row: dict[str, Any] = {
                "policy_trial": trial_id,
                "prediction_index": prediction_index,
                "metric": metric_name,
                "direction": direction,
                "rationale": prediction.get("rationale"),
                "evidence_trial_ids": evidence_ids,
                "visible_prompt_trial_ids": (
                    sorted(visible_trials) if visible_trials is not None else None
                ),
                "hidden_evidence_trial_ids": hidden_evidence,
                "evidence_visible_in_prompt": (
                    not hidden_evidence if visible_trials is not None else None
                ),
                "reference_trial": (
                    int(reference["policy_trial"]) if reference is not None else None
                ),
                "reference_value": None,
                "observed_value": None,
                "observed_change": None,
                "metric_source": None,
                "scorable": False,
                "correct": None,
                "unscorable_reason": None,
            }
            if visible_trials is None:
                row["unscorable_reason"] = "persisted_prompt_missing_or_invalid"
                rows.append(row)
                continue
            if hidden_evidence:
                row["unscorable_reason"] = "evidence_trial_not_visible_in_prompt"
                rows.append(row)
                continue
            if reference is None:
                row["unscorable_reason"] = "no_valid_evidence_trial"
                rows.append(row)
                continue
            reference_value, reference_source = _metric(reference, metric_name)
            observed_value, observed_source = _metric(record, metric_name)
            if reference_value is None or observed_value is None:
                row["unscorable_reason"] = "metric_missing_from_evaluation"
                rows.append(row)
                continue
            if reference_source != observed_source:
                row["unscorable_reason"] = "metric_source_mismatch"
                rows.append(row)
                continue
            change = observed_value - reference_value
            if direction == "increase":
                correct = change > 0
            elif direction == "decrease":
                correct = change < 0
            elif direction == "no_change":
                correct = abs(change) <= no_change_tolerance
            else:
                row["unscorable_reason"] = "invalid_direction"
                rows.append(row)
                continue
            row.update({
                "reference_value": reference_value,
                "observed_value": observed_value,
                "observed_change": change,
                "metric_source": reference_source,
                "scorable": True,
                "correct": bool(correct),
            })
            rows.append(row)
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scorable = [row for row in rows if row["scorable"]]
    correct = [row for row in scorable if row["correct"]]
    return {
        "predictions": len(rows),
        "scorable_predictions": len(scorable),
        "correct_predictions": len(correct),
        "scorable_fraction": len(scorable) / len(rows) if rows else None,
        "accuracy_conditional_on_scorable": (
            len(correct) / len(scorable) if scorable else None
        ),
        "evidence_protocol_violations": len({
            int(row["policy_trial"])
            for row in rows
            if row.get("hidden_evidence_trial_ids")
        }),
        "unauditable_prompt_trials": len({
            int(row["policy_trial"])
            for row in rows
            if row.get("visible_prompt_trial_ids") is None
        }),
    }


def score_study(study_dir: Path) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    runs: dict[str, Any] = {}
    for method in OPRO_METHODS:
        for run_dir in sorted((study_dir / "policies").glob(f"{method}_repeat*")):
            paths = sorted((run_dir / "trials").glob("trial_*.json"))
            if not paths:
                continue
            rows = score_run(paths)
            repetition = int(run_dir.name.rsplit("repeat", 1)[1])
            enriched = [
                {"method": method, "repetition": repetition, **row} for row in rows
            ]
            all_rows.extend(enriched)
            runs[run_dir.name] = _summary(rows)
    methods = {
        method: _summary([row for row in all_rows if row["method"] == method])
        for method in OPRO_METHODS
        if any(row["method"] == method for row in all_rows)
    }
    return {
        "reference_rule": (
            "lowest-objective valid cited evidence trial; earliest trial breaks ties"
        ),
        "evidence_visibility_rule": (
            "every cited evidence trial must occur in the observations of the exact "
            "persisted prompt; hidden-ledger citations are protocol violations"
        ),
        "increase_decrease_rule": "strict sign of observed metric change",
        "no_change_tolerance": 1e-3,
        "runs": runs,
        "methods": methods,
        "rows": all_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-dir", type=Path,
        default=Path("results/optimization/icaif_smm_v4_development"),
    )
    args = parser.parse_args()
    study_dir = args.study_dir.resolve()
    result = score_study(study_dir)
    output_dir = study_dir / "report"
    _atomic_json(output_dir / "opro_prediction_scores.json", result)
    flat_rows = []
    for row in result["rows"]:
        flat = dict(row)
        flat["evidence_trial_ids"] = json.dumps(flat["evidence_trial_ids"])
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(
        output_dir / "opro_prediction_scores.csv", index=False
    )
    print(f"OPRO prediction scores written to: {output_dir}")


if __name__ == "__main__":
    main()
