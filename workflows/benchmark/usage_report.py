#!/usr/bin/env python3
"""Aggregate OPRO proposal-runtime usage across ticker-specific calibrations.

Provider-neutral: the runtime/protocol label and cost-source description are
derived from each ledger record's own ``proposal_metadata`` rather than
hardcoded, so a study using a non-OpenRouter proposal runtime (e.g. Azure AI
Foundry) is reported under its own protocol, not mislabeled as OpenRouter.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def _sum_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "input_tokens", "cache_read_tokens", "cache_write_tokens",
        "output_tokens", "reasoning_tokens", "total_tokens", "reported_cost_usd",
    )
    usage = []
    request_ids: set[str] = set()
    transport_attempts = 0
    transient_retries = 0
    transport_statuses: Counter[str] = Counter()
    for record in records:
        metadata = record.get("proposal_metadata", {})
        if metadata.get("model_usage"):
            usage.append(metadata["model_usage"])
        if metadata.get("model_request_id"):
            request_ids.add(str(metadata["model_request_id"]))
        attempts = metadata.get("transport_attempts", [])
        if isinstance(attempts, list):
            transport_attempts += len(attempts)
            transient_retries += max(0, len(attempts) - 1)
            for attempt in attempts:
                if isinstance(attempt, dict):
                    status = attempt.get("status")
                    transport_statuses[
                        f"HTTP {status}" if status is not None
                        else str(attempt.get("error", "transport_error"))
                    ] += 1
    values = {key: sum(float(item.get(key, 0)) for item in usage) for key in fields}
    return {
        "model_calls": len(request_ids),
        "transport_attempts": transport_attempts,
        "transient_retries": transient_retries,
        "transport_statuses": dict(transport_statuses),
        **{
            key: values[key] if key == "reported_cost_usd" else int(values[key])
            for key in fields
        },
    }


def summarize(study_dir: Path) -> dict[str, Any]:
    tickers: dict[str, Any] = {}
    model_keys: set[str] = set()
    requested_models: set[str] = set()
    resolved_models: set[str] = set()
    providers: set[str] = set()
    protocols: set[str] = set()
    cost_sources: set[str] = set()
    for calibration_dir in sorted(study_dir.glob("*/calibration")):
        methods: dict[str, Any] = {}
        for method in ("opro_plain", "opro_heuristic"):
            records = [
                json.loads(path.read_text())
                for path in sorted((calibration_dir / "policies").glob(
                    f"{method}_repeat*/trials/trial_*.json"
                ))
            ]
            for record in records:
                metadata = record.get("proposal_metadata", {})
                for target, key in (
                    (model_keys, "model_key"),
                    (requested_models, "model_requested"),
                    (resolved_models, "model_resolved"),
                    (providers, "model_provider"),
                ):
                    if metadata.get(key):
                        target.add(str(metadata[key]))
                identity = metadata.get("model_runtime_identity", {})
                if identity.get("protocol"):
                    protocols.add(str(identity["protocol"]))
                usage_cost_source = metadata.get("model_usage", {}).get("cost_source")
                if usage_cost_source:
                    cost_sources.add(str(usage_cost_source))
            methods[method] = _sum_usage(records)
        tickers[calibration_dir.parent.name] = methods
    if not tickers:
        raise RuntimeError("No ticker calibration directories were found")
    if len(model_keys) != 1 or len(requested_models) != 1:
        raise RuntimeError("Expected exactly one configured OPRO model profile")
    if len(protocols) != 1:
        raise RuntimeError(
            "Expected exactly one proposal-runtime protocol across all records: "
            f"found {sorted(protocols)}"
        )
    # Frozen v9 OpenRouter records never populated a per-completion
    # cost_source field (reported cost came directly from OpenRouter's
    # usage.cost with no adapter-recorded provenance string); newer adapters
    # do populate it. Prefer the recorded value when present, but do not
    # require it -- that would be a false requirement on the immutable v9
    # adapter's historical record shape.
    cost_source_label = (
        f"sum of usage.cost fields (source: {next(iter(cost_sources))})"
        if len(cost_sources) == 1
        else f"sum of usage.cost fields (multiple sources: {sorted(cost_sources)})"
        if cost_sources
        else "sum of usage.cost fields (source not recorded per-completion)"
    )
    fields = (
        "model_calls", "transport_attempts", "transient_retries", "input_tokens",
        "cache_read_tokens", "cache_write_tokens", "output_tokens",
        "reasoning_tokens", "total_tokens", "reported_cost_usd",
    )
    totals = {
        key: sum(
            float(method[key])
            for methods in tickers.values() for method in methods.values()
        )
        for key in fields
    }
    return {
        "scope": "five ticker-specific optimizer studies; zero-slot preflight excluded",
        "runtime": next(iter(protocols)),
        "model_key": next(iter(model_keys)),
        "requested_model": next(iter(requested_models)),
        "resolved_models": sorted(resolved_models),
        "providers": sorted(providers),
        "tickers": tickers,
        "totals": {
            key: value if key == "reported_cost_usd" else int(value)
            for key, value in totals.items()
        },
        "cost_source": cost_source_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=Path(
            "results/optimization/icaif_smm_v9_ticker_specific_10x30m"
        ),
    )
    args = parser.parse_args()
    report = summarize(args.study_dir.resolve())
    output = args.study_dir.resolve() / "report" / "opro_model_usage.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"OPRO model usage written to: {output}")


if __name__ == "__main__":
    main()
