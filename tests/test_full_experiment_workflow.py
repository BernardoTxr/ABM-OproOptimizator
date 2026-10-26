import hashlib
import json
from pathlib import Path

import yaml

from workflows.benchmark.audit_component_stage import METHODS
from workflows.benchmark.run_full_experiment import (
    DEFAULT_COMPARISON,
    DEFAULT_EXECUTION,
    FullExperiment,
    pending_stage_methods,
    stage_audit_is_current,
)
from workflows.benchmark.ticker_books import TICKERS


def _write_stage_audit(
    path: Path,
    *,
    ticker: str,
    method: str,
    comparison_sha256: str,
    optimizer_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "passed": True,
        "ticker": ticker,
        "method": method,
        "comparison_config_sha256": comparison_sha256,
        "optimizer_config_sha256": optimizer_sha256,
    }))


def test_full_experiment_is_pinned_to_qwen_and_xeon_schedule() -> None:
    runner = FullExperiment(DEFAULT_COMPARISON, DEFAULT_EXECUTION)

    assert (
        runner.comparison["llm_model_key"]
        == "qwen_qwen3_8_flash_8192_routed"
    )
    assert runner.execution["execution"]["cpu_affinity"] == list(range(11))
    assert runner.execution["execution"]["reserved_logical_cpus"] == [11]

    registry_path = runner.optimizer_path.parent / runner.optimizer["llm"][
        "model_registry"
    ]
    registry = yaml.safe_load(registry_path.read_text())
    profile = registry["models"][runner.comparison["llm_model_key"]]
    assert profile["model_id"] == "qwen/qwen3.8-flash"
    assert profile["request"]["max_tokens"] == 8192
    assert profile["request"]["reasoning"] == {"max_tokens": 1024}
    assert profile["provider"]["order"] == ["makora", "alibaba"]
    assert profile["provider"]["only"] == ["makora", "alibaba"]
    assert profile["provider"]["allow_fallbacks"] is True
    gateway = yaml.safe_load(
        runner.optimizer_path.parent.joinpath("openrouter.yaml").read_text()
    )
    reliability = gateway["transport_reliability"]
    assert reliability["max_attempts"] == 6
    assert reliability["retry_http_statuses"] == [429, 500, 502, 503, 504]
    assert reliability["coordination_lock"] == "../../tmp/openrouter-request.lock"


def test_pending_stage_methods_only_returns_missing_or_stale_audits(
    tmp_path: Path,
) -> None:
    comparison_sha256 = hashlib.sha256(b"comparison").hexdigest()
    optimizer_sha256 = hashlib.sha256(b"optimizer").hexdigest()
    plan = {
        "comparison_config_sha256": comparison_sha256,
        "calibration_config_sha256": optimizer_sha256,
        "tickers": {
            ticker: {
                "calibration_study_dir": str(tmp_path / ticker / "calibration")
            }
            for ticker in TICKERS
        },
    }

    for ticker in TICKERS:
        for method in METHODS:
            if ticker == TICKERS[-1] and method == "opro_heuristic":
                continue
            _write_stage_audit(
                tmp_path
                / ticker
                / "calibration"
                / "report"
                / f"component_aware_stage_audit_{method}.json",
                ticker=ticker,
                method=method,
                comparison_sha256=comparison_sha256,
                optimizer_sha256=optimizer_sha256,
            )

    assert pending_stage_methods(plan) == ("opro_heuristic",)

    stale = (
        tmp_path
        / TICKERS[0]
        / "calibration"
        / "report"
        / "component_aware_stage_audit_sobol.json"
    )
    value = json.loads(stale.read_text())
    value["optimizer_config_sha256"] = "stale"
    stale.write_text(json.dumps(value))
    assert pending_stage_methods(plan) == ("sobol", "opro_heuristic")


def test_stage_audit_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    path.write_text("not-json")
    assert not stage_audit_is_current(
        path,
        ticker="ASAI3",
        method="sobol",
        comparison_sha256="comparison",
        optimizer_sha256="optimizer",
    )
