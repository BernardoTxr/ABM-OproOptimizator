import csv
import json
from pathlib import Path
from statistics import stdev

import pytest

from workflows.benchmark import published_summary as summary_module
from workflows.benchmark.published_summary import (
    BASELINE_METHODS,
    METHODS,
    OPRO_METHODS,
    build_summary,
    write_summary,
)
from workflows.benchmark.reproduce_published_study import build_plan
from workflows.benchmark.ticker_books import TICKERS


def _write_test_report(root: Path, methods: tuple[str, ...], offset: float) -> None:
    payload = {
        "tickers": {
            ticker: {
                "methods": {
                    method: {
                        "test_objective": offset
                        + ticker_index * 100
                        + METHODS.index(method),
                        "calibration_objective": (
                            ticker_index * 100 + METHODS.index(method)
                        ),
                        "selected_repetition": 0,
                        "selected_policy_trial": 49,
                        "candidate": {
                            "method": method,
                            "ticker": ticker,
                            "repetition": 0,
                        },
                    }
                    for method in methods
                }
            }
            for ticker_index, ticker in enumerate(TICKERS)
        }
    }
    path = root / "report" / "ticker_test_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))


def _fake_runs(path: str | Path, *, incomplete: bool = False):
    path = Path(path)
    ticker = path.parent.name
    methods = OPRO_METHODS if "followup" in str(path) else BASELINE_METHODS
    rows = []
    for method in methods:
        for repetition in range(5):
            rows.append({
                "sampler": method,
                "symbol": ticker,
                "split": "calibration",
                "repetition": repetition,
                "proposal_slots": (
                    49 if incomplete and ticker == TICKERS[0]
                    and method == methods[0] and repetition == 0 else 50
                ),
                "configured_budget": 50,
                "valid_proposals": 50,
                "invalid_or_error_proposals": 0,
                "model_calls": 38 if method in OPRO_METHODS else 0,
                "best_objective": (
                    TICKERS.index(ticker) * 100 + METHODS.index(method) + repetition
                ),
                "best_policy_trial": 49,
                "best_candidate": {
                    "method": method,
                    "ticker": ticker,
                    "repetition": repetition,
                },
                "best_so_far": [1000.0 - slot for slot in range(12)]
                + [
                    float(
                        TICKERS.index(ticker) * 100
                        + METHODS.index(method)
                        + repetition
                    )
                ] * 38,
            })
    return rows


def test_published_summary_validates_and_combines_both_studies(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    followup = tmp_path / "followup"
    _write_test_report(baseline, BASELINE_METHODS, 1_000.0)
    _write_test_report(followup, OPRO_METHODS, 1_000.0)
    monkeypatch.setattr(summary_module, "load_policy_runs", _fake_runs)

    result = build_summary(baseline, followup)

    assert tuple(result["methods"]) == METHODS
    assert set(result["tickers"]) == set(TICKERS)
    first = result["tickers"][TICKERS[0]]["methods"]["sobol"]
    assert first["calibration_best_loss_mean"] == pytest.approx(2.0)
    assert first["calibration_best_loss_std"] == pytest.approx(stdev(range(5)))
    assert first["calibration_rank"] == 1.0
    assert first["test_rank"] == 1.0
    assert result["methods"]["sobol"]["mean_calibration_rank"] == 1.0
    assert result["methods"]["opro_heuristic"]["mean_test_rank"] == 8.0

    json_path, csv_path = write_summary(result, tmp_path / "output")
    assert json.loads(json_path.read_text())["expected_slots_per_repetition"] == 50
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == (len(TICKERS) + 1) * len(METHODS)
    assert set(rows[0]) == {
        "scope", "ticker", "method", "calibration_best_loss_mean",
        "calibration_best_loss_std", "calibration_rank", "test_objective",
        "test_rank", "mean_calibration_best_loss",
        "mean_calibration_best_loss_std", "mean_calibration_rank",
        "mean_test_objective", "mean_test_rank",
    }
    assert rows[-1]["scope"] == "aggregate"
    assert float(rows[-1]["mean_test_rank"]) == 8.0


def test_published_summary_rejects_incomplete_calibration(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    followup = tmp_path / "followup"
    _write_test_report(baseline, BASELINE_METHODS, 0.0)
    _write_test_report(followup, OPRO_METHODS, 0.0)
    monkeypatch.setattr(
        summary_module,
        "load_policy_runs",
        lambda path: _fake_runs(path, incomplete=True),
    )

    with pytest.raises(RuntimeError, match="49 slots, expected 50"):
        build_summary(baseline, followup)


def _command(step):
    return step["command"]


def test_portable_plan_uses_only_published_methods_and_recorded_opro_workers():
    plan = build_plan(mode="portable")

    baseline_command = _command(plan["baselines"][-1])
    assert "workflows.benchmark.run_component_aware" in baseline_command
    selected = [
        baseline_command[index + 1]
        for index, value in enumerate(baseline_command[:-1]) if value == "--method"
    ]
    assert tuple(selected) == BASELINE_METHODS
    assert not set(OPRO_METHODS) & set(selected)

    opro_stage = _command(plan["opro"][1])
    assert "workflows.benchmark.opro_pair_followup_parallel" in opro_stage
    assert opro_stage[opro_stage.index("--ticker-workers") + 1] == "3"
    assert opro_stage[opro_stage.index("--simulation-workers") + 1] == "2"
    opro_test = _command(plan["evaluate"][1])
    assert opro_test[opro_test.index("--simulation-workers") + 1] == "2"
    assert "condition" in plan["prepare"][1]
    assert "condition" in plan["opro"][0]


def test_reference_plan_uses_frozen_parallel_baseline_launcher():
    plan = build_plan(mode="reference")
    baseline_command = _command(plan["baselines"][-1])

    assert "workflows.benchmark.run_component_aware_parallel" in baseline_command
    assert "configs/paper/xeon-e2146g-execution.yaml" in baseline_command
    assert "--simulation-workers" not in baseline_command
