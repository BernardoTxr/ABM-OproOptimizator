"""Run an offline, no-data smoke check of the simulator and OPRO contract."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from lob_sim.optimization.feedback import SMM_COMPONENTS
from lob_sim.optimization.model_runtime import ModelCompletion
from lob_sim.optimization.opro_policy import OPROPolicy
from lob_sim.optimization.space import SearchSpace
from lob_sim.simulation.runner import SimulationRunner
from workflows import REPO_ROOT
from workflows.benchmark.preflight_openrouter import representative_preflight_history


class OfflineSmokeRuntime:
    """Return one valid proposal without credentials or network access."""

    model_key = "offline_smoke"
    runtime_identity = {
        "protocol": "offline_smoke_v1",
        "model_key": model_key,
        "network": False,
    }

    def __init__(self, candidate: dict[str, Any]) -> None:
        self.candidate = copy.deepcopy(candidate)
        self.calls = 0

    def complete(
        self, prompt: str, response_schema: dict[str, Any]
    ) -> ModelCompletion:
        self.calls += 1
        payload = {
            "hypothesis": "Offline smoke proposal",
            "evidence_trial_ids": [0],
            "candidate": copy.deepcopy(self.candidate),
            "predictions": [{
                "metric": SMM_COMPONENTS[0],
                "direction": "decrease",
                "rationale": "Exercises strict response parsing without a provider.",
            }],
        }
        text = json.dumps(payload, sort_keys=True)
        request = json.dumps(
            {"prompt": prompt, "response_schema": response_schema},
            sort_keys=True,
            separators=(",", ":"),
        )
        return ModelCompletion(
            text=text,
            request_id=f"offline-{self.calls}",
            requested_model=self.model_key,
            resolved_model=self.model_key,
            provider="offline",
            finish_reason="stop",
            usage={
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "total_tokens": 0,
                "reported_cost_usd": 0.0,
                "cost_status": "true_zero_offline",
                "cost_currency": "USD",
                "cost_source": "no_external_request",
            },
            request_sha256=hashlib.sha256(request.encode()).hexdigest(),
            response_sha256=hashlib.sha256(text.encode()).hexdigest(),
            raw_response={"offline": True},
            transport_attempts=({
                "attempt": 1,
                "status": "offline",
                "duration_seconds": 0.0,
            },),
        )


def _run_simulator_smoke(temporary_root: Path) -> dict[str, Any]:
    source_path = REPO_ROOT / "configs/examples/smoke.yaml"
    source = yaml.safe_load(source_path.read_text())
    config = copy.deepcopy(source)
    for key in ("agents_yaml", "environment_yaml"):
        config[key] = str((source_path.parent / source[key]).resolve())
    config["output_directory"] = str(temporary_root / "simulation-output")
    config_path = temporary_root / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    SimulationRunner().run(str(config_path))
    scenario_dir = (
        Path(config["output_directory"])
        / config["scenario_name"]
        / "scenario_0000"
    )
    required = (
        "mid_price.parquet",
        "lob_snapshots.parquet",
        "metadata.yaml",
    )
    missing = [name for name in required if not (scenario_dir / name).is_file()]
    if missing:
        raise RuntimeError("Simulator smoke output is incomplete: " + ", ".join(missing))
    return {
        "scenario_count": int(config["n_scenarios"]),
        "steps": int(config["environment_overrides"]["end_time"]),
        "required_outputs": list(required),
    }


def _run_opro_smoke() -> dict[str, Any]:
    optimizer_path = REPO_ROOT / "configs/paper/optimizer.yaml"
    optimizer = yaml.safe_load(optimizer_path.read_text())
    space = SearchSpace.from_config(optimizer["search_space"])
    candidate = space.validate(optimizer["initial_design"][0])
    history = representative_preflight_history(optimizer)
    heuristic_path = optimizer_path.parent / optimizer["llm"]["heuristic_config"]
    heuristic = yaml.safe_load(heuristic_path.read_text())

    conditions: dict[str, Any] = {}
    for name, specification in (("plain", None), ("heuristic", heuristic)):
        runtime = OfflineSmokeRuntime(candidate)
        proposal = OPROPolicy(space, runtime, specification).ask(
            len(history),
            history=history,
            remaining_budget=int(optimizer["trial_budget"]) - len(history),
        )
        if proposal.candidate != candidate or runtime.calls != 1:
            raise RuntimeError(f"Offline {name} OPRO proposal contract failed")
        prompt = json.loads(proposal.metadata["prompt"])
        if len(prompt.get("observations", [])) != len(history):
            raise RuntimeError(f"Offline {name} OPRO history is incomplete")
        expected_heuristic = specification is not None
        if prompt.get("heuristic_condition") is not expected_heuristic:
            raise RuntimeError(f"Offline {name} OPRO condition is incorrect")
        prompt_heuristics = prompt.get("heuristics")
        if expected_heuristic:
            if not isinstance(prompt_heuristics, dict) or not prompt_heuristics.get(
                "version"
            ):
                raise RuntimeError("Offline heuristic OPRO registry is missing")
        elif prompt_heuristics is not None:
            raise RuntimeError("Offline plain OPRO unexpectedly contains heuristics")
        remaining = int(optimizer["trial_budget"]) - len(history)
        if prompt.get("remaining_proposal_budget") != remaining:
            raise RuntimeError(f"Offline {name} OPRO remaining budget is incorrect")
        conditions[name] = {
            "heuristic_condition": prompt["heuristic_condition"],
            "heuristic_registry_version": (
                prompt_heuristics.get("version") if expected_heuristic else None
            ),
            "history_observations": len(history),
            "remaining_proposal_budget": remaining,
            "prompt_sha256": proposal.metadata["prompt_hash"],
            "runtime_calls": runtime.calls,
            "cost_status": proposal.metadata["model_usage"]["cost_status"],
        }
    return {
        "conditions": conditions,
        "network_calls": 0,
        "optimizer_slots_consumed": 0,
        "private_data_accessed": False,
    }


def run_smoke() -> dict[str, Any]:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"Python 3.12 is required; found {sys.version_info.major}."
            f"{sys.version_info.minor}"
        )
    with TemporaryDirectory(prefix="lob-smoke-") as directory:
        simulator = _run_simulator_smoke(Path(directory))
    return {
        "status": "ready",
        "python": sys.version.split()[0],
        "simulator": simulator,
        "opro": _run_opro_smoke(),
    }


def main() -> None:
    result = run_smoke()
    print(json.dumps(result, indent=2, sort_keys=True))
    print("SMOKE CHECK PASSED: environment is ready for OPRO runtime development.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"SMOKE CHECK FAILED: {type(exc).__name__}: {exc}") from exc
