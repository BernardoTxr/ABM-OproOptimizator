"""One reproducible evaluator shared by every calibration proposal policy."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from lob_sim.loss.aggregate import aggregate_all
from lob_sim.loss.compare import LossSpec, build_real_kde, compute_loss
from lob_sim.loss.real_facts import compute_real_facts, discover_dates, resolve_date_split
from lob_sim.loss.smm import (
    SUPPORTED_SMM_VERSIONS,
    SMMSpec,
    SMMTarget,
    compute_smm_loss,
    load_or_build_empirical_target,
)
from lob_sim.optimization.space import SearchSpace
from lob_sim.paths import find_project_root
from lob_sim.simulation.runner import SimulationRunner


def _resolve(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _pooled_aggregate_for_loss(
    simulation_dir: Path,
    loss_version: str,
) -> dict[str, Any] | None:
    """Build legacy pooled summaries only for loss functions that consume them."""
    if loss_version in SUPPORTED_SMM_VERSIONS:
        return None
    return aggregate_all(str(simulation_dir), force=True)


def use_frozen_smm_weights(
    evaluation_target: SMMTarget,
    weight_target: SMMTarget,
    weight_target_cache: Path,
) -> SMMTarget:
    """Combine held-out target means with calibration-only SMM weights."""
    if weight_target.moment_names != evaluation_target.moment_names:
        raise ValueError(
            "Frozen SMM weight target moment order does not match the "
            "evaluation target"
        )
    return replace(
        evaluation_target,
        standard_errors=dict(weight_target.standard_errors),
        covariance=[list(row) for row in weight_target.covariance],
        metadata={
            **evaluation_target.metadata,
            "weight_source": "calibration_target",
            "weight_target_cache": str(weight_target_cache.resolve()),
            "weight_target_signature": weight_target.metadata.get("signature"),
        },
    )


def apply_simulation_output_retention(
    simulation_dir: Path, retention: str,
) -> bool:
    """Apply an explicit post-score retention policy to one trial output."""
    if retention == "full":
        return False
    if retention != "summary_only":
        raise ValueError(
            "simulation_output_retention must be 'full' or 'summary_only'"
        )
    resolved = simulation_dir.resolve()
    if resolved.name != "simulation" or resolved.parent == resolved:
        raise ValueError(f"Refusing to compact unsafe simulation path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
        return True
    return False


def validate_smm_analysis_archive(
    smm_details: dict[str, Any],
    scenario_schedule: list[dict[str, Any]],
    validity: dict[str, Any],
    moment_names: tuple[str, ...],
    expected_scenarios: int,
) -> dict[str, Any]:
    """Refuse raw-output deletion unless the plot-ready SMM record is complete."""
    expected_moments = set(moment_names)
    for key in (
        "simulated_moments",
        "target_moments",
        "raw_differences",
        "standardized_residuals",
    ):
        values = smm_details.get(key)
        if not isinstance(values, dict) or set(values) != expected_moments:
            raise ValueError(f"Incomplete SMM analysis archive field: {key}")
        numeric = np.asarray(list(values.values()), dtype=float)
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"Non-finite SMM analysis archive field: {key}")

    rows = smm_details.get("scenario_moments")
    if not isinstance(rows, list) or len(rows) != expected_scenarios:
        raise ValueError(
            "SMM analysis archive does not contain every scenario moment row"
        )
    scenario_names: set[str] = set()
    required_row_keys = {"scenario", "date", "window_start_utc", *moment_names}
    for row in rows:
        if not isinstance(row, dict) or not required_row_keys.issubset(row):
            raise ValueError("SMM scenario-moment row is incomplete")
        scenario_names.add(str(row["scenario"]))
        numeric = np.asarray([row[name] for name in moment_names], dtype=float)
        if not np.all(np.isfinite(numeric)):
            raise ValueError("SMM scenario-moment row contains non-finite values")
    if len(scenario_names) != expected_scenarios:
        raise ValueError("SMM scenario identifiers are not unique")

    if len(scenario_schedule) != expected_scenarios:
        raise ValueError("SMM analysis archive has an incomplete scenario schedule")
    scheduled_names = {
        f"scenario_{int(row['scenario_idx']):04d}"
        for row in scenario_schedule
        if isinstance(row, dict) and "scenario_idx" in row
    }
    if scheduled_names != scenario_names:
        raise ValueError("Scenario moments and metadata schedule do not align")
    if any(
        not {"real_date", "window_start_utc", "random_seed"}.issubset(row)
        for row in scenario_schedule
    ):
        raise ValueError("Scenario schedule lacks date, clock, or seed metadata")

    validity_rows = validity.get("scenarios")
    if not isinstance(validity_rows, list) or len(validity_rows) != expected_scenarios:
        raise ValueError("SMM analysis archive has incomplete validity diagnostics")
    return {
        "version": "smm_analysis_archive_v1",
        "moment_count": len(moment_names),
        "scenario_count": expected_scenarios,
        "contains_aggregate_moments": True,
        "contains_standardized_residuals": True,
        "contains_scenario_moments": True,
        "contains_scenario_schedule": True,
        "contains_validity_diagnostics": True,
    }


def _code_revision(repo_root: Path) -> str:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet"], cwd=repo_root,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode != 0
        return revision + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _evaluation_fingerprint(
    repo_root: Path,
    input_files: list[Path],
    data_files: list[Path],
    extra: dict[str, Any],
) -> str:
    """Hash every local input that can change a cached candidate objective."""
    digest = hashlib.sha256()
    digest.update(b"icaif-evaluator-v2\0")
    sources = sorted((repo_root / "lob_sim").rglob("*.py"))
    for path in sorted({item.resolve() for item in [*sources, *input_files]}):
        try:
            label = str(path.relative_to(repo_root.resolve()))
        except ValueError:
            label = path.name
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    data_stats = []
    for path in sorted({item.resolve() for item in data_files}):
        stat = path.stat()
        label = str(Path(*path.parts[-4:])) if len(path.parts) >= 4 else path.name
        data_stats.append((label, stat.st_size, stat.st_mtime_ns))
    digest.update(json.dumps(
        {"extra": extra, "data_stats": data_stats},
        sort_keys=True,
        separators=(",", ":"),
    ).encode())
    return digest.hexdigest()


def signed_diagnostics(
    sim: dict,
    real: dict,
    winsor_quantiles: tuple[float, float] = (0.01, 0.99),
) -> dict[str, Any]:
    """Signed, scale-free discrepancies for proposal policies and auditing."""
    diagnostics: dict[str, Any] = {}
    for fact in ("log_returns", "spread", "imbalance", "trade_sizes", "inter_arrival"):
        sim_values = np.asarray(sim.get(fact, {}).get("samples", []), dtype=float)
        real_values = np.asarray(real.get(fact, {}).get("samples", []), dtype=float)
        sim_values = sim_values[np.isfinite(sim_values)]
        real_values = real_values[np.isfinite(real_values)]
        if not sim_values.size or not real_values.size:
            continue
        low, high = np.quantile(real_values, winsor_quantiles)
        scale = max(float(high - low), np.finfo(float).eps)
        diagnostics[f"{fact}_mean_scaled_error"] = float(
            (np.mean(sim_values) - np.mean(real_values)) / scale
        )
        diagnostics[f"{fact}_std_ratio"] = float(
            np.std(sim_values) / max(np.std(real_values), np.finfo(float).eps)
        )

    sim_acf = sim.get("acf_abs_returns", {}).get("mean")
    real_acf = real.get("acf_abs_returns", {}).get("mean")
    if sim_acf is not None and real_acf is not None:
        length = min(len(sim_acf), len(real_acf), 11)
        if length > 1:
            diagnostics["acf_1_10_mean_error"] = float(
                np.mean(np.asarray(sim_acf)[1:length] - np.asarray(real_acf)[1:length])
            )

    real_depth = real.get("depth", {})
    for side, real_key in (("bid", "bid_mean"), ("ask", "ask_mean")):
        sim_values = sim.get(f"depth_{side}", {}).get("mean")
        real_values = real_depth.get(real_key)
        if sim_values is None or real_values is None:
            continue
        length = min(len(sim_values), len(real_values))
        relative = (
            np.asarray(sim_values[:length], dtype=float)
            - np.asarray(real_values[:length], dtype=float)
        ) / np.maximum(np.abs(real_values[:length]), np.finfo(float).eps)
        diagnostics[f"depth_{side}_relative_errors"] = relative.tolist()

    if "book_coverage" in sim and "book_coverage" in real:
        diagnostics["book_coverage_error"] = float(
            sim["book_coverage"]["mean"] - real["book_coverage"]["mean"]
        )
    return diagnostics


def validate_simulation(
    simulation_dir: Path,
    expected_scenarios: int,
    min_book_coverage: float,
    min_trades_per_scenario: int,
    required_depth_levels: int = 0,
    min_depth_coverage: float = 0.0,
) -> dict[str, Any]:
    """Apply prespecified market-validity gates before ranking a trial."""
    scenario_dirs = sorted(path for path in simulation_dir.glob("scenario_*") if path.is_dir())
    violations: list[str] = []
    scenario_rows: list[dict[str, Any]] = []
    if len(scenario_dirs) != expected_scenarios:
        violations.append(
            f"scenario_count:{len(scenario_dirs)}!={expected_scenarios}"
        )

    for scenario_dir in scenario_dirs:
        mid_path = scenario_dir / "mid_price.parquet"
        trade_path = scenario_dir / "trade_tape.parquet"
        lob_path = scenario_dir / "lob_snapshots.parquet"
        if not mid_path.exists() or not trade_path.exists() or (
            required_depth_levels and not lob_path.exists()
        ):
            violations.append(f"{scenario_dir.name}:missing_output")
            continue
        mid = pd.read_parquet(mid_path)
        trades = pd.read_parquet(trade_path)
        coverage = (
            float(mid["mid_price"].notna().mean())
            if not mid.empty and "mid_price" in mid else 0.0
        )
        trade_count = int(len(trades))
        row_violations: list[str] = []
        if coverage < min_book_coverage:
            row_violations.append("coverage")
        if trade_count < min_trades_per_scenario:
            row_violations.append("trades")
        depth_coverage = None
        if required_depth_levels:
            lob = pd.read_parquet(lob_path)
            required = set(range(1, required_depth_levels + 1))
            if lob.empty or not {"timestamp", "level"}.issubset(lob.columns):
                row_violations.append("depth_schema")
                depth_coverage = 0.0
            else:
                level_sets = lob.groupby("timestamp")["level"].agg(
                    lambda values: set(int(value) for value in values)
                )
                depth_coverage = float(np.mean([
                    required.issubset(level_set) for level_set in level_sets
                ])) if len(level_sets) else 0.0
                if depth_coverage < min_depth_coverage:
                    row_violations.append("depth_coverage")
        if not mid.empty:
            for column in ("mid_price", "best_bid", "best_ask"):
                if column not in mid:
                    row_violations.append(f"missing_{column}")
                    continue
                values = mid[column].dropna().to_numpy(dtype=float)
                if values.size and (not np.all(np.isfinite(values)) or np.any(values <= 0)):
                    row_violations.append(f"invalid_{column}")
            two_sided = mid.dropna(subset=["best_bid", "best_ask"])
            if not two_sided.empty and np.any(
                two_sided["best_bid"].to_numpy() >= two_sided["best_ask"].to_numpy()
            ):
                row_violations.append("crossed_or_locked_book")
        for violation in row_violations:
            violations.append(f"{scenario_dir.name}:{violation}")
        scenario_rows.append({
            "scenario": scenario_dir.name,
            "book_coverage": coverage,
            "depth_coverage": depth_coverage,
            "trade_count": trade_count,
            "valid": not row_violations,
        })
    return {
        "valid": not violations,
        "violations": violations,
        "scenarios": scenario_rows,
    }


class TrialEvaluator:
    """Materialize, run, validate, score, and persist one candidate."""

    def __init__(
        self,
        config_path: str,
        runtime_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.config_dir = self.config_path.parent
        self.config = yaml.safe_load(self.config_path.read_text())
        self.runtime_overrides = dict(runtime_overrides or {})
        allowed_overrides = {
            "study_name",
            "symbol",
            "split",
            "replications_per_trial",
            "simulation_workers",
            "evaluation_seed",
            "llm_model_key",
            "proposal_runtime",
            "smm_target_cache",
            "smm_window_moment_cache",
            "smm_weight_target_cache",
        }
        unknown_overrides = sorted(set(self.runtime_overrides) - allowed_overrides)
        if unknown_overrides:
            raise ValueError(f"Unsupported evaluator runtime overrides: {unknown_overrides}")
        self.config.update(self.runtime_overrides)
        self.repo_root = find_project_root(self.config_path)
        self.space = SearchSpace.from_config(self.config["search_space"])

        self.base_scenario_path = _resolve(self.config["base_config"], self.config_dir)
        self.base_scenario = yaml.safe_load(self.base_scenario_path.read_text())
        self.base_agents_path = _resolve(
            self.base_scenario["agents_yaml"], self.base_scenario_path.parent
        )
        self.base_agents = yaml.safe_load(self.base_agents_path.read_text())
        self.environment_path = _resolve(
            self.base_scenario["environment_yaml"], self.base_scenario_path.parent
        )
        self.real_config_path = _resolve(self.config["real_config"], self.config_dir)
        self.real_config = yaml.safe_load(self.real_config_path.read_text())
        data_dir = Path(self.real_config["data_dir"])
        if not data_dir.is_absolute():
            candidates = [
                self.repo_root / data_dir,
                self.real_config_path.parent.parent / data_dir,
                self.real_config_path.parent / data_dir,
            ]
            self.real_config["data_dir"] = str(next(
                (path.resolve() for path in candidates if path.exists()),
                candidates[0].resolve(),
            ))

        self.symbol = str(self.config["symbol"]).upper()
        self.split = str(self.config.get("split", "development"))
        dates = discover_dates(self.symbol, self.real_config)
        self.evaluation_dates = resolve_date_split(dates, self.real_config, self.split)
        self.loss_config_path = _resolve(self.config["loss_config"], self.config_dir)
        loss_values = yaml.safe_load(self.loss_config_path.read_text())
        self.loss_version = str(loss_values.get("version", "pooled_v2"))
        self.loss_spec: LossSpec | SMMSpec
        self.smm_target: SMMTarget | None = None
        target_cache: Path | None = None
        weight_target_cache: Path | None = None
        if self.loss_version in SUPPORTED_SMM_VERSIONS:
            self.loss_spec = SMMSpec.from_dict(loss_values)
            configured_target = self.config.get(
                "smm_target_cache",
                f"../results/real/{self.symbol}/smm_target_{self.split}_{self.loss_version}.json",
            )
            target_cache = _resolve(str(configured_target), self.config_dir)
            configured_window_cache = self.config.get("smm_window_moment_cache")
            window_cache = (
                _resolve(str(configured_window_cache), self.config_dir)
                if configured_window_cache else None
            )
            configured_weight_target = self.config.get("smm_weight_target_cache")
            weight_target_cache = (
                _resolve(str(configured_weight_target), self.config_dir)
                if configured_weight_target else None
            )
            weight_target = None
            if weight_target_cache is not None:
                if not weight_target_cache.is_file():
                    raise ValueError(
                        "Frozen SMM weight target is missing: "
                        f"{weight_target_cache}"
                    )
                weight_target = SMMTarget.from_dict(
                    json.loads(weight_target_cache.read_text())
                )
            self.smm_target = load_or_build_empirical_target(
                self.symbol,
                self.evaluation_dates,
                self.real_config,
                self.loss_spec,
                target_cache,
                window_cache,
                weight_target,
            )
            if weight_target_cache is not None:
                self.smm_target = use_frozen_smm_weights(
                    self.smm_target, weight_target, weight_target_cache
                )
            self.real_facts = None
            self.real_aggregate = None
        else:
            self.loss_spec = LossSpec.from_dict(loss_values)
            cache = _resolve(self.config["real_facts_cache"], self.config_dir)
            self.real_facts = compute_real_facts(
                self.symbol, self.evaluation_dates, self.real_config, str(cache)
            )
            self.real_aggregate = build_real_kde(self.real_facts)

        raw_depth = int(self.real_config.get("depth", 10))
        data_files = [
            Path(self.real_config["data_dir"]) / self.symbol / date
            / f"d{raw_depth}" / "events.parquet"
            for date in self.evaluation_dates
        ]
        self.evaluator_fingerprint = _evaluation_fingerprint(
            self.repo_root,
            [
                self.config_path,
                self.base_scenario_path,
                self.base_agents_path,
                self.environment_path,
                self.real_config_path,
                self.loss_config_path,
                *([target_cache] if target_cache is not None else []),
                *([weight_target_cache] if weight_target_cache is not None else []),
            ],
            data_files,
            {
                "symbol": self.symbol,
                "split": self.split,
                "dates": self.evaluation_dates,
                "runtime_overrides": self.runtime_overrides,
            },
        )

        output_root = _resolve(
            self.config.get("output_directory", "../results/optimization"),
            self.config_dir,
        )
        self.study_dir = output_root / str(self.config["study_name"])
        self.study_dir.mkdir(parents=True, exist_ok=True)
        self.replications = int(self.config.get("replications_per_trial", 10))
        self.simulation_workers = int(self.config.get("simulation_workers", 1))
        self.evaluation_seed = int(self.config["evaluation_seed"])
        self.simulation_output_retention = str(
            self.config.get("simulation_output_retention", "full")
        )
        if self.simulation_output_retention not in {"full", "summary_only"}:
            raise ValueError(
                "simulation_output_retention must be 'full' or 'summary_only'"
            )
        gates = self.config.get("validity_gates", {})
        self.min_book_coverage = float(gates.get("min_book_coverage", 0.99))
        self.min_trades_per_scenario = int(gates.get("min_trades_per_scenario", 1))
        self.required_depth_levels = int(gates.get("required_depth_levels", 0))
        self.min_depth_coverage = float(gates.get("min_depth_coverage", 0.0))

    def evaluate(self, candidate: dict[str, Any], trial_id: int | str) -> dict[str, Any]:
        normalized = self.space.validate(candidate)
        candidate_key = self.space.canonical_key(normalized)
        candidate_hash = hashlib.sha256(json.dumps({
            "candidate": candidate_key,
            "evaluator_fingerprint": self.evaluator_fingerprint,
        }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        # With a frozen seed schedule the evaluator is deterministic. Cache by
        # candidate plus a content fingerprint of code, configs, and data-file
        # identities so common designs are reused without accepting stale
        # objectives after an implementation change.
        trial_dir = self.study_dir / "evaluations" / candidate_hash
        result_path = trial_dir / "result.json"
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous.get("candidate_hash") != candidate_hash:
                raise ValueError(f"Candidate cache collision for {candidate_hash}")
            previous["resumed"] = True
            return previous

        trial_dir.mkdir(parents=True, exist_ok=True)
        agents = self.space.apply(self.base_agents, normalized)
        agents_path = trial_dir / "agents.yaml"
        agents_path.write_text(yaml.safe_dump(agents, sort_keys=False))

        scenario = dict(self.base_scenario)
        scenario["real_window_sampling"] = dict(scenario["real_window_sampling"])
        scenario["real_window_sampling"].update({
            "symbol": self.symbol,
            "real_config": str(self.real_config_path),
            "split": self.split,
        })
        scenario.update({
            "scenario_name": "simulation",
            "agents_yaml": str(agents_path),
            "environment_yaml": str(self.environment_path),
            "n_scenarios": self.replications,
            "n_workers": min(self.simulation_workers, self.replications),
            "random_seed": self.evaluation_seed,
            "output_directory": str(trial_dir),
        })
        scenario_path = trial_dir / "scenario.yaml"
        scenario_path.write_text(yaml.safe_dump(scenario, sort_keys=False))

        started = time.time()
        SimulationRunner().run(str(scenario_path))
        runtime_seconds = time.time() - started
        simulation_dir = trial_dir / "simulation"
        aggregate = _pooled_aggregate_for_loss(simulation_dir, self.loss_version)
        validity = validate_simulation(
            simulation_dir,
            self.replications,
            self.min_book_coverage,
            self.min_trades_per_scenario,
            self.required_depth_levels,
            self.min_depth_coverage,
        )
        smm_details: dict[str, Any] | None = None
        if self.loss_version in SUPPORTED_SMM_VERSIONS:
            assert isinstance(self.loss_spec, SMMSpec)
            assert self.smm_target is not None
            smm_loss = compute_smm_loss(
                sorted(path for path in simulation_dir.glob("scenario_*") if path.is_dir()),
                self.smm_target,
                self.loss_spec,
            )
            loss_total = smm_loss.total
            component_losses = smm_loss.per_group
            loss_specification = smm_loss.specification
            diagnostics = smm_loss.standardized_residuals
            smm_details = {
                "simulated_moments": smm_loss.simulated_moments,
                "target_moments": smm_loss.target_moments,
                "raw_differences": smm_loss.raw_differences,
                "standardized_residuals": smm_loss.standardized_residuals,
                "scenario_moments": smm_loss.scenario_moments,
                "target_metadata": self.smm_target.metadata,
            }
        else:
            assert isinstance(self.loss_spec, LossSpec)
            assert self.real_aggregate is not None
            assert aggregate is not None
            pooled_loss = compute_loss(
                aggregate["kde_data"], self.real_aggregate,
                specification=self.loss_spec,
            )
            loss_total = pooled_loss.total
            component_losses = pooled_loss.per_fact
            loss_specification = pooled_loss.specification
            diagnostics = signed_diagnostics(
                aggregate["kde_data"],
                self.real_aggregate,
                self.loss_spec.winsor_quantiles,
            )
        metadata = []
        for path in sorted(simulation_dir.glob("scenario_*/metadata.yaml")):
            metadata.append(yaml.safe_load(path.read_text()))
        analysis_archive = None
        if smm_details is not None:
            assert isinstance(self.loss_spec, SMMSpec)
            analysis_archive = validate_smm_analysis_archive(
                smm_details,
                metadata,
                validity,
                self.loss_spec.moment_names,
                self.replications,
            )
        result = {
            "first_trial_id": str(trial_id),
            "candidate": normalized,
            "candidate_hash": candidate_hash,
            "valid": bool(validity["valid"]),
            "validity": validity,
            # Invalid trials have no rankable objective. Sampler adapters must
            # report them as failed while still charging the trial budget.
            "objective": float(loss_total) if validity["valid"] else None,
            "diagnostic_loss": float(loss_total),
            "component_losses": {
                key: float(value) for key, value in component_losses.items()
            },
            "loss_specification": loss_specification,
            "signed_diagnostics": diagnostics,
            "smm": smm_details,
            "analysis_archive": analysis_archive,
            "runtime_seconds": float(runtime_seconds),
            "replications": self.replications,
            "evaluation_seed": self.evaluation_seed,
            "symbol": self.symbol,
            "split": self.split,
            "dates": self.evaluation_dates,
            "scenario_schedule": metadata,
            "code_revision": _code_revision(self.repo_root),
            "evaluator_fingerprint": self.evaluator_fingerprint,
            "evaluator_config": str(self.config_path),
            "simulation_output_retention": self.simulation_output_retention,
            "resumed": False,
        }
        # Commit the complete analysis record before any bulky raw output is
        # removed. A process interruption can therefore leave extra Parquet
        # files, but cannot leave only compacted output without its SMM archive.
        result["simulation_outputs_removed"] = False
        _atomic_json(result_path, result)
        result["simulation_outputs_removed"] = apply_simulation_output_retention(
            simulation_dir, self.simulation_output_retention
        )
        if result["simulation_outputs_removed"]:
            _atomic_json(result_path, result)
        return result
