from __future__ import annotations
import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml
from tqdm import tqdm

from lob_sim.agents.factory import AgentFactory
from lob_sim.environment import Environment, EnvironmentConfig
from lob_sim.paths import find_project_root
from lob_sim.simulation.output import OutputWriter


# ---------------------------------------------------------------------------
# Module-level worker function (must be top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _run_scenario(args: tuple) -> str:
    """Run one simulation scenario and return its output directory path.

    All arguments are plain Python types so the function is safely picklable
    and can be dispatched to a ``ProcessPoolExecutor`` worker.
    """
    (
        scenario_idx,
        seed,
        agents_yaml,
        env_cfg_dict,
        time_step,
        output_dir,
        scenario_name,
        n_workers,
        scenario_metadata,
        write_diagnostics,
    ) = args

    env_cfg   = _build_env_config(env_cfg_dict)
    timesteps = np.arange(env_cfg.start_time, env_cfg.end_time, time_step)

    agents = AgentFactory.from_yaml(agents_yaml, time_step, seed)
    env    = Environment(env_cfg)
    rng    = np.random.default_rng(seed)

    # Per-scenario step bar.  position rotates across worker slots (1-based so
    # it sits below the outer "Scenarios" bar at position 0).
    step_bar = tqdm(
        timesteps,
        desc=f"  scenario_{scenario_idx:04d}",
        unit="step",
        position=(scenario_idx % n_workers) + 1,
        leave=False,
        dynamic_ncols=True,
    )
    for t in step_bar:
        env.advance_time(float(t))
        shuffled = list(agents)
        rng.shuffle(shuffled)
        for agent in shuffled:
            obs     = env.get_observation(agent.agent_id)
            actions = agent.act(obs)
            for action in actions:
                env.submit_action(agent.agent_id, action)
        env.record_state(float(t))

    state_dfs    = env.get_state_dataframes()
    scenario_dir = OutputWriter.write(
        output_dir=output_dir,
        scenario_name=scenario_name,
        scenario_idx=scenario_idx,
        state_dfs=state_dfs,
        agents=agents,
    )
    meta = {
        "scenario_name": scenario_name,
        "scenario_idx":  scenario_idx,
        "time_step":     time_step,
        "random_seed":   seed,
        "n_agents":      len(agents),
        "start_time":    float(env_cfg.start_time),
        "end_time":      float(env_cfg.end_time),
        "tick_size":     float(env_cfg.tick_size),
        "n_steps":       len(timesteps),
    }
    meta.update(scenario_metadata)
    with open(os.path.join(scenario_dir, "metadata.yaml"), "w") as stream:
        yaml.safe_dump(meta, stream, sort_keys=False)
    if write_diagnostics:
        # Paper calibration does not consume per-book figures or reports, so
        # load Matplotlib only for runs that explicitly request diagnostics.
        from lob_sim.simulation.visualize import save_scenario_outputs

        save_scenario_outputs(scenario_dir, meta=meta)
    return scenario_dir


def _build_env_config(cfg: dict) -> EnvironmentConfig:
    lob_depth = int(cfg.get("lob_depth", 5))
    return EnvironmentConfig(
        tick_size=float(cfg["tick_size"]),
        min_order_size=float(cfg["min_order_size"]),
        start_time=float(cfg["start_time"]),
        end_time=float(cfg["end_time"]),
        lob_depth=lob_depth,
        book_storage_depth=max(lob_depth, int(cfg.get("book_storage_depth", 50))),
        seed_order_lifetime=(
            float(cfg["seed_order_lifetime"])
            if cfg.get("seed_order_lifetime") is not None else None
        ),
        initial_snapshot=cfg.get("initial_snapshot", {}),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class SimulationRunner:
    """
    Orchestrates one or more scenario runs as defined in a scenario YAML file.

    Scenario config keys:
      scenario_name:        str
      scenario_description: str  (optional)
      n_scenarios:          int
      agents_yaml:          str  (path)
      environment_yaml:     str  (path)
      time_step:            float (seconds)
      random_seed:          int
      output_directory:     str  (path)
      n_workers:            int  (default 1; >1 enables multiprocessing)
      write_diagnostics:    bool (default true; per-scenario PNGs and report)
    """

    def run(self, config_path: str) -> None:
        config = self._load_yaml(config_path)

        n_scenarios:   int   = int(config["n_scenarios"])
        scenario_name: str   = config["scenario_name"]
        time_step:     float = float(config["time_step"])
        random_seed:   int   = int(config["random_seed"])
        output_dir:    str   = config["output_directory"]
        n_workers:     int   = int(config.get("n_workers", 1))
        write_diagnostics_value = config.get("write_diagnostics", True)
        if not isinstance(write_diagnostics_value, bool):
            raise ValueError("write_diagnostics must be a YAML boolean")
        write_diagnostics: bool = write_diagnostics_value

        base_dir    = os.path.dirname(os.path.abspath(config_path))
        agents_yaml = self._resolve(config["agents_yaml"], base_dir)
        env_yaml    = self._resolve(config["environment_yaml"], base_dir)

        with open(env_yaml, "r") as f:
            env_cfg_dict = yaml.safe_load(f)
        env_cfg_dict.update(config.get("environment_overrides", {}))

        environment_variants = self._prepare_environment_variants(
            config=config,
            config_base_dir=base_dir,
            base_environment=env_cfg_dict,
            n_scenarios=n_scenarios,
            random_seed=random_seed,
        )

        env_cfg   = _build_env_config(environment_variants[0][0])
        n_steps   = len(np.arange(env_cfg.start_time, env_cfg.end_time, time_step))

        print(f"\nSimulation: {scenario_name}")
        print(f"  Scenarios : {n_scenarios}")
        print(f"  Time step : {time_step}s  |  Steps/scenario: {n_steps}")
        print(f"  Workers   : {n_workers}")
        print(f"  Output    : {output_dir}\n")
        if "real_window_sampling" in config:
            starts = sorted({meta["window_start_utc"] for _, meta in environment_variants})
            print(
                f"  Real windows: {len(starts)} starts across "
                f"{len({meta['real_date'] for _, meta in environment_variants})} "
                "sampled dates\n"
            )

        all_args = [
            (
                i,
                random_seed + i,
                agents_yaml,
                environment_variants[i][0],
                time_step,
                output_dir,
                scenario_name,
                n_workers,
                environment_variants[i][1],
                write_diagnostics,
            )
            for i in range(n_scenarios)
        ]

        if n_workers == 1:
            self._run_serial(all_args, n_scenarios)
        else:
            self._run_parallel(all_args, n_scenarios, n_workers)

        print(f"\nDone. Results in: {os.path.abspath(output_dir)}/{scenario_name}/")

    # ------------------------------------------------------------------

    @staticmethod
    def _run_serial(all_args: list, n_scenarios: int) -> None:
        with tqdm(total=n_scenarios, desc="Scenarios", unit="run") as pbar:
            for args in all_args:
                scenario_dir = _run_scenario(args)
                pbar.set_postfix({"last": os.path.basename(scenario_dir)})
                pbar.update(1)

    @staticmethod
    def _run_parallel(all_args: list, n_scenarios: int, n_workers: int) -> None:
        with tqdm(total=n_scenarios, desc="Scenarios", unit="run") as pbar:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(_run_scenario, args): args[0]
                    for args in all_args
                }
                for future in as_completed(futures):
                    scenario_dir = future.result()  # re-raises worker exceptions
                    pbar.set_postfix({"last": os.path.basename(scenario_dir)})
                    pbar.update(1)

    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _resolve(path: str, base_dir: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(base_dir, path))

    @staticmethod
    def _prepare_environment_variants(
        config: dict,
        config_base_dir: str,
        base_environment: dict,
        n_scenarios: int,
        random_seed: int,
    ) -> list[tuple[dict, dict]]:
        """Build one environment per sampled empirical intraday window."""
        sampling = config.get("real_window_sampling")
        if sampling is None:
            return [(copy.deepcopy(base_environment), {}) for _ in range(n_scenarios)]

        from lob_sim.loss.real_facts import (
            discover_dates,
            cached_initial_snapshots,
            resolve_date_split,
            resolve_measurement_windows,
            select_window_assignments,
        )

        real_config_path = SimulationRunner._resolve(
            str(sampling["real_config"]), config_base_dir
        )
        with open(real_config_path, "r") as stream:
            real_config = yaml.safe_load(stream)
        data_dir = str(real_config["data_dir"])
        if not os.path.isabs(data_dir):
            project_root = find_project_root(real_config_path)
            candidates = [
                os.path.join(project_root, data_dir),
                os.path.join(os.path.dirname(real_config_path), data_dir),
            ]
            real_config["data_dir"] = next(
                (os.path.abspath(path) for path in candidates if os.path.exists(path)),
                os.path.abspath(candidates[0]),
            )

        symbol = str(sampling["symbol"]).upper()
        split = str(sampling.get("split", "all"))
        available = discover_dates(symbol, real_config)
        dates = resolve_date_split(available, real_config, split)
        windows = resolve_measurement_windows(real_config)
        duration_seconds = float(real_config["window_duration_hours"]) * 3600.0
        simulation_seconds = (
            float(base_environment["end_time"]) - float(base_environment["start_time"])
        )
        if not np.isclose(duration_seconds, simulation_seconds):
            raise ValueError(
                "Real measurement-window duration must equal simulated duration: "
                f"{duration_seconds} != {simulation_seconds} seconds"
            )
        assignments = select_window_assignments(
            dates, windows, n_scenarios, random_seed
        )
        cache_label = str(real_config.get("cache_label", "windows"))
        snapshot_cache = sampling.get("snapshot_cache")
        if snapshot_cache is None:
            snapshot_cache = os.path.join(
                find_project_root(real_config_path), "results", "real", symbol,
                f"initial_books_{split}_{cache_label}_all.json",
            )
        elif not os.path.isabs(str(snapshot_cache)):
            snapshot_cache = os.path.join(config_base_dir, str(snapshot_cache))
        # Cache the complete date × clock-window opening-book grid once.  A
        # candidate, optimizer policy, or validation seed then changes only
        # which cached openings are selected; empirical reconstruction never
        # occurs inside an optimization trial.
        cache_assignments = [
            (date, window_start)
            for date in dates
            for window_start, _ in windows
        ]
        cached_snapshots = cached_initial_snapshots(
            symbol,
            cache_assignments,
            real_config,
            int(base_environment.get("lob_depth", real_config.get("lob_depth", 5))),
            str(snapshot_cache),
        )
        snapshot_lookup = {
            (date, window_start): snapshot
            for (date, window_start), snapshot
            in zip(cache_assignments, cached_snapshots)
        }
        snapshots = [snapshot_lookup[assignment] for assignment in assignments]
        variants: list[tuple[dict, dict]] = []
        for (date, window_start), snapshot in zip(assignments, snapshots):
            environment = copy.deepcopy(base_environment)
            environment["initial_snapshot"] = snapshot
            metadata = {
                "real_symbol": symbol,
                "real_split": split,
                "real_date": date,
                "window_start_utc": window_start.isoformat(),
                "window_duration_seconds": duration_seconds,
            }
            variants.append((environment, metadata))
        return variants
