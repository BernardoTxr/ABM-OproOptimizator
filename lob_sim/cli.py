from __future__ import annotations
import json
import os
import sys

import click

from lob_sim.simulation.runner import SimulationRunner


@click.group()
def cli() -> None:
    """Agent-based LOB simulator and calibration tools."""


@cli.command("run")
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the scenario YAML configuration file.",
)
def cmd_run(config: str) -> None:
    """Run a simulation scenario (or ensemble of scenarios)."""
    SimulationRunner().run(config)


@cli.command("analyze")
@click.option(
    "--output-dir", "-o",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Path to the scenario output directory (e.g. results/base_scenario).",
)
@click.option(
    "--mode",
    type=click.Choice(["single", "ensemble"], case_sensitive=False),
    default="ensemble",
    show_default=True,
    help="Plot a single scenario run or the full ensemble (mean ± std).",
)
@click.option(
    "--scenario-idx",
    type=int,
    default=0,
    show_default=True,
    help="Which scenario index to plot in single mode.",
)
@click.option(
    "--save",
    type=click.Path(dir_okay=False),
    default=None,
    help="Save the figure to this path instead of displaying it.",
)
def cmd_analyze(output_dir: str, mode: str, scenario_idx: int, save: str | None) -> None:
    """Compute and plot stylized facts from simulation output."""
    from lob_sim.loss.facts import compute_facts
    from lob_sim.loss.plots import plot_single, plot_ensemble

    if mode == "single":
        scenario_dir = os.path.join(output_dir, f"scenario_{scenario_idx:04d}")
        if not os.path.isdir(scenario_dir):
            click.echo(f"Error: directory not found: {scenario_dir}", err=True)
            sys.exit(1)
        facts = compute_facts(scenario_dir)
        plot_single(facts, title=os.path.basename(output_dir), save_path=save)
    else:
        # Collect all scenario_NNNN directories
        scenario_dirs = sorted([
            os.path.join(output_dir, d)
            for d in os.listdir(output_dir)
            if d.startswith("scenario_") and os.path.isdir(os.path.join(output_dir, d))
        ])
        if not scenario_dirs:
            click.echo(f"Error: no scenario directories found in {output_dir}", err=True)
            sys.exit(1)

        click.echo(f"Loading {len(scenario_dirs)} scenario(s)...")
        all_facts = [compute_facts(d) for d in scenario_dirs]
        plot_ensemble(all_facts, title=os.path.basename(output_dir), save_path=save)


@cli.command("aggregate")
@click.option(
    "--output-dir", "-o",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing scenario_NNNN subdirectories.",
)
@click.option(
    "--save-dir", "-s",
    default=None,
    type=click.Path(file_okay=False),
    help="Directory to write figures and report. Defaults to --output-dir.",
)
@click.option(
    "--force-recompute",
    is_flag=True,
    default=False,
    help="Recompute facts even if a cache already exists.",
)
def cmd_aggregate(
    output_dir: str,
    save_dir: str | None,
    force_recompute: bool,
) -> None:
    """Aggregate stylized facts across all scenarios and produce ensemble figures + report."""
    from lob_sim.loss.aggregate import aggregate_all
    import lob_sim.loss.ensemble_plots as ep
    from lob_sim.loss.report import generate_report

    if save_dir is None:
        save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)

    click.echo(f"Aggregating scenarios in: {output_dir}")
    result = aggregate_all(output_dir, force=force_recompute)
    scalars_df = result["scalars_df"]
    kde_data = result["kde_data"]
    n = len(scalars_df)
    click.echo(f"Loaded {n} scenarios.")

    # Typical number of observations per scenario for ACF confidence interval
    n_obs = int(scalars_df["n_obs_returns"].median()) if "n_obs_returns" in scalars_df.columns else 3599

    figures = [
        ("ensemble_log_returns.png",   lambda p: ep.plot_log_returns(kde_data, n, save_path=p)),
        ("ensemble_acf.png",           lambda p: ep.plot_acf(kde_data, n, n_obs=n_obs, save_path=p)),
        ("ensemble_spread.png",        lambda p: ep.plot_spread(kde_data, n, save_path=p)),
        ("ensemble_depth.png",         lambda p: ep.plot_depth(kde_data, n, save_path=p)),
        ("ensemble_imbalance.png",     lambda p: ep.plot_imbalance(kde_data, n, save_path=p)),
        ("ensemble_trade_sizes.png",   lambda p: ep.plot_trade_sizes(kde_data, n, save_path=p)),
        ("ensemble_inter_arrival.png", lambda p: ep.plot_inter_arrival(kde_data, n, save_path=p)),
    ]
    for fname, fn in figures:
        fn(os.path.join(save_dir, fname))

    report_path = os.path.join(save_dir, "ensemble_report.md")
    generate_report(scalars_df, report_path, n_scenarios=n)
    click.echo(f"Done. {len(figures)} figures + report written to: {save_dir}")


@cli.command("compare")
@click.option(
    "--scenario-dir", "-s",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Scenario output directory containing kde_cache.npz (e.g. results/base_scenario).",
)
@click.option(
    "--symbol",
    required=True,
    type=str,
    help="Ticker to compare (e.g. ASAI3).",
)
@click.option(
    "--real-config", "-r",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to real_analysis.yaml.",
)
@click.option(
    "--weights",
    default=None,
    type=str,
    help='Optional JSON string of per-fact weights, e.g. \'{"log_returns": 2.0, "spread": 1.0}\'.',
)
@click.option(
    "--loss-config",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Versioned loss YAML; optional --weights override its component weights.",
)
@click.option(
    "--force-recompute",
    is_flag=True,
    default=False,
    help="Recompute real facts even if a cache exists.",
)
@click.option(
    "--split",
    type=click.Choice(["all", "development", "test"], case_sensitive=False),
    default="all",
    show_default=True,
    help="Named date split from the real-data configuration.",
)
def cmd_compare(
    scenario_dir: str,
    symbol: str,
    real_config: str,
    weights: str | None,
    loss_config: str | None,
    force_recompute: bool,
    split: str,
) -> None:
    """Compare simulation ensemble against real BTG-OBD data for a given ticker.

    Loads the simulation KDE cache, extracts and pools real stylized facts
    across all available trading dates for SYMBOL, computes robust per-fact
    losses, and saves comparison plots and pooled statistics.

    Output is written to:  {scenario_dir}/comparison/{symbol}/
    """
    import yaml
    from lob_sim.loss.aggregate import aggregate_all
    from lob_sim.loss.real_facts import (
        compute_real_facts,
        compute_real_level_prices,
        discover_dates,
        resolve_date_split,
    )
    from lob_sim.loss.compare import (
        build_real_kde,
        compute_sim_level_prices,
        compute_loss,
        LossSpec,
        save_comparison_plots,
        save_loss_report,
        save_statistics_report,
    )

    # ---- load scenario KDE cache ----
    kde_cache_path = os.path.join(scenario_dir, "kde_cache.npz")
    if not os.path.exists(kde_cache_path):
        click.echo(
            f"Error: kde_cache.npz not found in {scenario_dir}. "
            "Run `python3 -m lob_sim aggregate` first.",
            err=True,
        )
        sys.exit(1)

    click.echo(f"Loading simulation KDE cache from: {kde_cache_path}")
    result = aggregate_all(scenario_dir, force=force_recompute)
    sim_kde_data = result["kde_data"]
    n_scenarios = len(result["scalars_df"])
    sim_n_obs = int(result["scalars_df"]["n_obs_returns"].median())

    # ---- parse real data config ----
    with open(real_config, "r") as f:
        config = yaml.safe_load(f)

    # ---- discover available dates ----
    available_dates = discover_dates(symbol, config)
    if not available_dates:
        click.echo(
            f"Error: no event files found for {symbol} under {config['data_dir']}.",
            err=True,
        )
        sys.exit(1)
    try:
        dates = resolve_date_split(available_dates, config, split.lower())
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    click.echo(f"Found {len(dates)} trading date(s) for {symbol}: {dates[0]} … {dates[-1]}")

    # ---- compute / load real facts ----
    output_dir = os.path.join(scenario_dir, "comparison", symbol)
    if split.lower() != "all":
        output_dir = os.path.join(output_dir, split.lower())
    os.makedirs(output_dir, exist_ok=True)

    cache_label = str(config.get("cache_label", "")).strip()
    cache_parts = [
        part for part in (split.lower(), cache_label)
        if part and part != "all"
    ]
    cache_suffix = "" if not cache_parts else "_" + "_".join(cache_parts)
    real_cache = os.path.join(
        "results", "real", symbol, f"facts_cache{cache_suffix}.npz"
    )
    if force_recompute and os.path.exists(real_cache):
        os.remove(real_cache)

    click.echo(f"Extracting real stylized facts for {symbol} …")
    real_facts = compute_real_facts(symbol, dates, config, cache_path=real_cache)

    # ---- build real KDE ----
    real_kde_data = build_real_kde(real_facts)
    real_level_prices = compute_real_level_prices(
        symbol, dates, config, real_facts=real_facts
    )
    sim_level_prices = compute_sim_level_prices(scenario_dir)

    # ---- parse weights ----
    parsed_weights: dict[str, float] | None = None
    if weights:
        try:
            parsed_weights = {k: float(v) for k, v in json.loads(weights).items()}
        except (json.JSONDecodeError, ValueError) as exc:
            click.echo(f"Error parsing --weights JSON: {exc}", err=True)
            sys.exit(1)

    # ---- compute loss ----
    specification = None
    if loss_config is not None:
        with open(loss_config, "r") as stream:
            specification = LossSpec.from_dict(yaml.safe_load(stream))
    loss = compute_loss(
        sim_kde_data,
        real_kde_data,
        weights=parsed_weights,
        specification=specification,
    )
    click.echo(f"Total loss: {loss.total:.6f}")
    for fact, value in sorted(loss.per_fact.items()):
        click.echo(f"  {fact:25s} {value:.6f}")

    # ---- save outputs ----
    save_comparison_plots(
        sim_kde_data,
        real_kde_data,
        output_dir,
        n_scenarios,
        sim_level_prices=sim_level_prices,
        real_level_prices=real_level_prices,
        sim_n_obs=sim_n_obs,
        real_n_obs=max(
            1, int(round(float(config["window_duration_hours"]) * 3600)) - 1
        ),
    )
    save_loss_report(loss, output_dir)
    save_statistics_report(sim_kde_data, real_kde_data, output_dir)
    click.echo(f"Done. Plots and reports written to: {output_dir}")


@cli.command("bootstrap-loss")
@click.option("--baseline-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--candidate-dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--real-cache", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--loss-config", default=None, type=click.Path(exists=True, dir_okay=False))
@click.option("--n-bootstrap", default=500, show_default=True, type=click.IntRange(min=1))
@click.option("--seed", default=2026, show_default=True, type=int)
@click.option("--output", default=None, type=click.Path(dir_okay=False))
def cmd_bootstrap_loss(
    baseline_dir: str,
    candidate_dir: str,
    real_cache: str,
    loss_config: str | None,
    n_bootstrap: int,
    seed: int,
    output: str | None,
) -> None:
    """Estimate uncertainty in a matched candidate-minus-baseline loss."""
    from pathlib import Path
    import numpy as np
    import yaml
    from lob_sim.loss.bootstrap import paired_bootstrap_loss
    from lob_sim.loss.compare import LossSpec

    cached = np.load(real_cache, allow_pickle=False)
    real_facts = {key: cached[key] for key in cached.files if not key.startswith("__")}
    specification = (
        LossSpec.from_dict(yaml.safe_load(Path(loss_config).read_text()))
        if loss_config is not None else None
    )
    result = paired_bootstrap_loss(
        baseline_dir,
        candidate_dir,
        real_facts,
        n_bootstrap=n_bootstrap,
        seed=seed,
        specification=specification,
    )
    if output is None:
        output = os.path.join(candidate_dir, "paired_bootstrap_loss.json")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    low, high = result["candidate_minus_baseline_ci95"]
    click.echo(
        f"Candidate − baseline: {result['candidate_minus_baseline']:.6f} "
        f"(95% bootstrap CI [{low:.6f}, {high:.6f}])"
    )
    click.echo(f"Result written to: {output}")


@cli.command("optimize")
@click.option(
    "--config", "config_path", required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Sampler-neutral optimization YAML configuration.",
)
@click.option(
    "--sampler", "sampler_name", required=True,
    type=click.Choice(
        ["random", "tpe", "opro_plain", "opro_heuristic"],
        case_sensitive=False,
    ),
)
@click.option("--repetition", default=0, show_default=True, type=click.IntRange(min=0))
@click.option("--budget", default=None, type=click.IntRange(min=1))
def cmd_optimize(
    config_path: str,
    sampler_name: str,
    repetition: int,
    budget: int | None,
) -> None:
    """Run or resume one equal-budget calibration-policy repetition."""
    from lob_sim.optimization.study import run_study

    records, output_dir = run_study(
        config_path, sampler_name.lower(), repetition, budget
    )
    objectives = [record["objective"] for record in records if record["objective"] is not None]
    best = min(objectives) if objectives else None
    click.echo(f"Completed proposals: {len(records)}; best valid loss: {best}")
    click.echo(f"Study written to: {output_dir}")


@cli.command("optimization-report")
@click.option(
    "--study-dir", required=True,
    type=click.Path(exists=True, file_okay=False),
)
def cmd_optimization_report(study_dir: str) -> None:
    """Generate benchmark tables and best-so-far curves from trial ledgers."""
    from lob_sim.optimization.report import write_benchmark_report

    summary, output_dir = write_benchmark_report(study_dir)
    click.echo(f"Methods found: {', '.join(summary['methods'])}")
    click.echo(f"Report written to: {output_dir}")
