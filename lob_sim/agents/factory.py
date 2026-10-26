from __future__ import annotations
import hashlib
from typing import Any

import numpy as np
import yaml

from lob_sim.archetypes.base import BaseArchetype
from lob_sim.archetypes.market_maker import MarketMakerAgent
from lob_sim.archetypes.multi_level_market_maker import MultiLevelMarketMakerAgent
from lob_sim.archetypes.momentum_trader import MomentumTraderAgent
from lob_sim.archetypes.mean_reversion_trader import MeanReversionTraderAgent
from lob_sim.archetypes.noise_trader import NoiseTraderAgent
from lob_sim.archetypes.random_limit_order_trader import RandomLimitOrderTraderAgent
from lob_sim.archetypes.value_trader import ValueTraderAgent
from lob_sim.core.models import Side


# Supported distribution types for parameter sampling
_DISTRIBUTIONS = {"fixed", "uniform", "normal", "lognormal", "exponential"}


def _agent_rng(seed: int, population_id: str, instance_index: int) -> np.random.Generator:
    """Return a stream stable to population reordering and leave-one-out tests."""
    payload = f"{int(seed)}\0{population_id}\0{int(instance_index)}".encode()
    digest = hashlib.sha256(payload).digest()
    entropy = [
        int(seed) & 0xFFFFFFFF,
        *[
            int.from_bytes(digest[offset:offset + 4], "little")
            for offset in range(0, 16, 4)
        ],
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _sample(spec: Any, rng: np.random.Generator) -> Any:
    """
    Sample a parameter value from a spec dict or return a scalar directly.

    Supported spec shapes:
      {type: fixed,       value: <number or string>}
      {type: uniform,     low: ...,  high: ...}
      {type: normal,      mean: ..., std: ...}
      {type: lognormal,   mean: ..., std: ...}
      {type: exponential, scale: ...}
    """
    if not isinstance(spec, dict):
        return spec   # bare scalar/string in YAML

    dist = spec.get("type", "fixed")
    if dist == "fixed":
        return spec["value"]
    elif dist == "uniform":
        return float(rng.uniform(spec["low"], spec["high"]))
    elif dist == "normal":
        return float(rng.normal(spec["mean"], spec["std"]))
    elif dist == "lognormal":
        return float(rng.lognormal(spec["mean"], spec["std"]))
    elif dist == "exponential":
        return float(rng.exponential(spec["scale"]))
    else:
        raise ValueError(f"Unknown distribution type '{dist}'. Choices: {_DISTRIBUTIONS}")


class AgentFactory:
    """
    Instantiates a list of agents from a YAML configuration file.

    Each entry in the YAML `agents` list must specify:
      - id:         stable population identifier (unique within the file)
      - archetype:  a registered market-agent implementation
      - count:      number of agents to create
      - parameters: mapping of parameter name → scalar or distribution spec

    Parameters marked as distributions are independently re-sampled for each
    agent instance, enabling heterogeneous populations. Random streams are
    derived from ``(scenario seed, population id, instance index)`` so adding,
    removing, or reordering another population does not silently change them.
    """

    @staticmethod
    def from_yaml(
        agents_yaml: str,
        sim_time_step: float,
        seed: int,
    ) -> list[BaseArchetype]:
        with open(agents_yaml, "r") as f:
            config = yaml.safe_load(f)

        agents: list[BaseArchetype] = []
        population_ids: set[str] = set()

        for entry in config.get("agents", []):
            archetype_name: str = entry["archetype"]
            population_id = str(entry.get("id", "")).strip()
            if not population_id:
                raise ValueError(
                    f"Agent population '{archetype_name}' requires a stable id"
                )
            if population_id in population_ids:
                raise ValueError(f"Duplicate agent population id: {population_id}")
            population_ids.add(population_id)
            count: int = int(entry.get("count", 1))
            params: dict = entry.get("parameters", {})

            for instance_index in range(count):
                agent_id = f"{population_id}_{instance_index:04d}"
                agent = AgentFactory._build(
                    archetype_name=archetype_name,
                    agent_id=agent_id,
                    params=params,
                    sim_time_step=sim_time_step,
                    rng=_agent_rng(seed, population_id, instance_index),
                )
                agents.append(agent)

        return agents

    @staticmethod
    def _build(
        archetype_name: str,
        agent_id: str,
        params: dict,
        sim_time_step: float,
        rng: np.random.Generator,
    ) -> BaseArchetype:
        p = {k: _sample(v, rng) for k, v in params.items()}

        name = archetype_name.lower().replace("_", "").replace(" ", "")

        if name in ("noisetrader", "noise_trader", "noisetrader"):
            agent_rng = np.random.default_rng(rng.integers(0, 2**32))
            side_name = str(p.get("side", "random")).lower()
            side = None if side_name in ("random", "both") else Side(side_name)
            return NoiseTraderAgent(
                agent_id=agent_id,
                side=side,
                order_frequency=float(p["activity"]),
                order_size=float(p["order_size"]),
                rng=agent_rng,
                buy_probability=float(p.get("buy_probability", 0.5)),
                order_size_sigma=float(p.get("order_size_sigma", 0.0)),
                max_order_size=float(p["max_order_size"]) if "max_order_size" in p else None,
                lot_size=float(p.get("lot_size", 100.0)),
                burst_size=float(p.get("burst_size", 0.0)),
                pause_probability=float(p.get("pause_probability", 0.0)),
                mean_pause_steps=float(p.get("mean_pause_steps", 1.0)),
                side_persistence=float(p.get("side_persistence", 0.0)),
            )

        elif name in ("marketmaker", "market_maker"):
            return MarketMakerAgent(
                agent_id=agent_id,
                half_spread=float(p["spread"]),
                order_size=float(p["order_size"]),
                max_inventory=float(p["inventory_limit"]),
            )

        elif name in ("multilevelmarketmaker", "multi_level_market_maker", "multilevelmarketmaker"):
            agent_rng = np.random.default_rng(rng.integers(0, 2**32))
            return MultiLevelMarketMakerAgent(
                agent_id=agent_id,
                n_levels=int(p["levels"]),
                half_spread=float(p["spread"]),
                level_spacing=2 * float(p["spread"]),
                order_size=float(p["order_size"]),
                max_inventory=float(p["inventory_limit"]),
                tick_size=float(p.get("tick_size", 0.01)),
                inventory_skew_ticks=int(p.get("inventory_skew_ticks", 0)),
                level_size_growth=float(p.get("level_size_growth", 0.0)),
                rng=agent_rng,
                wide_spread_probability=float(p.get("wide_spread_probability", 0.0)),
                spread_state_persistence=float(p.get("spread_state_persistence", 0.95)),
                level_sizes=p.get("level_sizes"),
                inventory_size_skew=float(p.get("inventory_size_skew", 0.0)),
                recenter_rate=float(p.get("recenter_rate", 0.0)),
                recenter_threshold_ticks=float(
                    p.get("recenter_threshold_ticks", 1.0)
                ),
            )

        elif name in ("momentumtrader", "momentum_trader"):
            agent_rng = np.random.default_rng(rng.integers(0, 2**32))
            return MomentumTraderAgent(
                agent_id=agent_id,
                lookback_steps=int(p["lookback"]),
                order_size=float(p["order_size"]),
                entry_threshold=float(p["threshold"]),
                order_frequency=float(p.get("activity", 1.0)),
                max_inventory=float(p.get("inventory_limit", float("inf"))),
                limit_timeout=int(p.get("limit_timeout", 5)),
                rng=agent_rng,
            )

        elif name in ("randomlimitordertrader", "random_limit_order_trader"):
            agent_rng = np.random.default_rng(rng.integers(0, 2**32))
            return RandomLimitOrderTraderAgent(
                agent_id=agent_id,
                order_frequency=float(p["activity"]),
                order_size=float(p["order_size"]),
                max_offset=float(p["price_offset"]),
                rng=agent_rng,
                tick_size=float(p.get("tick_size", 0.01)),
                inside_probability=float(p.get("inside_probability", 0.1)),
                cancel_probability=float(p.get("cancel_probability", 0.01)),
                best_probability=float(p.get("best_probability", 0.0)),
                imbalance_sensitivity=float(p.get("imbalance_sensitivity", 0.0)),
            )

        elif name in ("valuetrader", "value_trader"):
            agent_rng = np.random.default_rng(rng.integers(0, 2**32))
            return ValueTraderAgent(
                agent_id=agent_id,
                fundamental_price=(
                    float(p["fundamental_price"])
                    if p.get("fundamental_price") is not None else None
                ),
                value_volatility=float(p.get("value_volatility", 0.0)),
                signal_noise=float(p.get("signal_noise", 0.0)),
                order_frequency=float(p["activity"]),
                order_size=float(p["order_size"]),
                threshold=float(p.get("threshold", 0.0)),
                max_inventory=float(p["inventory_limit"]),
                rng=agent_rng,
            )

        elif name in ("meanreversiontrader", "mean_reversion_trader"):
            agent_rng = np.random.default_rng(rng.integers(0, 2**32))
            return MeanReversionTraderAgent(
                agent_id=agent_id,
                reference_horizon=int(p["reference_horizon"]),
                order_frequency=float(p["activity"]),
                order_size=float(p["order_size"]),
                threshold=float(p.get("threshold", 0.0)),
                max_inventory=float(p["inventory_limit"]),
                rng=agent_rng,
            )

        else:
            raise ValueError(
                f"Unknown archetype '{archetype_name}'. "
                "Valid options: NoiseTrader, MarketMaker, MultiLevelMarketMaker, "
                "MomentumTrader, MeanReversionTrader, "
                "ValueTrader, RandomLimitOrderTrader"
            )
