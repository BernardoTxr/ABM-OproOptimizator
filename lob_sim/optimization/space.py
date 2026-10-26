"""Typed, bounded population-level parameter spaces."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SearchDimension:
    """One optimizer-visible variable and the agent fields it controls."""

    name: str
    kind: str
    targets: tuple[tuple[str, str], ...]
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False
    choices: tuple[Any, ...] = ()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SearchDimension":
        name = str(config["name"])
        kind = str(config["type"]).lower()
        if kind not in {"float", "int", "categorical"}:
            raise ValueError(f"Unsupported search type for {name}: {kind}")
        targets = tuple(
            (str(target["agent"]), str(target["field"]))
            for target in config.get("targets", [])
        )
        if not targets:
            raise ValueError(f"Search dimension '{name}' requires at least one target")

        if kind == "categorical":
            choices = tuple(config.get("choices", []))
            if not choices:
                raise ValueError(f"Categorical dimension '{name}' has no choices")
            return cls(name=name, kind=kind, targets=targets, choices=choices)

        low = config.get("low")
        high = config.get("high")
        if low is None or high is None or float(low) > float(high):
            raise ValueError(f"Numeric dimension '{name}' has invalid bounds")
        step = config.get("step")
        if step is not None and float(step) <= 0:
            raise ValueError(f"Numeric dimension '{name}' requires step > 0")
        if bool(config.get("log", False)) and float(low) <= 0:
            raise ValueError(f"Log dimension '{name}' requires low > 0")
        numeric = int if kind == "int" else float
        return cls(
            name=name,
            kind=kind,
            targets=targets,
            low=numeric(low),
            high=numeric(high),
            step=numeric(step) if step is not None else None,
            log=bool(config.get("log", False)),
        )

    def validate(self, value: Any) -> Any:
        """Return a normalized value or raise for a malformed proposal."""
        if self.kind == "categorical":
            for choice in self.choices:
                if value == choice:
                    return copy.deepcopy(choice)
            raise ValueError(f"{self.name} is not one of the declared choices")

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{self.name} must be numeric")
        normalized: float | int
        if self.kind == "int":
            if float(value) != int(value):
                raise ValueError(f"{self.name} must be an integer")
            normalized = int(value)
        else:
            normalized = float(value)
        assert self.low is not None and self.high is not None
        tolerance = 1e-10 * max(1.0, abs(float(self.low)), abs(float(self.high)))
        if normalized < float(self.low) - tolerance or normalized > float(self.high) + tolerance:
            raise ValueError(
                f"{self.name}={normalized} outside [{self.low}, {self.high}]"
            )
        normalized = min(float(self.high), max(float(self.low), normalized))
        if self.kind == "int":
            normalized = int(round(normalized))
        if self.step is not None:
            offset = (float(normalized) - float(self.low)) / float(self.step)
            if not np.isclose(offset, round(offset), atol=1e-8):
                raise ValueError(f"{self.name} does not lie on its declared step")
            snapped = float(self.low) + round(offset) * float(self.step)
            snapped = min(float(self.high), max(float(self.low), snapped))
            normalized = int(round(snapped)) if self.kind == "int" else float(snapped)
        return normalized

    def sample(self, rng: np.random.Generator) -> Any:
        if self.kind == "categorical":
            return copy.deepcopy(self.choices[int(rng.integers(len(self.choices)))])
        assert self.low is not None and self.high is not None
        if self.kind == "int":
            step = int(self.step or 1)
            count = (int(self.high) - int(self.low)) // step + 1
            return int(self.low) + step * int(rng.integers(count))
        if self.step is not None:
            # Draw grid points directly. Rounding a continuous uniform draw
            # would give the two endpoints only half the mass of interior
            # values, weakening the random-search baseline.
            count = int(round(
                (float(self.high) - float(self.low)) / float(self.step)
            )) + 1
            index = int(rng.integers(count))
            return min(
                float(self.high),
                float(self.low) + index * float(self.step),
            )
        if self.log:
            value = float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
        else:
            value = float(rng.uniform(self.low, self.high))
        return value

    def from_unit_interval(self, value: float) -> Any:
        """Map one point in ``[0, 1)`` into this dimension's prior measure."""
        unit = float(value)
        if not 0.0 <= unit < 1.0:
            raise ValueError("Unit-cube coordinates must satisfy 0 <= value < 1")
        if self.kind == "categorical":
            return copy.deepcopy(self.choices[min(int(unit * len(self.choices)), len(self.choices) - 1)])
        assert self.low is not None and self.high is not None
        if self.kind == "int":
            step = int(self.step or 1)
            count = (int(self.high) - int(self.low)) // step + 1
            return int(self.low) + step * min(int(unit * count), count - 1)
        if self.step is not None:
            count = int(round(
                (float(self.high) - float(self.low)) / float(self.step)
            )) + 1
            index = min(int(unit * count), count - 1)
            return min(float(self.high), float(self.low) + index * float(self.step))
        if self.log:
            return float(np.exp(
                np.log(float(self.low))
                + unit * (np.log(float(self.high)) - np.log(float(self.low)))
            ))
        return float(self.low) + unit * (float(self.high) - float(self.low))


class SearchSpace:
    """Validated grouped variables applied to agent-population YAML."""

    def __init__(self, dimensions: list[SearchDimension]) -> None:
        if not dimensions:
            raise ValueError("Search space must contain at least one dimension")
        names = [dimension.name for dimension in dimensions]
        if len(names) != len(set(names)):
            raise ValueError("Search dimension names must be unique")
        self.dimensions = tuple(dimensions)

    @classmethod
    def from_config(cls, values: list[dict[str, Any]]) -> "SearchSpace":
        return cls([SearchDimension.from_config(value) for value in values])

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(dimension.name for dimension in self.dimensions)

    def validate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        missing = sorted(set(self.names) - set(candidate))
        extra = sorted(set(candidate) - set(self.names))
        if missing or extra:
            raise ValueError(f"Candidate schema mismatch: missing={missing}, extra={extra}")
        return {
            dimension.name: dimension.validate(candidate[dimension.name])
            for dimension in self.dimensions
        }

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        return {
            dimension.name: dimension.sample(rng)
            for dimension in self.dimensions
        }

    def sobol(self, n_candidates: int, seed: int) -> list[dict[str, Any]]:
        """Return a reproducible scrambled Sobol space-filling design."""
        if n_candidates <= 0:
            raise ValueError("n_candidates must be positive")
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=len(self.dimensions), scramble=True, seed=int(seed))
        exponent = int(np.ceil(np.log2(n_candidates)))
        points = sampler.random_base2(exponent)[:n_candidates]
        return [
            self.validate({
                dimension.name: dimension.from_unit_interval(float(point[index]))
                for index, dimension in enumerate(self.dimensions)
            })
            for point in points
        ]

    def apply(self, agents_config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        normalized = self.validate(candidate)
        result = copy.deepcopy(agents_config)
        entries = {entry.get("id"): entry for entry in result.get("agents", [])}
        if None in entries:
            raise ValueError("Every optimizer-visible agent requires a stable id")
        for dimension in self.dimensions:
            value = normalized[dimension.name]
            for agent_id, field in dimension.targets:
                if agent_id not in entries:
                    raise KeyError(f"Unknown optimizer target agent: {agent_id}")
                current: dict[str, Any] = entries[agent_id]
                parts = field.split(".")
                for part in parts[:-1]:
                    current = current[part]
                current[parts[-1]] = copy.deepcopy(value)
        return result

    def canonical_key(self, candidate: dict[str, Any]) -> str:
        normalized = self.validate(candidate)
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"))

    def prompt_schema(self) -> list[dict[str, Any]]:
        """JSON-serializable bounds supplied identically to both LLM arms."""
        schema: list[dict[str, Any]] = []
        for dimension in self.dimensions:
            item: dict[str, Any] = {"name": dimension.name, "type": dimension.kind}
            if dimension.kind == "categorical":
                item["choices"] = copy.deepcopy(list(dimension.choices))
            else:
                item.update({"low": dimension.low, "high": dimension.high})
                if dimension.step is not None:
                    item["step"] = dimension.step
                if dimension.log:
                    item["log"] = True
            schema.append(item)
        return schema
