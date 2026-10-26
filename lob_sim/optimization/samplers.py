"""Proposal-policy adapters for the common calibration evaluator."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lob_sim.optimization.space import SearchSpace


@dataclass
class Proposal:
    candidate: dict[str, Any]
    token: Any = None
    metadata: dict[str, Any] | None = None


class OPROProposalError(ValueError):
    """Malformed provider output carrying enough metadata for an audit trail."""

    def __init__(self, message: str, metadata: dict[str, Any]) -> None:
        super().__init__(message)
        self.proposal_metadata = metadata


class RandomPolicy:
    """Independent seeded draws, indexed so interrupted runs resume exactly."""

    name = "random"
    feedback_mode = "none"

    def __init__(self, space: SearchSpace, seed: int) -> None:
        self.space = space
        self.seed = int(seed)

    def ask(
        self,
        trial_index: int,
        fixed: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        remaining_budget: int | None = None,
    ) -> Proposal:
        if fixed is not None:
            return Proposal(self.space.validate(fixed), metadata={"source": "initial_design"})
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, trial_index]))
        return Proposal(self.space.sample(rng), metadata={"source": "random"})

    def tell(self, proposal: Proposal, result: dict[str, Any]) -> None:
        return None


class OptunaTPEPolicy:
    """Persistent Optuna TPE adapter using the same explicit ask/tell loop."""

    name = "tpe"
    feedback_mode = "scalar"

    def __init__(
        self,
        space: SearchSpace,
        seed: int,
        storage_path: Path,
        study_name: str,
        initial_design: list[dict[str, Any]],
    ) -> None:
        try:
            import optuna
        except ImportError as exc:
            raise RuntimeError(
                "Optuna is required for the TPE arm; install the project dependencies"
            ) from exc
        self.optuna = optuna
        self.space = space
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        sampler = optuna.samplers.TPESampler(
            seed=int(seed),
            n_startup_trials=len(initial_design),
            multivariate=True,
        )
        self.study = optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{storage_path}",
            direction="minimize",
            sampler=sampler,
            load_if_exists=True,
        )
        if not self.study.trials:
            for candidate in initial_design:
                self.study.enqueue_trial(self._encode(self.space.validate(candidate)))

    def _encode(self, candidate: dict[str, Any]) -> dict[str, Any]:
        encoded = dict(candidate)
        for dimension in self.space.dimensions:
            if dimension.kind == "categorical":
                encoded[dimension.name] = next(
                    index for index, choice in enumerate(dimension.choices)
                    if candidate[dimension.name] == choice
                )
        return encoded

    def ask(
        self,
        trial_index: int,
        fixed: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        remaining_budget: int | None = None,
    ) -> Proposal:
        trial = self.study.ask()
        candidate: dict[str, Any] = {}
        for dimension in self.space.dimensions:
            if dimension.kind == "float":
                candidate[dimension.name] = trial.suggest_float(
                    dimension.name,
                    float(dimension.low),
                    float(dimension.high),
                    step=float(dimension.step) if dimension.step is not None else None,
                    log=dimension.log,
                )
            elif dimension.kind == "int":
                candidate[dimension.name] = trial.suggest_int(
                    dimension.name,
                    int(dimension.low),
                    int(dimension.high),
                    step=int(dimension.step or 1),
                    log=dimension.log,
                )
            else:
                choice_index = trial.suggest_categorical(
                    dimension.name, list(range(len(dimension.choices)))
                )
                candidate[dimension.name] = dimension.choices[int(choice_index)]
        candidate = self.space.validate(candidate)
        if fixed is not None and self.space.canonical_key(candidate) != self.space.canonical_key(fixed):
            raise RuntimeError("Persistent TPE initial-design queue is out of sync")
        return Proposal(
            candidate,
            token=trial,
            metadata={"source": "initial_design" if fixed is not None else "tpe", "optuna_trial": trial.number},
        )

    def tell(self, proposal: Proposal, result: dict[str, Any]) -> None:
        if proposal.token is None:
            raise ValueError("Missing Optuna trial token")
        feedback = result.get("optimizer_feedback")
        if not isinstance(feedback, dict) or "final_loss" not in feedback:
            raise ValueError("TPE requires scalar optimizer feedback")
        self.study.tell(proposal.token, float(feedback["final_loss"]))
