"""Optional policy adapters for classical and component-aware comparisons."""
from __future__ import annotations

from pathlib import Path
import queue
import threading
from typing import Any

import numpy as np

from lob_sim.optimization.feedback import SMM_COMPONENTS
from lob_sim.optimization.samplers import Proposal
from lob_sim.optimization.space import SearchSpace


class _ExtensionPolicy:
    name = "extension"
    feedback_mode = "none"

    def __init__(self, extension_config_sha256: str) -> None:
        self.extension_config_sha256 = str(extension_config_sha256)

    def _metadata(self, source: str, **values: Any) -> dict[str, Any]:
        return {
            "source": source,
            "extension_config_sha256": self.extension_config_sha256,
            **values,
        }

    def _validate_resume(self, history: list[dict[str, Any]]) -> None:
        hashes = {
            row.get("proposal_metadata", {}).get("extension_config_sha256")
            for row in history
            if row.get("sampler") == self.name
            and row.get("proposal_metadata", {}).get("source") != "initial_design"
        }
        hashes.discard(None)
        if hashes and hashes != {self.extension_config_sha256}:
            raise RuntimeError(
                f"{self.name} extension configuration changed since its ledger began"
            )


class SobolPolicy(_ExtensionPolicy):
    """Continue the frozen Sobol design in disjoint repetition blocks."""

    name = "sobol"

    def __init__(
        self,
        space: SearchSpace,
        sequence_seed: int,
        initial_design_count: int,
        adaptive_start: int,
        extension_config_sha256: str,
    ) -> None:
        super().__init__(extension_config_sha256)
        self.space = space
        self.sequence_seed = int(sequence_seed)
        self.initial_design_count = int(initial_design_count)
        self.adaptive_start = int(adaptive_start)

    def ask(
        self,
        trial_index: int,
        fixed: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        remaining_budget: int | None = None,
    ) -> Proposal:
        if fixed is not None:
            return Proposal(
                self.space.validate(fixed),
                metadata=self._metadata("initial_design"),
            )
        self._validate_resume(history or [])
        sequence_index = self.adaptive_start + (
            int(trial_index) - self.initial_design_count
        )
        candidate = self.space.sobol(
            sequence_index + 1, self.sequence_seed
        )[sequence_index]
        return Proposal(candidate, metadata=self._metadata(
            self.name,
            sobol_seed=self.sequence_seed,
            sobol_index=sequence_index,
        ))

    def tell(self, proposal: Proposal, result: dict[str, Any]) -> None:
        return None


def _require_optuna(feature: str):
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            f"Optuna is required for {feature}; install the primary dependencies"
        ) from exc
    return optuna


class _PersistentOptunaPolicy(_ExtensionPolicy):
    """Shared persistent adapter for scalar Optuna ask/tell samplers."""

    feedback_mode = "scalar"

    def __init__(
        self,
        space: SearchSpace,
        storage_path: Path,
        study_name: str,
        initial_design: list[dict[str, Any]],
        sampler: Any,
        optuna: Any,
        extension_config_sha256: str,
        directions: list[str] | None = None,
    ) -> None:
        super().__init__(extension_config_sha256)
        self.optuna = optuna
        self.space = space
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        study_options = {
            "study_name": study_name,
            "storage": f"sqlite:///{storage_path}",
            "sampler": sampler,
            "load_if_exists": True,
        }
        if directions is None:
            study_options["direction"] = "minimize"
        else:
            study_options["directions"] = directions
        self.study = optuna.create_study(**study_options)
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
        if fixed is None:
            self._validate_resume(history or [])
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
        if (
            fixed is not None
            and self.space.canonical_key(candidate)
            != self.space.canonical_key(fixed)
        ):
            raise RuntimeError(
                f"Persistent {self.name} initial-design queue is out of sync"
            )
        return Proposal(
            candidate,
            token=trial,
            metadata=self._metadata(
                "initial_design" if fixed is not None else self.name,
                optuna_trial=trial.number,
            ),
        )

    def tell(self, proposal: Proposal, result: dict[str, Any]) -> None:
        if proposal.token is None:
            raise ValueError("Missing Optuna trial token")
        feedback = result.get("optimizer_feedback")
        if not isinstance(feedback, dict) or "final_loss" not in feedback:
            raise ValueError(f"{self.name} requires scalar optimizer feedback")
        self.study.tell(proposal.token, float(feedback["final_loss"]))


class OptunaGPPolicy(_PersistentOptunaPolicy):
    """Matérn-5/2 ARD GP Bayesian optimization with deterministic log-EI."""

    name = "gp_bo"

    def __init__(
        self,
        space: SearchSpace,
        seed: int,
        storage_path: Path,
        study_name: str,
        initial_design: list[dict[str, Any]],
        extension_config_sha256: str,
    ) -> None:
        optuna = _require_optuna("GP-BO")
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "GP-BO requires CPU PyTorch; install optimizer-extension.lock.txt"
            ) from exc
        sampler = optuna.samplers.GPSampler(
            seed=int(seed),
            n_startup_trials=len(initial_design),
            deterministic_objective=True,
        )
        super().__init__(
            space, storage_path, study_name, initial_design, sampler, optuna,
            extension_config_sha256,
        )


class MultiObjectiveTPEPolicy(_PersistentOptunaPolicy):
    """Optuna MOTPE over the six grouped losses with scalar-loss auditing."""

    name = "mo_tpe"
    feedback_mode = "component"

    def __init__(
        self,
        space: SearchSpace,
        seed: int,
        storage_path: Path,
        study_name: str,
        initial_design: list[dict[str, Any]],
        extension_config_sha256: str,
    ) -> None:
        optuna = _require_optuna("multi-objective TPE")
        sampler = optuna.samplers.TPESampler(
            seed=int(seed),
            n_startup_trials=len(initial_design),
            multivariate=True,
        )
        super().__init__(
            space,
            storage_path,
            study_name,
            initial_design,
            sampler,
            optuna,
            extension_config_sha256,
            directions=["minimize"] * len(SMM_COMPONENTS),
        )

    def tell(self, proposal: Proposal, result: dict[str, Any]) -> None:
        if proposal.token is None:
            raise ValueError("Missing Optuna trial token")
        feedback = result.get("optimizer_feedback")
        if not isinstance(feedback, dict):
            raise ValueError("Multi-objective TPE requires component feedback")
        components = feedback.get("component_losses")
        if not isinstance(components, dict) or tuple(
            feedback.get("component_order", ())
        ) != SMM_COMPONENTS:
            raise ValueError("Multi-objective TPE received an invalid component vector")
        proposal.token.set_user_attr("final_loss", float(feedback["final_loss"]))
        proposal.token.set_user_attr("feasible", bool(feedback["feasible"]))
        self.study.tell(
            proposal.token,
            [float(components[name]) for name in SMM_COMPONENTS],
        )


class ComponentAwareGPPolicy(_ExtensionPolicy):
    """Six independent GPs with EI on the known equal-weight final loss."""

    name = "component_gp_bo"
    feedback_mode = "component"

    def __init__(
        self,
        space: SearchSpace,
        seed: int,
        initial_design_count: int,
        extension_config_sha256: str,
        *,
        candidate_pool_size: int = 4096,
        exploration: float = 0.01,
    ) -> None:
        super().__init__(extension_config_sha256)
        if any(
            dimension.kind != "float" or dimension.step is not None
            for dimension in space.dimensions
        ):
            raise ValueError(
                "Component-aware GP-BO requires a continuous, unstepped search space"
            )
        if candidate_pool_size <= 0 or candidate_pool_size & (candidate_pool_size - 1):
            raise ValueError("candidate_pool_size must be a positive power of two")
        self.space = space
        self.seed = int(seed)
        self.initial_design_count = int(initial_design_count)
        self.candidate_pool_size = int(candidate_pool_size)
        self.exploration = float(exploration)

    def _to_unit(self, candidate: dict[str, Any]) -> np.ndarray:
        normalized = self.space.validate(candidate)
        values = []
        for dimension in self.space.dimensions:
            assert dimension.low is not None and dimension.high is not None
            value = float(normalized[dimension.name])
            low, high = float(dimension.low), float(dimension.high)
            if dimension.log:
                value, low, high = np.log(value), np.log(low), np.log(high)
            values.append((value - low) / (high - low))
        return np.asarray(values, dtype=float)

    def _fit_models(self, x: np.ndarray, y: np.ndarray, trial_index: int):
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        except ImportError as exc:
            raise RuntimeError(
                "Component-aware GP-BO requires scikit-learn; install "
                "component-aware.lock.txt"
            ) from exc
        models = []
        for component_index in range(len(SMM_COMPONENTS)):
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * Matern(
                    length_scale=np.ones(x.shape[1]),
                    length_scale_bounds=(1e-3, 1e3),
                    nu=2.5,
                )
                + WhiteKernel(1e-6, (1e-10, 1e1))
            )
            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-10,
                normalize_y=True,
                n_restarts_optimizer=1,
                random_state=self.seed + 1000 * int(trial_index) + component_index,
            )
            model.fit(x, y[:, component_index])
            models.append(model)
        return models

    def ask(
        self,
        trial_index: int,
        fixed: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        remaining_budget: int | None = None,
    ) -> Proposal:
        if fixed is not None:
            return Proposal(
                self.space.validate(fixed),
                metadata=self._metadata("initial_design"),
            )
        history = history or []
        self._validate_resume(history)
        usable = [
            record for record in history
            if record.get("candidate") is not None
            and isinstance(record.get("optimizer_feedback"), dict)
        ]
        if len(usable) < self.initial_design_count:
            raise RuntimeError(
                "Component-aware GP-BO requires the complete common initial design"
            )
        x = np.asarray([self._to_unit(record["candidate"]) for record in usable])
        y = np.asarray([
            [
                float(record["optimizer_feedback"]["component_losses"][name])
                for name in SMM_COMPONENTS
            ]
            for record in usable
        ])
        models = self._fit_models(x, y, trial_index)
        candidates = self.space.sobol(
            self.candidate_pool_size,
            seed=self.seed + 100_000 + int(trial_index),
        )
        seen = {
            record.get("candidate_key")
            or self.space.canonical_key(record["candidate"])
            for record in usable
        }
        candidates = [
            candidate for candidate in candidates
            if self.space.canonical_key(candidate) not in seen
        ]
        if not candidates:
            raise RuntimeError("Component-aware GP-BO candidate pool is exhausted")
        pool = np.asarray([self._to_unit(candidate) for candidate in candidates])
        means, variances = [], []
        for model in models:
            mean, standard_deviation = model.predict(pool, return_std=True)
            means.append(mean)
            variances.append(np.square(standard_deviation))
        aggregate_mean = np.mean(np.asarray(means), axis=0)
        aggregate_std = np.sqrt(np.sum(np.asarray(variances), axis=0)) / len(
            SMM_COMPONENTS
        )
        best = min(
            float(record["optimizer_feedback"]["final_loss"])
            for record in usable
        )
        improvement = best - aggregate_mean - self.exploration
        safe_std = np.maximum(aggregate_std, np.finfo(float).eps)
        z = improvement / safe_std
        try:
            from scipy.special import ndtr
        except ImportError as exc:
            raise RuntimeError("Component-aware GP-BO requires SciPy") from exc
        expected_improvement = (
            improvement * ndtr(z)
            + safe_std * np.exp(-0.5 * np.square(z)) / np.sqrt(2.0 * np.pi)
        )
        selected_index = int(np.argmax(expected_improvement))
        return Proposal(
            candidates[selected_index],
            metadata=self._metadata(
                self.name,
                acquisition="expected_improvement_of_equal_weight_component_sum",
                component_order=list(SMM_COMPONENTS),
                independent_output_gps=True,
                candidate_pool_size=self.candidate_pool_size,
                exploration=self.exploration,
                acquisition_value=float(expected_improvement[selected_index]),
            ),
        )

    def tell(self, proposal: Proposal, result: dict[str, Any]) -> None:
        feedback = result.get("optimizer_feedback")
        if not isinstance(feedback, dict) or not isinstance(
            feedback.get("component_losses"), dict
        ):
            raise ValueError("Component-aware GP-BO requires component feedback")
        if tuple(feedback.get("component_order", ())) != SMM_COMPONENTS:
            raise ValueError("Component-aware GP-BO received an invalid component order")


class OptunaCMAESPolicy(_PersistentOptunaPolicy):
    """CMA-ES with the common design counted as its startup phase."""

    name = "cma_es"

    def __init__(
        self,
        space: SearchSpace,
        seed: int,
        storage_path: Path,
        study_name: str,
        initial_design: list[dict[str, Any]],
        extension_config_sha256: str,
    ) -> None:
        optuna = _require_optuna("CMA-ES")
        try:
            import cmaes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "CMA-ES requires cmaes; install optimizer-extension.lock.txt"
            ) from exc
        sampler = optuna.samplers.CmaEsSampler(
            seed=int(seed),
            n_startup_trials=len(initial_design),
            independent_sampler=optuna.samplers.RandomSampler(seed=int(seed)),
        )
        super().__init__(
            space, storage_path, study_name, initial_design, sampler, optuna,
            extension_config_sha256,
        )


class _BADSBridgeError(RuntimeError):
    pass


class BADSAskTellPolicy(_ExtensionPolicy):
    """Adapt PyBADS' synchronous callback to the persistent ask/tell loop."""

    name = "bads"
    feedback_mode = "scalar"

    def __init__(
        self,
        space: SearchSpace,
        seed: int,
        initial_design_count: int,
        extension_config_sha256: str,
        invalid_penalty: float = 1e6,
        bads_factory: Any | None = None,
    ) -> None:
        super().__init__(extension_config_sha256)
        if any(
            dimension.kind != "float" or dimension.step is not None
            for dimension in space.dimensions
        ):
            raise ValueError("BADS requires a continuous, unstepped search space")
        self.space = space
        self.seed = int(seed)
        self.initial_design_count = int(initial_design_count)
        self.invalid_penalty = float(invalid_penalty)
        if not np.isfinite(self.invalid_penalty):
            raise ValueError("BADS invalid penalty must be finite")
        self._bads_factory = bads_factory
        self._requests: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None

    def _to_unit(self, candidate: dict[str, Any]) -> np.ndarray:
        values = []
        normalized = self.space.validate(candidate)
        for dimension in self.space.dimensions:
            assert dimension.low is not None and dimension.high is not None
            value = float(normalized[dimension.name])
            low, high = float(dimension.low), float(dimension.high)
            if dimension.log:
                value, low, high = np.log(value), np.log(low), np.log(high)
            values.append((value - low) / (high - low))
        return np.asarray(values, dtype=float)

    def _from_unit(self, values: Any) -> dict[str, Any]:
        point = np.asarray(values, dtype=float).reshape(-1)
        if point.size != len(self.space.dimensions):
            raise ValueError("PyBADS returned a candidate with the wrong dimension")
        candidate: dict[str, Any] = {}
        for value, dimension in zip(point, self.space.dimensions):
            assert dimension.low is not None and dimension.high is not None
            unit = min(1.0, max(0.0, float(value)))
            low, high = float(dimension.low), float(dimension.high)
            if dimension.log:
                candidate[dimension.name] = float(np.exp(
                    np.log(low) + unit * (np.log(high) - np.log(low))
                ))
            else:
                candidate[dimension.name] = low + unit * (high - low)
        return self.space.validate(candidate)

    @staticmethod
    def _optimizer_value(record: dict[str, Any], invalid_penalty: float) -> float:
        feedback = record.get("optimizer_feedback")
        if isinstance(feedback, dict):
            value = feedback.get("final_loss")
            if value is not None and np.isfinite(float(value)):
                return float(value)
        objective = record.get("objective")
        if objective is not None and np.isfinite(float(objective)):
            return float(objective)
        stored = record.get("proposal_metadata", {}).get("optimizer_objective")
        if stored is not None and np.isfinite(float(stored)):
            return float(stored)
        return float(invalid_penalty)

    def _load_factory(self):
        if self._bads_factory is not None:
            return self._bads_factory
        try:
            from pybads import BADS
        except ImportError as exc:
            raise RuntimeError(
                "BADS requires pybads; install optimizer-extension.lock.txt"
            ) from exc
        return BADS

    def _start(
        self, history: list[dict[str, Any]], remaining_budget: int
    ) -> None:
        self._validate_resume(history)
        initial = history[: self.initial_design_count]
        ranked = [
            record for record in initial
            if record.get("objective") is not None
            and np.isfinite(float(record["objective"]))
        ]
        if not ranked:
            raise RuntimeError("BADS requires a valid common-design incumbent")
        incumbent = min(ranked, key=lambda record: float(record["objective"]))
        x0 = self._to_unit(incumbent["candidate"])
        replay = history[self.initial_design_count :]
        total_adaptive = len(replay) + int(remaining_budget)
        factory = self._load_factory()

        def worker() -> None:
            replay_index = 0

            def objective(values: Any) -> float:
                nonlocal replay_index
                candidate = self._from_unit(values)
                key = self.space.canonical_key(candidate)
                if replay_index < len(replay):
                    expected = replay[replay_index]
                    replay_index += 1
                    if key != expected.get("candidate_key"):
                        raise _BADSBridgeError(
                            "PyBADS replay diverged from the persisted candidate ledger"
                        )
                    return self._optimizer_value(expected, self.invalid_penalty)
                response: queue.Queue[float] = queue.Queue(maxsize=1)
                self._requests.put(("candidate", (candidate, response)))
                return float(response.get())

            try:
                options = {
                    "display": "off",
                    "uncertainty_handling": False,
                    "nonlinear_scaling": False,
                    "random_seed": self.seed,
                    "fun_eval_start": 0,
                    "max_fun_evals": total_adaptive,
                    "tol_mesh": 1e-12,
                }
                optimizer = factory(
                    objective,
                    x0,
                    np.zeros_like(x0),
                    np.ones_like(x0),
                    np.full_like(x0, 0.05),
                    np.full_like(x0, 0.95),
                    options=options,
                )
                optimizer.optimize()
                self._requests.put(("finished", None))
            except BaseException as exc:
                self._requests.put(("error", exc))

        self._thread = threading.Thread(
            target=worker, name=f"bads-{self.seed}", daemon=True
        )
        self._thread.start()

    def ask(
        self,
        trial_index: int,
        fixed: dict[str, Any] | None = None,
        history: list[dict[str, Any]] | None = None,
        remaining_budget: int | None = None,
    ) -> Proposal:
        if fixed is not None:
            return Proposal(
                self.space.validate(fixed),
                metadata=self._metadata("initial_design"),
            )
        if self._thread is None:
            self._start(history or [], int(remaining_budget or 0))
        kind, payload = self._requests.get()
        if kind == "error":
            raise RuntimeError(f"PyBADS worker failed: {payload}") from payload
        if kind == "finished":
            raise RuntimeError("PyBADS terminated before exhausting the charged budget")
        candidate, response = payload
        return Proposal(
            candidate,
            token=response,
            metadata=self._metadata(
                self.name,
                bads_seed=self.seed,
                invalid_penalty=self.invalid_penalty,
            ),
        )

    def tell(self, proposal: Proposal, result: dict[str, Any]) -> None:
        if proposal.token is None:
            if proposal.metadata and proposal.metadata.get("source") == "initial_design":
                return
            raise ValueError("Missing PyBADS response token")
        feedback = result.get("optimizer_feedback")
        if not isinstance(feedback, dict) or "final_loss" not in feedback:
            raise ValueError("BADS requires scalar optimizer feedback")
        value = float(feedback["final_loss"])
        hidden_constraint_violation = not bool(feedback.get("feasible"))
        if proposal.metadata is not None:
            proposal.metadata["optimizer_objective"] = value
            proposal.metadata["hidden_constraint_violation"] = (
                hidden_constraint_violation
            )
        proposal.token.put(value)
