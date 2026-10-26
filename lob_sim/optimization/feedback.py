"""Frozen numerical feedback contract shared by optimizer policies."""
from __future__ import annotations

from typing import Any

import numpy as np


SMM_COMPONENTS = (
    "returns",
    "spread",
    "imbalance",
    "trade_size",
    "inter_arrival",
    "depth",
)
FEEDBACK_MODES = frozenset({"none", "scalar", "component"})


def build_optimizer_feedback(
    result: dict[str, Any],
    *,
    component_names: tuple[str, ...] = SMM_COMPONENTS,
    invalid_penalty: float,
) -> dict[str, Any]:
    """Create the auditable six-loss/final-loss record for one evaluation.

    Scientific ranking continues to use ``result['objective']`` and excludes
    invalid simulations. The penalty exists only in the proposal-policy
    feedback channel so every adaptive method handles hidden feasibility under
    the same predeclared convention.
    """
    penalty = float(invalid_penalty)
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("The optimizer invalid penalty must be positive and finite")
    feasible = bool(result.get("valid")) and result.get("objective") is not None
    if feasible:
        raw_components = result.get("component_losses", {})
        if set(raw_components) != set(component_names):
            raise ValueError(
                "Evaluation component losses do not match the frozen feedback order"
            )
        components = {
            name: float(raw_components[name]) for name in component_names
        }
        if not all(np.isfinite(value) and value >= 0 for value in components.values()):
            raise ValueError("Optimizer component losses must be finite and nonnegative")
        final_loss = float(result["objective"])
        expected = float(np.mean(list(components.values())))
        if not np.isclose(final_loss, expected, rtol=1e-10, atol=1e-12):
            raise ValueError(
                "Final SMM loss is not the equal-weight mean of its six components"
            )
    else:
        components = {name: penalty for name in component_names}
        final_loss = penalty
    return {
        "version": "grouped_smm_feedback_v1",
        "feasible": feasible,
        "component_order": list(component_names),
        "component_losses": components,
        "final_loss": final_loss,
        "invalid_penalty_applied": not feasible,
    }


def policy_feedback(
    feedback: dict[str, Any], mode: str,
) -> dict[str, Any]:
    """Expose only the information declared for one proposal-policy family."""
    if mode not in FEEDBACK_MODES:
        raise ValueError(f"Unknown optimizer feedback mode: {mode}")
    if mode == "none":
        return {"optimizer_feedback": None}
    visible = {
        "version": feedback["version"],
        "feasible": feedback["feasible"],
        "final_loss": feedback["final_loss"],
        "invalid_penalty_applied": feedback["invalid_penalty_applied"],
    }
    if mode == "component":
        visible.update({
            "component_order": list(feedback["component_order"]),
            "component_losses": dict(feedback["component_losses"]),
        })
    return {"optimizer_feedback": visible}


def feedback_contract(config: dict[str, Any]) -> tuple[tuple[str, ...], float]:
    """Validate and return the executable feedback settings."""
    values = config.get("optimizer_feedback", {})
    if values.get("version") != "grouped_smm_feedback_v1":
        raise ValueError("optimizer_feedback.version must be grouped_smm_feedback_v1")
    names = tuple(str(name) for name in values.get("component_order", ()))
    if names != SMM_COMPONENTS:
        raise ValueError(f"optimizer_feedback.component_order must be {SMM_COMPONENTS}")
    if values.get("final_aggregation") != "equal_weight_mean":
        raise ValueError("optimizer_feedback.final_aggregation must be equal_weight_mean")
    penalty = float(values.get("invalid_penalty", float("nan")))
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("optimizer_feedback.invalid_penalty must be positive and finite")
    return names, penalty
