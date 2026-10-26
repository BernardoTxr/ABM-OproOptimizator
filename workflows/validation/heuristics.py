#!/usr/bin/env python3
"""Validate component-level mechanism guidance against the search space."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from lob_sim.optimization.feedback import SMM_COMPONENTS


def validate_registry(registry: dict, optimizer: dict) -> dict[str, int | str]:
    version = str(registry.get("version", ""))
    if version != "component_mechanisms_smm_v1":
        raise ValueError("Expected the component-level SMM mechanism registry")
    dimensions = {
        str(item["name"]): item for item in optimizer.get("search_space", [])
    }
    parameter_names = set(dimensions)
    mechanisms = registry.get("mechanisms", [])
    if not mechanisms:
        raise ValueError("Heuristic registry requires at least one mechanism")
    ids = [str(mechanism.get("id", "")) for mechanism in mechanisms]
    if any(not mechanism_id for mechanism_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("Heuristic mechanism IDs must be non-empty and unique")
    control_count = 0
    covered_parameters: set[str] = set()
    allowed_components = set(SMM_COMPONENTS)
    for mechanism in mechanisms:
        forbidden = {"diagnostic_keys", "trigger", "signed_residuals"} & set(
            mechanism
        )
        if forbidden:
            raise ValueError(
                f"Mechanism {mechanism['id']} uses forbidden residual fields: "
                f"{sorted(forbidden)}"
            )
        primary = {str(value) for value in mechanism.get("primary_components", [])}
        collateral = {
            str(value) for value in mechanism.get("collateral_components", [])
        }
        unknown_components = (primary | collateral) - allowed_components
        if not primary or unknown_components:
            raise ValueError(
                f"Mechanism {mechanism['id']} has invalid components: "
                f"{sorted(unknown_components)}"
            )
        if not str(mechanism.get("mechanism", "")).strip():
            raise ValueError(f"Mechanism {mechanism['id']} requires an explanation")
        controls = mechanism.get("controls", [])
        if not controls:
            raise ValueError(f"Mechanism {mechanism['id']} requires at least one control")
        for control in controls:
            parameter = str(control.get("parameter", ""))
            fraction = control.get("maximum_fraction_of_range")
            if parameter not in parameter_names:
                raise ValueError(
                    f"Mechanism {mechanism['id']} names unknown parameter {parameter}"
                )
            if not str(
                control.get("simulated_feature_effect_when_increased", "")
            ).strip():
                raise ValueError(
                    f"Mechanism {mechanism['id']} must explain the increased-control effect"
                )
            if (
                isinstance(fraction, bool)
                or not isinstance(fraction, (int, float))
                or not 0 < float(fraction) <= 1
            ):
                raise ValueError(
                    f"Mechanism {mechanism['id']} requires a maximum fraction in (0, 1]"
                )
            control_count += 1
            covered_parameters.add(parameter)
    if covered_parameters != parameter_names:
        raise ValueError(
            "Mechanism registry must cover every search parameter exactly as a usable "
            f"control set; missing={sorted(parameter_names - covered_parameters)}, "
            f"unknown={sorted(covered_parameters - parameter_names)}"
        )
    return {
        "version": version,
        "mechanism_count": len(mechanisms),
        "control_count": control_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--heuristics", type=Path, default=Path("configs/paper/heuristics.yaml")
    )
    parser.add_argument(
        "--optimizer", type=Path, default=Path("configs/paper/optimizer.yaml")
    )
    args = parser.parse_args()
    summary = validate_registry(
        yaml.safe_load(args.heuristics.read_text()),
        yaml.safe_load(args.optimizer.read_text()),
    )
    print(yaml.safe_dump(summary, sort_keys=False).strip())


if __name__ == "__main__":
    main()
