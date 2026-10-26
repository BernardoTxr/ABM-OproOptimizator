#!/usr/bin/env python3
"""Validate exact direct dependencies for the frozen ICAIF benchmark."""
from __future__ import annotations

import argparse
from importlib import metadata
from pathlib import Path
import platform
from typing import Any

import yaml


DEFAULT_LOCK = Path("configs/paper/environment-lock.yaml")


def validate_environment(lock: dict[str, Any]) -> dict[str, Any]:
    if lock.get("version") != "icaif_environment_v1":
        raise RuntimeError("Unsupported ICAIF environment lock version")
    expected_python = str(lock.get("python_major_minor", ""))
    actual_python = platform.python_version()
    mismatches = []
    if not actual_python.startswith(expected_python + "."):
        mismatches.append(
            f"python expected {expected_python}.x, found {actual_python}"
        )
    actual_packages = {}
    packages = lock.get("packages", {})
    if not isinstance(packages, dict) or not packages:
        raise RuntimeError("ICAIF environment lock has no packages")
    for package, expected in packages.items():
        try:
            actual = metadata.version(str(package))
        except metadata.PackageNotFoundError:
            actual = None
        actual_packages[str(package)] = actual
        if actual != str(expected):
            mismatches.append(
                f"{package} expected {expected}, found {actual or 'MISSING'}"
            )
    if mismatches:
        raise RuntimeError(
            "ICAIF environment does not match the frozen lock: "
            + "; ".join(mismatches)
        )
    return {
        "lock_version": str(lock["version"]),
        "python": actual_python,
        "packages": actual_packages,
    }


def validate_extension_environment(lock: dict[str, Any]) -> dict[str, str]:
    """Validate exact add-on packages without changing the primary paper lock."""
    supported = {
        "icaif_optimizer_extension_environment_v1",
        "icaif_component_aware_environment_v1",
    }
    if lock.get("version") not in supported:
        raise RuntimeError("Unsupported optimizer-extension environment lock version")
    packages = lock.get("packages", {})
    if not isinstance(packages, dict) or not packages:
        raise RuntimeError("Optimizer-extension environment lock has no packages")
    actual_packages: dict[str, str] = {}
    mismatches = []
    for package, expected in packages.items():
        try:
            actual = metadata.version(str(package))
        except metadata.PackageNotFoundError:
            actual = None
        if actual is not None:
            actual_packages[str(package)] = actual
        if actual != str(expected):
            mismatches.append(
                f"{package} expected {expected}, found {actual or 'MISSING'}"
            )
    if mismatches:
        raise RuntimeError(
            "Optimizer-extension environment does not match the lock: "
            + "; ".join(mismatches)
        )
    return actual_packages


def validate_lock(lock: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one frozen lock to its matching validator."""
    if lock.get("version") == "icaif_environment_v1":
        return validate_environment(lock)
    packages = validate_extension_environment(lock)
    return {
        "lock_version": str(lock["version"]),
        "packages": packages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    result = validate_lock(yaml.safe_load(args.lock.read_text()))
    print(yaml.safe_dump(result, sort_keys=False).strip())


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
