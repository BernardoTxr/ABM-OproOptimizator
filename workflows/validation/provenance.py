#!/usr/bin/env python3
"""Freeze or verify provenance of empirical reconstruction and paper caches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


from workflows import REPO_ROOT
DEFAULT_OUTPUT = (
    REPO_ROOT / "results/optimization/icaif_smm_v4_development/report"
    / "reconstruction_provenance.json"
)
DEFAULT_ARTIFACTS = {
    "development_smm_target": (
        REPO_ROOT / "results/real/ASAI3/smm_target_development_v4.json"
    ),
    "development_window_moments": (
        REPO_ROOT
        / "results/real/ASAI3/smm_target_development_v4.windows.parquet"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _combined_sha256(records: list[dict[str, Any]]) -> str:
    identities = [(row["relative_path"], row["sha256"]) for row in records]
    return hashlib.sha256(
        json.dumps(identities, separators=(",", ":")).encode()
    ).hexdigest()


def source_records(source_root: Path) -> list[dict[str, Any]]:
    """Content-address every Python source in the reconstruction package."""
    source_root = source_root.resolve()
    records = []
    for path in sorted(source_root.rglob("*.py")):
        try:
            label = str(path.relative_to(REPO_ROOT))
        except ValueError:
            label = str(path.relative_to(source_root))
        records.append({
            "relative_path": label,
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        })
    if not records:
        raise RuntimeError(f"No Python sources found under {source_root}")
    return records


def artifact_records(artifacts: dict[str, Path]) -> list[dict[str, Any]]:
    """Content-address derived empirical artifacts used by the evaluator."""
    records = []
    for name, path in sorted(artifacts.items()):
        path = path.resolve()
        if not path.is_file():
            raise RuntimeError(f"Required empirical artifact is missing: {path}")
        try:
            label = str(path.relative_to(REPO_ROOT))
        except ValueError:
            label = str(path)
        records.append({
            "name": name,
            "relative_path": label,
            "bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        })
    return records


def _git_reconstruction_state() -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "real_lob_inference"],
            cwd=REPO_ROOT,
            text=True,
        ).splitlines()
        return {
            "revision": revision,
            "reconstruction_tree_clean": not status,
            "reconstruction_changed_paths": status,
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "revision": None,
            "reconstruction_tree_clean": None,
            "reconstruction_changed_paths": [],
        }


def build_provenance(
    source_root: Path | None = None,
    artifacts: dict[str, Path] | None = None,
) -> dict[str, Any]:
    sources = source_records(source_root or REPO_ROOT / "real_lob_inference")
    derived = artifact_records(artifacts or DEFAULT_ARTIFACTS)
    return {
        "purpose": (
            "ICAIF 2026 companion lock for empirical reconstruction sources "
            "and development caches"
        ),
        "scope_note": (
            "The evaluator fingerprint covers simulator and configuration inputs; "
            "this file additionally freezes the reconstruction package and SMM "
            "target artifacts."
        ),
        "reconstruction_sources": sources,
        "reconstruction_sources_combined_sha256": _combined_sha256(sources),
        "derived_empirical_artifacts": derived,
        "derived_empirical_artifacts_combined_sha256": _combined_sha256(derived),
        "git": _git_reconstruction_state(),
    }


def verify_provenance(
    expected_path: Path = DEFAULT_OUTPUT,
    source_root: Path | None = None,
    artifacts: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Raise when reconstruction sources or derived artifacts changed."""
    if not expected_path.is_file():
        raise RuntimeError(
            f"Reconstruction provenance lock is missing: {expected_path}. "
            "Create it before starting the frozen OPRO arms."
        )
    expected = json.loads(expected_path.read_text())
    current = build_provenance(source_root=source_root, artifacts=artifacts)
    keys = (
        "reconstruction_sources_combined_sha256",
        "derived_empirical_artifacts_combined_sha256",
    )
    mismatches = [key for key in keys if expected.get(key) != current.get(key)]
    if mismatches:
        raise RuntimeError(
            "Reconstruction provenance differs from the frozen lock: "
            + ", ".join(mismatches)
        )
    return {
        "verified": True,
        "lock_path": str(expected_path.resolve()),
        **{key: current[key] for key in keys},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.verify:
        print(json.dumps(verify_provenance(output), indent=2, sort_keys=True))
        return
    if output.exists():
        raise RuntimeError(
            f"Refusing to overwrite existing provenance lock: {output}. "
            "Delete it explicitly only when beginning a new benchmark."
        )
    payload = build_provenance()
    if payload["git"]["reconstruction_tree_clean"] is not True:
        raise RuntimeError(
            "Refusing to freeze a changed real_lob_inference source tree"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Reconstruction provenance frozen at: {output}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
