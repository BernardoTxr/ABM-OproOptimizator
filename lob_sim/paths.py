"""Project-path helpers shared by config-driven entry points."""

from __future__ import annotations

from pathlib import Path


def find_project_root(anchor: str | Path) -> Path:
    """Find the checkout root without assuming how deeply a config is nested."""
    path = Path(anchor).resolve()
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if (candidate / "lob_sim").is_dir() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError(f"Could not locate project root from {path}")

