"""Validated CPU-affinity and execution settings for paper workflows."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def parse_cpu_set(value: str) -> tuple[int, ...]:
    """Parse Linux CPU-list syntax such as ``0-3,6,8-9``."""
    cpus: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError("CPU set contains an empty item")
        if "-" in part:
            fields = part.split("-")
            if len(fields) != 2:
                raise ValueError(f"Invalid CPU range: {part}")
            start, stop = (int(field) for field in fields)
            if start < 0 or stop < start:
                raise ValueError(f"Invalid CPU range: {part}")
            cpus.update(range(start, stop + 1))
        else:
            cpu = int(part)
            if cpu < 0:
                raise ValueError("CPU identifiers must be non-negative")
            cpus.add(cpu)
    if not cpus:
        raise ValueError("CPU set cannot be empty")
    return tuple(sorted(cpus))


def format_cpu_set(cpus: tuple[int, ...] | list[int]) -> str:
    """Return a stable comma-separated CPU list for child commands."""
    return ",".join(str(cpu) for cpu in cpus)


def apply_cpu_affinity(cpus: tuple[int, ...]) -> None:
    """Restrict this process and all subsequently created children."""
    if not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("CPU affinity requires Linux sched_setaffinity")
    available = set(os.sched_getaffinity(0))
    unavailable = sorted(set(cpus) - available)
    if unavailable:
        raise RuntimeError(
            "Requested CPUs are unavailable to this process: "
            + ", ".join(str(cpu) for cpu in unavailable)
        )
    os.sched_setaffinity(0, set(cpus))


def load_execution_config(path: Path) -> dict[str, Any]:
    """Load and validate the checked-in Xeon scheduling configuration."""
    values = yaml.safe_load(path.read_text())
    if values.get("version") != "xeon_e2146g_parallel_v1":
        raise ValueError("Unsupported parallel execution configuration")
    machine = values.get("machine", {})
    execution = values.get("execution", {})
    logical_cpus = int(machine.get("logical_cpus", 0))
    affinity = tuple(int(cpu) for cpu in execution.get("cpu_affinity", ()))
    reserved = tuple(
        int(cpu) for cpu in execution.get("reserved_logical_cpus", ())
    )
    if logical_cpus <= 0:
        raise ValueError("machine.logical_cpus must be positive")
    if len(affinity) != len(set(affinity)) or len(reserved) != len(set(reserved)):
        raise ValueError("CPU affinity lists cannot contain duplicates")
    if set(affinity) & set(reserved):
        raise ValueError("Active and reserved CPU sets must be disjoint")
    if set(affinity) | set(reserved) != set(range(logical_cpus)):
        raise ValueError("Active and reserved CPU sets must cover the machine")
    if len(reserved) != 1:
        raise ValueError("The Xeon paper configuration must reserve one logical CPU")
    ticker_workers = int(execution.get("ticker_workers", 0))
    simulation_workers = int(execution.get("simulation_workers_per_ticker", 0))
    thread_limit = int(execution.get("numerical_library_threads", 0))
    if ticker_workers < 1 or ticker_workers > 5:
        raise ValueError("ticker_workers must be between one and five")
    if simulation_workers < 1 or thread_limit != 1:
        raise ValueError("Worker counts must be positive and numerical threads must be one")
    return values

