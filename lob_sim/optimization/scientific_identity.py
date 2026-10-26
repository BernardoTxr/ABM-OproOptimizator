"""A second, additive identity hash scoped to the frozen scientific surface.

``TrialEvaluator._evaluation_fingerprint`` (in ``evaluator.py``) hashes every
``.py`` file under ``lob_sim/`` plus the study/model overrides, so it
changes whenever *any* code changes -- including adding a new proposal-
runtime adapter that never runs for a given study. That fingerprint must
stay exactly as strict as it is today for resume-safety; it is not touched
here.

``scientific_identity_sha256`` is a narrower, purely additive hash: it
covers only the simulator core, environment, agent archetypes/population,
loss, and the search-space/seed/validity-gate portion of the evaluator
config -- explicitly excluding every ``lob_sim/optimization/*_runtime.py``
adapter, ``runtime_factory.py``, and the config's ``llm``/``proposal_runtime``
blocks. Two studies with equal ``scientific_identity_sha256`` but different
``evaluator_fingerprint`` values changed only their proposal runtime/model,
never the scientific pipeline -- exactly the claim this OPRO follow-up must
be able to prove against v9.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lob_sim.optimization.evaluator import TrialEvaluator


SCIENTIFIC_IDENTITY_VERSION = "icaif-scientific-identity-v1"

_SOURCE_GLOBS = (
    "lob_sim/core/**/*.py",
    "lob_sim/environment.py",
    "lob_sim/archetypes/**/*.py",
    "lob_sim/agents/**/*.py",
    "lob_sim/loss/**/*.py",
    "lob_sim/optimization/evaluator.py",
    "lob_sim/optimization/space.py",
    "lob_sim/optimization/feedback.py",
)

_SCOPED_CONFIG_KEYS = (
    "search_space",
    "validity_gates",
    "optimizer_seed",
    "evaluation_seed",
    "initial_design",
    "trial_budget",
    "optimizer_repetitions",
)


def scientific_identity_sha256(repo_root: Path, evaluator: "TrialEvaluator") -> str:
    digest = hashlib.sha256()
    digest.update(SCIENTIFIC_IDENTITY_VERSION.encode() + b"\0")
    for pattern in _SOURCE_GLOBS:
        for path in sorted(repo_root.glob(pattern)):
            digest.update(str(path.relative_to(repo_root)).encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    frozen_configs = (
        evaluator.base_agents_path,
        evaluator.environment_path,
        evaluator.base_scenario_path,
        evaluator.loss_config_path,
    )
    for path in frozen_configs:
        digest.update(str(path).encode() + b"\0" + path.read_bytes() + b"\0")
    scoped = {key: evaluator.config.get(key) for key in _SCOPED_CONFIG_KEYS}
    digest.update(json.dumps(scoped, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()
