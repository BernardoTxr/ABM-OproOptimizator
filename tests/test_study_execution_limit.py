import json
from pathlib import Path

from lob_sim.optimization import study
from lob_sim.optimization.samplers import Proposal


class _Space:
    @staticmethod
    def validate(candidate):
        return dict(candidate)

    @staticmethod
    def canonical_key(candidate):
        return json.dumps(candidate, sort_keys=True)


class _Evaluator:
    def __init__(self, study_dir: Path):
        self.config = {
            "trial_budget": 50,
            "optimizer_seed": 1,
            "initial_design": [{"x": 0.0}, {"x": 1.0}],
        }
        self.config_dir = study_dir
        self.study_dir = study_dir
        self.space = _Space()
        self.evaluator_fingerprint = "fingerprint"

    def evaluate(self, candidate, _trial_name):
        return {"valid": True, "objective": float(candidate["x"])}


class _Policy:
    feedback_mode = "scalar"

    def __init__(self):
        self.remaining_budgets = []

    def ask(self, trial_index, fixed=None, history=None, remaining_budget=None):
        self.remaining_budgets.append(remaining_budget)
        return Proposal(dict(fixed) if fixed is not None else {"x": trial_index + 1.0})

    def tell(self, _proposal, _result):
        return None


def test_execution_limit_does_not_shrink_policy_visible_budget(
    tmp_path: Path, monkeypatch
) -> None:
    evaluator = _Evaluator(tmp_path)
    policy = _Policy()
    monkeypatch.setattr(study, "TrialEvaluator", lambda *_args, **_kwargs: evaluator)
    monkeypatch.setattr(study, "_policy", lambda *_args, **_kwargs: policy)
    monkeypatch.setattr(study, "feedback_contract", lambda _config: (("x",), 1e6))
    monkeypatch.setattr(
        study,
        "build_optimizer_feedback",
        lambda result, **_kwargs: {
            "version": "test",
            "feasible": True,
            "component_order": ["x"],
            "component_losses": {"x": result["objective"]},
            "final_loss": result["objective"],
            "invalid_penalty_applied": False,
        },
    )

    records, run_dir = study.run_study(
        "unused.yaml",
        "test_policy",
        0,
        budget=50,
        execution_limit=3,
    )

    assert len(records) == 3
    assert policy.remaining_budgets == [50, 49, 48]
    summary = json.loads((Path(run_dir) / "summary.json").read_text())
    assert summary["budget"] == 50
    assert summary["completed_proposals"] == 3


def test_execution_limit_cannot_exceed_policy_budget(
    tmp_path: Path, monkeypatch
) -> None:
    evaluator = _Evaluator(tmp_path)
    monkeypatch.setattr(study, "TrialEvaluator", lambda *_args, **_kwargs: evaluator)

    try:
        study.run_study(
            "unused.yaml",
            "test_policy",
            0,
            budget=50,
            execution_limit=51,
        )
    except ValueError as exc:
        assert "Execution limit" in str(exc)
    else:
        raise AssertionError("execution_limit above budget must be rejected")
