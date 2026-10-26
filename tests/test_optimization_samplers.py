import unittest
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from lob_sim.optimization.samplers import RandomPolicy
from lob_sim.optimization.space import SearchSpace
from workflows.benchmark.classical_optimizers import (
    BADSAskTellPolicy,
    ComponentAwareGPPolicy,
    MultiObjectiveTPEPolicy,
    OptunaCMAESPolicy,
    OptunaGPPolicy,
    SobolPolicy,
)


class OptimizationSamplerTests(unittest.TestCase):
    @staticmethod
    def _scalar_feedback(value, feasible=True):
        return {"optimizer_feedback": {
            "version": "grouped_smm_feedback_v1",
            "feasible": feasible,
            "final_loss": float(value),
            "invalid_penalty_applied": not feasible,
        }}

    @staticmethod
    def _component_feedback(value, feasible=True):
        names = (
            "returns", "spread", "imbalance", "trade_size",
            "inter_arrival", "depth",
        )
        return {"optimizer_feedback": {
            "version": "grouped_smm_feedback_v1",
            "feasible": feasible,
            "component_order": list(names),
            "component_losses": {name: float(value) for name in names},
            "final_loss": float(value),
            "invalid_penalty_applied": not feasible,
        }}

    @staticmethod
    def _continuous_space():
        return SearchSpace.from_config([
            {
                "name": "x", "type": "float", "low": 0.0, "high": 10.0,
                "targets": [{"agent": "a", "field": "parameters.x"}],
            },
            {
                "name": "scale", "type": "float", "low": 1.0, "high": 100.0,
                "log": True,
                "targets": [{"agent": "a", "field": "parameters.scale"}],
            },
        ])

    def test_random_policy_is_index_reproducible(self):
        space = SearchSpace.from_config([{
            "name": "x", "type": "int", "low": 0, "high": 10, "step": 1,
            "targets": [{"agent": "a", "field": "parameters.x"}],
        }])
        left = RandomPolicy(space, 123).ask(7).candidate
        right = RandomPolicy(space, 123).ask(7).candidate
        self.assertEqual(left, right)

    def test_sobol_policy_continues_at_declared_sequence_offset(self):
        space = self._continuous_space()
        proposal = SobolPolicy(space, 321, 3, 12, "frozen").ask(7)
        self.assertEqual(proposal.candidate, space.sobol(17, 321)[16])
        self.assertEqual(proposal.metadata["sobol_seed"], 321)
        self.assertEqual(proposal.metadata["sobol_index"], 16)

    def _assert_persistent_ask_tell_cycle(self, policy_type, name):
        space = self._continuous_space()
        initial = [
            {"x": 1.0, "scale": 10.0},
            {"x": 8.0, "scale": 20.0},
        ]
        with TemporaryDirectory() as temporary:
            policy = policy_type(
                space,
                seed=11,
                storage_path=Path(temporary) / f"{name}.sqlite3",
                study_name=name,
                initial_design=initial,
                extension_config_sha256="frozen",
            )
            history = []
            for trial_index, fixed in enumerate(initial):
                proposal = policy.ask(trial_index, fixed=fixed, history=history)
                policy.tell(proposal, self._scalar_feedback(3 - trial_index))
                history.append({
                    "sampler": policy.name,
                    "candidate": proposal.candidate,
                    "proposal_metadata": proposal.metadata,
                    "objective": 3 - trial_index,
                })
            adaptive = policy.ask(2, history=history)
            self.assertEqual(adaptive.candidate, space.validate(adaptive.candidate))
            self.assertEqual(adaptive.metadata["source"], policy.name)
            policy.tell(adaptive, self._scalar_feedback(1.0))

    def test_gp_bo_completes_persistent_ask_tell_cycle(self):
        self._assert_persistent_ask_tell_cycle(OptunaGPPolicy, "gp-bo")

    @unittest.skipUnless(
        find_spec("cmaes") is not None,
        "CMA-ES belongs to the optional optimizer-extension environment",
    )
    def test_cma_es_completes_persistent_ask_tell_cycle(self):
        self._assert_persistent_ask_tell_cycle(OptunaCMAESPolicy, "cma-es")

    def test_multiobjective_tpe_uses_six_objectives(self):
        space = self._continuous_space()
        initial = [{"x": 1.0, "scale": 10.0}, {"x": 8.0, "scale": 20.0}]
        with TemporaryDirectory() as temporary:
            policy = MultiObjectiveTPEPolicy(
                space, 17, Path(temporary) / "mo.sqlite3", "mo", initial, "frozen"
            )
            history = []
            for index, fixed in enumerate(initial):
                proposal = policy.ask(index, fixed=fixed, history=history)
                policy.tell(proposal, self._component_feedback(2 - index))
                history.append({
                    "sampler": policy.name,
                    "candidate": proposal.candidate,
                    "proposal_metadata": proposal.metadata,
                    "optimizer_feedback": self._component_feedback(2 - index)[
                        "optimizer_feedback"
                    ],
                })
            adaptive = policy.ask(2, history=history)
            policy.tell(adaptive, self._component_feedback(0.5))
            self.assertEqual(len(policy.study.directions), 6)

    def test_component_gp_acquires_on_equal_weight_component_sum(self):
        space = self._continuous_space()
        policy = ComponentAwareGPPolicy(
            space, 7, 2, "frozen", candidate_pool_size=8
        )

        class FakeModel:
            def __init__(self, offset):
                self.offset = offset

            def predict(self, points, return_std):
                return points[:, 0] + self.offset, np.full(len(points), 0.1)

        policy._fit_models = lambda x, y, trial_index: [
            FakeModel(index * 0.01) for index in range(6)
        ]
        history = []
        for index, candidate in enumerate((
            {"x": 1.0, "scale": 10.0}, {"x": 8.0, "scale": 20.0}
        )):
            feedback = self._component_feedback(2 - index)["optimizer_feedback"]
            history.append({
                "sampler": policy.name,
                "candidate": candidate,
                "candidate_key": space.canonical_key(candidate),
                "proposal_metadata": {"source": "initial_design"},
                "optimizer_feedback": feedback,
            })
        proposal = policy.ask(2, history=history)
        self.assertEqual(proposal.metadata["component_order"], [
            "returns", "spread", "imbalance", "trade_size", "inter_arrival", "depth"
        ])
        self.assertIn("equal_weight_component_sum", proposal.metadata["acquisition"])
        policy.tell(proposal, self._component_feedback(0.8))

    def test_bads_bridge_uses_common_incumbent_and_exposes_one_candidate(self):
        space = self._continuous_space()

        class FakeBADS:
            def __init__(self, fun, x0, lb, ub, plb, pub, options):
                self.fun = fun
                self.x0 = x0
                self.options = options
                np.testing.assert_array_equal(lb, np.zeros(2))
                np.testing.assert_array_equal(ub, np.ones(2))

            def optimize(self):
                self.fun(self.x0)
                for index in range(self.options["max_fun_evals"] - 1):
                    self.fun(np.array([0.2 + 0.1 * index, 0.7 - 0.1 * index]))

        initial = [
            {
                "policy_trial": 0, "candidate": {"x": 1.0, "scale": 10.0},
                "candidate_key": space.canonical_key({"x": 1.0, "scale": 10.0}),
                "objective": 4.0, "proposal_metadata": {"source": "initial_design"},
            },
            {
                "policy_trial": 1, "candidate": {"x": 8.0, "scale": 20.0},
                "candidate_key": space.canonical_key({"x": 8.0, "scale": 20.0}),
                "objective": 2.0, "proposal_metadata": {"source": "initial_design"},
            },
        ]
        policy = BADSAskTellPolicy(
            space, seed=9, initial_design_count=2,
            extension_config_sha256="frozen", bads_factory=FakeBADS,
        )
        fixed = policy.ask(0, fixed=initial[0]["candidate"])
        policy.tell(fixed, self._scalar_feedback(4.0))
        proposal = policy.ask(2, history=initial, remaining_budget=1)
        self.assertAlmostEqual(proposal.candidate["x"], 8.0)
        self.assertAlmostEqual(proposal.candidate["scale"], 20.0)
        policy.tell(proposal, self._scalar_feedback(1.5))
        self.assertEqual(proposal.metadata["optimizer_objective"], 1.5)
        self.assertFalse(proposal.metadata["hidden_constraint_violation"])

    def test_bads_bridge_penalizes_hidden_constraint_failure(self):
        space = self._continuous_space()

        class OneCandidateBADS:
            def __init__(self, fun, x0, lb, ub, plb, pub, options):
                self.fun, self.x0 = fun, x0

            def optimize(self):
                self.fun(self.x0)
                self.fun(np.array([0.5, 0.5]))

        initial = [{
            "policy_trial": 0, "candidate": {"x": 2.0, "scale": 10.0},
            "candidate_key": space.canonical_key({"x": 2.0, "scale": 10.0}),
            "objective": 1.0, "proposal_metadata": {"source": "initial_design"},
        }]
        policy = BADSAskTellPolicy(
            space, 1, 1, "frozen", invalid_penalty=999.0,
            bads_factory=OneCandidateBADS,
        )
        proposal = policy.ask(1, history=initial, remaining_budget=1)
        policy.tell(proposal, self._scalar_feedback(999.0, feasible=False))
        self.assertEqual(proposal.metadata["optimizer_objective"], 999.0)
        self.assertTrue(proposal.metadata["hidden_constraint_violation"])


if __name__ == "__main__":
    unittest.main()
