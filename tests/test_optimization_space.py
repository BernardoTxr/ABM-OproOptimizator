import unittest

import numpy as np

from lob_sim.optimization.space import SearchSpace


class OptimizationSpaceTests(unittest.TestCase):
    def test_sobol_design_is_reproducible_and_valid(self):
        space = SearchSpace.from_config([
            {
                "name": "rate", "type": "float", "low": 0.1, "high": 10.0,
                "log": True, "targets": [{"agent": "a", "field": "parameters.rate"}],
            },
            {
                "name": "probability", "type": "float", "low": 0.0,
                "high": 1.0, "targets": [{"agent": "a", "field": "parameters.p"}],
            },
        ])
        first = space.sobol(7, seed=42)
        second = space.sobol(7, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)
        self.assertEqual(len({space.canonical_key(item) for item in first}), 7)
        for candidate in first:
            self.assertEqual(candidate, space.validate(candidate))

    def setUp(self):
        self.space = SearchSpace.from_config([
            {
                "name": "shared_activity",
                "type": "float",
                "low": 0.01,
                "high": 0.05,
                "step": 0.01,
                "targets": [
                    {"agent": "value", "field": "parameters.activity"},
                    {"agent": "momentum", "field": "parameters.activity"},
                ],
            },
            {
                "name": "depth_ladder",
                "type": "categorical",
                "choices": [[100.0, 200.0], [200.0, 300.0]],
                "targets": [{"agent": "maker", "field": "parameters.level_sizes"}],
            },
        ])
        self.agents = {
            "agents": [
                {"id": "value", "parameters": {"activity": 0.02}},
                {"id": "momentum", "parameters": {"activity": 0.02}},
                {"id": "maker", "parameters": {"level_sizes": [1.0, 2.0]}},
            ]
        }

    def test_grouped_parameter_is_applied_to_every_target(self):
        result = self.space.apply(self.agents, {
            "shared_activity": 0.03,
            "depth_ladder": [200.0, 300.0],
        })
        self.assertEqual(result["agents"][0]["parameters"]["activity"], 0.03)
        self.assertEqual(result["agents"][1]["parameters"]["activity"], 0.03)
        self.assertEqual(result["agents"][2]["parameters"]["level_sizes"], [200.0, 300.0])
        self.assertEqual(self.agents["agents"][0]["parameters"]["activity"], 0.02)

    def test_candidate_schema_and_step_are_strict(self):
        with self.assertRaises(ValueError):
            self.space.validate({"shared_activity": 0.025, "depth_ladder": [100.0, 200.0]})
        with self.assertRaises(ValueError):
            self.space.validate({"shared_activity": 0.02})

    def test_float_grid_normalization_is_idempotent_at_upper_bound(self):
        space = SearchSpace.from_config([{
            "name": "x", "type": "float", "low": 0.0, "high": 0.6, "step": 0.1,
            "targets": [{"agent": "value", "field": "parameters.activity"}],
        }])
        once = space.validate({"x": 0.6})
        twice = space.validate(once)
        self.assertEqual(once, twice)
        self.assertEqual(twice["x"], 0.6)

    def test_seeded_random_sampling_is_reproducible(self):
        left = self.space.sample(np.random.default_rng(7))
        right = self.space.sample(np.random.default_rng(7))
        self.assertEqual(left, right)
        self.space.validate(left)

    def test_stepped_float_sampling_is_uniform_over_grid_points(self):
        space = SearchSpace.from_config([{
            "name": "x", "type": "float", "low": 0.0, "high": 0.3,
            "step": 0.1,
            "targets": [{"agent": "value", "field": "parameters.activity"}],
        }])
        rng = np.random.default_rng(12)
        counts = {value: 0 for value in (0.0, 0.1, 0.2, 0.3)}
        for _ in range(8000):
            value = round(space.sample(rng)["x"], 1)
            counts[value] += 1
        self.assertTrue(all(1750 < count < 2250 for count in counts.values()))


if __name__ == "__main__":
    unittest.main()
