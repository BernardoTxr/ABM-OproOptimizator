import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lob_sim.loss.smm import (
    SMMSpec,
    SMMTarget,
    compute_moment_vector,
    estimate_target_from_window_moments,
    aggregate_target_from_window_moments,
    score_smm_moments,
)


ROOT = Path(__file__).resolve().parents[1]


def _spec() -> SMMSpec:
    return SMMSpec.from_dict(
        yaml.safe_load((ROOT / "configs/paper/loss.yaml").read_text())
    )


def _target(spec: SMMSpec, target: float = 0.0, se: float = 1.0) -> SMMTarget:
    size = len(spec.moment_names)
    return SMMTarget(
        moment_names=spec.moment_names,
        target={name: target for name in spec.moment_names},
        standard_errors={name: se for name in spec.moment_names},
        covariance=(np.eye(size) * se ** 2).tolist(),
        date_moments=[],
        metadata={},
    )


class SMMLossTests(unittest.TestCase):
    def test_moment_vector_has_signed_quantiles_probabilities_and_depth(self):
        spec = _spec()
        facts = {
            "log_returns": np.array([-0.02, -0.01, 0.01, 0.02]),
            "spread": np.array([0.01, 0.01, 0.02, 0.03]),
            "imbalance": np.array([-1.0, -0.5, 0.5, 1.0]),
            "trade_sizes": np.array([100.0, 200.0, 300.0, 1000.0]),
            "inter_arrival": np.array([0.0, 1.0, 2.0, 3.0]),
            "acf_abs_returns": np.linspace(1.0, 0.0, 51),
            "depth_bid": np.arange(1.0, 6.0) * 100.0,
            "depth_ask": np.arange(1.0, 6.0) * 200.0,
        }
        moments = compute_moment_vector(facts, spec)
        self.assertEqual(tuple(moments), spec.moment_names)
        self.assertAlmostEqual(moments["return_mean"], 0.0)
        self.assertAlmostEqual(moments["spread_p1"], 0.5)
        self.assertAlmostEqual(moments["inter_arrival_p0"], 0.25)
        self.assertAlmostEqual(moments["inter_arrival_cdf1"], 0.5)
        self.assertEqual(moments["inter_arrival_cdf5"], 1.0)
        self.assertEqual(moments["depth_ask_l5"], 1000.0)

    def test_group_dimension_does_not_give_depth_extra_priority(self):
        spec = _spec()
        simulated = {name: 1.0 for name in spec.moment_names}
        for name in spec.groups["depth"]:
            simulated[name] = 2.0
        result = score_smm_moments(simulated, _target(spec), spec)
        self.assertAlmostEqual(result.per_group["returns"], 1.0)
        self.assertAlmostEqual(result.per_group["depth"], 4.0)
        self.assertAlmostEqual(result.total, (5.0 + 4.0) / 6.0)

    def test_total_reconstructs_exactly_from_group_scores(self):
        spec = _spec()
        simulated = {
            name: float(index + 1) / 10.0
            for index, name in enumerate(spec.moment_names)
        }
        result = score_smm_moments(simulated, _target(spec), spec)
        self.assertAlmostEqual(result.total, np.mean(list(result.per_group.values())))

    def test_synthetic_target_prefers_its_generating_moments(self):
        spec = _spec()
        target = _target(spec, target=0.25, se=0.5)
        generating = {name: 0.25 for name in spec.moment_names}
        perturbed = dict(generating)
        perturbed["return_std"] += 0.5
        recovered = score_smm_moments(generating, target, spec)
        displaced = score_smm_moments(perturbed, target, spec)
        self.assertEqual(recovered.total, 0.0)
        self.assertGreater(displaced.total, recovered.total)

    def test_paper_objective_excludes_unstable_dependence_group(self):
        spec = _spec()
        self.assertNotIn("dependence", spec.groups)
        self.assertFalse(any("acf" in name for name in spec.moment_names))

    def test_previous_smm_protocol_is_not_supported(self):
        values = yaml.safe_load((ROOT / "configs/paper/loss.yaml").read_text())
        values["version"] = "smm_v3"
        with self.assertRaises(ValueError):
            SMMSpec.from_dict(values)

    def test_t_statistic_score_is_invariant_to_a_unit_change(self):
        spec = _spec()
        target = _target(spec, target=2.0, se=0.5)
        simulated = {name: 3.0 for name in spec.moment_names}
        baseline = score_smm_moments(simulated, target, spec)

        name = "depth_bid_l1"
        changed_target = SMMTarget(
            moment_names=target.moment_names,
            target={**target.target, name: target.target[name] * 100.0},
            standard_errors={
                **target.standard_errors,
                name: target.standard_errors[name] * 100.0,
            },
            covariance=target.covariance,
            date_moments=[],
            metadata={},
        )
        changed_simulated = {**simulated, name: simulated[name] * 100.0}
        changed = score_smm_moments(changed_simulated, changed_target, spec)
        self.assertAlmostEqual(baseline.total, changed.total)
        self.assertAlmostEqual(
            baseline.standardized_residuals[name],
            changed.standardized_residuals[name],
        )

    def test_date_cluster_bootstrap_is_deterministic(self):
        spec = _spec()
        rows = []
        for date_index, date in enumerate(("d1", "d2", "d3"), start=1):
            for window in ("w1", "w2"):
                rows.append({
                    "date": date,
                    "window_start_utc": window,
                    **{
                        name: float(date_index) + moment_index / 1000.0
                        for moment_index, name in enumerate(spec.moment_names)
                    },
                })
        frame = pd.DataFrame(rows)
        first = estimate_target_from_window_moments(frame, spec)
        second = estimate_target_from_window_moments(frame, spec)
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(np.array_equal(first[2], second[2]))
        self.assertAlmostEqual(first[0]["return_mean"], 2.0)

    def test_degenerate_standard_error_fails_instead_of_silent_flooring(self):
        spec = _spec()
        rows = []
        for date in ("d1", "d2"):
            rows.append({
                "date": date,
                "window_start_utc": "w1",
                **{name: 1.0 for name in spec.moment_names},
            })
        with self.assertRaisesRegex(ValueError, "degenerate standard errors"):
            estimate_target_from_window_moments(pd.DataFrame(rows), spec)

    def test_target_means_can_be_aggregated_with_external_weights(self):
        spec = _spec()
        rows = []
        for date in ("d1", "d2"):
            rows.append({
                "date": date,
                "window_start_utc": "w1",
                **{name: 1.0 for name in spec.moment_names},
            })
        target, date_moments = aggregate_target_from_window_moments(
            pd.DataFrame(rows), spec
        )
        self.assertEqual(target, {name: 1.0 for name in spec.moment_names})
        self.assertEqual(len(date_moments), 2)


if __name__ == "__main__":
    unittest.main()
