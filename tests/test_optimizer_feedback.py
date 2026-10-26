import unittest

from lob_sim.optimization.feedback import (
    SMM_COMPONENTS,
    build_optimizer_feedback,
    policy_feedback,
)


class OptimizerFeedbackTests(unittest.TestCase):
    def test_valid_result_preserves_six_groups_and_known_mean(self):
        components = {name: float(index + 1) for index, name in enumerate(SMM_COMPONENTS)}
        feedback = build_optimizer_feedback({
            "valid": True,
            "objective": 3.5,
            "component_losses": components,
            "signed_diagnostics": {"private": 99.0},
        }, invalid_penalty=1e6)
        self.assertEqual(feedback["component_losses"], components)
        self.assertEqual(feedback["final_loss"], 3.5)
        self.assertNotIn("signed_diagnostics", feedback)

    def test_invalid_result_maps_to_common_finite_penalty(self):
        feedback = build_optimizer_feedback(
            {"valid": False, "objective": None}, invalid_penalty=123.0
        )
        self.assertFalse(feedback["feasible"])
        self.assertEqual(feedback["final_loss"], 123.0)
        self.assertEqual(set(feedback["component_losses"].values()), {123.0})

    def test_information_firewall(self):
        feedback = build_optimizer_feedback({
            "valid": True,
            "objective": 1.0,
            "component_losses": {name: 1.0 for name in SMM_COMPONENTS},
        }, invalid_penalty=123.0)
        self.assertIsNone(policy_feedback(feedback, "none")["optimizer_feedback"])
        scalar = policy_feedback(feedback, "scalar")["optimizer_feedback"]
        self.assertNotIn("component_losses", scalar)
        component = policy_feedback(feedback, "component")["optimizer_feedback"]
        self.assertEqual(tuple(component["component_losses"]), SMM_COMPONENTS)

    def test_inconsistent_final_loss_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal-weight mean"):
            build_optimizer_feedback({
                "valid": True,
                "objective": 2.0,
                "component_losses": {name: 1.0 for name in SMM_COMPONENTS},
            }, invalid_penalty=123.0)


if __name__ == "__main__":
    unittest.main()
