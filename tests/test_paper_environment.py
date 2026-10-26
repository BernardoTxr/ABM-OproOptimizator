from __future__ import annotations

import unittest
from unittest.mock import patch

from workflows.validation.environment import (
    validate_environment,
    validate_extension_environment,
    validate_lock,
)


class ICAIFEnvironmentTests(unittest.TestCase):
    def test_exact_versions_pass(self) -> None:
        lock = {
            "version": "icaif_environment_v1",
            "python_major_minor": "3.12",
            "packages": {"numpy": "1.2.3", "pandas": "4.5.6"},
        }
        with (
            patch("workflows.validation.environment.platform.python_version", return_value="3.12.9"),
            patch("workflows.validation.environment.metadata.version", side_effect=["1.2.3", "4.5.6"]),
        ):
            result = validate_environment(lock)
        self.assertEqual(result["packages"]["numpy"], "1.2.3")

    def test_mismatch_is_actionable(self) -> None:
        lock = {
            "version": "icaif_environment_v1",
            "python_major_minor": "3.12",
            "packages": {"numpy": "1.2.3"},
        }
        with patch(
            "workflows.validation.environment.metadata.version",
            return_value="9.9.9",
        ):
            with self.assertRaisesRegex(RuntimeError, "numpy expected 1.2.3"):
                validate_environment(lock)

    def test_optimizer_extension_versions_are_checked_separately(self) -> None:
        lock = {
            "version": "icaif_optimizer_extension_environment_v1",
            "packages": {"cmaes": "1.2.3", "pybads": "4.5.6"},
        }
        with patch(
            "workflows.validation.environment.metadata.version",
            side_effect=["1.2.3", "4.5.6"],
        ):
            result = validate_extension_environment(lock)
        self.assertEqual(result, {"cmaes": "1.2.3", "pybads": "4.5.6"})

    def test_lock_dispatches_component_environment(self) -> None:
        lock = {
            "version": "icaif_component_aware_environment_v1",
            "packages": {"torch": "1.2.3"},
        }
        with patch(
            "workflows.validation.environment.metadata.version",
            return_value="1.2.3",
        ):
            result = validate_lock(lock)
        self.assertEqual(
            result,
            {
                "lock_version": "icaif_component_aware_environment_v1",
                "packages": {"torch": "1.2.3"},
            },
        )


if __name__ == "__main__":
    unittest.main()
