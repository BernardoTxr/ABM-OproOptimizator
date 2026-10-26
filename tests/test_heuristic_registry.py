import copy
import unittest
from pathlib import Path

import yaml

from workflows.validation.heuristics import validate_registry


class HeuristicRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        cls.registry = yaml.safe_load(
            (root / "configs/paper/heuristics.yaml").read_text()
        )
        cls.optimizer = yaml.safe_load(
            (root / "configs/paper/optimizer.yaml").read_text()
        )

    def test_frozen_registry_matches_search_space(self):
        summary = validate_registry(self.registry, self.optimizer)
        self.assertEqual(summary["version"], "component_mechanisms_smm_v1")
        self.assertEqual(summary["mechanism_count"], 11)
        self.assertEqual(summary["control_count"], 11)

    def test_unknown_parameter_is_rejected(self):
        registry = copy.deepcopy(self.registry)
        registry["mechanisms"][0]["controls"][0]["parameter"] = "magic_alpha"
        with self.assertRaises(ValueError):
            validate_registry(registry, self.optimizer)

    def test_signed_residual_trigger_contract_is_rejected(self):
        registry = copy.deepcopy(self.registry)
        registry["mechanisms"][0]["diagnostic_keys"] = ["spread_p1"]
        registry["mechanisms"][0]["trigger"] = "spread_p1 > 0"
        with self.assertRaisesRegex(ValueError, "forbidden residual fields"):
            validate_registry(registry, self.optimizer)

if __name__ == "__main__":
    unittest.main()
