import tempfile
import unittest
from pathlib import Path

from workflows.common.local_env import load_env_file


class LocalEnvironmentTests(unittest.TestCase):
    def test_loads_quoted_values_without_overriding_exported_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env.local"
            path.write_text(
                "# local secrets\n"
                "OPENROUTER_API_KEY='local key'\n"
                "export EXISTING=local\n"
            )
            environment = {"EXISTING": "exported"}
            loaded = load_env_file(path, environment=environment)
        self.assertEqual(loaded, ("OPENROUTER_API_KEY",))
        self.assertEqual(environment["OPENROUTER_API_KEY"], "local key")
        self.assertEqual(environment["EXISTING"], "exported")

    def test_missing_file_is_a_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = {}
            loaded = load_env_file(
                Path(directory) / "missing.env", environment=environment
            )
        self.assertEqual(loaded, ())
        self.assertEqual(environment, {})


if __name__ == "__main__":
    unittest.main()
