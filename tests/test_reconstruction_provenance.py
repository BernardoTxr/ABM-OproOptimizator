import json
import tempfile
import unittest
from pathlib import Path

from workflows.validation.provenance import (
    _combined_sha256,
    artifact_records,
    source_records,
    verify_provenance,
)


class ReconstructionProvenanceTests(unittest.TestCase):
    def test_source_and_artifact_records_are_content_addressed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "sources"
            sources.mkdir()
            (sources / "a.py").write_text("x = 1\n")
            (sources / "ignored.txt").write_text("not source\n")
            artifact = root / "facts.npz"
            artifact.write_bytes(b"facts")
            source_rows = source_records(sources)
            artifact_rows = artifact_records({"facts": artifact})
            self.assertEqual(len(source_rows), 1)
            self.assertEqual(len(source_rows[0]["sha256"]), 64)
            self.assertEqual(artifact_rows[0]["bytes"], 5)
            self.assertEqual(len(_combined_sha256(source_rows)), 64)

    def test_verification_detects_source_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "sources"
            sources.mkdir()
            source = sources / "a.py"
            source.write_text("x = 1\n")
            artifact = root / "facts.npz"
            artifact.write_bytes(b"facts")
            source_rows = source_records(sources)
            artifact_rows = artifact_records({"facts": artifact})
            lock = root / "lock.json"
            lock.write_text(json.dumps({
                "reconstruction_sources_combined_sha256": _combined_sha256(source_rows),
                "derived_empirical_artifacts_combined_sha256": _combined_sha256(artifact_rows),
            }))
            verified = verify_provenance(
                lock, source_root=sources, artifacts={"facts": artifact}
            )
            self.assertTrue(verified["verified"])
            source.write_text("x = 2\n")
            with self.assertRaisesRegex(RuntimeError, "Reconstruction provenance differs"):
                verify_provenance(
                    lock, source_root=sources, artifacts={"facts": artifact}
                )


if __name__ == "__main__":
    unittest.main()
