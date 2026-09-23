import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class SupplyChainContractTests(unittest.TestCase):
    def test_runtime_and_ci_use_hash_locked_dependencies(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("--require-hashes -r requirements.lock", dockerfile)
        self.assertIn("--require-hashes -r requirements-ci.lock", dockerfile)

    def test_lock_files_exist_and_contain_sha256_hashes(self) -> None:
        for filename in ("requirements.lock", "requirements-ci.lock"):
            content = (ROOT / filename).read_text(encoding="utf-8")
            self.assertIn("--hash=sha256:", content)
            self.assertGreater(len(content.splitlines()), 10)

    def test_manifest_declares_lock_gate_satisfied_by_design(self) -> None:
        manifest = json.loads((ROOT / "production-manifest.json").read_text(encoding="utf-8"))
        supply_chain = manifest["supply_chain"]
        self.assertTrue(supply_chain["transitive_dependency_lock_with_hashes"])
        self.assertEqual(supply_chain["runtime_lock"], "requirements.lock")
        self.assertEqual(supply_chain["ci_lock"], "requirements-ci.lock")

    def test_lock_drift_workflow_regenerates_and_compares(self) -> None:
        workflow = (
            ROOT / ".github" / "workflows" / "dependency-lock-candidate.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("requirements.lock.generated", workflow)
        self.assertIn("requirements-ci.lock.generated", workflow)
        self.assertIn("cmp requirements.lock requirements.lock.generated", workflow)
        self.assertIn("cmp requirements-ci.lock requirements-ci.lock.generated", workflow)


if __name__ == "__main__":
    unittest.main()
