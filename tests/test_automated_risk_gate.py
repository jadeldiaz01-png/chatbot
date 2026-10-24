"""Regression tests: SHA binding, path categories and fail-closed behavior."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import automated_risk_gate as gate

HEAD = "0123456789abcdef0123456789abcdef01234567"
OTHER = "fedcba9876543210fedcba9876543210fedcba98"


class AutomatedRiskGateTests(unittest.TestCase):
    def test_documentation_is_low(self):
        self.assertEqual(gate.classify_path("docs/overview.md"), "LOW")

    def test_regular_code_is_medium(self):
        self.assertEqual(gate.classify_path("src/analysis.py"), "MEDIUM")

    def test_workflow_changes_are_high(self):
        self.assertEqual(gate.classify_path(".github/workflows/ci.yml"), "HIGH")

    def test_financial_workflow_is_high(self):
        self.assertEqual(gate.classify_path("src/paypal_capture.py"), "HIGH")

    def test_security_and_migrations_are_high(self):
        self.assertEqual(gate.classify_path("security/access.rego"), "HIGH")
        self.assertEqual(gate.classify_path("migrations/001_init.sql"), "HIGH")

    def test_high_risk_blocks_without_exact_sha(self):
        result = gate.evaluate(["src/paypal_capture.py"], HEAD, "")
        self.assertEqual(result["decision"], "BLOCKED_HUMAN_AUTHORIZATION_REQUIRED")
        self.assertFalse(result["financial_execution_authorized"])

    def test_approval_of_previous_commit_is_invalid(self):
        result = gate.evaluate([".github/workflows/deploy.yml"], HEAD, OTHER)
        self.assertEqual(result["decision"], "BLOCKED_HUMAN_AUTHORIZATION_REQUIRED")

    def test_exact_sha_allows_merge_gate_only(self):
        result = gate.evaluate(["src/paypal_capture.py"], HEAD, HEAD)
        self.assertEqual(result["decision"], "PASS")
        self.assertFalse(result["financial_execution_authorized"])
        self.assertFalse(result["production_deploy_authorized"])

    def test_medium_code_does_not_require_reviewer(self):
        result = gate.evaluate(["src/analysis.py"], HEAD)
        self.assertEqual(result["decision"], "PASS")

    def test_empty_pr_fails_closed(self):
        with self.assertRaises(ValueError):
            gate.evaluate([], HEAD)

    def test_invalid_sha_fails_closed(self):
        with self.assertRaises(ValueError):
            gate.evaluate(["README.md"], "main")


    def test_trusted_api_checks_original_renamed_path(self):
        from unittest.mock import patch
        replies = [
            {"head": {"sha": HEAD}},
            [{"filename": "docs/harmless.md", "previous_filename": ".github/workflows/deploy.yml"}],
            {"head": {"sha": HEAD}},
        ]
        with patch.object(gate, "github_api_json", side_effect=replies):
            paths = gate.trusted_pr_paths("owner/repo", "4", HEAD, "read-token")
        self.assertEqual(gate.evaluate(paths, HEAD)["risk"], "HIGH")

    def test_trusted_api_rejects_pr_head_change(self):
        from unittest.mock import patch
        with patch.object(gate, "github_api_json", return_value={"head": {"sha": OTHER}}):
            with self.assertRaisesRegex(ValueError, "changed"):
                gate.trusted_pr_paths("owner/repo", "4", HEAD, "read-token")

    def test_trusted_api_rejects_invalid_repository(self):
        with self.assertRaisesRegex(ValueError, "Invalid repository"):
            gate.trusted_pr_paths("../repo", "4", HEAD, "read-token")



if __name__ == "__main__":
    unittest.main()
