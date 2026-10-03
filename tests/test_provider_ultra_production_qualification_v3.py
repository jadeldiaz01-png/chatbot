import importlib.util
import pathlib
import unittest

MODULE_PATH = pathlib.Path("scripts/provider_ultra_production_qualification_v3.py")
SPEC = importlib.util.spec_from_file_location(
    "provider_ultra_production_qualification_v3",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class ProductionQualificationV3Tests(unittest.TestCase):
    def test_candidate_system_instructions_are_hash_pinned(self) -> None:
        value = MODULE.load_candidate_system_instructions()
        self.assertIn("Never request passwords", value)
        self.assertIn("explicitly recommend human follow-up", value)
        self.assertEqual(
            MODULE.SYSTEM_INSTRUCTIONS_SHA256,
            "79f6e3eb01e7edaaed0fd63628a5c40fe2209211631db3bf357490ee2e7c843a",
        )

    def test_v3_changes_only_system_message_relative_to_v2_candidate(self) -> None:
        old_system = MODULE.baseline.load_system_instructions()
        new_system = MODULE.load_candidate_system_instructions()
        prompt = MODULE.baseline.PROMPT_CASES[0][1]

        old_request = MODULE.v2.build_request(old_system, prompt)
        new_request = MODULE.build_request(new_system, prompt)

        self.assertEqual(old_request["max_tokens"], 128)
        self.assertEqual(new_request["max_tokens"], 128)

        for key in set(old_request) | set(new_request):
            if key == "messages":
                continue
            self.assertEqual(old_request.get(key), new_request.get(key))

        self.assertEqual(old_request["messages"][1], new_request["messages"][1])
        self.assertNotEqual(old_request["messages"][0], new_request["messages"][0])

    def test_unexpected_model_identity_is_rejected(self) -> None:
        self.assertTrue(
            MODULE.expected_model_identity(MODULE.baseline.MODEL)
        )
        self.assertFalse(
            MODULE.expected_model_identity("nvidia/another-model")
        )
        self.assertFalse(MODULE.expected_model_identity(None))

    def test_same_prompt_corpus_and_transport_contract(self) -> None:
        self.assertEqual(MODULE.TOTAL_PLANNED_CALLS, 20)
        self.assertEqual(MODULE.REPETITIONS_PER_PROMPT, 4)
        self.assertEqual(MODULE.MAX_TOKENS, 128)
        self.assertEqual(MODULE.TARGET_P95_SECONDS, 8.0)
        self.assertEqual(MODULE.MIN_PASS_RATE, 1.0)
        self.assertFalse(MODULE.baseline.STREAM)
        self.assertEqual(MODULE.baseline.MAX_RETRIES, 2)
        self.assertEqual(MODULE.baseline.REQUEST_TIMEOUT_SECONDS, 45.0)

    def test_candidate_instructions_explicitly_cover_failed_v2_domains(self) -> None:
        text = MODULE.load_candidate_system_instructions().lower()
        for phrase in (
            "human approval",
            "not to send or share it",
            "cannot perform or confirm that action",
        ):
            self.assertIn(phrase, text)

    def test_summary_requires_all_twenty_quality_passes(self) -> None:
        results = []
        for repetition in range(1, 5):
            for prompt_id, _ in MODULE.baseline.PROMPT_CASES:
                results.append(
                    {
                        "sample_id": f"{prompt_id}-r{repetition}",
                        "prompt_id": prompt_id,
                        "status": "completed",
                        "quality_passed": True,
                        "total_seconds": 2.0,
                        "retries": 0,
                    }
                )
        summary = MODULE.summarize(results)
        self.assertTrue(summary["quality_gate_pass"])
        self.assertTrue(summary["latency_gate_pass"])
        self.assertEqual(summary["quality_pass_rate"], 1.0)

        results[-1]["quality_passed"] = False
        summary = MODULE.summarize(results)
        self.assertFalse(summary["quality_gate_pass"])

    def test_report_governance_stays_fail_closed(self) -> None:
        report = MODULE.build_report(
            results=[],
            created_at="2026-10-03T00:00:00+00:00",
        )
        self.assertEqual(report["experiment"], MODULE.EXPERIMENT_NAME)
        self.assertEqual(
            report["system_instructions_sha256"],
            MODULE.SYSTEM_INSTRUCTIONS_SHA256,
        )
        self.assertEqual(report["request_contract"]["max_tokens"], 128)
        self.assertFalse(report["production_configuration_changed"])
        self.assertFalse(report["production_runtime_parameter_change_authorized"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["sensitive_payloads_recorded"])
        MODULE.validate_report(report)


if __name__ == "__main__":
    unittest.main()
