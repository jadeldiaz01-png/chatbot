import importlib.util
import pathlib
import unittest

MODULE_PATH = pathlib.Path("scripts/provider_ultra_production_shaped_latency.py")
SPEC = importlib.util.spec_from_file_location("provider_ultra_production_shaped_latency", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class ProductionShapedLatencyTests(unittest.TestCase):
    def test_system_instructions_are_bound_to_current_runtime_literal(self) -> None:
        value = MODULE.load_system_instructions("streamlit_app.py")
        self.assertEqual(
            MODULE.prompt_digest(value),
            MODULE.SYSTEM_INSTRUCTIONS_SHA256,
        )

    def test_frozen_request_contract(self) -> None:
        system = MODULE.load_system_instructions("streamlit_app.py")
        request = MODULE.build_request(system, "hello")
        self.assertEqual(request["model"], "nvidia/nemotron-3-ultra-550b-a55b")
        self.assertEqual(request["messages"][0], {"role": "system", "content": system})
        self.assertEqual(request["messages"][1], {"role": "user", "content": "hello"})
        self.assertEqual(request["max_tokens"], 512)
        self.assertEqual(request["temperature"], 1.0)
        self.assertEqual(request["top_p"], 0.95)
        self.assertFalse(request["stream"])
        self.assertFalse(
            request["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        )

    def test_preregistered_sample_size_and_synthetic_cases(self) -> None:
        self.assertEqual(MODULE.REPETITIONS_PER_PROMPT, 4)
        self.assertEqual(len(MODULE.PROMPT_CASES), 5)
        self.assertEqual(MODULE.TOTAL_PLANNED_CALLS, 20)
        self.assertEqual(
            {item[0] for item in MODULE.PROMPT_CASES},
            {
                "service_scope_es",
                "human_followup_es",
                "authority_boundary_es",
                "privacy_boundary_es",
                "service_scope_en",
            },
        )

    def test_summary_requires_complete_error_free_p95(self) -> None:
        results = []
        for index in range(20):
            results.append(
                {
                    "status": "completed",
                    "total_seconds": 1.0 + index / 100.0,
                    "http_attempts": 1,
                    "retries": 0,
                }
            )
        summary = MODULE.summarize(results)
        self.assertEqual(summary["observed"], 20)
        self.assertEqual(summary["errors"], 0)
        self.assertTrue(summary["strict_slo_met"])
        self.assertEqual(summary["calls_over_slo"], 0)

        results[-1] = {
            "status": "infrastructure_error",
            "total_seconds": 2.0,
            "http_attempts": 3,
            "retries": 2,
        }
        summary = MODULE.summarize(results)
        self.assertEqual(summary["errors"], 1)
        self.assertFalse(summary["strict_slo_met"])
        self.assertEqual(summary["total_retries"], 2)

    def test_governance_flags_remain_fail_closed(self) -> None:
        self.assertFalse(MODULE.FUNCTIONAL_CONFIGURATION_CHANGED)
        self.assertFalse(MODULE.PRODUCTION_RUNTIME_PARAMETER_CHANGE_AUTHORIZED)
        self.assertFalse(MODULE.PROMOTION_AUTHORIZED)
        self.assertFalse(MODULE.QUALITY_EVALUATION)
        self.assertFalse(MODULE.SENSITIVE_PAYLOADS_RECORDED)
        self.assertEqual(MODULE.REQUEST_TIMEOUT_SECONDS, 45.0)
        self.assertEqual(MODULE.MAX_RETRIES, 2)
        self.assertEqual(MODULE.PACING_SECONDS, 12.0)
        self.assertEqual(MODULE.SLO_SECONDS, 8.0)


if __name__ == "__main__":
    unittest.main()
