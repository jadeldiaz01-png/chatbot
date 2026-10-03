import importlib.util
import pathlib
import unittest

MODULE_PATH = pathlib.Path("scripts/provider_ultra_production_shaped_latency.py")
SPEC = importlib.util.spec_from_file_location("provider_ultra_production_shaped_latency", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class ProductionShapedLatencyTests(unittest.TestCase):
    def test_system_instructions_are_bound_to_historical_snapshot(self) -> None:
        value = MODULE.load_system_instructions()
        self.assertEqual(
            MODULE.prompt_digest(value),
            MODULE.SYSTEM_INSTRUCTIONS_SHA256,
        )
        self.assertEqual(
            MODULE.SYSTEM_INSTRUCTIONS_SOURCE,
            "evidence/provider-ultra-system-instructions-v1.json:system_instructions",
        )

    def test_current_runtime_prompt_is_independent_from_historical_contract(self) -> None:
        runtime_source = pathlib.Path("streamlit_app.py").read_text(encoding="utf-8")
        self.assertIn("SYSTEM_INSTRUCTIONS", runtime_source)
        historical = MODULE.load_system_instructions()
        self.assertNotEqual(
            MODULE.prompt_digest(historical),
            "79f6e3eb01e7edaaed0fd63628a5c40fe2209211631db3bf357490ee2e7c843a",
        )

    def test_frozen_request_contract(self) -> None:
        system = MODULE.load_system_instructions()
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

    def test_attempt_tracker_attributes_retry_and_header_phases(self) -> None:
        from types import SimpleNamespace

        now = [10.0]

        def clock() -> float:
            return now[0]

        tracker = MODULE.AttemptTracker(clock=clock)
        tracker.begin("sample-1", sample_started_at=10.0)

        now[0] = 10.2
        tracker.on_request(None)

        now[0] = 11.2
        tracker.on_response(
            SimpleNamespace(
                status_code=503,
                headers={"x-request-id": "req-1"},
            )
        )

        now[0] = 12.0
        tracker.on_request(None)

        now[0] = 12.5
        tracker.on_response(
            SimpleNamespace(
                status_code=200,
                headers={"x-request-id": "req-2"},
            )
        )

        snapshot = tracker.snapshot(
            "sample-1",
            call_completed_at=13.0,
        )

        self.assertEqual(snapshot["http_attempts"], 2)
        self.assertEqual(snapshot["retries"], 1)
        self.assertEqual(snapshot["status_codes"], [503, 200])
        self.assertEqual(snapshot["request_ids"], ["req-1", "req-2"])

        attribution = snapshot["latency_attribution"]
        self.assertEqual(
            attribution["schema_version"],
            MODULE.ATTRIBUTION_SCHEMA_VERSION,
        )
        self.assertTrue(attribution["attribution_available"])
        self.assertEqual(attribution["pre_first_request_seconds"], 0.2)
        self.assertEqual(attribution["first_request_to_completion_seconds"], 2.8)
        self.assertEqual(attribution["final_headers_to_completion_seconds"], 0.5)
        self.assertEqual(attribution["retry_path_seconds"], 1.8)
        self.assertEqual(len(attribution["attempts"]), 2)

        first, second = attribution["attempts"]
        self.assertEqual(first["attempt_index"], 1)
        self.assertEqual(first["outcome"], "retry_after_headers")
        self.assertEqual(first["request_to_headers_seconds"], 1.0)
        self.assertEqual(first["request_to_next_attempt_or_end_seconds"], 1.8)
        self.assertEqual(first["headers_to_next_attempt_or_end_seconds"], 0.8)

        self.assertEqual(second["attempt_index"], 2)
        self.assertEqual(second["outcome"], "call_completed_after_headers")
        self.assertEqual(second["request_to_headers_seconds"], 0.5)
        self.assertEqual(second["request_to_next_attempt_or_end_seconds"], 1.0)
        self.assertEqual(second["headers_to_next_attempt_or_end_seconds"], 0.5)

    def test_summary_attributes_retry_and_output_token_cohorts(self) -> None:
        results = []
        token_sizes = [32, 96, 192, 384]
        for index in range(20):
            retries = 1 if index in {3, 7, 11} else 0
            total_seconds = 12.0 if retries else 4.0 + (index % 3)
            output_tokens = token_sizes[index % len(token_sizes)]
            results.append(
                {
                    "status": "completed",
                    "total_seconds": total_seconds,
                    "http_attempts": retries + 1,
                    "retries": retries,
                    "output_tokens": output_tokens,
                    "latency_attribution": {
                        "schema_version": MODULE.ATTRIBUTION_SCHEMA_VERSION,
                        "attribution_available": True,
                        "pre_first_request_seconds": 0.1,
                        "first_request_to_completion_seconds": total_seconds - 0.1,
                        "final_headers_to_completion_seconds": 0.5,
                        "retry_path_seconds": 7.0 if retries else 0.0,
                        "attempts": [],
                    },
                }
            )

        summary = MODULE.summarize(results)
        attribution = summary["latency_attribution"]

        self.assertEqual(
            attribution["schema_version"],
            MODULE.ATTRIBUTION_SCHEMA_VERSION,
        )
        self.assertEqual(attribution["observed_with_attempt_timing"], 20)
        self.assertEqual(attribution["retry_cohort"]["observed"], 3)
        self.assertEqual(attribution["no_retry_cohort"]["observed"], 17)
        self.assertEqual(attribution["output_token_observations"], 20)
        self.assertEqual(
            set(attribution["output_token_buckets"]),
            {"000-064", "065-128", "129-256", "257-512"},
        )
        self.assertEqual(
            sum(
                bucket["observed"]
                for bucket in attribution["output_token_buckets"].values()
            ),
            20,
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

    def test_incremental_report_persists_partial_and_complete_state(self) -> None:
        import json
        import tempfile

        created_at = "2026-10-01T00:00:00+00:00"
        completed_result = {
            "sample_id": "service_scope_es-r1",
            "prompt_id": "service_scope_es",
            "prompt_sha256": "0" * 64,
            "status": "completed",
            "total_seconds": 1.25,
            "output_chars": 42,
            "output_tokens": 10,
            "total_tokens": 20,
            "response_model": MODULE.MODEL,
            "http_attempts": 1,
            "retries": 0,
            "status_codes": [200],
            "request_ids": ["req-1"],
        }

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "report.json"

            initial = MODULE.persist_report(
                path,
                base_url=MODULE.DEFAULT_BASE_URL,
                results=[],
                created_at=created_at,
            )
            persisted_initial = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(initial["complete"])
            self.assertFalse(persisted_initial["complete"])
            self.assertEqual(persisted_initial["summary"]["observed"], 0)
            self.assertEqual(persisted_initial["results"], [])

            partial = MODULE.persist_report(
                path,
                base_url=MODULE.DEFAULT_BASE_URL,
                results=[completed_result],
                created_at=created_at,
            )
            persisted_partial = json.loads(path.read_text(encoding="utf-8"))
            self.assertFalse(partial["complete"])
            self.assertFalse(persisted_partial["complete"])
            self.assertEqual(persisted_partial["summary"]["observed"], 1)
            self.assertEqual(persisted_partial["results"][0]["request_ids"], ["req-1"])

            final_results = [dict(completed_result) for _ in range(MODULE.TOTAL_PLANNED_CALLS)]
            for index, item in enumerate(final_results, start=1):
                item["sample_id"] = f"sample-{index:02d}"
            final = MODULE.persist_report(
                path,
                base_url=MODULE.DEFAULT_BASE_URL,
                results=final_results,
                created_at=created_at,
            )
            persisted_final = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(final["complete"])
            self.assertTrue(persisted_final["complete"])
            self.assertEqual(
                persisted_final["summary"]["observed"],
                MODULE.TOTAL_PLANNED_CALLS,
            )
            self.assertTrue(persisted_final["summary"]["strict_slo_met"])

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
        self.assertEqual(MODULE.ATTRIBUTION_SCHEMA_VERSION, "2026-10-02.1")


if __name__ == "__main__":
    unittest.main()
