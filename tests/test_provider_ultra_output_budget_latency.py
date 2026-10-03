import importlib.util
import pathlib
import unittest

MODULE_PATH = pathlib.Path("scripts/provider_ultra_output_budget_latency.py")
SPEC = importlib.util.spec_from_file_location(
    "provider_ultra_output_budget_latency",
    MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class OutputBudgetLatencyTests(unittest.TestCase):
    def test_baseline_contract_is_reused_without_mutation(self) -> None:
        system = MODULE.load_system_instructions()
        baseline_request = MODULE.baseline.build_request(system, "hello")

        self.assertEqual(MODULE.MODEL, MODULE.baseline.MODEL)
        self.assertEqual(
            MODULE.SYSTEM_INSTRUCTIONS_SHA256,
            MODULE.baseline.SYSTEM_INSTRUCTIONS_SHA256,
        )
        self.assertEqual(MODULE.TEMPERATURE, 1.0)
        self.assertEqual(MODULE.TOP_P, 0.95)
        self.assertFalse(MODULE.ENABLE_THINKING)
        self.assertFalse(MODULE.STREAM)
        self.assertEqual(MODULE.REQUEST_TIMEOUT_SECONDS, 45.0)
        self.assertEqual(MODULE.MAX_RETRIES, 2)
        self.assertEqual(MODULE.PACING_SECONDS, 12.0)
        self.assertEqual(MODULE.SLO_SECONDS, 8.0)

        for budget in MODULE.OUTPUT_BUDGETS:
            request = MODULE.build_request(system, "hello", max_tokens=budget)
            changed = {
                key
                for key in set(baseline_request) | set(request)
                if baseline_request.get(key) != request.get(key)
            }
            self.assertTrue(changed.issubset({"max_tokens"}))
            if budget == baseline_request["max_tokens"]:
                self.assertEqual(changed, set())
            else:
                self.assertEqual(changed, {"max_tokens"})
            self.assertEqual(request["max_tokens"], budget)

    def test_invalid_budget_fails_closed(self) -> None:
        system = MODULE.load_system_instructions()
        with self.assertRaises(ValueError):
            MODULE.build_request(system, "hello", max_tokens=64)

    def test_plan_has_same_corpus_and_twenty_calls_per_budget(self) -> None:
        plan = MODULE.planned_samples()
        self.assertEqual(len(plan), 60)
        self.assertEqual(len({item["sample_id"] for item in plan}), 60)

        for budget in MODULE.OUTPUT_BUDGETS:
            cohort = [item for item in plan if item["max_tokens"] == budget]
            self.assertEqual(len(cohort), 20)
            self.assertEqual(
                {(item["prompt_id"], item["repetition"]) for item in cohort},
                {
                    (prompt_id, repetition)
                    for repetition in range(1, 5)
                    for prompt_id, _ in MODULE.PROMPT_CASES
                },
            )

    def test_budget_order_is_deterministic_and_counterbalanced(self) -> None:
        plan = MODULE.planned_samples()
        blocks = {}
        for item in plan:
            blocks.setdefault(item["block_index"], []).append(item)

        self.assertEqual(len(blocks), 20)
        first_positions = {budget: 0 for budget in MODULE.OUTPUT_BUDGETS}
        for block_index, items in blocks.items():
            ordered = sorted(items, key=lambda item: item["order_in_block"])
            expected = list(MODULE.counterbalanced_budgets(block_index))
            self.assertEqual([item["max_tokens"] for item in ordered], expected)
            first_positions[ordered[0]["max_tokens"]] += 1

        self.assertLessEqual(
            max(first_positions.values()) - min(first_positions.values()),
            1,
        )

    def test_summary_records_required_metrics_per_cohort(self) -> None:
        results = []
        for budget in MODULE.OUTPUT_BUDGETS:
            for index in range(20):
                retries = 1 if index == 19 else 0
                total_seconds = 4.0 + (budget / 512.0) + index / 100.0
                results.append(
                    {
                        "sample_id": f"sample-{budget}-{index}",
                        "prompt_id": "service_scope_es",
                        "prompt_sha256": "0" * 64,
                        "repetition": 1,
                        "max_tokens": budget,
                        "order_in_block": 1,
                        "block_index": index,
                        "status": "completed",
                        "total_seconds": total_seconds,
                        "output_tokens": min(budget, 100 + index),
                        "http_attempts": retries + 1,
                        "retries": retries,
                        "latency_attribution": {
                            "schema_version": MODULE.ATTRIBUTION_SCHEMA_VERSION,
                            "attribution_available": True,
                            "pre_first_request_seconds": 0.01,
                            "first_request_to_completion_seconds": total_seconds,
                            "final_headers_to_completion_seconds": 0.2,
                            "retry_path_seconds": 2.0 if retries else 0.0,
                            "attempts": [
                                {
                                    "attempt_index": 1,
                                    "status_code": 200,
                                    "request_id": "req",
                                    "outcome": "call_completed_after_headers",
                                    "request_to_headers_seconds": total_seconds - 0.2,
                                    "request_to_next_attempt_or_end_seconds": total_seconds,
                                    "headers_to_next_attempt_or_end_seconds": 0.2,
                                }
                            ],
                        },
                    }
                )

        summary = MODULE.summarize(results)
        self.assertEqual(summary["planned"], 60)
        self.assertEqual(summary["observed"], 60)
        self.assertEqual(set(summary["cohorts"]), {"128", "256", "512"})

        for budget in MODULE.OUTPUT_BUDGETS:
            cohort = summary["cohorts"][str(budget)]
            self.assertEqual(cohort["planned"], 20)
            self.assertEqual(cohort["observed"], 20)
            self.assertEqual(cohort["completed"], 20)
            self.assertEqual(cohort["errors"], 0)
            self.assertEqual(cohort["total_retries"], 1)
            self.assertEqual(cohort["samples_with_retry"], 1)
            self.assertEqual(cohort["output_token_observations"], 20)
            self.assertIsNotNone(cohort["p50_total_seconds"])
            self.assertIsNotNone(cohort["p95_total_seconds"])
            self.assertIsNotNone(cohort["max_total_seconds"])
            self.assertIsNotNone(cohort["p95_request_to_headers_seconds"])
            self.assertIsNotNone(cohort["p95_headers_to_completion_seconds"])
            self.assertIsNotNone(cohort["p95_output_tokens"])

    def test_report_governance_remains_fail_closed(self) -> None:
        report = MODULE.build_report(
            base_url=MODULE.DEFAULT_BASE_URL,
            results=[],
            created_at="2026-10-02T00:00:00+00:00",
        )
        self.assertEqual(report["experimental_parameter"], "max_tokens")
        self.assertEqual(report["output_budgets"], [128, 256, 512])
        self.assertEqual(report["calls_per_budget"], 20)
        self.assertEqual(report["total_planned_calls"], 60)
        self.assertFalse(report["production_configuration_changed"])
        self.assertFalse(report["production_runtime_parameter_change_authorized"])
        self.assertFalse(report["promotion_authorized"])
        self.assertFalse(report["quality_evaluation"])
        self.assertFalse(report["sensitive_payloads_recorded"])

    def test_validate_report_accepts_partial_preregistered_envelope(self) -> None:
        report = MODULE.build_report(
            base_url=MODULE.DEFAULT_BASE_URL,
            results=[],
            created_at="2026-10-02T00:00:00+00:00",
        )
        MODULE.validate_report(report)


if __name__ == "__main__":
    unittest.main()
