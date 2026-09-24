import unittest

from scripts.run_provider_latency_probe import percentile, summarize_samples


class ProviderLatencyProbeTests(unittest.TestCase):
    def test_percentile_uses_nearest_rank(self) -> None:
        values = [float(value) for value in range(1, 21)]
        self.assertEqual(percentile(values, 0.95), 19.0)

    def test_summary_tracks_latency_retries_and_slo(self) -> None:
        samples = [
            {
                "status": "completed",
                "stream_open_seconds": 0.2,
                "first_event_seconds": 0.4,
                "first_content_seconds": 0.5,
                "total_seconds": 1.0,
                "http_attempts": 1,
                "retries": 0,
            },
            {
                "status": "completed",
                "stream_open_seconds": 0.3,
                "first_event_seconds": 0.5,
                "first_content_seconds": 0.6,
                "total_seconds": 2.0,
                "http_attempts": 2,
                "retries": 1,
            },
        ]

        summary = summarize_samples(samples, slo_seconds=8.0)

        self.assertEqual(summary["planned_samples"], 2)
        self.assertEqual(summary["completed_samples"], 2)
        self.assertEqual(summary["infrastructure_errors"], 0)
        self.assertEqual(summary["total_http_attempts"], 3)
        self.assertEqual(summary["total_retries"], 1)
        self.assertEqual(summary["retried_samples"], 1)
        self.assertTrue(summary["p95_first_content_within_slo"])
        self.assertTrue(summary["p95_total_within_slo"])
        self.assertEqual(summary["metrics"]["total_seconds"]["p95"], 2.0)

    def test_summary_fails_slo_without_calling_it_infrastructure(self) -> None:
        samples = [
            {
                "status": "completed",
                "stream_open_seconds": 0.5,
                "first_event_seconds": 1.0,
                "first_content_seconds": 9.0,
                "total_seconds": 10.0,
                "http_attempts": 1,
                "retries": 0,
            }
        ]

        summary = summarize_samples(samples, slo_seconds=8.0)

        self.assertEqual(summary["infrastructure_errors"], 0)
        self.assertFalse(summary["p95_first_content_within_slo"])
        self.assertFalse(summary["p95_total_within_slo"])

    def test_infrastructure_error_remains_separate(self) -> None:
        samples = [
            {
                "status": "infrastructure_error",
                "total_seconds": 3.0,
                "http_attempts": 3,
                "retries": 2,
                "status_codes": [503, 503, 503],
            }
        ]

        summary = summarize_samples(samples, slo_seconds=8.0)

        self.assertEqual(summary["completed_samples"], 0)
        self.assertEqual(summary["infrastructure_errors"], 1)
        self.assertEqual(summary["total_retries"], 2)
        self.assertFalse(summary["p95_total_within_slo"])


if __name__ == "__main__":
    unittest.main()
