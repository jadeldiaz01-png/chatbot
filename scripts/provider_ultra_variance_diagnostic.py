from __future__ import annotations

"""Read-only analysis helpers for completed provider A/B artifacts."""

from collections import Counter
from datetime import UTC, datetime
from typing import Any

DIAGNOSTIC_NAME = "provider_ultra_variance_diagnostic"
DIAGNOSTIC_SPEC_VERSION = "2026-09-27.1"
EXPECTED_CONDITIONS = ("reuse", "no_keepalive")
EXPECTED_REPLICATES = tuple(range(1, 6))


def _typed_error(item: dict[str, Any]) -> str:
    values = (item.get("error_type"), item.get("error_phase"), item.get("status_code"))
    parts = [str(value) for value in values if value is not None]
    return ":".join(parts) if parts else "unknown"


def analyze_report(report: dict[str, Any]) -> dict[str, Any]:
    condition = report.get("connection_reuse_condition")
    replicate = report.get("replicate")
    results = report.get("results")
    if condition not in EXPECTED_CONDITIONS:
        raise ValueError("unexpected connection-reuse condition")
    if replicate not in EXPECTED_REPLICATES:
        raise ValueError("unexpected replicate")
    if not isinstance(results, list):
        raise ValueError("report results must be a list")

    error_classes: Counter[str] = Counter()
    http_versions: Counter[str] = Counter()
    status_codes: Counter[int] = Counter()
    temporal: list[dict[str, Any]] = []
    for index, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            raise ValueError("report result must be an object")
        version = item.get("http_version")
        if isinstance(version, str) and version:
            http_versions[version] += 1
        for code in item.get("status_codes", []):
            if isinstance(code, int):
                status_codes[code] += 1
        if item.get("status") != "completed":
            error_classes[_typed_error(item)] += 1
        temporal.append({
            "call_index": index,
            "round": item.get("round"),
            "order_in_round": item.get("order_in_round"),
            "mode": item.get("mode"),
            "status": item.get("status"),
            "response_headers_seconds": item.get("response_headers_seconds"),
            "total_seconds": item.get("total_seconds"),
            "http_version": item.get("http_version"),
            "transport_bytes_downloaded": item.get("transport_bytes_downloaded"),
            "error_class": None if item.get("status") == "completed" else _typed_error(item),
        })

    return {
        "diagnostic": DIAGNOSTIC_NAME,
        "diagnostic_spec_version": DIAGNOSTIC_SPEC_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "source_complete": report.get("complete") is True,
        "source_commit_sha": report.get("git_sha"),
        "condition": condition,
        "replicate": replicate,
        "calls_observed": len(results),
        "http_versions": dict(sorted(http_versions.items())),
        "status_codes": dict(sorted(status_codes.items())),
        "typed_error_classes": dict(sorted(error_classes.items())),
        "temporal_calls": temporal,
        "sensitive_payloads_recorded": False,
        "functional_configuration_changed": False,
        "promotion_authorized": False,
    }
