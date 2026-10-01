from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
from openai import DefaultHttpxClient, OpenAI

from provider_ultra_raw_http_stream_vs_nonstream import (
    API_BASE_URL as RAW_API_BASE_URL,
    FUNCTIONAL_CONFIGURATION_CHANGED as RAW_FUNCTIONAL_CONFIGURATION_CHANGED,
    MAX_RETRIES as RAW_MAX_RETRIES,
    PACING_SECONDS as RAW_PACING_SECONDS,
    PROMOTION_AUTHORIZED as RAW_PROMOTION_AUTHORIZED,
    REQUEST_TIMEOUT_SECONDS as RAW_REQUEST_TIMEOUT_SECONDS,
    SENSITIVE_PAYLOADS_RECORDED as RAW_SENSITIVE_PAYLOADS_RECORDED,
    SLO_SECONDS as RAW_SLO_SECONDS,
    ULTRA_MODEL as RAW_ULTRA_MODEL,
    raw_request_payload,
    run_raw_call,
)
from provider_ultra_transport_diagnostic import (
    API_BASE_URL as SDK_API_BASE_URL,
    MAX_RETRIES as SDK_MAX_RETRIES,
    PACING_SECONDS as SDK_PACING_SECONDS,
    REQUEST_TIMEOUT_SECONDS as SDK_REQUEST_TIMEOUT_SECONDS,
    ULTRA_MODEL as SDK_ULTRA_MODEL,
    RequestTracker,
    request_spec as sdk_request_spec,
    run_nonstream_call as run_sdk_nonstream_call,
    run_stream_call as run_sdk_stream_call,
)

EXPERIMENT_NAME = "provider_ultra_sdk_vs_raw_http"
EXPERIMENT_SPEC_VERSION = "2026-10-01.1"
REPETITIONS = 20
SLO_SECONDS = 8.0
PACING_SECONDS = 12.0
REQUEST_TIMEOUT_SECONDS = 15.0
MAX_RETRIES = 0
CONDITIONS = ("sdk_stream", "raw_stream", "sdk_nonstream", "raw_nonstream")
ORDER_SCHEDULE = (
    ("sdk_stream", "raw_stream", "sdk_nonstream", "raw_nonstream"),
    ("raw_stream", "sdk_nonstream", "raw_nonstream", "sdk_stream"),
    ("sdk_nonstream", "raw_nonstream", "sdk_stream", "raw_stream"),
    ("raw_nonstream", "sdk_stream", "raw_stream", "sdk_nonstream"),
)
EXECUTION_MODE = "four_condition_latin_square_sequential_paced"
FUNCTIONAL_CONFIGURATION_CHANGED = False
PROMOTION_AUTHORIZED = False
SENSITIVE_PAYLOADS_RECORDED = False


def frozen_request_payload(*, stream: bool) -> dict[str, Any]:
    return {
        "model": "nvidia/nemotron-3-ultra-550b-a55b",
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": stream,
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(p * len(ordered)) - 1)], 4)


def normalized_sdk_payload(*, stream: bool) -> dict[str, Any]:
    spec = sdk_request_spec()
    return {
        "model": spec["model"],
        "messages": spec["messages"],
        "max_tokens": spec["max_tokens"],
        "temperature": spec["temperature"],
        "top_p": spec["top_p"],
        "chat_template_kwargs": spec["extra_body"]["chat_template_kwargs"],
        "stream": stream,
    }


def assert_frozen_base_contracts() -> None:
    if not (SDK_API_BASE_URL == RAW_API_BASE_URL == "https://integrate.api.nvidia.com/v1"):
        raise RuntimeError("API base drift")
    if not (SDK_ULTRA_MODEL == RAW_ULTRA_MODEL == "nvidia/nemotron-3-ultra-550b-a55b"):
        raise RuntimeError("model drift")
    if not (SDK_REQUEST_TIMEOUT_SECONDS == RAW_REQUEST_TIMEOUT_SECONDS == REQUEST_TIMEOUT_SECONDS):
        raise RuntimeError("timeout drift")
    if not (SDK_MAX_RETRIES == RAW_MAX_RETRIES == MAX_RETRIES):
        raise RuntimeError("retry drift")
    if not (SDK_PACING_SECONDS == RAW_PACING_SECONDS == PACING_SECONDS):
        raise RuntimeError("pacing drift")
    if RAW_SLO_SECONDS != SLO_SECONDS:
        raise RuntimeError("SLO drift")
    if any((
        RAW_FUNCTIONAL_CONFIGURATION_CHANGED,
        RAW_PROMOTION_AUTHORIZED,
        RAW_SENSITIVE_PAYLOADS_RECORDED,
    )):
        raise RuntimeError("raw governance contract drift")
    for stream in (True, False):
        expected = frozen_request_payload(stream=stream)
        sdk_payload = normalized_sdk_payload(stream=stream)
        raw_payload = raw_request_payload(stream=stream)
        if sdk_payload != expected:
            raise RuntimeError(f"SDK payload drift from frozen contract stream={stream}")
        if raw_payload != expected:
            raise RuntimeError(f"raw payload drift from frozen contract stream={stream}")


def normalize_comparison_result(condition: str, item: dict[str, Any]) -> dict[str, Any]:
    result = dict(item)
    source_status = str(result.get("status") or "unknown")
    source_error_type = result.get("error_type")

    if condition == "sdk_stream":
        events = result.get("stream_events_observed")
        comparison_complete = (
            source_status == "completed"
            and isinstance(events, int)
            and events > 0
        )
    elif condition == "raw_stream":
        status_code = result.get("status_code")
        events = result.get("sse_events_observed")
        comparison_complete = (
            isinstance(status_code, int)
            and 200 <= status_code < 300
            and isinstance(events, int)
            and events > 0
        )
    elif condition == "sdk_nonstream":
        comparison_complete = source_status == "completed"
    elif condition == "raw_nonstream":
        comparison_complete = (
            source_status == "completed"
            and result.get("protocol_complete") is True
        )
    else:
        raise ValueError(f"unknown condition: {condition}")

    comparison_status = "completed" if comparison_complete else source_status
    if comparison_status == "completed" and not comparison_complete:
        comparison_status = "protocol_error"
    if source_status == "completed" and not comparison_complete:
        comparison_status = "protocol_error"

    result.update({
        "source_status": source_status,
        "source_error_type": source_error_type,
        "comparison_status": comparison_status,
        "comparison_protocol_complete": comparison_complete,
        "comparison_error_type": (
            None
            if comparison_complete
            else source_error_type or "IncompleteComparableResponse"
        ),
    })
    return result


def order_for_round(round_number: int) -> tuple[str, ...]:
    if round_number < 1:
        raise ValueError("round_number must be positive")
    return ORDER_SCHEDULE[(round_number - 1) % len(ORDER_SCHEDULE)]


def execute_condition(
    condition: str,
    *,
    sdk_client: OpenAI,
    sdk_tracker: RequestTracker,
    raw_client: httpx2.Client,
    api_key: str,
    round_number: int,
    order_in_round: int,
) -> dict[str, Any]:
    if condition == "sdk_stream":
        item = run_sdk_stream_call(
            sdk_client, sdk_tracker,
            round_number=round_number, order_in_round=order_in_round,
        )
        first_signal = item.get("first_event_seconds")
    elif condition == "sdk_nonstream":
        item = run_sdk_nonstream_call(
            sdk_client, sdk_tracker,
            round_number=round_number, order_in_round=order_in_round,
        )
        first_signal = None
    elif condition == "raw_stream":
        item = run_raw_call(
            raw_client, api_key=api_key, mode="stream",
            round_number=round_number, order_in_round=order_in_round,
        )
        first_signal = item.get("first_sse_event_seconds")
    elif condition == "raw_nonstream":
        item = run_raw_call(
            raw_client, api_key=api_key, mode="nonstream",
            round_number=round_number, order_in_round=order_in_round,
        )
        first_signal = None
    else:
        raise ValueError(f"unknown condition: {condition}")

    result = dict(item)
    transport, mode = condition.split("_", 1)
    result.update({
        "condition": condition,
        "transport": transport,
        "mode": mode,
        "round": round_number,
        "order_in_round": order_in_round,
        "first_signal_seconds": first_signal,
    })
    return normalize_comparison_result(condition, result)


def summarize(results: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    items = [x for x in results if x.get("condition") == condition]
    completed = [x for x in items if x.get("comparison_status") == "completed"]
    errors = [x for x in items if x.get("comparison_status") != "completed"]

    def metric(name: str) -> list[float]:
        return [float(x[name]) for x in completed if isinstance(x.get(name), (int, float))]

    totals = metric("total_seconds")
    p95 = percentile(totals, 0.95)
    return {
        "planned": REPETITIONS,
        "observed": len(items),
        "completed": len(completed),
        "errors": len(errors),
        "response_headers_p95": percentile(metric("response_headers_seconds"), 0.95),
        "first_signal_p95": percentile(metric("first_signal_seconds"), 0.95),
        "total_p50": percentile(totals, 0.50),
        "total_p95": p95,
        "completed_calls_over_slo": sum(v > SLO_SECONDS for v in totals),
        "strict_slo_met": (
            len(items) == REPETITIONS and not errors
            and p95 is not None and p95 <= SLO_SECONDS
        ),
    }


def paired_delta(
    results: list[dict[str, Any]],
    left: str,
    right: str,
    field: str,
) -> dict[str, Any]:
    by_round: dict[int, dict[str, dict[str, Any]]] = {}
    for item in results:
        by_round.setdefault(int(item["round"]), {})[str(item["condition"])] = item
    values: list[float] = []
    paired_completed = 0
    for round_number in range(1, REPETITIONS + 1):
        pair = by_round.get(round_number, {})
        lhs, rhs = pair.get(left), pair.get(right)
        if not lhs or not rhs:
            continue
        if lhs.get("comparison_status") == rhs.get("comparison_status") == "completed":
            paired_completed += 1
            a, b = lhs.get(field), rhs.get(field)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                values.append(round(float(a) - float(b), 4))
    return {
        "left": left,
        "right": right,
        "field": field,
        "paired_completed": paired_completed,
        "delta_left_minus_right_p50": percentile(values, 0.50),
        "delta_left_minus_right_p95": percentile(values, 0.95),
    }


def build_report(results: list[dict[str, Any]], *, complete: bool) -> dict[str, Any]:
    position_counts = {c: [0, 0, 0, 0] for c in CONDITIONS}
    for round_number in range(1, REPETITIONS + 1):
        for position, condition in enumerate(order_for_round(round_number)):
            position_counts[condition][position] += 1

    return {
        "experiment": EXPERIMENT_NAME,
        "experiment_spec_version": EXPERIMENT_SPEC_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "provider": "nvidia_nim",
        "model": SDK_ULTRA_MODEL,
        "execution_mode": EXECUTION_MODE,
        "conditions": list(CONDITIONS),
        "order_schedule": [list(x) for x in ORDER_SCHEDULE],
        "condition_position_counts": position_counts,
        "repetitions_per_condition": REPETITIONS,
        "total_planned_calls": REPETITIONS * len(CONDITIONS),
        "configured_pacing_seconds": PACING_SECONDS,
        "configured_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "configured_max_retries": MAX_RETRIES,
        "slo_seconds": SLO_SECONDS,
        "functional_configuration_changed": FUNCTIONAL_CONFIGURATION_CHANGED,
        "promotion_authorized": PROMOTION_AUTHORIZED,
        "sensitive_payloads_recorded": SENSITIVE_PAYLOADS_RECORDED,
        "quality_evaluation": False,
        "complete": complete,
        "frozen_request_contract": {
            "model": SDK_ULTRA_MODEL,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "enable_thinking": False,
        },
        "summary": {c: summarize(results, c) for c in CONDITIONS},
        "comparisons": {
            "sdk_minus_raw_stream_total": paired_delta(
                results, "sdk_stream", "raw_stream", "total_seconds"
            ),
            "sdk_minus_raw_nonstream_total": paired_delta(
                results, "sdk_nonstream", "raw_nonstream", "total_seconds"
            ),
            "sdk_minus_raw_stream_headers": paired_delta(
                results, "sdk_stream", "raw_stream", "response_headers_seconds"
            ),
            "sdk_minus_raw_nonstream_headers": paired_delta(
                results, "sdk_nonstream", "raw_nonstream", "response_headers_seconds"
            ),
            "sdk_minus_raw_stream_first_signal": paired_delta(
                results, "sdk_stream", "raw_stream", "first_signal_seconds"
            ),
            "sdk_stream_minus_nonstream_total": paired_delta(
                results, "sdk_stream", "sdk_nonstream", "total_seconds"
            ),
            "raw_stream_minus_nonstream_total": paired_delta(
                results, "raw_stream", "raw_nonstream", "total_seconds"
            ),
        },
        "results": results,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="provider-ultra-sdk-vs-raw-http/report.json")
    args = parser.parse_args()
    assert_frozen_base_contracts()

    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        print("NVIDIA_API_KEY is required for the SDK vs raw HTTP diagnostic")
        return 2

    tracker = RequestTracker()
    sdk_client = OpenAI(
        base_url=SDK_API_BASE_URL,
        api_key=api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        http_client=DefaultHttpxClient(event_hooks={
            "request": [tracker.request_hook],
            "response": [tracker.response_hook],
        }),
    )
    raw_client = httpx2.Client(
        transport=httpx2.HTTPTransport(retries=MAX_RETRIES),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    output = Path(args.output)
    results: list[dict[str, Any]] = []
    write_report(output, build_report(results, complete=False))

    call_index = 0
    try:
        for round_number in range(1, REPETITIONS + 1):
            for order_in_round, condition in enumerate(order_for_round(round_number), start=1):
                pacing = 0.0
                if call_index:
                    started = time.perf_counter()
                    time.sleep(PACING_SECONDS)
                    pacing = round(time.perf_counter() - started, 4)
                item = execute_condition(
                    condition,
                    sdk_client=sdk_client,
                    sdk_tracker=tracker,
                    raw_client=raw_client,
                    api_key=api_key,
                    round_number=round_number,
                    order_in_round=order_in_round,
                )
                item["pacing_seconds_before_call"] = pacing
                results.append(item)
                call_index += 1
                write_report(output, build_report(results, complete=False))
    finally:
        raw_client.close()
        sdk_client.close()

    report = build_report(results, complete=True)
    write_report(output, report)
    print(json.dumps({
        c: {
            "completed": report["summary"][c]["completed"],
            "errors": report["summary"][c]["errors"],
            "total_p95": report["summary"][c]["total_p95"],
            "strict_slo_met": report["summary"][c]["strict_slo_met"],
        }
        for c in CONDITIONS
    }, sort_keys=True))
    return 2 if any(report["summary"][c]["errors"] for c in CONDITIONS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
