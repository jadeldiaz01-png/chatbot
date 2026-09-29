from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2

API_BASE_URL = "https://integrate.api.nvidia.com/v1"
ULTRA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
REQUEST_TIMEOUT_SECONDS = 15.0
MAX_RETRIES = 0
PACING_SECONDS = 12.0
MODES = ("stream", "nonstream")

EXPERIMENT_NAME = "provider_ultra_raw_http_stream_vs_nonstream"
EXPERIMENT_SPEC_VERSION = "2026-09-29.1"
RAW_CHAT_COMPLETIONS_URL = f"{API_BASE_URL}/chat/completions"
SLO_SECONDS = 8.0
REPETITIONS = 20
FUNCTIONAL_CONFIGURATION_CHANGED = False
PROMOTION_AUTHORIZED = False
SENSITIVE_PAYLOADS_RECORDED = False
SDK_BYPASSED = True
EXECUTION_MODE = "paired_sequential_alternating_order_paced"
REQUEST_PATH = "raw_httpx2_no_openai_sdk"


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(p * len(ordered)) - 1)
    return round(ordered[index], 4)


def raw_request_payload(*, stream: bool) -> dict[str, Any]:
    return {
        "model": ULTRA_MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
        "stream": stream,
    }


def safe_error_metadata(exc: Exception) -> dict[str, Any]:
    return {"error_type": type(exc).__name__}


def _normalized_content_type(response: httpx2.Response) -> str | None:
    value = response.headers.get("content-type")
    if not isinstance(value, str) or not value:
        return None
    return value.split(";", 1)[0].strip().lower() or None


def _request_id(response: httpx2.Response) -> str | None:
    for header in ("x-request-id", "x-nvidia-request-id", "request-id"):
        value = response.headers.get(header)
        if isinstance(value, str) and value:
            return value
    return None


def _feed_sse_lines(
    buffer: bytes,
    chunk: bytes,
    *,
    started: float,
    first_event_seconds: float | None,
    events_observed: int,
    done_observed: bool,
) -> tuple[bytes, float | None, int, bool]:
    data = buffer + chunk
    lines = data.split(b"\n")
    remainder = lines.pop()
    for raw_line in lines:
        line = raw_line.rstrip(b"\r")
        if not line.startswith(b"data:"):
            continue
        events_observed += 1
        if first_event_seconds is None:
            first_event_seconds = round(time.perf_counter() - started, 4)
        if line.strip() == b"data: [DONE]":
            done_observed = True
    return remainder, first_event_seconds, events_observed, done_observed


def run_raw_call(
    client: httpx2.Client,
    *,
    api_key: str,
    mode: str,
    round_number: int,
    order_in_round: int,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")

    stream = mode == "stream"
    payload = raw_request_payload(stream=stream)
    started = time.perf_counter()
    headers_seconds: float | None = None
    first_body_byte_seconds: float | None = None
    first_sse_event_seconds: float | None = None
    body_bytes_observed = 0
    body_chunks_observed = 0
    sse_events_observed = 0
    sse_done_observed = False
    sse_buffer = b""
    status_code: int | None = None
    http_version: str | None = None
    content_type: str | None = None
    request_id: str | None = None
    error_phase = "connect_or_headers"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        with client.stream(
            "POST",
            RAW_CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
        ) as response:
            headers_seconds = round(time.perf_counter() - started, 4)
            error_phase = "body"
            status_code = response.status_code
            http_version = response.http_version
            content_type = _normalized_content_type(response)
            request_id = _request_id(response)

            for chunk in response.iter_raw():
                if not chunk:
                    continue
                now = time.perf_counter()
                if first_body_byte_seconds is None:
                    first_body_byte_seconds = round(now - started, 4)
                body_chunks_observed += 1
                body_bytes_observed += len(chunk)
                if stream:
                    (
                        sse_buffer,
                        first_sse_event_seconds,
                        sse_events_observed,
                        sse_done_observed,
                    ) = _feed_sse_lines(
                        sse_buffer,
                        chunk,
                        started=started,
                        first_event_seconds=first_sse_event_seconds,
                        events_observed=sse_events_observed,
                        done_observed=sse_done_observed,
                    )

            total_seconds = round(time.perf_counter() - started, 4)
            status = "completed" if 200 <= status_code < 300 else "http_error"
            error_type = None if status == "completed" else "HTTPStatusError"
            return {
                "mode": mode,
                "round": round_number,
                "order_in_round": order_in_round,
                "status": status,
                "error_type": error_type,
                "error_phase": None if status == "completed" else "response_status",
                "status_code": status_code,
                "response_headers_seconds": headers_seconds,
                "first_body_byte_seconds": first_body_byte_seconds,
                "headers_to_first_body_byte_seconds": (
                    round(first_body_byte_seconds - headers_seconds, 4)
                    if isinstance(first_body_byte_seconds, (int, float))
                    and isinstance(headers_seconds, (int, float))
                    else None
                ),
                "first_sse_event_seconds": first_sse_event_seconds if stream else None,
                "body_chunks_observed": body_chunks_observed,
                "body_bytes_observed": body_bytes_observed,
                "sse_events_observed": sse_events_observed if stream else None,
                "sse_done_observed": sse_done_observed if stream else None,
                "http_version": http_version,
                "response_content_type": content_type,
                "request_id": request_id,
                "total_seconds": total_seconds,
            }
    except Exception as exc:
        total_seconds = round(time.perf_counter() - started, 4)
        return {
            "mode": mode,
            "round": round_number,
            "order_in_round": order_in_round,
            "status": "infrastructure_error",
            "error_phase": error_phase,
            "status_code": status_code,
            "response_headers_seconds": headers_seconds,
            "first_body_byte_seconds": first_body_byte_seconds,
            "headers_to_first_body_byte_seconds": (
                round(first_body_byte_seconds - headers_seconds, 4)
                if isinstance(first_body_byte_seconds, (int, float))
                and isinstance(headers_seconds, (int, float))
                else None
            ),
            "first_sse_event_seconds": first_sse_event_seconds if stream else None,
            "body_chunks_observed": body_chunks_observed,
            "body_bytes_observed": body_bytes_observed,
            "sse_events_observed": sse_events_observed if stream else None,
            "sse_done_observed": sse_done_observed if stream else None,
            "http_version": http_version,
            "response_content_type": content_type,
            "request_id": request_id,
            "total_seconds": total_seconds,
            **safe_error_metadata(exc),
        }


def summarize_mode(
    results: list[dict[str, Any]],
    *,
    mode: str,
    repetitions: int,
    slo_seconds: float,
) -> dict[str, Any]:
    items = [item for item in results if item["mode"] == mode]
    completed = [item for item in items if item["status"] == "completed"]
    errors = [item for item in items if item["status"] != "completed"]

    def metric(name: str) -> list[float]:
        return [
            float(item[name])
            for item in completed
            if isinstance(item.get(name), (int, float))
        ]

    total_values = metric("total_seconds")
    total_p95 = percentile(total_values, 0.95)
    error_types = Counter(
        str(item["error_type"])
        for item in errors
        if isinstance(item.get("error_type"), str)
    )
    error_phases = Counter(
        str(item["error_phase"])
        for item in errors
        if isinstance(item.get("error_phase"), str)
    )
    status_codes = Counter(
        int(item["status_code"])
        for item in items
        if isinstance(item.get("status_code"), int)
    )

    return {
        "planned": repetitions,
        "observed": len(items),
        "completed": len(completed),
        "errors": len(errors),
        "response_headers_seconds": {
            "p50": percentile(metric("response_headers_seconds"), 0.50),
            "p95": percentile(metric("response_headers_seconds"), 0.95),
        },
        "first_body_byte_seconds": {
            "p50": percentile(metric("first_body_byte_seconds"), 0.50),
            "p95": percentile(metric("first_body_byte_seconds"), 0.95),
        },
        "headers_to_first_body_byte_seconds": {
            "p50": percentile(metric("headers_to_first_body_byte_seconds"), 0.50),
            "p95": percentile(metric("headers_to_first_body_byte_seconds"), 0.95),
        },
        "total_seconds": {
            "p50": percentile(total_values, 0.50),
            "p95": total_p95,
            "min": round(min(total_values), 4) if total_values else None,
            "max": round(max(total_values), 4) if total_values else None,
        },
        "first_sse_event_seconds": (
            {
                "p50": percentile(metric("first_sse_event_seconds"), 0.50),
                "p95": percentile(metric("first_sse_event_seconds"), 0.95),
            }
            if mode == "stream"
            else None
        ),
        "status_codes": dict(sorted(status_codes.items())),
        "error_types": dict(sorted(error_types.items())),
        "error_phases": dict(sorted(error_phases.items())),
        "completed_calls_over_slo": sum(
            value > slo_seconds for value in total_values
        ),
        "post_header_no_body_errors": sum(
            item.get("status") != "completed"
            and isinstance(item.get("response_headers_seconds"), (int, float))
            and item.get("first_body_byte_seconds") is None
            for item in items
        ),
        "strict_slo_met": (
            len(items) == repetitions
            and not errors
            and total_p95 is not None
            and total_p95 <= slo_seconds
        ),
    }


def build_comparison(
    results: list[dict[str, Any]],
    *,
    repetitions: int,
) -> dict[str, Any]:
    by_round: dict[int, dict[str, dict[str, Any]]] = {}
    for item in results:
        by_round.setdefault(int(item["round"]), {})[str(item["mode"])] = item

    counts = {
        "paired_rounds_observed": 0,
        "both_completed": 0,
        "stream_error_nonstream_completed": 0,
        "nonstream_error_stream_completed": 0,
        "both_error": 0,
    }
    header_deltas: list[float] = []
    first_byte_deltas: list[float] = []
    total_deltas: list[float] = []

    for round_number in range(1, repetitions + 1):
        pair = by_round.get(round_number, {})
        stream = pair.get("stream")
        nonstream = pair.get("nonstream")
        if stream is None or nonstream is None:
            continue
        counts["paired_rounds_observed"] += 1

        stream_ok = stream["status"] == "completed"
        nonstream_ok = nonstream["status"] == "completed"
        if stream_ok and nonstream_ok:
            counts["both_completed"] += 1
            for target, field in (
                (header_deltas, "response_headers_seconds"),
                (first_byte_deltas, "first_body_byte_seconds"),
                (total_deltas, "total_seconds"),
            ):
                left = stream.get(field)
                right = nonstream.get(field)
                if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                    target.append(round(float(left) - float(right), 4))
        elif not stream_ok and nonstream_ok:
            counts["stream_error_nonstream_completed"] += 1
        elif stream_ok and not nonstream_ok:
            counts["nonstream_error_stream_completed"] += 1
        else:
            counts["both_error"] += 1

    def delta_summary(values: list[float]) -> dict[str, float | None]:
        return {
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "min": round(min(values), 4) if values else None,
            "max": round(max(values), 4) if values else None,
        }

    return {
        **counts,
        "stream_minus_nonstream_response_headers_seconds": delta_summary(header_deltas),
        "stream_minus_nonstream_first_body_byte_seconds": delta_summary(first_byte_deltas),
        "stream_minus_nonstream_total_seconds": delta_summary(total_deltas),
    }


def build_report(
    results: list[dict[str, Any]],
    *,
    repetitions: int,
    slo_seconds: float,
    complete: bool,
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_NAME,
        "experiment_spec_version": EXPERIMENT_SPEC_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "provider": "nvidia_nim",
        "request_path": REQUEST_PATH,
        "sdk_bypassed": SDK_BYPASSED,
        "api_base_url": API_BASE_URL,
        "raw_chat_completions_url": RAW_CHAT_COMPLETIONS_URL,
        "model": ULTRA_MODEL,
        "execution_mode": EXECUTION_MODE,
        "repetitions_per_mode": repetitions,
        "configured_pacing_seconds": PACING_SECONDS,
        "configured_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "configured_max_retries": MAX_RETRIES,
        "slo_seconds": slo_seconds,
        "functional_configuration_changed": FUNCTIONAL_CONFIGURATION_CHANGED,
        "promotion_authorized": PROMOTION_AUTHORIZED,
        "sensitive_payloads_recorded": SENSITIVE_PAYLOADS_RECORDED,
        "quality_evaluation": False,
        "complete": complete,
        "frozen_request_contract": {
            "model": ULTRA_MODEL,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "enable_thinking": False,
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "max_retries": MAX_RETRIES,
            "pacing_seconds": PACING_SECONDS,
            "slo_seconds": slo_seconds,
        },
        "summary": {
            mode: summarize_mode(
                results,
                mode=mode,
                repetitions=repetitions,
                slo_seconds=slo_seconds,
            )
            for mode in MODES
        },
        "comparison": build_comparison(results, repetitions=repetitions),
        "results": results,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="provider-ultra-raw-http-stream-vs-nonstream/report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        print("NVIDIA_API_KEY is required for the raw HTTP diagnostic")
        return 2

    transport = httpx2.HTTPTransport(retries=MAX_RETRIES)
    client = httpx2.Client(
        transport=transport,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    output_path = Path(args.output)
    results: list[dict[str, Any]] = []
    write_report(
        output_path,
        build_report(
            results,
            repetitions=REPETITIONS,
            slo_seconds=SLO_SECONDS,
            complete=False,
        ),
    )

    call_index = 0
    try:
        for round_number in range(1, REPETITIONS + 1):
            order = MODES if round_number % 2 else tuple(reversed(MODES))
            for order_in_round, mode in enumerate(order, start=1):
                pacing_seconds_before_call = 0.0
                if call_index:
                    pacing_started = time.perf_counter()
                    time.sleep(PACING_SECONDS)
                    pacing_seconds_before_call = round(
                        time.perf_counter() - pacing_started,
                        4,
                    )

                result = run_raw_call(
                    client,
                    api_key=api_key,
                    mode=mode,
                    round_number=round_number,
                    order_in_round=order_in_round,
                )
                result["pacing_seconds_before_call"] = pacing_seconds_before_call
                results.append(result)
                call_index += 1
                write_report(
                    output_path,
                    build_report(
                        results,
                        repetitions=REPETITIONS,
                        slo_seconds=SLO_SECONDS,
                        complete=False,
                    ),
                )
    finally:
        client.close()

    report = build_report(
        results,
        repetitions=REPETITIONS,
        slo_seconds=SLO_SECONDS,
        complete=True,
    )
    write_report(output_path, report)

    concise = {
        "stream": {
            "completed": report["summary"]["stream"]["completed"],
            "errors": report["summary"]["stream"]["errors"],
            "total_p95": report["summary"]["stream"]["total_seconds"]["p95"],
        },
        "nonstream": {
            "completed": report["summary"]["nonstream"]["completed"],
            "errors": report["summary"]["nonstream"]["errors"],
            "total_p95": report["summary"]["nonstream"]["total_seconds"]["p95"],
        },
        "comparison": report["comparison"],
    }
    print(json.dumps(concise, sort_keys=True))

    if any(report["summary"][mode]["errors"] for mode in MODES):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
