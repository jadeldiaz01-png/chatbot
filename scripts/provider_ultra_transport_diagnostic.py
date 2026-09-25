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

from openai import DefaultHttpxClient, OpenAI

DIAGNOSTIC_SCHEMA_VERSION = "1.0"
DIAGNOSTIC_SPEC_VERSION = "2026-09-25.1"
API_BASE_URL = "https://integrate.api.nvidia.com/v1"
ULTRA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
REQUEST_ID_HEADERS = ("x-request-id", "x-nvidia-request-id", "request-id")
REQUEST_TIMEOUT_SECONDS = 15.0
MAX_RETRIES = 0
PACING_SECONDS = 12.0
MODES = ("stream", "nonstream")


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(p * len(ordered)) - 1)
    return round(ordered[index], 4)


class RequestTracker:
    def __init__(self) -> None:
        self.current: dict[str, Any] | None = None

    def begin(self, label: str) -> None:
        self.current = {
            "label": label,
            "started": time.perf_counter(),
            "attempts": 0,
            "status_codes": [],
            "request_ids": [],
            "response_content_types": [],
            "response_headers_seconds": None,
        }

    def request_hook(self, _request: Any) -> None:
        if self.current is not None:
            self.current["attempts"] += 1

    def response_hook(self, response: Any) -> None:
        if self.current is None:
            return

        if self.current["response_headers_seconds"] is None:
            self.current["response_headers_seconds"] = round(
                time.perf_counter() - self.current["started"],
                4,
            )

        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            self.current["status_codes"].append(status_code)

        headers = getattr(response, "headers", None)
        if headers is None:
            return

        content_type = headers.get("content-type")
        if isinstance(content_type, str) and content_type:
            normalized_content_type = content_type.split(";", 1)[0].strip().lower()
            if (
                normalized_content_type
                and normalized_content_type not in self.current["response_content_types"]
            ):
                self.current["response_content_types"].append(normalized_content_type)

        for header in REQUEST_ID_HEADERS:
            value = headers.get(header)
            if isinstance(value, str) and value:
                if value not in self.current["request_ids"]:
                    self.current["request_ids"].append(value)
                break

    def finish(self) -> dict[str, Any]:
        if self.current is None:
            raise RuntimeError("request tracker has no active request")
        data = self.current
        self.current = None
        attempts = int(data["attempts"])
        return {
            "response_headers_seconds": data["response_headers_seconds"],
            "http_attempts": attempts,
            "retries": max(attempts - 1, 0),
            "status_codes": list(data["status_codes"]),
            "request_ids": list(data["request_ids"]),
            "response_content_types": list(data["response_content_types"]),
        }


def request_spec() -> dict[str, Any]:
    return {
        "model": ULTRA_MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": False,
            }
        },
    }


def extract_stream_content(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    content = getattr(delta, "content", None)
    return content if isinstance(content, str) else ""


def extract_nonstream_content(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    return content if isinstance(content, str) else ""


def safe_error_metadata(exc: Exception) -> dict[str, Any]:
    item: dict[str, Any] = {"error_type": type(exc).__name__}

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        item["status_code"] = status_code

    request_id = getattr(exc, "request_id", None)
    if isinstance(request_id, str) and request_id:
        item["request_id"] = request_id

    for attr, field in (
        ("code", "error_code"),
        ("type", "error_api_type"),
        ("param", "error_param"),
    ):
        value = getattr(exc, attr, None)
        if isinstance(value, (str, int, float, bool)):
            item[field] = value

    response = getattr(exc, "response", None)
    if response is not None:
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            item["error_response_status_code"] = response_status

        headers = getattr(response, "headers", None)
        if headers is not None:
            content_type = headers.get("content-type")
            if isinstance(content_type, str) and content_type:
                item["error_response_content_type"] = (
                    content_type.split(";", 1)[0].strip().lower()
                )

    return item


def run_stream_call(
    client: OpenAI,
    tracker: RequestTracker,
    *,
    round_number: int,
    order_in_round: int,
) -> dict[str, Any]:
    spec = request_spec()
    tracker.begin("stream")
    started = time.perf_counter()
    stream_ready_seconds: float | None = None
    first_event_seconds: float | None = None
    first_content_seconds: float | None = None
    stream_events_observed = 0
    stream_content_chunks_observed = 0
    content_chars = 0
    error_phase = "create"

    try:
        stream = client.chat.completions.create(**spec, stream=True)
        stream_ready_seconds = round(time.perf_counter() - started, 4)
        error_phase = "iterate"
        try:
            for chunk in stream:
                now = time.perf_counter()
                stream_events_observed += 1
                if first_event_seconds is None:
                    first_event_seconds = round(now - started, 4)
                content = extract_stream_content(chunk)
                if content:
                    stream_content_chunks_observed += 1
                    if first_content_seconds is None:
                        first_content_seconds = round(now - started, 4)
                    content_chars += len(content)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

        total_seconds = round(time.perf_counter() - started, 4)
        transport = tracker.finish()
        return {
            "mode": "stream",
            "round": round_number,
            "order_in_round": order_in_round,
            "model": spec["model"],
            "status": "completed",
            "max_tokens": spec["max_tokens"],
            "stream_ready_seconds": stream_ready_seconds,
            "first_event_seconds": first_event_seconds,
            "first_content_seconds": first_content_seconds,
            "stream_events_observed": stream_events_observed,
            "stream_content_chunks_observed": stream_content_chunks_observed,
            "total_seconds": total_seconds,
            "content_chars_observed": content_chars,
            **transport,
        }
    except Exception as exc:
        total_seconds = round(time.perf_counter() - started, 4)
        transport = tracker.finish()
        return {
            "mode": "stream",
            "round": round_number,
            "order_in_round": order_in_round,
            "model": spec["model"],
            "status": "infrastructure_error",
            "max_tokens": spec["max_tokens"],
            "error_phase": error_phase,
            "stream_ready_seconds": stream_ready_seconds,
            "first_event_seconds": first_event_seconds,
            "first_content_seconds": first_content_seconds,
            "stream_events_observed": stream_events_observed,
            "stream_content_chunks_observed": stream_content_chunks_observed,
            "content_chars_observed": content_chars,
            "total_seconds": total_seconds,
            **transport,
            **safe_error_metadata(exc),
        }


def run_nonstream_call(
    client: OpenAI,
    tracker: RequestTracker,
    *,
    round_number: int,
    order_in_round: int,
) -> dict[str, Any]:
    spec = request_spec()
    tracker.begin("nonstream")
    started = time.perf_counter()

    try:
        response = client.chat.completions.create(**spec, stream=False)
        total_seconds = round(time.perf_counter() - started, 4)
        content = extract_nonstream_content(response)
        transport = tracker.finish()
        return {
            "mode": "nonstream",
            "round": round_number,
            "order_in_round": order_in_round,
            "model": spec["model"],
            "status": "completed",
            "max_tokens": spec["max_tokens"],
            "total_seconds": total_seconds,
            "content_chars_observed": len(content),
            **transport,
        }
    except Exception as exc:
        total_seconds = round(time.perf_counter() - started, 4)
        transport = tracker.finish()
        return {
            "mode": "nonstream",
            "round": round_number,
            "order_in_round": order_in_round,
            "model": spec["model"],
            "status": "infrastructure_error",
            "max_tokens": spec["max_tokens"],
            "error_phase": "request",
            "total_seconds": total_seconds,
            **transport,
            **safe_error_metadata(exc),
        }


def summarize_mode(
    results: list[dict[str, Any]],
    mode: str,
    slo_seconds: float,
    planned: int,
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

    status_codes = Counter(
        code
        for item in items
        for code in item.get("status_codes", [])
        if isinstance(code, int)
    )
    total_values = metric("total_seconds")
    total_p95 = percentile(total_values, 0.95)
    over_slo = sum(1 for value in total_values if value > slo_seconds)
    error_types = Counter(
        str(item["error_type"])
        for item in errors
        if isinstance(item.get("error_type"), str)
    )
    error_codes = Counter(
        str(item["error_code"])
        for item in errors
        if isinstance(item.get("error_code"), (str, int, float, bool))
    )
    error_phases = Counter(
        str(item["error_phase"])
        for item in errors
        if isinstance(item.get("error_phase"), str)
    )
    error_status_codes = Counter(
        int(item["status_code"])
        for item in errors
        if isinstance(item.get("status_code"), int)
    )
    response_content_types = Counter(
        content_type
        for item in items
        for content_type in item.get("response_content_types", [])
        if isinstance(content_type, str)
    )

    summary: dict[str, Any] = {
        "model": ULTRA_MODEL,
        "planned": planned,
        "completed": len(completed),
        "errors": len(errors),
        "response_headers_seconds": {
            "p50": percentile(metric("response_headers_seconds"), 0.50),
            "p95": percentile(metric("response_headers_seconds"), 0.95),
        },
        "total_seconds": {
            "p50": percentile(total_values, 0.50),
            "p95": total_p95,
            "min": round(min(total_values), 4) if total_values else None,
            "max": round(max(total_values), 4) if total_values else None,
        },
        "completed_calls_over_slo": over_slo,
        "total_http_attempts": sum(
            int(item.get("http_attempts", 0))
            for item in items
            if isinstance(item.get("http_attempts"), int)
        ),
        "total_retries": sum(
            int(item.get("retries", 0))
            for item in items
            if isinstance(item.get("retries"), int)
        ),
        "status_codes": dict(sorted(status_codes.items())),
        "response_content_types": dict(sorted(response_content_types.items())),
        "error_types": dict(sorted(error_types.items())),
        "error_codes": dict(sorted(error_codes.items())),
        "error_phases": dict(sorted(error_phases.items())),
        "error_status_codes": dict(sorted(error_status_codes.items())),
        "slo_seconds": slo_seconds,
        "strict_slo_met": (
            len(items) == planned
            and not errors
            and total_p95 is not None
            and total_p95 <= slo_seconds
        ),
    }

    if mode == "stream":
        summary["stream_ready_seconds"] = {
            "p50": percentile(metric("stream_ready_seconds"), 0.50),
            "p95": percentile(metric("stream_ready_seconds"), 0.95),
        }
        summary["first_event_seconds"] = {
            "p50": percentile(metric("first_event_seconds"), 0.50),
            "p95": percentile(metric("first_event_seconds"), 0.95),
        }
        summary["first_content_seconds"] = {
            "p50": percentile(metric("first_content_seconds"), 0.50),
            "p95": percentile(metric("first_content_seconds"), 0.95),
        }

    return summary


def build_comparison(
    results: list[dict[str, Any]],
    *,
    repetitions: int,
    slo_seconds: float,
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
        "stream_over_slo_nonstream_within": 0,
        "nonstream_over_slo_stream_within": 0,
        "both_over_slo": 0,
    }
    completed_deltas: list[float] = []

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
            stream_total = float(stream["total_seconds"])
            nonstream_total = float(nonstream["total_seconds"])
            completed_deltas.append(round(stream_total - nonstream_total, 4))
            stream_over = stream_total > slo_seconds
            nonstream_over = nonstream_total > slo_seconds
            if stream_over and nonstream_over:
                counts["both_over_slo"] += 1
            elif stream_over:
                counts["stream_over_slo_nonstream_within"] += 1
            elif nonstream_over:
                counts["nonstream_over_slo_stream_within"] += 1
        elif not stream_ok and nonstream_ok:
            counts["stream_error_nonstream_completed"] += 1
        elif stream_ok and not nonstream_ok:
            counts["nonstream_error_stream_completed"] += 1
        else:
            counts["both_error"] += 1

    return {
        **counts,
        "stream_minus_nonstream_total_seconds": {
            "p50": percentile(completed_deltas, 0.50),
            "p95": percentile(completed_deltas, 0.95),
            "min": round(min(completed_deltas), 4) if completed_deltas else None,
            "max": round(max(completed_deltas), 4) if completed_deltas else None,
        },
    }


def build_report(
    results: list[dict[str, Any]],
    *,
    repetitions: int,
    slo_seconds: float,
    complete: bool,
) -> dict[str, Any]:
    return {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_spec_version": DIAGNOSTIC_SPEC_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "provider": "nvidia_nim",
        "api_base_url": API_BASE_URL,
        "model": ULTRA_MODEL,
        "repetitions_per_mode": repetitions,
        "execution_mode": "paired_sequential_alternating_order_paced",
        "configured_pacing_seconds": PACING_SECONDS,
        "quality_evaluation": False,
        "chatbot_functional_configuration_changed": False,
        "complete": complete,
        "configured_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "configured_max_retries": MAX_RETRIES,
        "slo_seconds": slo_seconds,
        "summary": {
            mode: summarize_mode(results, mode, slo_seconds, repetitions)
            for mode in MODES
        },
        "comparison": build_comparison(
            results,
            repetitions=repetitions,
            slo_seconds=slo_seconds,
        ),
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
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--slo-seconds", type=float, default=8.0)
    parser.add_argument(
        "--output",
        default="provider-ultra-transport-diagnostic/report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 5 <= args.repetitions <= 20:
        raise ValueError("--repetitions must be between 5 and 20")
    if not math.isfinite(args.slo_seconds) or args.slo_seconds <= 0:
        raise ValueError("--slo-seconds must be finite and positive")

    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        print("NVIDIA_API_KEY is required for the Ultra transport diagnostic")
        return 2

    tracker = RequestTracker()
    http_client = DefaultHttpxClient(
        event_hooks={
            "request": [tracker.request_hook],
            "response": [tracker.response_hook],
        }
    )
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        http_client=http_client,
    )

    output_path = Path(args.output)
    results: list[dict[str, Any]] = []
    write_report(
        output_path,
        build_report(
            results,
            repetitions=args.repetitions,
            slo_seconds=args.slo_seconds,
            complete=False,
        ),
    )

    call_index = 0
    for round_number in range(1, args.repetitions + 1):
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

            if mode == "stream":
                result = run_stream_call(
                    client,
                    tracker,
                    round_number=round_number,
                    order_in_round=order_in_round,
                )
            else:
                result = run_nonstream_call(
                    client,
                    tracker,
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
                    repetitions=args.repetitions,
                    slo_seconds=args.slo_seconds,
                    complete=False,
                ),
            )

    report = build_report(
        results,
        repetitions=args.repetitions,
        slo_seconds=args.slo_seconds,
        complete=True,
    )
    write_report(output_path, report)

    concise = {
        "stream": {
            "completed": report["summary"]["stream"]["completed"],
            "errors": report["summary"]["stream"]["errors"],
            "total_p95": report["summary"]["stream"]["total_seconds"]["p95"],
            "strict_slo_met": report["summary"]["stream"]["strict_slo_met"],
        },
        "nonstream": {
            "completed": report["summary"]["nonstream"]["completed"],
            "errors": report["summary"]["nonstream"]["errors"],
            "total_p95": report["summary"]["nonstream"]["total_seconds"]["p95"],
            "strict_slo_met": report["summary"]["nonstream"]["strict_slo_met"],
        },
        "comparison": report["comparison"],
    }
    print(json.dumps(concise, sort_keys=True))

    if any(report["summary"][mode]["errors"] for mode in MODES):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
