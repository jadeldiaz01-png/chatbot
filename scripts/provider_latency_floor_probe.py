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

PROBE_SCHEMA_VERSION = "1.0"
PROBE_SPEC_VERSION = "2026-09-24.1"
API_BASE_URL = "https://integrate.api.nvidia.com/v1"
ULTRA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
SAFETY_MODEL = "nvidia/nemotron-3.5-content-safety"
REQUEST_ID_HEADERS = ("x-request-id", "x-nvidia-request-id", "request-id")
REQUEST_TIMEOUT_SECONDS = 15.0
MAX_RETRIES = 0


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
            "response_headers_seconds": None,
        }

    def request_hook(self, _request: Any) -> None:
        if self.current is None:
            return
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
        }


def extract_content(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    content = getattr(delta, "content", None)
    return content if isinstance(content, str) else ""


def request_spec(kind: str) -> dict[str, Any]:
    if kind == "ultra":
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

    if kind == "safety":
        return {
            "model": SAFETY_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello."}],
                }
            ],
            "max_tokens": 32,
            "temperature": 0.01,
            "top_p": 0.95,
            "extra_body": {
                "chat_template_kwargs": {
                    "request_categories": "/categories",
                    "enable_thinking": False,
                }
            },
        }

    raise ValueError(f"unknown probe kind: {kind}")


def safe_error_metadata(exc: Exception) -> dict[str, Any]:
    item: dict[str, Any] = {"error_type": type(exc).__name__}
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        item["status_code"] = status_code
    request_id = getattr(exc, "request_id", None)
    if isinstance(request_id, str) and request_id:
        item["request_id"] = request_id
    return item


def run_probe_call(
    client: OpenAI,
    tracker: RequestTracker,
    *,
    kind: str,
    round_number: int,
) -> dict[str, Any]:
    spec = request_spec(kind)
    tracker.begin(kind)
    started = time.perf_counter()
    first_event_seconds: float | None = None
    first_content_seconds: float | None = None
    content_chars = 0

    try:
        stream = client.chat.completions.create(**spec, stream=True)
        stream_ready_seconds = round(time.perf_counter() - started, 4)
        try:
            for chunk in stream:
                now = time.perf_counter()
                if first_event_seconds is None:
                    first_event_seconds = round(now - started, 4)
                content = extract_content(chunk)
                if content:
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
            "kind": kind,
            "round": round_number,
            "model": spec["model"],
            "status": "completed",
            "max_tokens": spec["max_tokens"],
            "stream_ready_seconds": stream_ready_seconds,
            "first_event_seconds": first_event_seconds,
            "first_content_seconds": first_content_seconds,
            "total_seconds": total_seconds,
            "content_chars_observed": content_chars,
            **transport,
        }
    except Exception as exc:
        total_seconds = round(time.perf_counter() - started, 4)
        transport = tracker.finish()
        return {
            "kind": kind,
            "round": round_number,
            "model": spec["model"],
            "status": "infrastructure_error",
            "max_tokens": spec["max_tokens"],
            "total_seconds": total_seconds,
            **transport,
            **safe_error_metadata(exc),
        }


def summarize(
    results: list[dict[str, Any]],
    kind: str,
    slo_seconds: float,
    planned: int,
) -> dict[str, Any]:
    items = [item for item in results if item["kind"] == kind]
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
    total_retries = sum(
        int(item.get("retries", 0))
        for item in items
        if isinstance(item.get("retries"), int)
    )
    total_attempts = sum(
        int(item.get("http_attempts", 0))
        for item in items
        if isinstance(item.get("http_attempts"), int)
    )
    total_values = metric("total_seconds")
    total_p95 = percentile(total_values, 0.95)

    return {
        "model": items[0]["model"] if items else None,
        "planned": planned,
        "completed": len(completed),
        "errors": len(errors),
        "response_headers_seconds": {
            "p50": percentile(metric("response_headers_seconds"), 0.50),
            "p95": percentile(metric("response_headers_seconds"), 0.95),
        },
        "stream_ready_seconds": {
            "p50": percentile(metric("stream_ready_seconds"), 0.50),
            "p95": percentile(metric("stream_ready_seconds"), 0.95),
        },
        "first_event_seconds": {
            "p50": percentile(metric("first_event_seconds"), 0.50),
            "p95": percentile(metric("first_event_seconds"), 0.95),
        },
        "first_content_seconds": {
            "p50": percentile(metric("first_content_seconds"), 0.50),
            "p95": percentile(metric("first_content_seconds"), 0.95),
        },
        "total_seconds": {
            "p50": percentile(total_values, 0.50),
            "p95": total_p95,
            "min": round(min(total_values), 4) if total_values else None,
            "max": round(max(total_values), 4) if total_values else None,
        },
        "total_http_attempts": total_attempts,
        "total_retries": total_retries,
        "retried_calls": sum(1 for item in items if int(item.get("retries", 0)) > 0),
        "status_codes": dict(sorted(status_codes.items())),
        "slo_seconds": slo_seconds,
        "slo_met_total_p95": (
            total_p95 is not None
            and not errors
            and total_p95 <= slo_seconds
        ),
    }


def build_report(
    results: list[dict[str, Any]],
    *,
    repetitions: int,
    slo_seconds: float,
    complete: bool,
) -> dict[str, Any]:
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "probe_spec_version": PROBE_SPEC_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "provider": "nvidia_nim",
        "api_base_url": API_BASE_URL,
        "repetitions_per_model": repetitions,
        "execution_mode": "sequential_alternating",
        "quality_evaluation": False,
        "complete": complete,
        "configured_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "configured_max_retries": MAX_RETRIES,
        "slo_seconds": slo_seconds,
        "summary": {
            "ultra": summarize(results, "ultra", slo_seconds, repetitions),
            "safety": summarize(results, "safety", slo_seconds, repetitions),
        },
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
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--slo-seconds", type=float, default=8.0)
    parser.add_argument("--output", default="provider-probe/report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 3 <= args.repetitions <= 20:
        raise ValueError("--repetitions must be between 3 and 20")
    if not math.isfinite(args.slo_seconds) or args.slo_seconds <= 0:
        raise ValueError("--slo-seconds must be finite and positive")

    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        print("NVIDIA_API_KEY is required for the provider latency probe")
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

    for round_number in range(1, args.repetitions + 1):
        # Alternate model probes within each round to reduce time-drift bias.
        for kind in ("ultra", "safety"):
            result = run_probe_call(
                client,
                tracker,
                kind=kind,
                round_number=round_number,
            )
            results.append(result)
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
        key: {
            "completed": value["completed"],
            "errors": value["errors"],
            "first_content_p95": value["first_content_seconds"]["p95"],
            "total_p95": value["total_seconds"]["p95"],
            "total_retries": value["total_retries"],
            "slo_met_total_p95": value["slo_met_total_p95"],
        }
        for key, value in report["summary"].items()
    }
    print(json.dumps(concise, sort_keys=True))

    if any(value["errors"] for value in report["summary"].values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
