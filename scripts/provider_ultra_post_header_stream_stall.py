from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import httpx2
from openai import DefaultHttpxClient, OpenAI

from provider_ultra_transport_diagnostic import (
    API_BASE_URL,
    MAX_RETRIES,
    PACING_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    RequestTracker,
    extract_stream_content,
    request_spec,
    safe_error_metadata,
)

EXPERIMENT_NAME = "provider_ultra_post_header_stream_stall"
EXPERIMENT_SPEC_VERSION = "2026-09-28.1"
CALLS_PER_REPLICATE = 20
REPLICATES = 5
SLO_SECONDS = 8.0
MAX_CONNECTIONS = 1000
KEEPALIVE_EXPIRY_SECONDS = 30.0
CONDITIONS = {
    "reuse": 100,
    "no_keepalive": 0,
}


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(p * len(ordered)) - 1)
    return round(ordered[index], 4)


def delta_seconds(later: Any, earlier: Any) -> float | None:
    if not isinstance(later, (int, float)) or not isinstance(earlier, (int, float)):
        return None
    return round(float(later) - float(earlier), 4)


class PostHeaderTracker(RequestTracker):
    def begin(self, label: str) -> None:
        super().begin(label)
        assert self.current is not None
        self.current["first_transport_body_chunk_seconds"] = None
        self.current["first_transport_body_chunk_size_bytes"] = None

    def note_transport_body_chunk(self, size_bytes: int) -> None:
        if self.current is None or size_bytes <= 0:
            return
        if self.current["first_transport_body_chunk_seconds"] is None:
            self.current["first_transport_body_chunk_seconds"] = round(
                time.perf_counter() - self.current["started"],
                4,
            )
            self.current["first_transport_body_chunk_size_bytes"] = size_bytes

    def finish(self) -> dict[str, Any]:
        if self.current is None:
            raise RuntimeError("request tracker has no active request")
        first_chunk_seconds = self.current["first_transport_body_chunk_seconds"]
        first_chunk_size = self.current["first_transport_body_chunk_size_bytes"]
        result = super().finish()
        result["first_transport_body_chunk_seconds"] = first_chunk_seconds
        result["first_transport_body_chunk_size_bytes"] = first_chunk_size
        return result


class FirstTransportBodyChunkStream(httpx2.SyncByteStream):
    def __init__(
        self,
        inner: Iterable[bytes],
        tracker: PostHeaderTracker,
    ) -> None:
        self._inner = inner
        self._tracker = tracker

    def __iter__(self):
        for chunk in self._inner:
            if chunk:
                self._tracker.note_transport_body_chunk(len(chunk))
            yield chunk

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()


class PostHeaderTimingTransport(httpx2.BaseTransport):
    def __init__(self, tracker: PostHeaderTracker, *, limits: httpx2.Limits) -> None:
        self._tracker = tracker
        self._inner = httpx2.HTTPTransport(limits=limits)

    def handle_request(self, request: httpx2.Request) -> httpx2.Response:
        response = self._inner.handle_request(request)
        return httpx2.Response(
            status_code=response.status_code,
            headers=response.headers,
            stream=FirstTransportBodyChunkStream(response.stream, self._tracker),
            extensions=response.extensions,
        )

    def close(self) -> None:
        self._inner.close()


def stage_gaps(item: dict[str, Any]) -> dict[str, float | None]:
    headers = item.get("response_headers_seconds")
    first_chunk = item.get("first_transport_body_chunk_seconds")
    first_event = item.get("first_event_seconds")
    first_content = item.get("first_content_seconds")
    total = item.get("total_seconds")
    return {
        "headers_to_first_transport_body_chunk_seconds": delta_seconds(first_chunk, headers),
        "first_transport_body_chunk_to_first_event_seconds": delta_seconds(
            first_event, first_chunk
        ),
        "first_event_to_first_content_seconds": delta_seconds(first_content, first_event),
        "first_content_to_finish_seconds": delta_seconds(total, first_content),
    }


def run_stream_call(
    client: OpenAI,
    tracker: PostHeaderTracker,
    *,
    call_number: int,
) -> dict[str, Any]:
    spec = request_spec()
    tracker.begin("stream")
    started = time.perf_counter()
    stream_ready_seconds: float | None = None
    first_event_seconds: float | None = None
    first_content_seconds: float | None = None
    stream_events_observed = 0
    stream_content_chunks_observed = 0
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
                if extract_stream_content(chunk):
                    stream_content_chunks_observed += 1
                    if first_content_seconds is None:
                        first_content_seconds = round(now - started, 4)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

        total_seconds = round(time.perf_counter() - started, 4)
        transport = tracker.finish()
        result = {
            "mode": "stream",
            "call_number": call_number,
            "model": spec["model"],
            "status": "completed",
            "max_tokens": spec["max_tokens"],
            "stream_ready_seconds": stream_ready_seconds,
            "first_event_seconds": first_event_seconds,
            "first_content_seconds": first_content_seconds,
            "stream_events_observed": stream_events_observed,
            "stream_content_chunks_observed": stream_content_chunks_observed,
            "total_seconds": total_seconds,
            **transport,
        }
        result.update(stage_gaps(result))
        return result
    except Exception as exc:
        total_seconds = round(time.perf_counter() - started, 4)
        transport = tracker.finish()
        result = {
            "mode": "stream",
            "call_number": call_number,
            "model": spec["model"],
            "status": "infrastructure_error",
            "max_tokens": spec["max_tokens"],
            "error_phase": error_phase,
            "stream_ready_seconds": stream_ready_seconds,
            "first_event_seconds": first_event_seconds,
            "first_content_seconds": first_content_seconds,
            "stream_events_observed": stream_events_observed,
            "stream_content_chunks_observed": stream_content_chunks_observed,
            "total_seconds": total_seconds,
            **transport,
            **safe_error_metadata(exc),
        }
        result.update(stage_gaps(result))
        return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in results if item.get("status") == "completed"]
    errors = [item for item in results if item.get("status") != "completed"]

    def metric(name: str, items: list[dict[str, Any]] = results) -> list[float]:
        return [
            float(item[name])
            for item in items
            if isinstance(item.get(name), (int, float))
        ]

    total_values = metric("total_seconds", completed)
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
    error_status_codes = Counter(
        int(item["status_code"])
        for item in errors
        if isinstance(item.get("status_code"), int)
    )

    post_header_no_body_errors = sum(
        1
        for item in errors
        if isinstance(item.get("response_headers_seconds"), (int, float))
        and item.get("first_transport_body_chunk_seconds") is None
    )
    post_body_pre_event_errors = sum(
        1
        for item in errors
        if isinstance(item.get("first_transport_body_chunk_seconds"), (int, float))
        and item.get("first_event_seconds") is None
    )
    post_event_pre_content_errors = sum(
        1
        for item in errors
        if isinstance(item.get("first_event_seconds"), (int, float))
        and item.get("first_content_seconds") is None
    )

    stage_metrics = {}
    for name in (
        "response_headers_seconds",
        "first_transport_body_chunk_seconds",
        "first_event_seconds",
        "first_content_seconds",
        "total_seconds",
        "headers_to_first_transport_body_chunk_seconds",
        "first_transport_body_chunk_to_first_event_seconds",
        "first_event_to_first_content_seconds",
        "first_content_to_finish_seconds",
    ):
        values = metric(name)
        stage_metrics[name] = {
            "observed": len(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
        }

    return {
        "planned": CALLS_PER_REPLICATE,
        "observed": len(results),
        "completed": len(completed),
        "errors": len(errors),
        "stage_metrics": stage_metrics,
        "typed_error_types": dict(sorted(error_types.items())),
        "error_phases": dict(sorted(error_phases.items())),
        "error_status_codes": dict(sorted(error_status_codes.items())),
        "post_header_no_body_errors": post_header_no_body_errors,
        "post_body_pre_event_errors": post_body_pre_event_errors,
        "post_event_pre_content_errors": post_event_pre_content_errors,
        "completed_calls_over_slo": sum(
            1 for value in total_values if value > SLO_SECONDS
        ),
        "slo_seconds": SLO_SECONDS,
        "strict_slo_met": (
            len(results) == CALLS_PER_REPLICATE
            and not errors
            and total_p95 is not None
            and total_p95 <= SLO_SECONDS
        ),
    }


def build_report(
    results: list[dict[str, Any]],
    *,
    condition: str,
    replicate: int,
    complete: bool,
) -> dict[str, Any]:
    spec = request_spec()
    return {
        "experiment": EXPERIMENT_NAME,
        "experiment_spec_version": EXPERIMENT_SPEC_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "complete": complete,
        "provider": "nvidia_nim",
        "api_base_url": API_BASE_URL,
        "model": spec["model"],
        "connection_reuse_condition": condition,
        "replicate": replicate,
        "calls_per_replicate": CALLS_PER_REPLICATE,
        "http_limits": {
            "max_connections": MAX_CONNECTIONS,
            "max_keepalive_connections": CONDITIONS[condition],
            "keepalive_expiry_seconds": KEEPALIVE_EXPIRY_SECONDS,
        },
        "frozen_request_contract": {
            "request_spec_source": "provider_ultra_transport_diagnostic.request_spec",
            "max_tokens": spec["max_tokens"],
            "temperature": spec["temperature"],
            "top_p": spec["top_p"],
            "enable_thinking": spec["extra_body"]["chat_template_kwargs"][
                "enable_thinking"
            ],
            "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "max_retries": MAX_RETRIES,
            "pacing_seconds": PACING_SECONDS,
            "slo_seconds": SLO_SECONDS,
        },
        "results": results,
        "summary": summarize(results),
        "sensitive_payloads_recorded": False,
        "functional_configuration_changed": False,
        "promotion_authorized": False,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.replicate <= REPLICATES:
        raise ValueError(f"--replicate must be between 1 and {REPLICATES}")

    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        print("NVIDIA_API_KEY is required for the post-header stall diagnostic")
        return 2

    tracker = PostHeaderTracker()
    limits = httpx2.Limits(
        max_connections=MAX_CONNECTIONS,
        max_keepalive_connections=CONDITIONS[args.condition],
        keepalive_expiry=KEEPALIVE_EXPIRY_SECONDS,
    )
    transport = PostHeaderTimingTransport(tracker, limits=limits)
    http_client = DefaultHttpxClient(
        transport=transport,
        event_hooks={
            "request": [tracker.request_hook],
            "response": [tracker.response_hook],
        },
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
            condition=args.condition,
            replicate=args.replicate,
            complete=False,
        ),
    )

    for call_number in range(1, CALLS_PER_REPLICATE + 1):
        pacing_seconds_before_call = 0.0
        if call_number > 1:
            pacing_started = time.perf_counter()
            time.sleep(PACING_SECONDS)
            pacing_seconds_before_call = round(
                time.perf_counter() - pacing_started,
                4,
            )

        result = run_stream_call(
            client,
            tracker,
            call_number=call_number,
        )
        result["pacing_seconds_before_call"] = pacing_seconds_before_call
        result["connection_reuse_condition"] = args.condition
        result["replicate"] = args.replicate
        results.append(result)

        write_report(
            output_path,
            build_report(
                results,
                condition=args.condition,
                replicate=args.replicate,
                complete=False,
            ),
        )

    report = build_report(
        results,
        condition=args.condition,
        replicate=args.replicate,
        complete=True,
    )
    write_report(output_path, report)
    print(json.dumps({
        "experiment": EXPERIMENT_NAME,
        "condition": args.condition,
        "replicate": args.replicate,
        "summary": report["summary"],
    }, sort_keys=True))

    if report["summary"]["errors"]:
        return 2
    if not report["summary"]["strict_slo_met"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
