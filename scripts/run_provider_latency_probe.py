from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from openai import DefaultHttpxClient, OpenAI

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_ULTRA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_SAFETY_MODEL = "nvidia/nemotron-3.5-content-safety"
DEFAULT_REPETITIONS = 20
DEFAULT_SLO_SECONDS = 8.0
DEFAULT_TIMEOUT_SECONDS = 45.0
DEFAULT_MAX_RETRIES = 2

_REQUEST_ID_HEADERS = ("x-request-id", "x-nvidia-request-id", "request-id")


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return round(ordered[index], 4)


def safe_error_metadata(exc: Exception) -> dict[str, Any]:
    metadata: dict[str, Any] = {"error_type": type(exc).__name__}

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        metadata["status_code"] = status_code

    request_id = getattr(exc, "request_id", None)
    if isinstance(request_id, str) and request_id:
        metadata["request_id"] = request_id

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        remote_error = body.get("error", body)
        if isinstance(remote_error, dict):
            code = remote_error.get("code")
            remote_type = remote_error.get("type")
            if isinstance(code, str) and code:
                metadata["remote_error_code"] = code
            if isinstance(remote_type, str) and remote_type:
                metadata["remote_error_type"] = remote_type

    return metadata


class AttemptTracker:
    def __init__(self) -> None:
        self.active_sample: str | None = None
        self.attempts: dict[str, int] = {}
        self.status_codes: dict[str, list[int]] = {}
        self.request_ids: dict[str, list[str]] = {}

    def begin(self, sample_id: str) -> None:
        self.active_sample = sample_id
        self.attempts[sample_id] = 0
        self.status_codes[sample_id] = []
        self.request_ids[sample_id] = []

    def end(self) -> None:
        self.active_sample = None

    def on_request(self, _request: Any) -> None:
        sample_id = self.active_sample
        if sample_id is None:
            return
        self.attempts[sample_id] = self.attempts.get(sample_id, 0) + 1

    def on_response(self, response: Any) -> None:
        sample_id = self.active_sample
        if sample_id is None:
            return

        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            self.status_codes.setdefault(sample_id, []).append(status_code)

        headers = getattr(response, "headers", None)
        if headers is None:
            return

        for header in _REQUEST_ID_HEADERS:
            value = headers.get(header)
            if isinstance(value, str) and value:
                request_ids = self.request_ids.setdefault(sample_id, [])
                if value not in request_ids:
                    request_ids.append(value)
                break

    def snapshot(self, sample_id: str) -> dict[str, Any]:
        attempts = self.attempts.get(sample_id, 0)
        return {
            "http_attempts": attempts,
            "retries": max(attempts - 1, 0),
            "status_codes": list(self.status_codes.get(sample_id, [])),
            "request_ids": list(self.request_ids.get(sample_id, [])),
        }


def _first_delta_has_content(chunk: Any) -> bool:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return False
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return False
    content = getattr(delta, "content", None)
    reasoning = getattr(delta, "reasoning_content", None)
    return bool(content) or bool(reasoning)


def run_stream_sample(
    *,
    client: OpenAI,
    tracker: AttemptTracker,
    sample_id: str,
    provider_stage: str,
    model: str,
    request_factory: Callable[[], dict[str, Any]],
    timeout_seconds: float,
    max_retries: int,
) -> dict[str, Any]:
    request = request_factory()
    started = time.perf_counter()
    first_event_seconds: float | None = None
    first_content_seconds: float | None = None
    stream_open_seconds: float | None = None
    output_chars = 0
    chunk_count = 0
    response_model: str | None = None

    tracker.begin(sample_id)
    try:
        stream = client.chat.completions.create(**request)
        stream_open_seconds = time.perf_counter() - started
        try:
            for chunk in stream:
                elapsed = time.perf_counter() - started
                chunk_count += 1
                choices = getattr(chunk, "choices", None)
                if choices and first_event_seconds is None:
                    first_event_seconds = elapsed

                chunk_model = getattr(chunk, "model", None)
                if isinstance(chunk_model, str) and chunk_model:
                    response_model = chunk_model

                if _first_delta_has_content(chunk) and first_content_seconds is None:
                    first_content_seconds = elapsed

                if choices:
                    delta = getattr(choices[0], "delta", None)
                    if delta is not None:
                        content = getattr(delta, "content", None)
                        reasoning = getattr(delta, "reasoning_content", None)
                        if isinstance(content, str):
                            output_chars += len(content)
                        if isinstance(reasoning, str):
                            output_chars += len(reasoning)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

        total_seconds = time.perf_counter() - started
        transport = tracker.snapshot(sample_id)
        if transport["http_attempts"] == 0:
            # Test doubles may bypass HTTP event hooks.
            transport["http_attempts"] = 1
            transport["retries"] = 0

        return {
            "sample_id": sample_id,
            "provider_stage": provider_stage,
            "model": model,
            "status": "completed",
            "stream_open_seconds": round(stream_open_seconds, 4)
            if stream_open_seconds is not None
            else None,
            "first_event_seconds": round(first_event_seconds, 4)
            if first_event_seconds is not None
            else None,
            "first_content_seconds": round(first_content_seconds, 4)
            if first_content_seconds is not None
            else None,
            "total_seconds": round(total_seconds, 4),
            "chunk_count": chunk_count,
            "output_chars": output_chars,
            "response_model": response_model,
            "configured_timeout_seconds": timeout_seconds,
            "configured_max_retries": max_retries,
            **transport,
        }
    except Exception as exc:
        total_seconds = time.perf_counter() - started
        return {
            "sample_id": sample_id,
            "provider_stage": provider_stage,
            "model": model,
            "status": "infrastructure_error",
            "stream_open_seconds": round(stream_open_seconds, 4)
            if stream_open_seconds is not None
            else None,
            "first_event_seconds": round(first_event_seconds, 4)
            if first_event_seconds is not None
            else None,
            "first_content_seconds": round(first_content_seconds, 4)
            if first_content_seconds is not None
            else None,
            "total_seconds": round(total_seconds, 4),
            "configured_timeout_seconds": timeout_seconds,
            "configured_max_retries": max_retries,
            **tracker.snapshot(sample_id),
            **safe_error_metadata(exc),
        }
    finally:
        tracker.end()


def summarize_samples(
    samples: list[dict[str, Any]],
    *,
    slo_seconds: float,
) -> dict[str, Any]:
    completed = [item for item in samples if item.get("status") == "completed"]
    errors = [item for item in samples if item.get("status") != "completed"]

    def observed(metric: str) -> list[float]:
        values: list[float] = []
        for item in completed:
            value = item.get(metric)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values

    metrics: dict[str, Any] = {}
    for metric in (
        "stream_open_seconds",
        "first_event_seconds",
        "first_content_seconds",
        "total_seconds",
    ):
        values = observed(metric)
        metrics[metric] = {
            "observed": len(values),
            "min": round(min(values), 4) if values else None,
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "max": round(max(values), 4) if values else None,
            "mean": round(sum(values) / len(values), 4) if values else None,
        }

    total_p95 = metrics["total_seconds"]["p95"]
    first_content_p95 = metrics["first_content_seconds"]["p95"]

    return {
        "planned_samples": len(samples),
        "completed_samples": len(completed),
        "infrastructure_errors": len(errors),
        "total_http_attempts": sum(
            int(item.get("http_attempts", 0))
            for item in samples
            if isinstance(item.get("http_attempts"), int)
        ),
        "total_retries": sum(
            int(item.get("retries", 0))
            for item in samples
            if isinstance(item.get("retries"), int)
        ),
        "retried_samples": sum(
            1 for item in samples if isinstance(item.get("retries"), int) and item["retries"] > 0
        ),
        "metrics": metrics,
        "slo_seconds": slo_seconds,
        "p95_first_content_within_slo": (
            first_content_p95 is not None and first_content_p95 <= slo_seconds
        ),
        "p95_total_within_slo": total_p95 is not None and total_p95 <= slo_seconds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="provider-latency-results/report.json")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=int(os.getenv("PROBE_REPETITIONS", str(DEFAULT_REPETITIONS))),
    )
    parser.add_argument(
        "--slo-seconds",
        type=float,
        default=float(os.getenv("PROBE_SLO_SECONDS", str(DEFAULT_SLO_SECONDS))),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 5 <= args.repetitions <= 30:
        raise ValueError("--repetitions must be between 5 and 30")
    if args.slo_seconds <= 0:
        raise ValueError("--slo-seconds must be positive")

    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not api_key:
        print("NVIDIA_API_KEY is required for the provider latency probe")
        return 2

    base_url = os.getenv("NVIDIA_API_BASE_URL", DEFAULT_BASE_URL)
    if base_url != DEFAULT_BASE_URL:
        raise ValueError("provider latency probe must use the approved NVIDIA trial endpoint")

    ultra_model = os.getenv("NVIDIA_MODEL", DEFAULT_ULTRA_MODEL)
    safety_model = os.getenv("NVIDIA_SAFETY_MODEL", DEFAULT_SAFETY_MODEL)
    if ultra_model != DEFAULT_ULTRA_MODEL:
        raise ValueError("unexpected NVIDIA_MODEL for latency-floor probe")
    if safety_model != DEFAULT_SAFETY_MODEL:
        raise ValueError("unexpected NVIDIA_SAFETY_MODEL for latency-floor probe")

    tracker = AttemptTracker()
    http_client = DefaultHttpxClient(
        event_hooks={
            "request": [tracker.on_request],
            "response": [tracker.on_response],
        }
    )
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        max_retries=DEFAULT_MAX_RETRIES,
        http_client=http_client,
    )

    samples: list[dict[str, Any]] = []

    def ultra_request() -> dict[str, Any]:
        return {
            "model": ultra_model,
            "messages": [{"role": "user", "content": "Reply exactly with: OK"}],
            "max_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
        }

    def safety_request() -> dict[str, Any]:
        return {
            "model": safety_model,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello."}],
                }
            ],
            "max_tokens": 16,
            "temperature": 0.01,
            "top_p": 0.95,
            "stream": True,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            },
        }

    for index in range(1, args.repetitions + 1):
        samples.append(
            run_stream_sample(
                client=client,
                tracker=tracker,
                sample_id=f"ultra-{index:02d}",
                provider_stage="main_model",
                model=ultra_model,
                request_factory=ultra_request,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
                max_retries=DEFAULT_MAX_RETRIES,
            )
        )
        samples.append(
            run_stream_sample(
                client=client,
                tracker=tracker,
                sample_id=f"safety-{index:02d}",
                provider_stage="content_safety",
                model=safety_model,
                request_factory=safety_request,
                timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
                max_retries=DEFAULT_MAX_RETRIES,
            )
        )

    ultra_samples = [item for item in samples if item["provider_stage"] == "main_model"]
    safety_samples = [item for item in samples if item["provider_stage"] == "content_safety"]

    ultra_summary = summarize_samples(ultra_samples, slo_seconds=args.slo_seconds)
    safety_summary = summarize_samples(safety_samples, slo_seconds=args.slo_seconds)

    any_errors = (
        ultra_summary["infrastructure_errors"] > 0
        or safety_summary["infrastructure_errors"] > 0
    )
    if any_errors:
        probe_status = "NOT_ADJUDICATED_INFRASTRUCTURE"
        necessary_condition_pass = False
    else:
        necessary_condition_pass = bool(
            ultra_summary["p95_total_within_slo"]
            and safety_summary["p95_total_within_slo"]
        )
        probe_status = "PASS" if necessary_condition_pass else "FAIL_SLO"

    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "provider": "nvidia_nim",
        "api_base_url": base_url,
        "ultra_model": ultra_model,
        "safety_model": safety_model,
        "repetitions_per_model": args.repetitions,
        "execution_order": "sequential_pairs_ultra_then_safety",
        "streaming": True,
        "synthetic_non_sensitive_inputs": True,
        "probe_status": probe_status,
        "target_p95_seconds": args.slo_seconds,
        "provider_floor_necessary_condition_pass": necessary_condition_pass,
        "interpretation": (
            "Necessary condition only: both provider components must satisfy the target "
            "individually before an end-to-end chatbot path can plausibly satisfy it."
        ),
        "summary": {
            "ultra": ultra_summary,
            "content_safety": safety_summary,
        },
        "samples": samples,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "probe_status": probe_status,
                "provider_floor_necessary_condition_pass": necessary_condition_pass,
                "target_p95_seconds": args.slo_seconds,
                "ultra": ultra_summary,
                "content_safety": safety_summary,
            },
            sort_keys=True,
        )
    )

    return 2 if any_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
