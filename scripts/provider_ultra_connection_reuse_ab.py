from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx2
from openai import DefaultHttpxClient, OpenAI

from provider_ultra_transport_diagnostic import (
    API_BASE_URL,
    MAX_RETRIES,
    MODES,
    PACING_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    RequestTracker,
    build_report,
    run_nonstream_call,
    run_stream_call,
)

EXPERIMENT_SPEC_VERSION = "2026-09-26.1"
EXPERIMENT_NAME = "provider_ultra_connection_reuse_ab"
REPETITIONS_PER_MODE = 20
SLO_SECONDS = 8.0
MAX_CONNECTIONS = 1000
KEEPALIVE_EXPIRY_SECONDS = 30.0
CONDITIONS = {
    "reuse": 100,
    "no_keepalive": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=tuple(CONDITIONS), required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def build_experiment_report(
    results: list[dict[str, Any]],
    *,
    condition: str,
    replicate: int,
    complete: bool,
) -> dict[str, Any]:
    report = build_report(
        results,
        repetitions=REPETITIONS_PER_MODE,
        slo_seconds=SLO_SECONDS,
        complete=complete,
    )
    report.update(
        {
            "experiment": EXPERIMENT_NAME,
            "experiment_spec_version": EXPERIMENT_SPEC_VERSION,
            "connection_reuse_condition": condition,
            "replicate": replicate,
            "http_limits": {
                "max_connections": MAX_CONNECTIONS,
                "max_keepalive_connections": CONDITIONS[condition],
                "keepalive_expiry_seconds": KEEPALIVE_EXPIRY_SECONDS,
            },
            "functional_configuration_changed": False,
            "promotion_authorized": False,
        }
    )
    return report


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if not 1 <= args.replicate <= 5:
        raise ValueError("--replicate must be between 1 and 5")

    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        print("NVIDIA_API_KEY is required for the connection-reuse A/B experiment")
        return 2

    max_keepalive_connections = CONDITIONS[args.condition]
    tracker = RequestTracker()
    http_client = DefaultHttpxClient(
        limits=httpx2.Limits(
            max_connections=MAX_CONNECTIONS,
            max_keepalive_connections=max_keepalive_connections,
            keepalive_expiry=KEEPALIVE_EXPIRY_SECONDS,
        ),
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
        build_experiment_report(
            results,
            condition=args.condition,
            replicate=args.replicate,
            complete=False,
        ),
    )

    call_index = 0
    for round_number in range(1, REPETITIONS_PER_MODE + 1):
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
            result["connection_reuse_condition"] = args.condition
            result["replicate"] = args.replicate
            results.append(result)
            call_index += 1

            write_report(
                output_path,
                build_experiment_report(
                    results,
                    condition=args.condition,
                    replicate=args.replicate,
                    complete=False,
                ),
            )

    report = build_experiment_report(
        results,
        condition=args.condition,
        replicate=args.replicate,
        complete=True,
    )
    write_report(output_path, report)

    concise = {
        "experiment": EXPERIMENT_NAME,
        "condition": args.condition,
        "replicate": args.replicate,
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
    }
    print(json.dumps(concise, sort_keys=True))

    if any(report["summary"][mode]["errors"] for mode in MODES):
        return 2
    if any(
        not report["summary"][mode]["strict_slo_met"]
        for mode in MODES
    ):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
