from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx2

from provider_ultra_raw_http_stream_vs_nonstream import (
    API_BASE_URL,
    EXECUTION_MODE,
    FUNCTIONAL_CONFIGURATION_CHANGED,
    MAX_RETRIES,
    MODES,
    PACING_SECONDS,
    PROMOTION_AUTHORIZED,
    REPETITIONS,
    REQUEST_PATH,
    REQUEST_TIMEOUT_SECONDS,
    SDK_BYPASSED,
    SENSITIVE_PAYLOADS_RECORDED,
    SLO_SECONDS,
    ULTRA_MODEL,
    build_report,
    run_raw_call,
)

EXPERIMENT_NAME = "provider_ultra_raw_http_stream_vs_nonstream_confirmatory"
EXPERIMENT_SPEC_VERSION = "2026-09-30.1"
BLOCKS = 3
BLOCK_PAUSE_SECONDS = PACING_SECONDS


def aggregate_blocks(blocks: list[dict[str, Any]], *, complete: bool) -> dict[str, Any]:
    all_results = [
        {**item, "block": block["block"]}
        for block in blocks
        for item in block["report"]["results"]
    ]
    aggregate = build_report(
        all_results,
        repetitions=BLOCKS * REPETITIONS,
        slo_seconds=SLO_SECONDS,
        complete=complete,
    )
    aggregate["experiment"] = EXPERIMENT_NAME
    aggregate["experiment_spec_version"] = EXPERIMENT_SPEC_VERSION
    aggregate["blocks_planned"] = BLOCKS
    aggregate["blocks_observed"] = len(blocks)
    aggregate["repetitions_per_mode_per_block"] = REPETITIONS
    aggregate["block_pause_seconds"] = BLOCK_PAUSE_SECONDS
    aggregate["block_summaries"] = [
        {
            "block": block["block"],
            "summary": block["report"]["summary"],
            "comparison": block["report"]["comparison"],
        }
        for block in blocks
    ]
    return aggregate


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_block(client: httpx2.Client, *, api_key: str, block_number: int) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    call_index = 0
    for round_number in range(1, REPETITIONS + 1):
        order = MODES if round_number % 2 else tuple(reversed(MODES))
        for order_in_round, mode in enumerate(order, start=1):
            pacing_seconds_before_call = 0.0
            if call_index:
                started = time.perf_counter()
                time.sleep(PACING_SECONDS)
                pacing_seconds_before_call = round(time.perf_counter() - started, 4)
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

    report = build_report(
        results,
        repetitions=REPETITIONS,
        slo_seconds=SLO_SECONDS,
        complete=True,
    )
    report["experiment"] = EXPERIMENT_NAME
    report["experiment_spec_version"] = EXPERIMENT_SPEC_VERSION
    report["block"] = block_number
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="provider-ultra-raw-http-confirmatory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        print("NVIDIA_API_KEY is required for confirmatory replication")
        return 2

    output_dir = Path(args.output_dir)
    transport = httpx2.HTTPTransport(retries=MAX_RETRIES)
    client = httpx2.Client(transport=transport, timeout=REQUEST_TIMEOUT_SECONDS)
    blocks: list[dict[str, Any]] = []

    try:
        for block_number in range(1, BLOCKS + 1):
            if blocks:
                time.sleep(BLOCK_PAUSE_SECONDS)
            report = run_block(client, api_key=api_key, block_number=block_number)
            write_json(output_dir / f"block-{block_number}.json", report)
            blocks.append({"block": block_number, "report": report})
            write_json(
                output_dir / "aggregate.json",
                aggregate_blocks(blocks, complete=False),
            )
    finally:
        client.close()

    aggregate = aggregate_blocks(blocks, complete=len(blocks) == BLOCKS)
    write_json(output_dir / "aggregate.json", aggregate)

    if any(
        block["report"]["summary"][mode]["errors"]
        for block in blocks
        for mode in MODES
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
