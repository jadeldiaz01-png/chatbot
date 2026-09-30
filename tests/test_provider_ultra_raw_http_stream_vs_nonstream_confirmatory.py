from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

import provider_ultra_raw_http_stream_vs_nonstream_confirmatory as confirm
from provider_ultra_raw_http_stream_vs_nonstream import (
    API_BASE_URL,
    EXECUTION_MODE,
    FUNCTIONAL_CONFIGURATION_CHANGED,
    MAX_RETRIES,
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
)


def synthetic_block(block: int) -> dict:
    results = []
    for round_number in range(1, REPETITIONS + 1):
        for order_in_round, mode in enumerate(
            ("stream", "nonstream") if round_number % 2 else ("nonstream", "stream"),
            start=1,
        ):
            results.append(
                {
                    "mode": mode,
                    "round": round_number,
                    "order_in_round": order_in_round,
                    "status": "completed",
                    "error_type": None,
                    "error_phase": None,
                    "protocol_complete": True,
                    "status_code": 200,
                    "response_headers_seconds": 0.2,
                    "first_body_byte_seconds": 0.3,
                    "headers_to_first_body_byte_seconds": 0.1,
                    "first_sse_event_seconds": 0.4 if mode == "stream" else None,
                    "total_seconds": 0.5,
                }
            )
    report = build_report(
        results,
        repetitions=REPETITIONS,
        slo_seconds=SLO_SECONDS,
        complete=True,
    )
    report["block"] = block
    return {"block": block, "report": report}


def test_frozen_contract() -> None:
    assert confirm.BLOCKS == 3
    assert confirm.BLOCK_PAUSE_SECONDS == PACING_SECONDS == 12.0
    assert REPETITIONS == 20
    assert API_BASE_URL == "https://integrate.api.nvidia.com/v1"
    assert ULTRA_MODEL == "nvidia/nemotron-3-ultra-550b-a55b"
    assert REQUEST_TIMEOUT_SECONDS == 15.0
    assert MAX_RETRIES == 0
    assert SLO_SECONDS == 8.0
    assert EXECUTION_MODE == "paired_sequential_alternating_order_paced"
    assert REQUEST_PATH == "raw_httpx2_no_openai_sdk"
    assert SDK_BYPASSED is True
    assert FUNCTIONAL_CONFIGURATION_CHANGED is False
    assert PROMOTION_AUTHORIZED is False
    assert SENSITIVE_PAYLOADS_RECORDED is False


def test_separate_and_aggregate_evidence() -> None:
    blocks = [synthetic_block(i) for i in range(1, confirm.BLOCKS + 1)]
    aggregate = confirm.aggregate_blocks(blocks, complete=True)
    assert aggregate["complete"] is True
    assert aggregate["blocks_planned"] == 3
    assert aggregate["blocks_observed"] == 3
    assert aggregate["repetitions_per_mode_per_block"] == 20
    assert len(aggregate["block_summaries"]) == 3
    assert len(aggregate["results"]) == 120
    assert sum(x["mode"] == "stream" for x in aggregate["results"]) == 60
    assert sum(x["mode"] == "nonstream" for x in aggregate["results"]) == 60
    assert {x["block"] for x in aggregate["results"]} == {1, 2, 3}


if __name__ == "__main__":
    test_frozen_contract()
    test_separate_and_aggregate_evidence()
    print("PROVIDER_ULTRA_RAW_HTTP_CONFIRMATORY_TEST=PASS")
