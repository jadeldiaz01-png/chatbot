from __future__ import annotations

import json
import time
from typing import Any, Iterable

import httpx2

from provider_ultra_post_header_stream_stall import (
    CONDITIONS,
    EXPERIMENT_SPEC_VERSION as PARENT_SPEC_VERSION,
    FirstTransportBodyChunkStream,
    PostHeaderTracker,
    build_report as build_parent_report,
    stage_gaps,
)

EXPERIMENT_NAME = "provider_ultra_post_header_first_byte_timing"
EXPERIMENT_SPEC_VERSION = "2026-09-28.1"
OBSERVATION_BOUNDARY = "response_headers_to_first_nonempty_transport_body_chunk"
FUNCTIONAL_CONFIGURATION_CHANGED = False
PROMOTION_AUTHORIZED = False
SENSITIVE_PAYLOADS_RECORDED = False
TIMEOUT_ERROR_TYPES = frozenset({
    "APITimeoutError",
    "ConnectTimeout",
    "PoolTimeout",
    "ReadTimeout",
    "TimeoutException",
    "WriteTimeout",
})


class FirstByteTimingStream(FirstTransportBodyChunkStream):
    """Observe timing metadata only; never retain transport payload bytes."""

    def __init__(self, inner: Iterable[bytes], tracker: PostHeaderTracker) -> None:
        super().__init__(inner, tracker)
        self._first_nonempty_chunk_observed = False

    def __iter__(self):
        for chunk in self._inner:
            if chunk and not self._first_nonempty_chunk_observed:
                self._tracker.note_transport_body_chunk(len(chunk))
                self._first_nonempty_chunk_observed = True
            yield chunk


def first_byte_observation(item: dict[str, Any]) -> dict[str, Any]:
    headers = item.get("response_headers_seconds")
    first_chunk = item.get("first_transport_body_chunk_seconds")
    status = item.get("status")
    error_type = item.get("error_type")
    return {
        "headers_observed": isinstance(headers, (int, float)),
        "first_body_byte_observed": isinstance(first_chunk, (int, float)),
        "headers_to_first_body_byte_seconds": (
            round(float(first_chunk) - float(headers), 4)
            if isinstance(headers, (int, float))
            and isinstance(first_chunk, (int, float))
            else None
        ),
        "post_header_first_byte_timeout": (
            status != "completed"
            and isinstance(headers, (int, float))
            and first_chunk is None
            and error_type in TIMEOUT_ERROR_TYPES
        ),
        "error_type": error_type,
        "error_phase": item.get("error_phase"),
    }


def enrich_result(item: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(item)
    enriched.update(stage_gaps(enriched))
    enriched["first_byte_observation"] = first_byte_observation(enriched)
    return enriched


def build_report(
    results: list[dict[str, Any]], *, condition: str, replicate: int, complete: bool
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown diagnostic condition: {condition}")
    report = build_parent_report(
        [enrich_result(item) for item in results],
        condition=condition,
        replicate=replicate,
        complete=complete,
    )
    report["experiment"] = EXPERIMENT_NAME
    report["experiment_spec_version"] = EXPERIMENT_SPEC_VERSION
    report["parent_instrumentation_spec_version"] = PARENT_SPEC_VERSION
    report["observation_boundary"] = OBSERVATION_BOUNDARY
    report["functional_configuration_changed"] = FUNCTIONAL_CONFIGURATION_CHANGED
    report["promotion_authorized"] = PROMOTION_AUTHORIZED
    report["sensitive_payloads_recorded"] = SENSITIVE_PAYLOADS_RECORDED
    observations = [item["first_byte_observation"] for item in report["results"]]
    report["first_byte_summary"] = {
        "observed": len(observations),
        "headers_observed": sum(bool(item["headers_observed"]) for item in observations),
        "first_body_byte_observed": sum(
            bool(item["first_body_byte_observed"]) for item in observations
        ),
        "post_header_first_byte_timeouts": sum(
            bool(item["post_header_first_byte_timeout"]) for item in observations
        ),
    }
    return report


def contract_snapshot() -> str:
    """Offline CI-readable snapshot proving this probe is metadata-only/non-promoting."""
    return json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "observation_boundary": OBSERVATION_BOUNDARY,
            "functional_configuration_changed": FUNCTIONAL_CONFIGURATION_CHANGED,
            "promotion_authorized": PROMOTION_AUTHORIZED,
            "sensitive_payloads_recorded": SENSITIVE_PAYLOADS_RECORDED,
            "conditions": sorted(CONDITIONS),
        },
        sort_keys=True,
    )
