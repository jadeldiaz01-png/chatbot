import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from provider_ultra_post_header_first_byte_timing import (
    OBSERVATION_BOUNDARY,
    build_report,
    first_byte_observation,
)


def test_first_byte_boundary_completed() -> None:
    observation = first_byte_observation({
        "status": "completed",
        "response_headers_seconds": 1.0,
        "first_transport_body_chunk_seconds": 1.25,
    })
    assert observation["headers_observed"] is True
    assert observation["first_body_byte_observed"] is True
    assert observation["headers_to_first_body_byte_seconds"] == 0.25
    assert observation["post_header_first_byte_timeout"] is False


def test_post_header_timeout_is_typed_without_payload() -> None:
    observation = first_byte_observation({
        "status": "infrastructure_error",
        "response_headers_seconds": 1.0,
        "first_transport_body_chunk_seconds": None,
        "error_type": "ReadTimeout",
        "error_phase": "iterate",
    })
    assert observation["post_header_first_byte_timeout"] is True
    assert observation["error_type"] == "ReadTimeout"
    assert observation["error_phase"] == "iterate"


def test_report_is_non_promoting_and_configuration_frozen() -> None:
    report = build_report([], condition="reuse", replicate=1, complete=False)
    assert report["observation_boundary"] == OBSERVATION_BOUNDARY
    assert report["functional_configuration_changed"] is False
    assert report["promotion_authorized"] is False
    assert report["sensitive_payloads_recorded"] is False
    frozen = report["frozen_request_contract"]
    assert frozen["max_tokens"] == 8
    assert frozen["temperature"] == 0.0
    assert frozen["top_p"] == 1.0
    assert frozen["enable_thinking"] is False
    assert frozen["request_timeout_seconds"] == 15.0
    assert frozen["max_retries"] == 0
    assert frozen["pacing_seconds"] == 12.0
    assert frozen["slo_seconds"] == 8.0


if __name__ == "__main__":
    test_first_byte_boundary_completed()
    test_post_header_timeout_is_typed_without_payload()
    test_report_is_non_promoting_and_configuration_frozen()
    print("PROVIDER_POST_HEADER_FIRST_BYTE_TIMING_TEST=PASS")
