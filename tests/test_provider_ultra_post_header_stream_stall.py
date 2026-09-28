import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import httpx2

from provider_ultra_post_header_stream_stall import (
    FirstTransportBodyChunkStream,
    PostHeaderTracker,
    build_report,
    stage_gaps,
)


class FakeStream(httpx2.SyncByteStream):
    def __iter__(self):
        yield b""
        yield b"abc"
        yield b"def"

    def close(self):
        return None


def test_first_transport_body_chunk_is_metadata_only() -> None:
    tracker = PostHeaderTracker()
    tracker.begin("stream")
    wrapped = FirstTransportBodyChunkStream(FakeStream(), tracker)
    assert list(wrapped) == [b"", b"abc", b"def"]
    metadata = tracker.finish()
    assert isinstance(metadata["first_transport_body_chunk_seconds"], float)
    assert metadata["first_transport_body_chunk_size_bytes"] == 3
    assert "content" not in metadata


def test_stage_gaps_are_derived_only_from_timestamps() -> None:
    gaps = stage_gaps({
        "response_headers_seconds": 1.0,
        "first_transport_body_chunk_seconds": 1.5,
        "first_event_seconds": 1.7,
        "first_content_seconds": 2.0,
        "total_seconds": 2.4,
    })
    assert gaps["headers_to_first_transport_body_chunk_seconds"] == 0.5
    assert gaps["first_transport_body_chunk_to_first_event_seconds"] == 0.2
    assert gaps["first_event_to_first_content_seconds"] == 0.3
    assert gaps["first_content_to_finish_seconds"] == 0.4


def test_report_remains_fail_closed_and_non_promoting() -> None:
    report = build_report([], condition="reuse", replicate=1, complete=False)
    assert report["functional_configuration_changed"] is False
    assert report["promotion_authorized"] is False
    assert report["sensitive_payloads_recorded"] is False
    assert report["frozen_request_contract"]["max_tokens"] == 8
    assert report["frozen_request_contract"]["request_timeout_seconds"] == 15.0
    assert report["frozen_request_contract"]["max_retries"] == 0
    assert report["frozen_request_contract"]["pacing_seconds"] == 12.0
    assert report["frozen_request_contract"]["slo_seconds"] == 8.0


if __name__ == "__main__":
    test_first_transport_body_chunk_is_metadata_only()
    test_stage_gaps_are_derived_only_from_timestamps()
    test_report_remains_fail_closed_and_non_promoting()
    print("PROVIDER_POST_HEADER_STREAM_STALL_TEST=PASS")
