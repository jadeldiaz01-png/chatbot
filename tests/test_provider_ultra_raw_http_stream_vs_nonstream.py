import json
import sys
from pathlib import Path

import httpx2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from provider_ultra_raw_http_stream_vs_nonstream import (
    EXECUTION_MODE,
    FUNCTIONAL_CONFIGURATION_CHANGED,
    MAX_RETRIES,
    PACING_SECONDS,
    PROMOTION_AUTHORIZED,
    RAW_CHAT_COMPLETIONS_URL,
    REQUEST_PATH,
    REQUEST_TIMEOUT_SECONDS,
    SDK_BYPASSED,
    SENSITIVE_PAYLOADS_RECORDED,
    SLO_SECONDS,
    ULTRA_MODEL,
    build_report,
    raw_request_payload,
    run_raw_call,
)


def test_raw_payload_differs_only_by_stream_flag() -> None:
    stream = raw_request_payload(stream=True)
    nonstream = raw_request_payload(stream=False)

    assert stream["model"] == ULTRA_MODEL
    assert stream["messages"] == [{"role": "user", "content": "Reply with exactly: OK"}]
    assert stream["max_tokens"] == 8
    assert stream["temperature"] == 0.0
    assert stream["top_p"] == 1.0
    assert stream["chat_template_kwargs"] == {"enable_thinking": False}

    stream_without_flag = dict(stream)
    nonstream_without_flag = dict(nonstream)
    assert stream_without_flag.pop("stream") is True
    assert nonstream_without_flag.pop("stream") is False
    assert stream_without_flag == nonstream_without_flag


def test_raw_http_stream_and_nonstream_use_mock_transport_only() -> None:
    seen_payloads = []

    class StaticStream(httpx2.SyncByteStream):
        def __init__(self, body: bytes) -> None:
            self.body = body

        def __iter__(self):
            yield self.body

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == RAW_CHAT_COMPLETIONS_URL
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json.loads(request.read())
        seen_payloads.append(payload)
        if payload["stream"]:
            return httpx2.Response(
                200,
                request=request,
                headers={
                    "content-type": "text/event-stream",
                    "x-request-id": "stream-test",
                },
                stream=StaticStream(b'data: {"choices":[]}\n\ndata: [DONE]\n\n'),
            )
        return httpx2.Response(
            200,
            request=request,
            headers={
                "content-type": "application/json",
                "x-request-id": "nonstream-test",
            },
            stream=StaticStream(b'{"choices":[]}'),
        )

    client = httpx2.Client(
        transport=httpx2.MockTransport(handler),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    try:
        stream = run_raw_call(
            client,
            api_key="test-key",
            mode="stream",
            round_number=1,
            order_in_round=1,
        )
        nonstream = run_raw_call(
            client,
            api_key="test-key",
            mode="nonstream",
            round_number=1,
            order_in_round=2,
        )
    finally:
        client.close()

    assert stream["status"] == "completed"
    assert stream["status_code"] == 200
    assert stream["first_body_byte_seconds"] is not None
    assert stream["sse_events_observed"] == 2
    assert stream["sse_done_observed"] is True
    assert stream["request_id"] == "stream-test"

    assert nonstream["status"] == "completed"
    assert nonstream["status_code"] == 200
    assert nonstream["first_body_byte_seconds"] is not None
    assert nonstream["sse_events_observed"] is None
    assert nonstream["request_id"] == "nonstream-test"

    assert len(seen_payloads) == 2
    left = dict(seen_payloads[0])
    right = dict(seen_payloads[1])
    assert left.pop("stream") is True
    assert right.pop("stream") is False
    assert left == right


def test_report_preserves_frozen_contract_and_no_promotion() -> None:
    report = build_report(
        [],
        repetitions=20,
        slo_seconds=SLO_SECONDS,
        complete=False,
    )
    frozen = report["frozen_request_contract"]

    assert report["request_path"] == REQUEST_PATH
    assert report["sdk_bypassed"] is SDK_BYPASSED is True
    assert report["execution_mode"] == EXECUTION_MODE
    assert report["functional_configuration_changed"] is FUNCTIONAL_CONFIGURATION_CHANGED is False
    assert report["promotion_authorized"] is PROMOTION_AUTHORIZED is False
    assert report["sensitive_payloads_recorded"] is SENSITIVE_PAYLOADS_RECORDED is False

    assert frozen["model"] == ULTRA_MODEL
    assert frozen["messages"] == [{"role": "user", "content": "Reply with exactly: OK"}]
    assert frozen["max_tokens"] == 8
    assert frozen["temperature"] == 0.0
    assert frozen["top_p"] == 1.0
    assert frozen["enable_thinking"] is False
    assert frozen["request_timeout_seconds"] == REQUEST_TIMEOUT_SECONDS == 15.0
    assert frozen["max_retries"] == MAX_RETRIES == 0
    assert frozen["pacing_seconds"] == PACING_SECONDS == 12.0
    assert frozen["slo_seconds"] == SLO_SECONDS == 8.0


if __name__ == "__main__":
    test_raw_payload_differs_only_by_stream_flag()
    test_raw_http_stream_and_nonstream_use_mock_transport_only()
    test_report_preserves_frozen_contract_and_no_promotion()
    print("PROVIDER_ULTRA_RAW_HTTP_STREAM_VS_NONSTREAM_TEST=PASS")
