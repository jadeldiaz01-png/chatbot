import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import provider_ultra_sdk_vs_raw_http as experiment


def test_frozen_contract_and_order_balance() -> None:
    experiment.assert_frozen_base_contracts()
    assert experiment.REPETITIONS == 20
    assert experiment.PACING_SECONDS == 12.0
    assert experiment.REQUEST_TIMEOUT_SECONDS == 15.0
    assert experiment.MAX_RETRIES == 0
    assert experiment.SLO_SECONDS == 8.0
    assert experiment.FUNCTIONAL_CONFIGURATION_CHANGED is False
    assert experiment.PROMOTION_AUTHORIZED is False
    assert experiment.SENSITIVE_PAYLOADS_RECORDED is False

    expected = set(experiment.CONDITIONS)
    for row in experiment.ORDER_SCHEDULE:
        assert set(row) == expected

    counts = {condition: [0, 0, 0, 0] for condition in experiment.CONDITIONS}
    for round_number in range(1, 21):
        for position, condition in enumerate(experiment.order_for_round(round_number)):
            counts[condition][position] += 1
    assert all(value == [5, 5, 5, 5] for value in counts.values())


def test_report_has_four_paired_conditions_and_no_promotion() -> None:
    results = []
    for round_number in range(1, 21):
        for order_in_round, condition in enumerate(
            experiment.order_for_round(round_number), start=1
        ):
            transport, mode = condition.split("_", 1)
            item = {
                "condition": condition,
                "transport": transport,
                "mode": mode,
                "round": round_number,
                "order_in_round": order_in_round,
                "status": "completed",
                "response_headers_seconds": 0.2,
                "first_signal_seconds": 0.3 if mode == "stream" else None,
                "total_seconds": 0.5,
            }
            if condition == "sdk_stream":
                item["stream_events_observed"] = 1
            elif condition == "raw_stream":
                item.update({
                    "status_code": 200,
                    "sse_events_observed": 1,
                    "protocol_complete": True,
                })
            elif condition == "raw_nonstream":
                item.update({"status_code": 200, "protocol_complete": True})
            results.append(experiment.normalize_comparison_result(condition, item))

    report = experiment.build_report(results, complete=True)
    assert report["complete"] is True
    assert report["total_planned_calls"] == 80
    assert len(report["results"]) == 80
    assert report["functional_configuration_changed"] is False
    assert report["promotion_authorized"] is False
    assert report["sensitive_payloads_recorded"] is False
    for condition in experiment.CONDITIONS:
        assert report["summary"][condition]["observed"] == 20
        assert report["summary"][condition]["strict_slo_met"] is True
        assert report["condition_position_counts"][condition] == [5, 5, 5, 5]
    assert report["comparisons"]["sdk_minus_raw_stream_total"]["paired_completed"] == 20
    assert report["comparisons"]["sdk_minus_raw_nonstream_total"]["paired_completed"] == 20
    assert report["comparisons"]["sdk_stream_minus_nonstream_total"]["paired_completed"] == 20
    assert report["comparisons"]["raw_stream_minus_nonstream_total"]["paired_completed"] == 20


def test_comparison_completion_is_transport_neutral_for_streaming() -> None:
    sdk_empty = experiment.normalize_comparison_result(
        "sdk_stream",
        {"status": "completed", "stream_events_observed": 0},
    )
    assert sdk_empty["source_status"] == "completed"
    assert sdk_empty["comparison_status"] == "protocol_error"
    assert sdk_empty["comparison_protocol_complete"] is False

    raw_without_done = experiment.normalize_comparison_result(
        "raw_stream",
        {
            "status": "protocol_error",
            "error_type": "IncompleteSSEStream",
            "status_code": 200,
            "sse_events_observed": 1,
            "protocol_complete": False,
        },
    )
    assert raw_without_done["source_status"] == "protocol_error"
    assert raw_without_done["comparison_status"] == "completed"
    assert raw_without_done["comparison_protocol_complete"] is True


def test_frozen_contract_rejects_common_builder_drift() -> None:
    original_sdk = experiment.sdk_request_spec
    original_raw = experiment.raw_request_payload
    bad = experiment.frozen_request_payload(stream=False)
    bad["max_tokens"] = 9

    def bad_sdk_spec():
        return {
            "model": bad["model"],
            "messages": bad["messages"],
            "max_tokens": bad["max_tokens"],
            "temperature": bad["temperature"],
            "top_p": bad["top_p"],
            "extra_body": {
                "chat_template_kwargs": bad["chat_template_kwargs"],
            },
        }

    def bad_raw_payload(*, stream: bool):
        payload = dict(bad)
        payload["stream"] = stream
        return payload

    experiment.sdk_request_spec = bad_sdk_spec
    experiment.raw_request_payload = bad_raw_payload
    try:
        try:
            experiment.assert_frozen_base_contracts()
        except RuntimeError as exc:
            assert "frozen contract" in str(exc)
        else:
            raise AssertionError("common SDK/raw payload drift was not rejected")
    finally:
        experiment.sdk_request_spec = original_sdk
        experiment.raw_request_payload = original_raw


if __name__ == "__main__":
    test_frozen_contract_and_order_balance()
    test_report_has_four_paired_conditions_and_no_promotion()
    test_comparison_completion_is_transport_neutral_for_streaming()
    test_frozen_contract_rejects_common_builder_drift()
    print("PROVIDER_ULTRA_SDK_VS_RAW_HTTP_TEST=PASS")
