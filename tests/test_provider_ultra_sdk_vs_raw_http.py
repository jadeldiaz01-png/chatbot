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
            results.append({
                "condition": condition,
                "transport": transport,
                "mode": mode,
                "round": round_number,
                "order_in_round": order_in_round,
                "status": "completed",
                "response_headers_seconds": 0.2,
                "first_signal_seconds": 0.3 if mode == "stream" else None,
                "total_seconds": 0.5,
            })

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


if __name__ == "__main__":
    test_frozen_contract_and_order_balance()
    test_report_has_four_paired_conditions_and_no_promotion()
    print("PROVIDER_ULTRA_SDK_VS_RAW_HTTP_TEST=PASS")
