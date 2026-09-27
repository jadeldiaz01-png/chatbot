import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "provider_ultra_variance_diagnostic.py"
spec = importlib.util.spec_from_file_location("provider_ultra_variance_diagnostic", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
analyze_report = module.analyze_report


def test_analyze_report_is_diagnostic_only() -> None:
    source_sha = "f0ae7cf8e6605b88c079fe7ca8c7c1a283dff549"
    report = {
        "complete": True,
        "commit_sha": source_sha,
        "connection_reuse_condition": "reuse",
        "replicate": 1,
        "results": [{
            "round": 1,
            "order_in_round": 1,
            "mode": "stream",
            "status": "infrastructure_error",
            "response_headers_seconds": 1.2,
            "total_seconds": 15.0,
            "http_version": "HTTP/1.1",
            "transport_bytes_downloaded": 0,
            "error_type": "APITimeoutError",
            "error_phase": "iterate",
            "status_codes": [],
        }],
    }
    result = analyze_report(report)
    assert result["source_complete"] is True
    assert result["source_commit_sha"] == source_sha
    assert result["functional_configuration_changed"] is False
    assert result["promotion_authorized"] is False
    assert result["sensitive_payloads_recorded"] is False
    assert result["typed_error_classes"] == {"APITimeoutError:iterate": 1}
    assert result["temporal_calls"][0]["round"] == 1


if __name__ == "__main__":
    test_analyze_report_is_diagnostic_only()
    print("PROVIDER_VARIANCE_DIAGNOSTIC_TEST=PASS")
