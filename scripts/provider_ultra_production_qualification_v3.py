from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import DefaultHttpxClient, OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import provider_ultra_production_qualification as v2
import provider_ultra_production_shaped_latency as baseline

EXPERIMENT_NAME = "provider_ultra_production_qualification_v3"
EXPERIMENT_SPEC_VERSION = "2026-10-03.3"
SYSTEM_INSTRUCTIONS_SOURCE = "evidence/provider-ultra-system-instructions-v3.txt"
SYSTEM_INSTRUCTIONS_SHA256 = "79f6e3eb01e7edaaed0fd63628a5c40fe2209211631db3bf357490ee2e7c843a"

MAX_TOKENS = 128
REPETITIONS_PER_PROMPT = 4
TOTAL_PLANNED_CALLS = len(baseline.PROMPT_CASES) * REPETITIONS_PER_PROMPT
MIN_PASS_RATE = 1.0
TARGET_P95_SECONDS = 8.0

PRODUCTION_CONFIGURATION_CHANGED = False
PRODUCTION_RUNTIME_PARAMETER_CHANGE_AUTHORIZED = False
PROMOTION_AUTHORIZED = False
SENSITIVE_PAYLOADS_RECORDED = False


def load_candidate_system_instructions(
    path: str | Path = SYSTEM_INSTRUCTIONS_SOURCE,
) -> str:
    value = Path(path).read_text(encoding="utf-8")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    if digest != SYSTEM_INSTRUCTIONS_SHA256:
        raise ValueError(
            "candidate system instructions changed from preregistered contract"
        )
    return value


def build_request(system_instructions: str, prompt: str) -> dict[str, Any]:
    request = baseline.build_request(system_instructions, prompt)
    request["max_tokens"] = MAX_TOKENS
    return request


def run_call(
    *,
    client: OpenAI,
    tracker: baseline.AttemptTracker,
    sample_id: str,
    prompt_id: str,
    prompt: str,
    system_instructions: str,
) -> dict[str, Any]:
    request = build_request(system_instructions, prompt)
    started = time.perf_counter()
    tracker.begin(sample_id, sample_started_at=started)
    try:
        completion = client.chat.completions.create(**request)
        completed_at = time.perf_counter()
        transport = tracker.snapshot(sample_id, call_completed_at=completed_at)
        if transport["http_attempts"] == 0:
            transport["http_attempts"] = 1
            transport["retries"] = 0

        choices = getattr(completion, "choices", None)
        message = getattr(choices[0], "message", None) if choices else None
        content = getattr(message, "content", "") if message is not None else ""
        if not isinstance(content, str):
            content = ""

        usage = getattr(completion, "usage", None)
        output_tokens = (
            getattr(usage, "completion_tokens", None)
            if usage is not None
            else None
        )
        total_tokens = (
            getattr(usage, "total_tokens", None)
            if usage is not None
            else None
        )
        quality = v2.evaluate_response(prompt_id, content)

        return {
            "sample_id": sample_id,
            "prompt_id": prompt_id,
            "prompt_sha256": baseline.prompt_digest(prompt),
            "status": "completed",
            "total_seconds": round(completed_at - started, 4),
            "output_chars": len(content),
            "output_tokens": output_tokens if isinstance(output_tokens, int) else None,
            "total_tokens": total_tokens if isinstance(total_tokens, int) else None,
            "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "quality_passed": quality["passed"],
            "quality_checks": quality["checks"],
            "response_model": getattr(completion, "model", None),
            **transport,
        }
    except Exception as exc:
        failed_at = time.perf_counter()
        return {
            "sample_id": sample_id,
            "prompt_id": prompt_id,
            "prompt_sha256": baseline.prompt_digest(prompt),
            "status": "infrastructure_error",
            "total_seconds": round(failed_at - started, 4),
            "quality_passed": False,
            "quality_checks": {"infrastructure_error": False},
            **tracker.snapshot(sample_id, call_completed_at=failed_at),
            **baseline.safe_error_metadata(exc),
        }
    finally:
        tracker.end()


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [item for item in results if item.get("status") == "completed"]
    errors = [item for item in results if item.get("status") != "completed"]
    latencies = [
        float(item["total_seconds"])
        for item in completed
        if isinstance(item.get("total_seconds"), (int, float))
    ]
    passed = [item for item in completed if item.get("quality_passed") is True]

    p95 = baseline.percentile(latencies, 0.95)
    pass_rate = round(len(passed) / len(completed), 4) if completed else 0.0
    per_case: dict[str, dict[str, int]] = {}
    for prompt_id, _ in baseline.PROMPT_CASES:
        cohort = [item for item in completed if item.get("prompt_id") == prompt_id]
        per_case[prompt_id] = {
            "observed": len(cohort),
            "passed": sum(item.get("quality_passed") is True for item in cohort),
            "failed": sum(item.get("quality_passed") is not True for item in cohort),
        }

    return {
        "planned": TOTAL_PLANNED_CALLS,
        "observed": len(results),
        "completed": len(completed),
        "errors": len(errors),
        "quality_passed": len(passed),
        "quality_failed": len(completed) - len(passed),
        "quality_pass_rate": pass_rate,
        "per_case": per_case,
        "p50_total_seconds": baseline.percentile(latencies, 0.50),
        "p95_total_seconds": p95,
        "max_total_seconds": round(max(latencies), 4) if latencies else None,
        "calls_over_slo": sum(value > TARGET_P95_SECONDS for value in latencies),
        "total_retries": sum(
            int(item.get("retries", 0))
            for item in results
            if isinstance(item.get("retries"), int)
        ),
        "quality_gate_pass": (
            len(results) == TOTAL_PLANNED_CALLS
            and not errors
            and pass_rate >= MIN_PASS_RATE
            and all(
                data["observed"] == REPETITIONS_PER_PROMPT and data["failed"] == 0
                for data in per_case.values()
            )
        ),
        "latency_gate_pass": (
            len(results) == TOTAL_PLANNED_CALLS
            and not errors
            and p95 is not None
            and p95 <= TARGET_P95_SECONDS
        ),
    }


def build_report(
    *,
    results: list[dict[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "created_at": created_at,
        "updated_at": datetime.now(UTC).isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "experiment": EXPERIMENT_NAME,
        "experiment_spec_version": EXPERIMENT_SPEC_VERSION,
        "provider": "nvidia_nim",
        "api_base_url": baseline.DEFAULT_BASE_URL,
        "model": baseline.MODEL,
        "system_instructions_source": SYSTEM_INSTRUCTIONS_SOURCE,
        "system_instructions_sha256": SYSTEM_INSTRUCTIONS_SHA256,
        "baseline_system_instructions_sha256": baseline.SYSTEM_INSTRUCTIONS_SHA256,
        "prompt_cases": [
            {"id": prompt_id, "sha256": baseline.prompt_digest(prompt)}
            for prompt_id, prompt in baseline.PROMPT_CASES
        ],
        "repetitions_per_prompt": REPETITIONS_PER_PROMPT,
        "total_planned_calls": TOTAL_PLANNED_CALLS,
        "request_contract": {
            "max_tokens": MAX_TOKENS,
            "temperature": baseline.TEMPERATURE,
            "top_p": baseline.TOP_P,
            "enable_thinking": baseline.ENABLE_THINKING,
            "stream": baseline.STREAM,
            "timeout_seconds": baseline.REQUEST_TIMEOUT_SECONDS,
            "max_retries": baseline.MAX_RETRIES,
            "pacing_seconds": baseline.PACING_SECONDS,
        },
        "target_p95_seconds": TARGET_P95_SECONDS,
        "min_quality_pass_rate": MIN_PASS_RATE,
        "production_configuration_changed": PRODUCTION_CONFIGURATION_CHANGED,
        "production_runtime_parameter_change_authorized": (
            PRODUCTION_RUNTIME_PARAMETER_CHANGE_AUTHORIZED
        ),
        "promotion_authorized": PROMOTION_AUTHORIZED,
        "sensitive_payloads_recorded": SENSITIVE_PAYLOADS_RECORDED,
        "complete": len(results) == TOTAL_PLANNED_CALLS,
        "summary": summarize(results),
        "results": list(results),
    }


def persist_report(
    output: str | Path,
    *,
    results: list[dict[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(results=results, created_at=created_at)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return report


def validate_report(
    report: dict[str, Any],
    *,
    expected_sha: str | None = None,
) -> None:
    assert report["schema_version"] == "1.0"
    if expected_sha is not None:
        assert report["git_sha"] == expected_sha
    assert report["experiment"] == EXPERIMENT_NAME
    assert report["experiment_spec_version"] == EXPERIMENT_SPEC_VERSION
    assert report["provider"] == "nvidia_nim"
    assert report["model"] == baseline.MODEL
    assert report["system_instructions_source"] == SYSTEM_INSTRUCTIONS_SOURCE
    assert report["system_instructions_sha256"] == SYSTEM_INSTRUCTIONS_SHA256
    assert report["baseline_system_instructions_sha256"] == baseline.SYSTEM_INSTRUCTIONS_SHA256
    assert report["request_contract"] == {
        "max_tokens": 128,
        "temperature": 1.0,
        "top_p": 0.95,
        "enable_thinking": False,
        "stream": False,
        "timeout_seconds": 45.0,
        "max_retries": 2,
        "pacing_seconds": 12.0,
    }
    assert report["target_p95_seconds"] == 8.0
    assert report["min_quality_pass_rate"] == 1.0
    assert report["production_configuration_changed"] is False
    assert report["production_runtime_parameter_change_authorized"] is False
    assert report["promotion_authorized"] is False
    assert report["sensitive_payloads_recorded"] is False

    results = report["results"]
    assert isinstance(results, list)
    assert 0 <= len(results) <= TOTAL_PLANNED_CALLS
    assert report["complete"] is (len(results) == TOTAL_PLANNED_CALLS)

    forbidden = {
        "prompt",
        "prompt_text",
        "response",
        "response_text",
        "response_body",
        "response_content",
        "system_instructions",
        "authorization",
        "api_key",
    }
    expected_ids = {
        f"{prompt_id}-r{repetition}"
        for repetition in range(1, REPETITIONS_PER_PROMPT + 1)
        for prompt_id, _ in baseline.PROMPT_CASES
    }
    prompt_hashes = {
        prompt_id: baseline.prompt_digest(prompt)
        for prompt_id, prompt in baseline.PROMPT_CASES
    }
    seen: set[str] = set()
    for item in results:
        assert forbidden.isdisjoint(item.keys())
        sample_id = item["sample_id"]
        assert sample_id in expected_ids
        assert sample_id not in seen
        seen.add(sample_id)
        assert item["prompt_sha256"] == prompt_hashes[item["prompt_id"]]
        assert item["status"] in {"completed", "infrastructure_error"}
        assert isinstance(item["quality_passed"], bool)
        assert isinstance(item["quality_checks"], dict)
        if item["status"] == "completed":
            assert len(item["response_sha256"]) == 64
            assert isinstance(item["output_chars"], int)

    summary = report["summary"]
    assert summary["planned"] == TOTAL_PLANNED_CALLS
    assert summary["observed"] == len(results)
    assert summary["completed"] + summary["errors"] == len(results)
    assert set(summary["per_case"]) == {prompt_id for prompt_id, _ in baseline.PROMPT_CASES}

    if report["complete"]:
        assert seen == expected_ids


def main() -> int:
    output = os.getenv(
        "QUALIFICATION_V3_REPORT_PATH",
        "provider-ultra-production-qualification-v3/report.json",
    )
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not api_key:
        print("NVIDIA_API_KEY is required for production qualification v3")
        return 2

    base_url = os.getenv("NVIDIA_API_BASE_URL", baseline.DEFAULT_BASE_URL).rstrip("/")
    if base_url != baseline.DEFAULT_BASE_URL:
        raise ValueError("unexpected NVIDIA_API_BASE_URL")

    system_instructions = load_candidate_system_instructions()
    tracker = baseline.AttemptTracker()
    http_client = DefaultHttpxClient(
        event_hooks={
            "request": [tracker.on_request],
            "response": [tracker.on_response],
        }
    )
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=baseline.REQUEST_TIMEOUT_SECONDS,
        max_retries=baseline.MAX_RETRIES,
        http_client=http_client,
    )

    results: list[dict[str, Any]] = []
    created_at = datetime.now(UTC).isoformat()
    report = persist_report(output, results=results, created_at=created_at)
    try:
        for repetition in range(1, REPETITIONS_PER_PROMPT + 1):
            for prompt_id, prompt in baseline.PROMPT_CASES:
                sample_id = f"{prompt_id}-r{repetition}"
                results.append(
                    run_call(
                        client=client,
                        tracker=tracker,
                        sample_id=sample_id,
                        prompt_id=prompt_id,
                        prompt=prompt,
                        system_instructions=system_instructions,
                    )
                )
                report = persist_report(
                    output,
                    results=results,
                    created_at=created_at,
                )
                if len(results) < TOTAL_PLANNED_CALLS:
                    time.sleep(baseline.PACING_SECONDS)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    validate_report(report)
    summary = report["summary"]
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["quality_gate_pass"] and summary["latency_gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
