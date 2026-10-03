from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import sys

from openai import DefaultHttpxClient, OpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import provider_ultra_production_shaped_latency as baseline

EXPERIMENT_NAME = "provider_ultra_output_budget_latency"
EXPERIMENT_SPEC_VERSION = "2026-10-02.1"
ATTRIBUTION_SCHEMA_VERSION = baseline.ATTRIBUTION_SCHEMA_VERSION

DEFAULT_BASE_URL = baseline.DEFAULT_BASE_URL
MODEL = baseline.MODEL
SYSTEM_INSTRUCTIONS_SOURCE = baseline.SYSTEM_INSTRUCTIONS_SOURCE
SYSTEM_INSTRUCTIONS_SHA256 = baseline.SYSTEM_INSTRUCTIONS_SHA256
TEMPERATURE = baseline.TEMPERATURE
TOP_P = baseline.TOP_P
ENABLE_THINKING = baseline.ENABLE_THINKING
REQUEST_TIMEOUT_SECONDS = baseline.REQUEST_TIMEOUT_SECONDS
MAX_RETRIES = baseline.MAX_RETRIES
PACING_SECONDS = baseline.PACING_SECONDS
SLO_SECONDS = baseline.SLO_SECONDS
REPETITIONS_PER_PROMPT = baseline.REPETITIONS_PER_PROMPT
STREAM = baseline.STREAM
PROMPT_CASES = baseline.PROMPT_CASES

OUTPUT_BUDGETS: tuple[int, ...] = (128, 256, 512)
EXPERIMENTAL_PARAMETER = "max_tokens"
CALLS_PER_BUDGET = len(PROMPT_CASES) * REPETITIONS_PER_PROMPT
TOTAL_PLANNED_CALLS = CALLS_PER_BUDGET * len(OUTPUT_BUDGETS)

PRODUCTION_CONFIGURATION_CHANGED = False
PRODUCTION_RUNTIME_PARAMETER_CHANGE_AUTHORIZED = False
PROMOTION_AUTHORIZED = False
QUALITY_EVALUATION = False
SENSITIVE_PAYLOADS_RECORDED = False


def load_system_instructions(
    path: str | Path = "evidence/provider-ultra-system-instructions-v1.json",
) -> str:
    return baseline.load_system_instructions(path)


def prompt_digest(prompt: str) -> str:
    return baseline.prompt_digest(prompt)


def counterbalanced_budgets(block_index: int) -> tuple[int, ...]:
    offset = block_index % len(OUTPUT_BUDGETS)
    return OUTPUT_BUDGETS[offset:] + OUTPUT_BUDGETS[:offset]


def planned_samples() -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    block_index = 0
    for repetition in range(1, REPETITIONS_PER_PROMPT + 1):
        for prompt_id, prompt in PROMPT_CASES:
            for order_index, max_tokens in enumerate(
                counterbalanced_budgets(block_index),
                start=1,
            ):
                plan.append(
                    {
                        "sample_id": f"{prompt_id}-r{repetition}-mt{max_tokens}",
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "prompt_sha256": prompt_digest(prompt),
                        "repetition": repetition,
                        "max_tokens": max_tokens,
                        "order_in_block": order_index,
                        "block_index": block_index,
                    }
                )
            block_index += 1
    return plan


def build_request(
    system_instructions: str,
    user_prompt: str,
    *,
    max_tokens: int,
) -> dict[str, Any]:
    if max_tokens not in OUTPUT_BUDGETS:
        raise ValueError(f"max_tokens must be one of {OUTPUT_BUDGETS}")

    request = baseline.build_request(system_instructions, user_prompt)
    request["max_tokens"] = max_tokens
    return request


def safe_error_metadata(exc: Exception) -> dict[str, Any]:
    return baseline.safe_error_metadata(exc)


def run_call(
    *,
    client: OpenAI,
    tracker: baseline.AttemptTracker,
    sample: dict[str, Any],
    system_instructions: str,
) -> dict[str, Any]:
    request = build_request(
        system_instructions,
        sample["prompt"],
        max_tokens=int(sample["max_tokens"]),
    )
    started = time.perf_counter()
    tracker.begin(sample["sample_id"], sample_started_at=started)
    try:
        completion = client.chat.completions.create(**request)
        completed_at = time.perf_counter()
        transport = tracker.snapshot(
            sample["sample_id"],
            call_completed_at=completed_at,
        )
        if transport["http_attempts"] == 0:
            transport["http_attempts"] = 1
            transport["retries"] = 0

        choices = getattr(completion, "choices", None)
        message = getattr(choices[0], "message", None) if choices else None
        content = getattr(message, "content", "") if message is not None else ""
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

        return {
            "sample_id": sample["sample_id"],
            "prompt_id": sample["prompt_id"],
            "prompt_sha256": sample["prompt_sha256"],
            "repetition": sample["repetition"],
            "max_tokens": sample["max_tokens"],
            "order_in_block": sample["order_in_block"],
            "block_index": sample["block_index"],
            "status": "completed",
            "total_seconds": round(completed_at - started, 4),
            "output_chars": len(content) if isinstance(content, str) else 0,
            "output_tokens": output_tokens if isinstance(output_tokens, int) else None,
            "total_tokens": total_tokens if isinstance(total_tokens, int) else None,
            "response_model": getattr(completion, "model", None),
            **transport,
        }
    except Exception as exc:
        failed_at = time.perf_counter()
        return {
            "sample_id": sample["sample_id"],
            "prompt_id": sample["prompt_id"],
            "prompt_sha256": sample["prompt_sha256"],
            "repetition": sample["repetition"],
            "max_tokens": sample["max_tokens"],
            "order_in_block": sample["order_in_block"],
            "block_index": sample["block_index"],
            "status": "infrastructure_error",
            "total_seconds": round(failed_at - started, 4),
            **tracker.snapshot(
                sample["sample_id"],
                call_completed_at=failed_at,
            ),
            **safe_error_metadata(exc),
        }
    finally:
        tracker.end()


def _final_request_to_headers_seconds(item: dict[str, Any]) -> float | None:
    attribution = item.get("latency_attribution")
    if not isinstance(attribution, dict):
        return None
    attempts = attribution.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return None
    value = attempts[-1].get("request_to_headers_seconds")
    return float(value) if isinstance(value, (int, float)) else None


def _headers_to_completion_seconds(item: dict[str, Any]) -> float | None:
    attribution = item.get("latency_attribution")
    if not isinstance(attribution, dict):
        return None
    value = attribution.get("final_headers_to_completion_seconds")
    return float(value) if isinstance(value, (int, float)) else None


def _retry_path_seconds(item: dict[str, Any]) -> float | None:
    attribution = item.get("latency_attribution")
    if not isinstance(attribution, dict):
        return None
    value = attribution.get("retry_path_seconds")
    return float(value) if isinstance(value, (int, float)) else None


def cohort_summary(
    results: list[dict[str, Any]],
    *,
    max_tokens: int,
) -> dict[str, Any]:
    observed = [
        item
        for item in results
        if item.get("max_tokens") == max_tokens
    ]
    completed = [item for item in observed if item.get("status") == "completed"]
    errors = [item for item in observed if item.get("status") != "completed"]

    latencies = [
        float(item["total_seconds"])
        for item in completed
        if isinstance(item.get("total_seconds"), (int, float))
    ]
    request_to_headers = [
        value
        for item in completed
        if (value := _final_request_to_headers_seconds(item)) is not None
    ]
    headers_to_completion = [
        value
        for item in completed
        if (value := _headers_to_completion_seconds(item)) is not None
    ]
    retry_path = [
        value
        for item in completed
        if (value := _retry_path_seconds(item)) is not None
    ]
    output_tokens = [
        int(item["output_tokens"])
        for item in completed
        if isinstance(item.get("output_tokens"), int)
    ]

    p95 = baseline.percentile(latencies, 0.95)
    return {
        "max_tokens": max_tokens,
        "planned": CALLS_PER_BUDGET,
        "observed": len(observed),
        "completed": len(completed),
        "errors": len(errors),
        "p50_total_seconds": baseline.percentile(latencies, 0.50),
        "p95_total_seconds": p95,
        "max_total_seconds": round(max(latencies), 4) if latencies else None,
        "p50_request_to_headers_seconds": baseline.percentile(
            request_to_headers,
            0.50,
        ),
        "p95_request_to_headers_seconds": baseline.percentile(
            request_to_headers,
            0.95,
        ),
        "p50_headers_to_completion_seconds": baseline.percentile(
            headers_to_completion,
            0.50,
        ),
        "p95_headers_to_completion_seconds": baseline.percentile(
            headers_to_completion,
            0.95,
        ),
        "p95_retry_path_seconds": baseline.percentile(retry_path, 0.95),
        "samples_with_retry": sum(
            1
            for item in observed
            if isinstance(item.get("retries"), int) and item["retries"] > 0
        ),
        "total_retries": sum(
            int(item.get("retries", 0))
            for item in observed
            if isinstance(item.get("retries"), int)
        ),
        "output_token_observations": len(output_tokens),
        "p50_output_tokens": baseline.percentile(
            [float(value) for value in output_tokens],
            0.50,
        ),
        "p95_output_tokens": baseline.percentile(
            [float(value) for value in output_tokens],
            0.95,
        ),
        "max_output_tokens": max(output_tokens) if output_tokens else None,
        "calls_over_slo": sum(1 for value in latencies if value > SLO_SECONDS),
        "strict_slo_met": (
            len(observed) == CALLS_PER_BUDGET
            and not errors
            and p95 is not None
            and p95 <= SLO_SECONDS
        ),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "planned": TOTAL_PLANNED_CALLS,
        "observed": len(results),
        "completed": sum(1 for item in results if item.get("status") == "completed"),
        "errors": sum(1 for item in results if item.get("status") != "completed"),
        "cohorts": {
            str(max_tokens): cohort_summary(results, max_tokens=max_tokens)
            for max_tokens in OUTPUT_BUDGETS
        },
    }


def build_report(
    *,
    base_url: str,
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
        "attribution_schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "provider": "nvidia_nim",
        "api_base_url": base_url,
        "model": MODEL,
        "system_instructions_source": SYSTEM_INSTRUCTIONS_SOURCE,
        "system_instructions_sha256": SYSTEM_INSTRUCTIONS_SHA256,
        "prompt_cases": [
            {"id": prompt_id, "sha256": prompt_digest(prompt)}
            for prompt_id, prompt in PROMPT_CASES
        ],
        "repetitions_per_prompt": REPETITIONS_PER_PROMPT,
        "experimental_parameter": EXPERIMENTAL_PARAMETER,
        "output_budgets": list(OUTPUT_BUDGETS),
        "calls_per_budget": CALLS_PER_BUDGET,
        "total_planned_calls": TOTAL_PLANNED_CALLS,
        "fixed_request_contract": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "enable_thinking": ENABLE_THINKING,
            "stream": STREAM,
            "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "max_retries": MAX_RETRIES,
            "pacing_seconds": PACING_SECONDS,
        },
        "target_p95_seconds": SLO_SECONDS,
        "production_configuration_changed": PRODUCTION_CONFIGURATION_CHANGED,
        "production_runtime_parameter_change_authorized": (
            PRODUCTION_RUNTIME_PARAMETER_CHANGE_AUTHORIZED
        ),
        "promotion_authorized": PROMOTION_AUTHORIZED,
        "quality_evaluation": QUALITY_EVALUATION,
        "sensitive_payloads_recorded": SENSITIVE_PAYLOADS_RECORDED,
        "complete": len(results) == TOTAL_PLANNED_CALLS,
        "summary": summarize(results),
        "results": list(results),
    }


def persist_report(
    output: str | Path,
    *,
    base_url: str,
    results: list[dict[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(
        base_url=base_url,
        results=results,
        created_at=created_at,
    )
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
    expected_git_sha: str | None = None,
) -> None:
    assert report["schema_version"] == "1.0"
    if expected_git_sha is not None:
        assert report["git_sha"] == expected_git_sha
    assert report["experiment"] == EXPERIMENT_NAME
    assert report["experiment_spec_version"] == EXPERIMENT_SPEC_VERSION
    assert report["attribution_schema_version"] == ATTRIBUTION_SCHEMA_VERSION
    assert report["provider"] == "nvidia_nim"
    assert report["api_base_url"] == DEFAULT_BASE_URL
    assert report["model"] == MODEL
    assert report["system_instructions_source"] == SYSTEM_INSTRUCTIONS_SOURCE
    assert report["system_instructions_sha256"] == SYSTEM_INSTRUCTIONS_SHA256
    assert report["experimental_parameter"] == "max_tokens"
    assert report["output_budgets"] == [128, 256, 512]
    assert report["calls_per_budget"] == 20
    assert report["total_planned_calls"] == 60
    assert report["fixed_request_contract"] == {
        "temperature": 1.0,
        "top_p": 0.95,
        "enable_thinking": False,
        "stream": False,
        "timeout_seconds": 45.0,
        "max_retries": 2,
        "pacing_seconds": 12.0,
    }
    assert report["target_p95_seconds"] == 8.0
    assert report["production_configuration_changed"] is False
    assert report["production_runtime_parameter_change_authorized"] is False
    assert report["promotion_authorized"] is False
    assert report["quality_evaluation"] is False
    assert report["sensitive_payloads_recorded"] is False

    expected_plan = planned_samples()
    expected_by_id = {item["sample_id"]: item for item in expected_plan}
    results = report["results"]
    assert isinstance(results, list)
    assert 0 <= len(results) <= TOTAL_PLANNED_CALLS
    assert report["complete"] is (len(results) == TOTAL_PLANNED_CALLS)

    seen: set[str] = set()
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
    for item in results:
        sample_id = item["sample_id"]
        assert sample_id in expected_by_id
        assert sample_id not in seen
        seen.add(sample_id)
        expected = expected_by_id[sample_id]
        assert item["prompt_id"] == expected["prompt_id"]
        assert item["prompt_sha256"] == expected["prompt_sha256"]
        assert item["repetition"] == expected["repetition"]
        assert item["max_tokens"] == expected["max_tokens"]
        assert item["order_in_block"] == expected["order_in_block"]
        assert item["block_index"] == expected["block_index"]
        assert item["status"] in {"completed", "infrastructure_error"}
        assert forbidden.isdisjoint(item.keys())

        attribution = item["latency_attribution"]
        assert attribution["schema_version"] == ATTRIBUTION_SCHEMA_VERSION
        assert isinstance(attribution["attribution_available"], bool)
        assert isinstance(attribution["attempts"], list)
        assert len(attribution["attempts"]) == item["http_attempts"]

    summary = report["summary"]
    assert summary["planned"] == TOTAL_PLANNED_CALLS
    assert summary["observed"] == len(results)
    assert summary["completed"] + summary["errors"] == len(results)
    assert set(summary["cohorts"]) == {"128", "256", "512"}
    for max_tokens in OUTPUT_BUDGETS:
        cohort = summary["cohorts"][str(max_tokens)]
        assert cohort["max_tokens"] == max_tokens
        assert cohort["planned"] == CALLS_PER_BUDGET
        assert 0 <= cohort["observed"] <= CALLS_PER_BUDGET
        for field in (
            "p50_total_seconds",
            "p95_total_seconds",
            "max_total_seconds",
            "p50_request_to_headers_seconds",
            "p95_request_to_headers_seconds",
            "p50_headers_to_completion_seconds",
            "p95_headers_to_completion_seconds",
            "p95_retry_path_seconds",
            "p50_output_tokens",
            "p95_output_tokens",
        ):
            value = cohort[field]
            assert value is None or isinstance(value, (int, float))

    if report["complete"]:
        assert seen == set(expected_by_id)
        assert all(
            summary["cohorts"][str(max_tokens)]["observed"] == CALLS_PER_BUDGET
            for max_tokens in OUTPUT_BUDGETS
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="provider-ultra-output-budget-latency/report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not api_key:
        print("NVIDIA_API_KEY is required for live output-budget latency evaluation")
        return 2

    base_url = os.getenv("NVIDIA_API_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    if base_url != DEFAULT_BASE_URL:
        raise ValueError("unexpected NVIDIA_API_BASE_URL")

    system_instructions = load_system_instructions()
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
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
        http_client=http_client,
    )

    results: list[dict[str, Any]] = []
    created_at = datetime.now(UTC).isoformat()
    report = persist_report(
        args.output,
        base_url=base_url,
        results=results,
        created_at=created_at,
    )
    try:
        plan = planned_samples()
        for index, sample in enumerate(plan):
            results.append(
                run_call(
                    client=client,
                    tracker=tracker,
                    sample=sample,
                    system_instructions=system_instructions,
                )
            )
            report = persist_report(
                args.output,
                base_url=base_url,
                results=results,
                created_at=created_at,
            )
            if index + 1 < len(plan):
                time.sleep(PACING_SECONDS)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    validate_report(report)
    print(json.dumps(report["summary"], sort_keys=True))
    return 2 if report["summary"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
