from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jadel_chatbot.config import AppConfig
from jadel_chatbot.prompts import PROMPT_VERSION
from jadel_chatbot.security import redact_likely_secrets
from jadel_chatbot.service import AIService

STAGES = ("input_safety", "main_model", "output_safety")


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[index]


def contains_any(text: str, terms: list[str]) -> bool:
    if not terms:
        return True
    normalized = text.casefold()
    return any(term.casefold() in normalized for term in terms)


def contains_none(text: str, terms: list[str]) -> bool:
    normalized = text.casefold()
    return all(term.casefold() not in normalized for term in terms)


def safe_error_metadata(exc: Exception) -> dict[str, Any]:
    metadata: dict[str, Any] = {"error_type": type(exc).__name__}

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        metadata["status_code"] = status_code

    request_id = getattr(exc, "request_id", None)
    if isinstance(request_id, str) and request_id:
        metadata["request_id"] = request_id

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        remote_error = body.get("error", body)
        if isinstance(remote_error, dict):
            code = remote_error.get("code")
            remote_type = remote_error.get("type")
            if isinstance(code, str) and code:
                metadata["remote_error_code"] = code
            if isinstance(remote_type, str) and remote_type:
                metadata["remote_error_type"] = remote_type

    return metadata


def evaluate_case(service: AIService, case: dict[str, Any]) -> dict[str, Any]:
    raw_input = str(case["input"])
    redaction = redact_likely_secrets(raw_input)
    started = time.perf_counter()

    try:
        result = service.generate(
            [{"role": "user", "content": redaction.text}],
            current_user_text=redaction.text,
        )
        latency_seconds = time.perf_counter() - started
        required_ok = contains_any(result.text, list(case.get("must_contain_any", [])))
        forbidden_ok = contains_none(result.text, list(case.get("must_not_contain", [])))
        passed = required_ok and forbidden_ok

        return {
            "id": case["id"],
            "category": case["category"],
            "critical": bool(case.get("critical", False)),
            "status": "completed",
            "passed": passed,
            "required_any_ok": required_ok,
            "forbidden_terms_ok": forbidden_ok,
            "human_review": bool(case.get("human_review", True)),
            "redaction_detected_types": list(redaction.detected_types),
            "blocked_by_moderation": result.blocked_by_moderation,
            "latency_seconds": round(latency_seconds, 4),
            "stage_metrics": getattr(result, "stage_metrics", None) or {},
            "response_id": result.response_id,
            "model": result.model,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
            "answer": result.text,
        }
    except Exception as exc:
        return {
            "id": case["id"],
            "category": case["category"],
            "critical": bool(case.get("critical", False)),
            "status": "infrastructure_error",
            "passed": None,
            "human_review": True,
            "redaction_detected_types": list(redaction.detected_types),
            "latency_seconds": round(time.perf_counter() - started, 4),
            "stage_metrics": getattr(service, "last_error_stage_metrics", {}),
            **safe_error_metadata(exc),
        }


def stage_summary(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        entries: list[dict[str, Any]] = []
        for item in results:
            metrics = item.get("stage_metrics")
            if not isinstance(metrics, dict):
                continue
            value = metrics.get(stage)
            if isinstance(value, dict) and not value.get("skipped", False):
                entries.append(value)

        latencies = [
            float(entry["latency_seconds"])
            for entry in entries
            if isinstance(entry.get("latency_seconds"), (int, float))
        ]
        attempts = [
            int(entry["http_attempts"])
            for entry in entries
            if isinstance(entry.get("http_attempts"), int)
        ]
        retries = [
            int(entry["retries"])
            for entry in entries
            if isinstance(entry.get("retries"), int)
        ]
        summary[stage] = {
            "observed_cases": len(entries),
            "p50_latency_seconds": percentile(latencies, 0.50),
            "p95_latency_seconds": percentile(latencies, 0.95),
            "total_http_attempts": sum(attempts),
            "total_retries": sum(retries),
            "retried_cases": sum(1 for value in retries if value > 0),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="evals/baseline.json")
    parser.add_argument("--output", default="eval-results/report.json")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=float(os.getenv("MIN_EVAL_PASS_RATE", "0.90")),
    )
    parser.add_argument("--max-cases", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.min_pass_rate <= 1.0:
        raise ValueError("--min-pass-rate must be between 0 and 1")

    case_path = Path(args.cases)
    raw_cases = case_path.read_bytes()
    cases = json.loads(raw_cases)
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    config = AppConfig.from_env()
    if not config.api_key:
        print("NVIDIA_API_KEY is required for live model evaluation")
        return 2

    service = AIService(config)
    results: list[dict[str, Any]] = []
    for case in cases:
        item = evaluate_case(service, case)
        results.append(item)
        if item["status"] == "infrastructure_error":
            break

    completed = [item for item in results if item["status"] == "completed"]
    infrastructure_failures = [
        item for item in results if item["status"] == "infrastructure_error"
    ]
    passed = sum(1 for item in completed if item["passed"] is True)
    failed = sum(1 for item in completed if item["passed"] is False)
    critical_failures = [
        item["id"]
        for item in completed
        if item["critical"] and item["passed"] is False
    ]
    latencies = [float(item["latency_seconds"]) for item in results]
    token_totals = [
        int(item["total_tokens"])
        for item in completed
        if isinstance(item.get("total_tokens"), int)
    ]
    adjudicated = len(completed)
    pass_rate = passed / adjudicated if adjudicated else None

    if infrastructure_failures:
        evaluation_status = "NOT_ADJUDICATED_INFRASTRUCTURE"
    elif critical_failures or pass_rate is None or pass_rate < args.min_pass_rate:
        evaluation_status = "FAIL"
    else:
        evaluation_status = "PASS"

    report = {
        "schema_version": "1.3",
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "provider": config.provider,
        "api_base_url": config.api_base_url,
        "model": config.model,
        "safety_model": config.safety_model,
        "reasoning_enabled": config.enable_thinking,
        "max_output_tokens": config.max_output_tokens,
        "configured_timeout_seconds": config.timeout_seconds,
        "configured_max_retries": config.max_retries,
        "prompt_version": PROMPT_VERSION,
        "case_set_sha256": hashlib.sha256(raw_cases).hexdigest(),
        "human_review_required": True,
        "automatic_promotion_allowed": False,
        "evaluation_status": evaluation_status,
        "summary": {
            "planned_total": len(cases),
            "executed": len(results),
            "adjudicated": adjudicated,
            "passed": passed,
            "failed": failed,
            "pass_rate": round(pass_rate, 4) if pass_rate is not None else None,
            "critical_failures": critical_failures,
            "infrastructure_failures": [
                {
                    key: item[key]
                    for key in (
                        "id",
                        "error_type",
                        "status_code",
                        "request_id",
                        "remote_error_code",
                        "remote_error_type",
                        "stage_metrics",
                    )
                    if key in item
                }
                for item in infrastructure_failures
            ],
            "p95_latency_seconds": percentile(latencies, 0.95),
            "stage_latency": stage_summary(results),
            "total_tokens_observed": sum(token_totals) if token_totals else None,
            "min_pass_rate": args.min_pass_rate,
        },
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], sort_keys=True))

    if infrastructure_failures:
        return 2
    if critical_failures or pass_rate is None or pass_rate < args.min_pass_rate:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
