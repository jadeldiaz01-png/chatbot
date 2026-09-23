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
            "passed": passed,
            "required_any_ok": required_ok,
            "forbidden_terms_ok": forbidden_ok,
            "human_review": bool(case.get("human_review", True)),
            "redaction_detected_types": list(redaction.detected_types),
            "blocked_by_moderation": result.blocked_by_moderation,
            "latency_seconds": round(latency_seconds, 4),
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
            "passed": False,
            "human_review": True,
            "redaction_detected_types": list(redaction.detected_types),
            "latency_seconds": round(time.perf_counter() - started, 4),
            "error_type": type(exc).__name__,
        }


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
        print("OPENAI_API_KEY is required for live model evaluation")
        return 2

    service = AIService(config)
    results = [evaluate_case(service, case) for case in cases]

    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    critical_failures = [
        item["id"] for item in results if item["critical"] and not item["passed"]
    ]
    latencies = [float(item["latency_seconds"]) for item in results]
    token_totals = [
        int(item["total_tokens"])
        for item in results
        if isinstance(item.get("total_tokens"), int)
    ]
    pass_rate = passed / total if total else 0.0

    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "model": config.model,
        "prompt_version": PROMPT_VERSION,
        "case_set_sha256": hashlib.sha256(raw_cases).hexdigest(),
        "human_review_required": True,
        "automatic_promotion_allowed": False,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(pass_rate, 4),
            "critical_failures": critical_failures,
            "p95_latency_seconds": percentile(latencies, 0.95),
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

    if critical_failures or pass_rate < args.min_pass_rate:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
