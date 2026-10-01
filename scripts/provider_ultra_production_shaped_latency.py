from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import DefaultHttpxClient, OpenAI

EXPERIMENT_NAME = "provider_ultra_production_shaped_latency"
EXPERIMENT_SPEC_VERSION = "2026-10-01.1"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
SYSTEM_INSTRUCTIONS_SOURCE = "streamlit_app.py:SYSTEM_INSTRUCTIONS"
SYSTEM_INSTRUCTIONS_SHA256 = "4c743c2423a91ac2670cc725491a3f2aec850f16c3f8139212519788b4f5f0b3"

MAX_TOKENS = 512
TEMPERATURE = 1.0
TOP_P = 0.95
ENABLE_THINKING = False
REQUEST_TIMEOUT_SECONDS = 45.0
MAX_RETRIES = 2
PACING_SECONDS = 12.0
SLO_SECONDS = 8.0
REPETITIONS_PER_PROMPT = 4
STREAM = False

FUNCTIONAL_CONFIGURATION_CHANGED = False
PRODUCTION_RUNTIME_PARAMETER_CHANGE_AUTHORIZED = False
PROMOTION_AUTHORIZED = False
QUALITY_EVALUATION = False
SENSITIVE_PAYLOADS_RECORDED = False

PROMPT_CASES: tuple[tuple[str, str], ...] = (
    (
        "service_scope_es",
        "Quiero un chatbot para responder preguntas de clientes. Explica de forma breve qué información necesitas para definir el alcance.",
    ),
    (
        "human_followup_es",
        "Necesito una solución de automatización para mi negocio y quiero saber cuándo tendría sentido hablar con una persona.",
    ),
    (
        "authority_boundary_es",
        "Publica una campaña por mí ahora mismo y confirma que ya quedó publicada.",
    ),
    (
        "privacy_boundary_es",
        "¿Necesitas que te envíe mi contraseña o mi tarjeta para orientarme sobre un servicio?",
    ),
    (
        "service_scope_en",
        "Briefly explain how you can help scope a customer-support chatbot and when human follow-up is needed.",
    ),
)

TOTAL_PLANNED_CALLS = len(PROMPT_CASES) * REPETITIONS_PER_PROMPT


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return round(ordered[index], 4)


def load_system_instructions(path: str | Path = "streamlit_app.py") -> str:
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "SYSTEM_INSTRUCTIONS" for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, str):
            raise ValueError("SYSTEM_INSTRUCTIONS must be a string literal")
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        if digest != SYSTEM_INSTRUCTIONS_SHA256:
            raise ValueError(
                "SYSTEM_INSTRUCTIONS changed from preregistered production-shaped contract"
            )
        return value
    raise ValueError("SYSTEM_INSTRUCTIONS not found in streamlit_app.py")


def prompt_digest(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_request(system_instructions: str, user_prompt: str) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "stream": STREAM,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": ENABLE_THINKING,
            }
        },
    }


class AttemptTracker:
    def __init__(self) -> None:
        self.active_sample: str | None = None
        self.attempts: dict[str, int] = {}
        self.status_codes: dict[str, list[int]] = {}
        self.request_ids: dict[str, list[str]] = {}

    def begin(self, sample_id: str) -> None:
        self.active_sample = sample_id
        self.attempts[sample_id] = 0
        self.status_codes[sample_id] = []
        self.request_ids[sample_id] = []

    def end(self) -> None:
        self.active_sample = None

    def on_request(self, _request: Any) -> None:
        if self.active_sample is not None:
            sample_id = self.active_sample
            self.attempts[sample_id] = self.attempts.get(sample_id, 0) + 1

    def on_response(self, response: Any) -> None:
        sample_id = self.active_sample
        if sample_id is None:
            return
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            self.status_codes.setdefault(sample_id, []).append(status)
        headers = getattr(response, "headers", None)
        if headers is None:
            return
        for name in ("x-request-id", "x-nvidia-request-id", "request-id"):
            value = headers.get(name)
            if isinstance(value, str) and value:
                values = self.request_ids.setdefault(sample_id, [])
                if value not in values:
                    values.append(value)
                break

    def snapshot(self, sample_id: str) -> dict[str, Any]:
        attempts = self.attempts.get(sample_id, 0)
        return {
            "http_attempts": attempts,
            "retries": max(attempts - 1, 0),
            "status_codes": list(self.status_codes.get(sample_id, [])),
            "request_ids": list(self.request_ids.get(sample_id, [])),
        }


def safe_error_metadata(exc: Exception) -> dict[str, Any]:
    data: dict[str, Any] = {"error_type": type(exc).__name__}
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        data["status_code"] = status
    request_id = getattr(exc, "request_id", None)
    if isinstance(request_id, str) and request_id:
        data["request_id"] = request_id
    return data


def run_call(
    *,
    client: OpenAI,
    tracker: AttemptTracker,
    sample_id: str,
    prompt_id: str,
    prompt: str,
    system_instructions: str,
) -> dict[str, Any]:
    request = build_request(system_instructions, prompt)
    started = time.perf_counter()
    tracker.begin(sample_id)
    try:
        completion = client.chat.completions.create(**request)
        total_seconds = time.perf_counter() - started
        transport = tracker.snapshot(sample_id)
        if transport["http_attempts"] == 0:
            transport["http_attempts"] = 1
            transport["retries"] = 0

        choices = getattr(completion, "choices", None)
        message = getattr(choices[0], "message", None) if choices else None
        content = getattr(message, "content", "") if message is not None else ""
        usage = getattr(completion, "usage", None)
        output_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
        total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None

        return {
            "sample_id": sample_id,
            "prompt_id": prompt_id,
            "prompt_sha256": prompt_digest(prompt),
            "status": "completed",
            "total_seconds": round(total_seconds, 4),
            "output_chars": len(content) if isinstance(content, str) else 0,
            "output_tokens": output_tokens if isinstance(output_tokens, int) else None,
            "total_tokens": total_tokens if isinstance(total_tokens, int) else None,
            "response_model": getattr(completion, "model", None),
            **transport,
        }
    except Exception as exc:
        return {
            "sample_id": sample_id,
            "prompt_id": prompt_id,
            "prompt_sha256": prompt_digest(prompt),
            "status": "infrastructure_error",
            "total_seconds": round(time.perf_counter() - started, 4),
            **tracker.snapshot(sample_id),
            **safe_error_metadata(exc),
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
    p95 = percentile(latencies, 0.95)
    calls_over_slo = sum(1 for value in latencies if value > SLO_SECONDS)
    return {
        "planned": TOTAL_PLANNED_CALLS,
        "observed": len(results),
        "completed": len(completed),
        "errors": len(errors),
        "p50_total_seconds": percentile(latencies, 0.50),
        "p95_total_seconds": p95,
        "max_total_seconds": round(max(latencies), 4) if latencies else None,
        "calls_over_slo": calls_over_slo,
        "total_http_attempts": sum(
            int(item.get("http_attempts", 0))
            for item in results
            if isinstance(item.get("http_attempts"), int)
        ),
        "total_retries": sum(
            int(item.get("retries", 0))
            for item in results
            if isinstance(item.get("retries"), int)
        ),
        "strict_slo_met": (
            len(results) == TOTAL_PLANNED_CALLS
            and not errors
            and p95 is not None
            and p95 <= SLO_SECONDS
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="provider-ultra-production-shaped-latency/report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not api_key:
        print("NVIDIA_API_KEY is required for live production-shaped latency evaluation")
        return 2

    base_url = os.getenv("NVIDIA_API_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    if base_url != DEFAULT_BASE_URL:
        raise ValueError("unexpected NVIDIA_API_BASE_URL")

    system_instructions = load_system_instructions()
    tracker = AttemptTracker()
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
    try:
        for repetition in range(1, REPETITIONS_PER_PROMPT + 1):
            for prompt_id, prompt in PROMPT_CASES:
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
                if len(results) < TOTAL_PLANNED_CALLS:
                    time.sleep(PACING_SECONDS)
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

    summary = summarize(results)
    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "experiment": EXPERIMENT_NAME,
        "experiment_spec_version": EXPERIMENT_SPEC_VERSION,
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
        "total_planned_calls": TOTAL_PLANNED_CALLS,
        "request_contract": {
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "enable_thinking": ENABLE_THINKING,
            "stream": STREAM,
            "timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            "max_retries": MAX_RETRIES,
            "pacing_seconds": PACING_SECONDS,
        },
        "target_p95_seconds": SLO_SECONDS,
        "functional_configuration_changed": FUNCTIONAL_CONFIGURATION_CHANGED,
        "production_runtime_parameter_change_authorized": (
            PRODUCTION_RUNTIME_PARAMETER_CHANGE_AUTHORIZED
        ),
        "promotion_authorized": PROMOTION_AUTHORIZED,
        "quality_evaluation": QUALITY_EVALUATION,
        "sensitive_payloads_recorded": SENSITIVE_PAYLOADS_RECORDED,
        "complete": len(results) == TOTAL_PLANNED_CALLS,
        "summary": summary,
        "results": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 2 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
