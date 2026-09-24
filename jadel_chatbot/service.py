from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from openai import DefaultHttpxClient, OpenAI

from .config import AppConfig
from .prompts import PROMPT_VERSION, SYSTEM_INSTRUCTIONS

logger = logging.getLogger("jadel_chatbot")

_SAFETY_RE = re.compile(
    r"^(User Safety|Response Safety):\s*(safe|unsafe)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_REQUEST_ID_HEADERS = ("x-request-id", "x-nvidia-request-id", "request-id")


@dataclass(frozen=True)
class GenerationResult:
    text: str
    blocked_by_moderation: bool = False
    response_id: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    stage_metrics: dict[str, dict[str, Any]] | None = None


class AIService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._active_stage: str | None = None
        self._attempt_counts: dict[str, int] = {}
        self._status_codes: dict[str, list[int]] = {}
        self._request_ids: dict[str, list[str]] = {}
        http_client = DefaultHttpxClient(
            event_hooks={
                "request": [self._record_http_attempt],
                "response": [self._record_http_response],
            }
        )
        self.client = OpenAI(
            base_url=config.api_base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
            http_client=http_client,
        )

    def _record_http_attempt(self, _request: Any) -> None:
        stage = getattr(self, "_active_stage", None)
        if stage is None:
            return
        self._attempt_counts[stage] = self._attempt_counts.get(stage, 0) + 1

    def _record_http_response(self, response: Any) -> None:
        stage = getattr(self, "_active_stage", None)
        if stage is None:
            return

        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            self._status_codes.setdefault(stage, []).append(status_code)

        headers = getattr(response, "headers", None)
        if headers is None:
            return
        for header in _REQUEST_ID_HEADERS:
            value = headers.get(header)
            if isinstance(value, str) and value:
                values = self._request_ids.setdefault(stage, [])
                if value not in values:
                    values.append(value)
                break

    @staticmethod
    def _usage_value(usage: Any, *names: str) -> int | None:
        for name in names:
            value = getattr(usage, name, None)
            if isinstance(value, int):
                return value
        return None

    @staticmethod
    def _message_text(completion: Any) -> str:
        choices = getattr(completion, "choices", None)
        if not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", "")
        return content.strip() if isinstance(content, str) else ""

    @staticmethod
    def _safety_verdict(text: str, label: str) -> bool:
        verdicts = {
            name.casefold(): value.casefold()
            for name, value in _SAFETY_RE.findall(text)
        }
        key = label.casefold()
        if key not in verdicts:
            raise RuntimeError("safety model returned an unrecognized verdict")
        return verdicts[key] == "unsafe"

    def _stage_completion(
        self,
        stage: str,
        **request: Any,
    ) -> tuple[Any, dict[str, Any]]:
        self._attempt_counts[stage] = 0
        self._status_codes[stage] = []
        self._request_ids[stage] = []
        self._active_stage = stage
        started = time.perf_counter()
        completion: Any | None = None
        try:
            completion = self.client.chat.completions.create(**request)
        except Exception as exc:
            metrics = self._finalize_stage_metrics(
                stage,
                request=request,
                latency_seconds=time.perf_counter() - started,
                completion=None,
            )
            try:
                setattr(exc, "stage_metrics", {stage: metrics})
            except Exception:
                pass
            raise
        finally:
            self._active_stage = None

        metrics = self._finalize_stage_metrics(
            stage,
            request=request,
            latency_seconds=time.perf_counter() - started,
            completion=completion,
        )
        return completion, metrics

    def _finalize_stage_metrics(
        self,
        stage: str,
        *,
        request: dict[str, Any],
        latency_seconds: float,
        completion: Any | None,
    ) -> dict[str, Any]:
        attempts = self._attempt_counts.pop(stage, 0)
        if completion is not None and attempts == 0:
            # Unit-test fakes do not execute the HTTP hooks.
            attempts = 1

        request_ids = self._request_ids.pop(stage, [])
        sdk_request_id = getattr(completion, "_request_id", None)
        if isinstance(sdk_request_id, str) and sdk_request_id:
            if sdk_request_id not in request_ids:
                request_ids.append(sdk_request_id)

        status_codes = self._status_codes.pop(stage, [])
        return {
            "latency_seconds": round(latency_seconds, 4),
            "http_attempts": attempts,
            "retries": max(attempts - 1, 0),
            "status_codes": status_codes,
            "request_ids": request_ids,
            "request_model": request.get("model"),
            "max_tokens": request.get("max_tokens"),
            "configured_timeout_seconds": self.config.timeout_seconds,
            "configured_max_retries": self.config.max_retries,
            "response_id": getattr(completion, "id", None),
            "response_model": getattr(completion, "model", None),
        }

    def _is_flagged(
        self,
        user_text: str,
        *,
        assistant_text: str | None = None,
        stage: str = "input_safety",
        stage_metrics: dict[str, dict[str, Any]] | None = None,
    ) -> bool:
        if not self.config.moderation_enabled:
            if stage_metrics is not None:
                stage_metrics[stage] = {
                    "latency_seconds": 0.0,
                    "http_attempts": 0,
                    "retries": 0,
                    "status_codes": [],
                    "request_ids": [],
                    "request_model": self.config.safety_model,
                    "max_tokens": 100,
                    "configured_timeout_seconds": self.config.timeout_seconds,
                    "configured_max_retries": self.config.max_retries,
                    "skipped": True,
                }
            return False

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": [{"type": "text", "text": user_text}]}
        ]
        label = "User Safety"
        if assistant_text is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                }
            )
            label = "Response Safety"

        completion, metrics = self._stage_completion(
            stage,
            model=self.config.safety_model,
            messages=messages,
            max_tokens=100,
            temperature=0.01,
            top_p=0.95,
            extra_body={
                "chat_template_kwargs": {
                    "request_categories": "/categories",
                    "enable_thinking": False,
                }
            },
        )
        if stage_metrics is not None:
            stage_metrics[stage] = metrics
        verdict = self._message_text(completion)
        return self._safety_verdict(verdict, label)

    def _response_metadata(self, completion: Any) -> dict[str, Any]:
        usage = getattr(completion, "usage", None)
        return {
            "response_id": getattr(completion, "id", None),
            "model": getattr(completion, "model", self.config.model),
            "input_tokens": self._usage_value(usage, "prompt_tokens", "input_tokens"),
            "output_tokens": self._usage_value(
                usage, "completion_tokens", "output_tokens"
            ),
            "total_tokens": self._usage_value(usage, "total_tokens"),
        }

    def _log_response_metadata(self, completion: Any) -> dict[str, Any]:
        metadata = self._response_metadata(completion)
        logger.info(
            json.dumps(
                {
                    "event": "llm_response",
                    "provider": self.config.provider,
                    "prompt_version": PROMPT_VERSION,
                    **metadata,
                },
                sort_keys=True,
            )
        )
        return metadata

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        current_user_text: str,
        image_data_url: str | None = None,
    ) -> GenerationResult:
        if image_data_url is not None:
            raise ValueError("Nemotron 3 Ultra production baseline is text-only")

        stage_metrics: dict[str, dict[str, Any]] = {}
        if self._is_flagged(
            current_user_text,
            stage="input_safety",
            stage_metrics=stage_metrics,
        ):
            return GenerationResult(
                "I cannot process or execute that request as written. "
                "No puedo procesar ni ejecutar esa solicitud tal como está. "
                "Puedes reformularla de forma segura.",
                blocked_by_moderation=True,
                stage_metrics=stage_metrics,
            )

        request_messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS}
        ]
        request_messages.extend(dict(item) for item in messages)

        completion, main_metrics = self._stage_completion(
            "main_model",
            model=self.config.model,
            messages=request_messages,
            max_tokens=self.config.max_output_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": self.config.enable_thinking,
                }
            },
        )
        stage_metrics["main_model"] = main_metrics
        metadata = self._log_response_metadata(completion)

        answer = self._message_text(completion)
        if not answer:
            answer = (
                "No pude generar una respuesta útil. Inténtalo nuevamente o solicita "
                "seguimiento humano."
            )

        if self._is_flagged(
            current_user_text,
            assistant_text=answer,
            stage="output_safety",
            stage_metrics=stage_metrics,
        ):
            return GenerationResult(
                "La respuesta generada fue retenida por los controles de seguridad. "
                "Solicita seguimiento humano si necesitas ayuda.",
                blocked_by_moderation=True,
                stage_metrics=stage_metrics,
                **metadata,
            )

        return GenerationResult(answer, stage_metrics=stage_metrics, **metadata)
