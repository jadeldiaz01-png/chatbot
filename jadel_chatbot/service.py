from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .config import AppConfig
from .prompts import PROMPT_VERSION, SYSTEM_INSTRUCTIONS

logger = logging.getLogger("jadel_chatbot")


@dataclass(frozen=True)
class GenerationResult:
    text: str
    blocked_by_moderation: bool = False


class AIService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    def _is_flagged(self, text: str) -> bool:
        if not self.config.moderation_enabled or not text.strip():
            return False
        result = self.client.moderations.create(
            model=self.config.moderation_model,
            input=text,
        )
        return bool(result.results and result.results[0].flagged)

    @staticmethod
    def _usage_value(usage: Any, name: str) -> int | None:
        value = getattr(usage, name, None)
        return value if isinstance(value, int) else None

    def _log_response_metadata(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        payload = {
            "event": "llm_response",
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", self.config.model),
            "prompt_version": PROMPT_VERSION,
            "input_tokens": self._usage_value(usage, "input_tokens"),
            "output_tokens": self._usage_value(usage, "output_tokens"),
            "total_tokens": self._usage_value(usage, "total_tokens"),
            "store": self.config.response_store,
        }
        logger.info(json.dumps(payload, sort_keys=True))

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        current_user_text: str,
        image_data_url: str | None = None,
    ) -> GenerationResult:
        if self._is_flagged(current_user_text):
            return GenerationResult(
                "No puedo procesar ese contenido tal como está. "
                "Puedes reformular la solicitud de forma segura.",
                blocked_by_moderation=True,
            )

        request_input: list[dict[str, Any]] = [dict(item) for item in messages]
        if image_data_url is not None:
            if not self.config.multimodal_enabled:
                raise ValueError("multimodal capability is disabled")
            if request_input and request_input[-1].get("role") == "user":
                request_input[-1] = {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": current_user_text},
                        {"type": "input_image", "image_url": image_data_url},
                    ],
                }

        response = self.client.responses.create(
            model=self.config.model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=request_input,
            max_output_tokens=self.config.max_output_tokens,
            store=self.config.response_store,
        )
        self._log_response_metadata(response)

        answer = (response.output_text or "").strip()
        if not answer:
            answer = (
                "No pude generar una respuesta útil. Inténtalo nuevamente o solicita "
                "seguimiento humano."
            )

        if self._is_flagged(answer):
            return GenerationResult(
                "La respuesta generada fue retenida por los controles de seguridad. "
                "Solicita seguimiento humano si necesitas ayuda.",
                blocked_by_moderation=True,
            )
        return GenerationResult(answer)
