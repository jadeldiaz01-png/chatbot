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
    response_id: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class AIService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

    def _is_flagged(self, text: str, image_data_url: str | None = None) -> bool:
        if not self.config.moderation_enabled:
            return False

        moderation_input: str | list[dict[str, Any]]
        if image_data_url is None:
            if not text.strip():
                return False
            moderation_input = text
        else:
            items: list[dict[str, Any]] = []
            if text.strip():
                items.append({"type": "text", "text": text})
            items.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url},
                }
            )
            moderation_input = items

        result = self.client.moderations.create(
            model=self.config.moderation_model,
            input=moderation_input,
        )
        return bool(result.results and result.results[0].flagged)

    @staticmethod
    def _usage_value(usage: Any, name: str) -> int | None:
        value = getattr(usage, name, None)
        return value if isinstance(value, int) else None

    def _response_metadata(self, response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        return {
            "response_id": getattr(response, "id", None),
            "model": getattr(response, "model", self.config.model),
            "input_tokens": self._usage_value(usage, "input_tokens"),
            "output_tokens": self._usage_value(usage, "output_tokens"),
            "total_tokens": self._usage_value(usage, "total_tokens"),
        }

    def _log_response_metadata(self, response: Any) -> dict[str, Any]:
        metadata = self._response_metadata(response)
        logger.info(
            json.dumps(
                {
                    "event": "llm_response",
                    "prompt_version": PROMPT_VERSION,
                    "store": self.config.response_store,
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
        if image_data_url is not None and not self.config.multimodal_enabled:
            raise ValueError("multimodal capability is disabled")

        if self._is_flagged(current_user_text, image_data_url):
            return GenerationResult(
                "No puedo procesar ese contenido tal como está. "
                "Puedes reformular la solicitud de forma segura.",
                blocked_by_moderation=True,
            )

        request_input: list[dict[str, Any]] = [dict(item) for item in messages]
        if image_data_url is not None and request_input:
            if request_input[-1].get("role") == "user":
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
        metadata = self._log_response_metadata(response)

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
                **metadata,
            )
        return GenerationResult(answer, **metadata)
