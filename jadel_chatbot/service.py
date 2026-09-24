from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .config import AppConfig
from .prompts import PROMPT_VERSION, SYSTEM_INSTRUCTIONS

logger = logging.getLogger("jadel_chatbot")

_SAFETY_RE = re.compile(
    r"^(User Safety|Response Safety):\s*(safe|unsafe)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


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
            base_url=config.api_base_url,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )

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

    def _is_flagged(
        self,
        user_text: str,
        *,
        assistant_text: str | None = None,
    ) -> bool:
        if not self.config.moderation_enabled:
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

        completion = self.client.chat.completions.create(
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

        if self._is_flagged(current_user_text):
            return GenerationResult(
                "No puedo procesar ese contenido tal como está. "
                "Puedes reformular la solicitud de forma segura.",
                blocked_by_moderation=True,
            )

        request_messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS}
        ]
        request_messages.extend(dict(item) for item in messages)

        completion = self.client.chat.completions.create(
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
        metadata = self._log_response_metadata(completion)

        answer = self._message_text(completion)
        if not answer:
            answer = (
                "No pude generar una respuesta útil. Inténtalo nuevamente o solicita "
                "seguimiento humano."
            )

        if self._is_flagged(current_user_text, assistant_text=answer):
            return GenerationResult(
                "La respuesta generada fue retenida por los controles de seguridad. "
                "Solicita seguimiento humano si necesitas ayuda.",
                blocked_by_moderation=True,
                **metadata,
            )

        return GenerationResult(answer, **metadata)
