from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _env_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class AppConfig:
    api_key: str
    model: str
    moderation_model: str
    max_input_chars: int
    max_history_messages: int
    max_output_tokens: int
    timeout_seconds: float
    max_retries: int
    session_requests_per_minute: int
    response_store: bool
    moderation_enabled: bool
    multimodal_enabled: bool
    max_image_bytes: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna").strip(),
            moderation_model=os.getenv(
                "OPENAI_MODERATION_MODEL", "omni-moderation-latest"
            ).strip(),
            max_input_chars=_env_int(
                "MAX_INPUT_CHARS", 4000, minimum=128, maximum=20000
            ),
            max_history_messages=_env_int(
                "MAX_HISTORY_MESSAGES", 20, minimum=2, maximum=100
            ),
            max_output_tokens=_env_int(
                "MAX_OUTPUT_TOKENS", 800, minimum=64, maximum=8192
            ),
            timeout_seconds=_env_float(
                "OPENAI_TIMEOUT_SECONDS", 30.0, minimum=1.0, maximum=120.0
            ),
            max_retries=_env_int("OPENAI_MAX_RETRIES", 2, minimum=0, maximum=5),
            session_requests_per_minute=_env_int(
                "SESSION_REQUESTS_PER_MINUTE", 10, minimum=1, maximum=60
            ),
            response_store=_env_bool("OPENAI_RESPONSE_STORE", False),
            moderation_enabled=_env_bool("ENABLE_MODERATION", True),
            multimodal_enabled=_env_bool("ENABLE_MULTIMODAL", False),
            max_image_bytes=_env_int(
                "MAX_IMAGE_BYTES",
                5 * 1024 * 1024,
                minimum=1024,
                maximum=20 * 1024 * 1024,
            ),
        )
