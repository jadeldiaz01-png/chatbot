from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
NVIDIA_DEFAULT_SAFETY_MODEL = "nvidia/nemotron-3.5-content-safety"


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


def _nvidia_base_url() -> str:
    value = os.getenv("NVIDIA_BASE_URL", NVIDIA_DEFAULT_BASE_URL).strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("NVIDIA_BASE_URL must be an absolute HTTPS URL")
    return value


@dataclass(frozen=True)
class AppConfig:
    api_key: str
    api_base_url: str
    provider: str
    model: str
    safety_model: str
    max_input_chars: int
    max_history_messages: int
    max_output_tokens: int
    timeout_seconds: float
    max_retries: int
    session_requests_per_minute: int
    moderation_enabled: bool
    multimodal_enabled: bool
    max_image_bytes: int
    enable_thinking: bool
    temperature: float
    top_p: float

    @classmethod
    def from_env(cls) -> AppConfig:
        multimodal_enabled = _env_bool("ENABLE_MULTIMODAL", False)
        if multimodal_enabled:
            raise ValueError(
                "ENABLE_MULTIMODAL cannot be enabled with the text-only Nemotron 3 Ultra baseline"
            )

        return cls(
            api_key=os.getenv("NVIDIA_API_KEY", "").strip(),
            api_base_url=_nvidia_base_url(),
            provider="nvidia_nim",
            model=os.getenv("NVIDIA_MODEL", NVIDIA_DEFAULT_MODEL).strip(),
            safety_model=os.getenv(
                "NVIDIA_SAFETY_MODEL", NVIDIA_DEFAULT_SAFETY_MODEL
            ).strip(),
            max_input_chars=_env_int(
                "MAX_INPUT_CHARS", 4000, minimum=128, maximum=20000
            ),
            max_history_messages=_env_int(
                "MAX_HISTORY_MESSAGES", 20, minimum=2, maximum=100
            ),
            max_output_tokens=_env_int(
                "MAX_OUTPUT_TOKENS", 1024, minimum=64, maximum=16384
            ),
            timeout_seconds=_env_float(
                "NVIDIA_TIMEOUT_SECONDS", 45.0, minimum=1.0, maximum=180.0
            ),
            max_retries=_env_int("NVIDIA_MAX_RETRIES", 2, minimum=0, maximum=5),
            session_requests_per_minute=_env_int(
                "SESSION_REQUESTS_PER_MINUTE", 10, minimum=1, maximum=60
            ),
            moderation_enabled=_env_bool("ENABLE_MODERATION", True),
            multimodal_enabled=multimodal_enabled,
            max_image_bytes=_env_int(
                "MAX_IMAGE_BYTES",
                5 * 1024 * 1024,
                minimum=1024,
                maximum=20 * 1024 * 1024,
            ),
            enable_thinking=_env_bool("NVIDIA_ENABLE_THINKING", False),
            temperature=_env_float(
                "NVIDIA_TEMPERATURE", 1.0, minimum=0.0, maximum=2.0
            ),
            top_p=_env_float("NVIDIA_TOP_P", 0.95, minimum=0.01, maximum=1.0),
        )
