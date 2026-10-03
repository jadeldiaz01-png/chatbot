from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable

from openai import OpenAI

OPENAI_PROVIDER = "openai"
NVIDIA_PROVIDER = "nvidia"
SUPPORTED_PROVIDERS = {OPENAI_PROVIDER, NVIDIA_PROVIDER}

DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

NVIDIA_MAX_TOKENS = 128
NVIDIA_TEMPERATURE = 1.0
NVIDIA_TOP_P = 0.95
NVIDIA_ENABLE_THINKING = False
NVIDIA_STREAM = False
NVIDIA_TIMEOUT_SECONDS = 45.0
NVIDIA_MAX_RETRIES = 2

GENERIC_UNAVAILABLE_MESSAGE = (
    "El servicio de IA no está disponible temporalmente. "
    "No se realizó ninguna acción externa."
)
SENSITIVE_INPUT_MESSAGE = (
    "Por privacidad, elimina contraseñas, claves, datos de pago, documentos de identidad, "
    "correos, teléfonos u otros datos personales sensibles antes de continuar."
)


@dataclass(frozen=True)
class SensitiveFinding:
    category: str


@dataclass(frozen=True)
class RuntimeConfig:
    provider: str
    model: str
    api_key: str
    base_url: str | None = None


def _normalize_provider(value: str) -> str:
    provider = value.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"unsupported CHATBOT_PROVIDER: {provider!r}")
    return provider


def runtime_config_from_env(env: dict[str, str] | None = None) -> RuntimeConfig:
    values = os.environ if env is None else env
    provider = _normalize_provider(values.get("CHATBOT_PROVIDER", OPENAI_PROVIDER))

    if provider == OPENAI_PROVIDER:
        return RuntimeConfig(
            provider=provider,
            model=values.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip()
            or DEFAULT_OPENAI_MODEL,
            api_key=values.get("OPENAI_API_KEY", "").strip(),
        )

    base_url = values.get("NVIDIA_API_BASE_URL", DEFAULT_NVIDIA_BASE_URL).rstrip("/")
    if base_url != DEFAULT_NVIDIA_BASE_URL:
        raise ValueError("unexpected NVIDIA_API_BASE_URL")
    return RuntimeConfig(
        provider=provider,
        model=values.get("NVIDIA_MODEL", DEFAULT_NVIDIA_MODEL).strip()
        or DEFAULT_NVIDIA_MODEL,
        api_key=values.get("NVIDIA_API_KEY", "").strip(),
        base_url=base_url,
    )


_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    re.IGNORECASE,
)
_LABELED_PASSWORD_RE = re.compile(
    r"(?i)\b(?:password|contrase(?:ñ|n)a|contrasena|clave)\s*[:=]\s*[^\s,;]{4,}"
)
_LABELED_SECRET_RE = re.compile(
    r"(?i)\b(?:api[_ -]?key|token|secret|clave[_ -]?api)\s*[:=]\s*[A-Za-z0-9_./+\-=]{12,}"
)
_TOKEN_PREFIX_RE = re.compile(
    r"(?i)\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|nvapi-[A-Za-z0-9_-]{16,})\b"
)
_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_LABELED_PASSPORT_RE = re.compile(
    r"(?i)\b(?:passport|pasaporte)\s*[:=#-]?\s*[A-Z0-9]{6,12}\b"
)
_LABELED_RECOVERY_RE = re.compile(
    r"(?i)\b(?:recovery\s*code|codigo\s*de\s*recuperacion)\s*[:=]\s*[A-Z0-9-]{6,}"
)
_DOMINICAN_ID_RE = re.compile(r"(?<!\d)\d{3}-?\d{7}-?\d(?!\d)")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[ .-]?)?(?:\(?\d{2,4}\)?[ .-]?)"
    r"\d{3}[ .-]?\d{4}(?!\w)"
)
_CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def _luhn_valid(raw: str) -> bool:
    digits = [int(ch) for ch in raw if ch.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        value = digit
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def detect_sensitive_input(text: str) -> tuple[SensitiveFinding, ...]:
    findings: list[SensitiveFinding] = []

    def add(category: str) -> None:
        if all(item.category != category for item in findings):
            findings.append(SensitiveFinding(category=category))

    if _PRIVATE_KEY_RE.search(text):
        add("private_key")
    if _LABELED_PASSWORD_RE.search(text):
        add("password")
    if _LABELED_SECRET_RE.search(text) or _TOKEN_PREFIX_RE.search(text):
        add("api_secret")
    if _EMAIL_RE.search(text):
        add("email")
    if _LABELED_PASSPORT_RE.search(text) or _DOMINICAN_ID_RE.search(text):
        add("government_id")
    if _LABELED_RECOVERY_RE.search(text):
        add("recovery_code")
    if _PHONE_RE.search(text):
        add("phone")
    if any(_luhn_valid(match.group(0)) for match in _CARD_CANDIDATE_RE.finditer(text)):
        add("payment_card")

    return tuple(findings)


def validate_messages_for_provider(messages: Iterable[dict[str, str]]) -> None:
    categories: set[str] = set()
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        categories.update(item.category for item in detect_sensitive_input(content))
    if categories:
        joined = ",".join(sorted(categories))
        raise ValueError(f"sensitive_input_blocked:{joined}")


def build_nvidia_request(
    *,
    model: str,
    system_instructions: str,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_instructions},
            *messages,
        ],
        "max_tokens": NVIDIA_MAX_TOKENS,
        "temperature": NVIDIA_TEMPERATURE,
        "top_p": NVIDIA_TOP_P,
        "stream": NVIDIA_STREAM,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": NVIDIA_ENABLE_THINKING,
            }
        },
    }


class ChatRuntime:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        if not config.api_key:
            raise ValueError(f"{config.provider}_api_key_missing")

        if config.provider == NVIDIA_PROVIDER:
            self.client = OpenAI(
                api_key=config.api_key,
                base_url=config.base_url,
                timeout=NVIDIA_TIMEOUT_SECONDS,
                max_retries=NVIDIA_MAX_RETRIES,
            )
        else:
            self.client = OpenAI(api_key=config.api_key)

    def generate(
        self,
        *,
        system_instructions: str,
        messages: list[dict[str, str]],
    ) -> str:
        validate_messages_for_provider(messages)

        if self.config.provider == NVIDIA_PROVIDER:
            completion = self.client.chat.completions.create(
                **build_nvidia_request(
                    model=self.config.model,
                    system_instructions=system_instructions,
                    messages=messages,
                )
            )
            choices = getattr(completion, "choices", None)
            message = getattr(choices[0], "message", None) if choices else None
            content = getattr(message, "content", "") if message is not None else ""
            return content.strip() if isinstance(content, str) else ""

        response = self.client.responses.create(
            model=self.config.model,
            instructions=system_instructions,
            input=messages,
        )
        content = getattr(response, "output_text", "")
        return content.strip() if isinstance(content, str) else ""

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
