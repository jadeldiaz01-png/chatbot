from __future__ import annotations

import base64
import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class RedactionResult:
    text: str
    detected_types: tuple[str, ...]


_PATTERN_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "private_key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----.*?"
            r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    (
        "github_token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    (
        "generic_secret_assignment",
        re.compile(
            r"(?i)\b(password|passwd|api[_-]?key|secret|token)\s*[:=]\s*([^\s,;]{8,})"
        ),
    ),
)


def redact_likely_secrets(text: str) -> RedactionResult:
    redacted = text
    detected: list[str] = []
    for rule_name, pattern in _PATTERN_RULES:
        if pattern.search(redacted):
            detected.append(rule_name)
            if rule_name == "generic_secret_assignment":
                redacted = pattern.sub(lambda m: f"{m.group(1)}=[REDACTED]", redacted)
            else:
                redacted = pattern.sub("[REDACTED]", redacted)
    return RedactionResult(redacted, tuple(sorted(set(detected))))


def prune_request_timestamps(
    timestamps: Iterable[float], now: float, *, window_seconds: float = 60.0
) -> list[float]:
    floor = now - window_seconds
    return [stamp for stamp in timestamps if stamp >= floor]


def allow_session_request(
    timestamps: Iterable[float],
    now: float,
    *,
    limit: int,
    window_seconds: float = 60.0,
) -> tuple[bool, list[float]]:
    active = prune_request_timestamps(timestamps, now, window_seconds=window_seconds)
    if len(active) >= limit:
        return False, active
    active.append(now)
    return True, active


_ALLOWED_IMAGE_MIME = {"image/png", "image/jpeg", "image/webp", "image/gif"}


def image_to_data_url(data: bytes, mime_type: str, *, max_bytes: int) -> str:
    if mime_type not in _ALLOWED_IMAGE_MIME:
        raise ValueError("unsupported image type")
    if not data:
        raise ValueError("empty image")
    if len(data) > max_bytes:
        raise ValueError("image exceeds configured size limit")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
