"""Environment-driven settings for the Braid demo server.

Everything is read once at import of `get_settings()` and cached, so a request
handler can never observe a half-changed configuration. Nothing here has a
default that reaches the network or a private repository.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

BACKEND_MOCK = "mock"
BACKEND_RUNTIME = "runtime"
BACKENDS = (BACKEND_MOCK, BACKEND_RUNTIME)


def _int(name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return value


def _float(name: str, default: float, *, minimum: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value <= minimum:
        raise ValueError(f"{name} must be > {minimum}, got {value}")
    return value


def _str(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _bool(name: str, default: bool = False) -> bool:
    raw = _str(name)
    if not raw:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    backend: str = BACKEND_MOCK

    # Model source. Exactly one of path/url is needed for the runtime backend.
    model_path: str = ""
    model_url: str = ""
    model_sha256: str = ""
    model_cache: Path = field(default_factory=lambda: Path("/var/cache/braid"))
    device: str = "auto"
    preset: str = ""  # optional: assert the checkpoint matches this preset

    # Request limits.
    max_prompt_bytes: int = 4096
    max_new_bytes: int = 1024
    default_new_bytes: int = 256
    max_concurrent: int = 1
    queue_limit: int = 8
    request_timeout: float = 60.0

    # Rate limiting (per client IP). Set requests to 0 to disable entirely and
    # let a reverse proxy own it.
    rate_limit_requests: int = 20
    rate_limit_window: float = 60.0

    # Presentation / provenance.
    runtime_version: str = ""
    model_name_override: str = ""
    site_url: str = "https://solexsis.ai"
    repo_url: str = "https://github.com/SolexsisAI/braid-demo"
    log_level: str = "INFO"
    trust_forwarded_for: bool = False

    @property
    def is_mock(self) -> bool:
        return self.backend == BACKEND_MOCK


def load_settings() -> Settings:
    backend = _str("BRAID_BACKEND", BACKEND_MOCK).lower()
    if backend not in BACKENDS:
        raise ValueError(f"BRAID_BACKEND must be one of {BACKENDS}, got {backend!r}")
    settings = Settings(
        backend=backend,
        model_path=_str("BRAID_MODEL_PATH"),
        model_url=_str("BRAID_MODEL_URL"),
        model_sha256=_str("BRAID_MODEL_SHA256").lower(),
        model_cache=Path(_str("BRAID_MODEL_CACHE", "/var/cache/braid")),
        device=_str("BRAID_DEVICE", "auto"),
        preset=_str("BRAID_PRESET"),
        max_prompt_bytes=_int("BRAID_MAX_PROMPT_BYTES", 4096, maximum=1 << 20),
        max_new_bytes=_int("BRAID_MAX_NEW_BYTES", 1024, maximum=1 << 16),
        default_new_bytes=_int("BRAID_DEFAULT_NEW_BYTES", 256),
        max_concurrent=_int("BRAID_MAX_CONCURRENT", 1, maximum=64),
        queue_limit=_int("BRAID_QUEUE_LIMIT", 8, minimum=0, maximum=1024),
        request_timeout=_float("BRAID_REQUEST_TIMEOUT", 60.0),
        rate_limit_requests=_int("BRAID_RATE_LIMIT_REQUESTS", 20, minimum=0),
        rate_limit_window=_float("BRAID_RATE_LIMIT_WINDOW", 60.0),
        runtime_version=_str("BRAID_RUNTIME_VERSION"),
        model_name_override=_str("BRAID_MODEL_NAME"),
        site_url=_str("BRAID_SITE_URL", "https://solexsis.ai"),
        repo_url=_str("BRAID_REPO_URL", "https://github.com/SolexsisAI/braid-demo"),
        log_level=_str("BRAID_LOG_LEVEL", "INFO").upper(),
        trust_forwarded_for=_bool("BRAID_TRUST_FORWARDED_FOR", False),
    )
    if settings.default_new_bytes > settings.max_new_bytes:
        # Lowering only the ceiling is the common case; clamp rather than refuse
        # to boot over a default the operator never set.
        settings = replace(settings, default_new_bytes=settings.max_new_bytes)
    if settings.backend == BACKEND_RUNTIME and not (settings.model_path or settings.model_url):
        raise ValueError("BRAID_BACKEND=runtime needs BRAID_MODEL_PATH or BRAID_MODEL_URL")
    if settings.model_url and not settings.model_sha256:
        raise ValueError(
            "BRAID_MODEL_URL requires BRAID_MODEL_SHA256 — downloads are verified "
            "before they are loaded, and fail closed"
        )
    if settings.model_sha256 and len(settings.model_sha256) != 64:
        raise ValueError("BRAID_MODEL_SHA256 must be a 64-character hex digest")
    return settings


_cached: Settings | None = None


def get_settings(refresh: bool = False) -> Settings:
    global _cached
    if _cached is None or refresh:
        _cached = load_settings()
    return _cached
