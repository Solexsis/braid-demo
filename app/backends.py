"""Generation backends.

Two implementations behind one interface:

* `MockBackend` — deterministic, dependency-free, instant. Selected with
  `BRAID_BACKEND=mock`. Used by tests and frontend development so neither needs
  the private runtime wheel or a checkpoint.
* `RuntimeBackend` — the real model, via `idklm.runtime.BraidRuntime` from the
  `braid-runtime` wheel. Selected with `BRAID_BACKEND=runtime`.

**No architecture code lives in this repository.** The runtime backend is an
adapter and nothing more; if you find yourself writing a tensor operation here,
it belongs in `idk-lm`.
"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import Settings
from .model_source import ModelSourceError, resolve_model

log = logging.getLogger("braid.backend")

MOCK_RUNTIME_VERSION = "mock-0.1.0"


class BackendError(RuntimeError):
    """The backend could not be created or could not serve a request."""


@dataclass
class Completion:
    text: str
    prompt_bytes: int
    generated_bytes: int
    duration_s: float
    bytes_per_second: float
    seed: int
    finish_reason: str


class Backend(Protocol):
    name: str
    ready: bool

    def info(self) -> dict: ...

    def generate(
        self,
        prompt: str,
        max_new_bytes: int,
        temperature: float,
        top_k: int,
        seed: int | None,
        deadline_s: float | None = None,
    ) -> Completion: ...

    def stream(
        self,
        prompt: str,
        max_new_bytes: int,
        temperature: float,
        top_k: int,
        seed: int | None,
        deadline_s: float | None = None,
    ) -> Iterator[str]: ...


# --------------------------------------------------------------------- mock

_MOCK_VOCAB = (
    "the machine woke beneath the city and counted its own heartbeats . "
    "bytes arrive one at a time ; a hierarchy folds them into phrases , then "
    "unfolds them again . nothing here is a transformer . the convolution does "
    "the local work and attention is spent only where the sequence is short . "
    "this is a research preview , so the text means very little and the "
    "plumbing means everything . "
).split()


class MockBackend:
    """Deterministic pseudo-text, byte-budgeted exactly like the real thing.

    It is *not* a model and never pretends to be: `info()["backend"]` says
    `mock` and the API surfaces that to the client.
    """

    name = "mock"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ready = True

    def info(self) -> dict:
        return {
            "backend": self.name,
            "model_name": self.settings.model_name_override or "braid-mock",
            "model_id": "braid-mock",
            "preset": "mock",
            "parameters": 0,
            "architecture": "mock backend (no model loaded)",
            "architecture_summary": (
                "deterministic placeholder; select BRAID_BACKEND=runtime for the "
                "real byte-level hourglass"
            ),
            "context_limit_bytes": self.settings.max_prompt_bytes,
            "checkpoint_version": None,
            "checkpoint_id": None,
            "runtime_version": MOCK_RUNTIME_VERSION,
            "device": "cpu",
            "precision": "n/a",
            "research_preview": True,
            "training_bytes": None,
            "dataset": None,
            "license": None,
            "known_limitations": [
                "This deployment is running the mock backend: output is generated "
                "text-shaped noise, not model output.",
            ],
        }

    def _rng(self, prompt: str, temperature: float, top_k: int, seed: int | None) -> random.Random:
        material = f"{prompt}|{temperature}|{top_k}|{seed}".encode()
        digest = hashlib.sha256(material).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def _tokens(
        self, prompt: str, max_new_bytes: int, temperature: float, top_k: int, seed: int | None
    ) -> Iterator[str]:
        rng = self._rng(prompt, temperature, top_k, seed)
        produced = 0
        while produced < max_new_bytes:
            word = rng.choice(_MOCK_VOCAB)
            piece = (" " if produced else "") + word
            encoded = piece.encode("utf-8")
            if produced + len(encoded) > max_new_bytes:
                remaining = max_new_bytes - produced
                piece = encoded[:remaining].decode("utf-8", errors="ignore")
                if not piece:
                    break
                produced = max_new_bytes
                yield piece
                break
            produced += len(encoded)
            yield piece

    def stream(
        self,
        prompt: str,
        max_new_bytes: int,
        temperature: float,
        top_k: int,
        seed: int | None,
        deadline_s: float | None = None,
    ) -> Iterator[str]:
        started = time.perf_counter()
        for piece in self._tokens(prompt, max_new_bytes, temperature, top_k, seed):
            yield piece
            if deadline_s is not None and time.perf_counter() - started > deadline_s:
                return

    def generate(
        self,
        prompt: str,
        max_new_bytes: int,
        temperature: float,
        top_k: int,
        seed: int | None,
        deadline_s: float | None = None,
    ) -> Completion:
        started = time.perf_counter()
        text = "".join(self.stream(prompt, max_new_bytes, temperature, top_k, seed, deadline_s))
        duration = time.perf_counter() - started
        produced = len(text.encode("utf-8"))
        return Completion(
            text=text,
            prompt_bytes=len(prompt.encode("utf-8")),
            generated_bytes=produced,
            duration_s=duration,
            bytes_per_second=produced / duration if duration > 0 else 0.0,
            seed=seed if seed is not None else -1,
            finish_reason="length" if produced >= max_new_bytes else "stopped",
        )


# ------------------------------------------------------------------ runtime


class RuntimeBackend:
    """Adapter over `idklm.runtime.BraidRuntime`. Contains no model code."""

    name = "runtime"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ready = False
        try:
            from idklm.runtime import RUNTIME_VERSION, BraidRuntime
        except ImportError as exc:  # pragma: no cover - depends on install
            raise BackendError(
                "BRAID_BACKEND=runtime needs the private `braid-runtime` wheel. "
                "Install it with `uv pip install braid_runtime-<version>-py3-none-any.whl`, "
                "or run with BRAID_BACKEND=mock."
            ) from exc

        expected = settings.runtime_version
        if expected and expected != RUNTIME_VERSION:
            raise BackendError(
                f"runtime version mismatch: BRAID_RUNTIME_VERSION={expected} but "
                f"the installed wheel provides {RUNTIME_VERSION}"
            )
        try:
            path = resolve_model(
                model_path=settings.model_path,
                model_url=settings.model_url,
                model_sha256=settings.model_sha256,
                cache_dir=settings.model_cache,
            )
        except ModelSourceError as exc:
            raise BackendError(str(exc)) from exc

        try:
            self.runtime = BraidRuntime.load(
                path,
                device=settings.device,
                expect_preset=settings.preset or None,
            )
        except Exception as exc:  # runtime raises its own CheckpointError family
            raise BackendError(f"could not load the model: {exc}") from exc
        self.runtime_version = RUNTIME_VERSION
        self.model_path = Path(path)
        self.ready = True
        log.info(
            "model loaded",
            extra={"path": str(path), "device": self.runtime.device},
        )

    def info(self) -> dict:
        info = dict(self.runtime.model_info())
        info["backend"] = self.name
        if self.settings.model_name_override:
            info["model_name"] = self.settings.model_name_override
        info.pop("torch_version", None)  # not useful publicly, and it is a version leak
        info.pop("trainable_parameters", None)
        info.pop("non_embedding_parameters", None)
        return info

    def warmup(self) -> None:
        try:
            self.runtime.warmup()
        except Exception:  # pragma: no cover - warmup must never break startup
            log.warning("warmup failed", exc_info=True)

    def stream(
        self,
        prompt: str,
        max_new_bytes: int,
        temperature: float,
        top_k: int,
        seed: int | None,
        deadline_s: float | None = None,
    ) -> Iterator[str]:
        return self.runtime.generate_stream(
            prompt,
            max_new_bytes=max_new_bytes,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            deadline_s=deadline_s,
        )

    def generate(
        self,
        prompt: str,
        max_new_bytes: int,
        temperature: float,
        top_k: int,
        seed: int | None,
        deadline_s: float | None = None,
    ) -> Completion:
        result = self.runtime.generate(
            prompt,
            max_new_bytes=max_new_bytes,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            deadline_s=deadline_s,
        )
        return Completion(
            text=result.text,
            prompt_bytes=result.prompt_bytes,
            generated_bytes=result.generated_bytes,
            duration_s=result.duration_s,
            bytes_per_second=result.bytes_per_second,
            seed=result.seed,
            finish_reason=result.finish_reason,
        )


def create_backend(settings: Settings) -> Backend:
    if settings.is_mock:
        log.warning("starting with the MOCK backend: responses are not model output")
        return MockBackend(settings)
    backend = RuntimeBackend(settings)
    backend.warmup()
    return backend
