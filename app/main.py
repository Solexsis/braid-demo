"""Braid research-preview demo API.

Endpoints:

    GET  /health                 liveness + whether the model is loaded
    GET  /v1/model               safe public metadata
    POST /v1/completions         one completion
    POST /v1/completions/stream  server-sent events (`?format=ndjson` also works)
    GET  /v1/benchmarks          the checked-in measured results
    GET  /                       the web UI

Nothing in this file knows how Braid works. It validates, admits, delegates to a
backend, and formats. Internal exceptions never reach the client verbatim.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from .backends import Backend, BackendError, create_backend
from .config import Settings, get_settings
from .limits import ConcurrencyGate, Overloaded, RateLimiter
from .schemas import (
    CompletionRequest,
    CompletionResponse,
    HealthResponse,
    ModelResponse,
    error_payload,
)

APP_VERSION = "0.1.0"
STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR = Path(__file__).parent / "data"

log = logging.getLogger("braid.api")


def configure_logging(level: str = "INFO") -> None:
    """Structured (JSON-per-line) logs on stdout; no secrets, no prompts."""

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            for key, value in record.__dict__.items():
                if key in (
                    "args",
                    "msg",
                    "levelname",
                    "levelno",
                    "pathname",
                    "filename",
                    "module",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "lineno",
                    "funcName",
                    "created",
                    "msecs",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "processName",
                    "process",
                    "name",
                    "taskName",
                    "message",
                ):
                    continue
                payload[key] = value
            if record.exc_info:
                payload["exc"] = self.formatException(record.exc_info)
            return json.dumps(payload, default=str)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level, logging.INFO))


class AppState:
    """Everything mutable, in one place, owned by the lifespan."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend: Backend | None = None
        self.backend_error: str | None = None
        self.started_at = time.monotonic()
        self.gate = ConcurrencyGate(settings.max_concurrent, settings.queue_limit)
        self.rate_limiter = RateLimiter(settings.rate_limit_requests, settings.rate_limit_window)
        self.shutting_down = False


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    state = AppState(settings)
    app.state.braid = state
    log.info(
        "starting",
        extra={"backend": settings.backend, "device": settings.device, "version": APP_VERSION},
    )
    try:
        # Loading can take tens of seconds on a large checkpoint; keep the event
        # loop responsive so the health endpoint answers during startup.
        state.backend = await asyncio.to_thread(create_backend, settings)
    except BackendError as exc:
        state.backend_error = str(exc)
        log.error("backend unavailable", extra={"reason": str(exc)})
    except Exception as exc:  # pragma: no cover - defensive
        state.backend_error = "backend failed to start"
        log.exception("unexpected backend failure: %s", exc)
    try:
        yield
    finally:
        state.shutting_down = True
        # Let in-flight generations finish before the process exits.
        deadline = time.monotonic() + min(30.0, settings.request_timeout)
        while state.gate.in_flight and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        log.info("shutdown complete", extra={"in_flight": state.gate.in_flight})


app = FastAPI(
    title="Braid research preview",
    version=APP_VERSION,
    description=(
        "Public demo API for Braid, a tokenizer-free byte-level hourglass "
        "language model. Research preview."
    ),
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)


def state_of(request: Request) -> AppState:
    return request.app.state.braid


def client_key(request: Request, settings: Settings) -> str:
    if settings.trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # Never leak a traceback or a filesystem path to a client.
        log.exception("unhandled error", extra={"request_id": request_id, "path": request.url.path})
        return JSONResponse(
            status_code=500,
            content=error_payload(
                "internal_error", "the server failed to handle this request", request_id
            ),
            headers={"x-request-id": request_id},
        )
    response.headers["x-request-id"] = request_id
    log.info(
        "request",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )
    return response


# ------------------------------------------------------------------- health


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    state = state_of(request)
    loaded = bool(state.backend and getattr(state.backend, "ready", False))
    return HealthResponse(
        status="ok" if loaded and not state.shutting_down else "degraded",
        model_loaded=loaded,
        backend=state.settings.backend,
        uptime_s=round(time.monotonic() - state.started_at, 3),
        in_flight=state.gate.in_flight,
        version=APP_VERSION,
    )


@app.get("/healthz", include_in_schema=False)
async def healthz(request: Request) -> HealthResponse:
    return await health(request)


# -------------------------------------------------------------------- model


def _require_backend(state: AppState) -> Backend | JSONResponse:
    if state.backend is None:
        return JSONResponse(
            status_code=503,
            content=error_payload(
                "model_unavailable",
                state.backend_error or "the model is not loaded",
            ),
        )
    return state.backend


@app.get("/v1/model", response_model=ModelResponse)
async def model_info(request: Request):
    state = state_of(request)
    backend = _require_backend(state)
    if isinstance(backend, JSONResponse):
        return backend
    info = backend.info()
    settings = state.settings
    return ModelResponse(
        model_name=info.get("model_name", "braid"),
        backend=info.get("backend", settings.backend),
        parameters=int(info.get("parameters") or 0),
        architecture=info.get("architecture", "unknown"),
        architecture_summary=info.get("architecture_summary", ""),
        context_limit_bytes=int(info.get("context_limit_bytes") or 0),
        max_prompt_bytes=settings.max_prompt_bytes,
        max_new_bytes=settings.max_new_bytes,
        checkpoint_version=info.get("checkpoint_version"),
        checkpoint_id=info.get("checkpoint_id"),
        runtime_version=str(info.get("runtime_version", "unknown")),
        device=str(info.get("device", "unknown")),
        precision=info.get("precision"),
        research_preview=True,
        preset=info.get("preset"),
        training_bytes=info.get("training_bytes"),
        dataset=info.get("dataset"),
        license=info.get("license"),
        known_limitations=list(info.get("known_limitations") or []),
    )


# --------------------------------------------------------------- completions


def _resolve_request(state: AppState, body: CompletionRequest) -> tuple[str, int, bool]:
    """Clamp a validated request to the server's configured limits."""
    settings = state.settings
    raw = body.prompt.encode("utf-8")
    truncated = False
    if len(raw) > settings.max_prompt_bytes:
        raw = raw[: settings.max_prompt_bytes]
        truncated = True
    prompt = raw.decode("utf-8", errors="ignore")
    requested = body.max_new_bytes or settings.default_new_bytes
    return prompt, min(requested, settings.max_new_bytes), truncated


def _admit(state: AppState, request: Request) -> JSONResponse | None:
    if state.shutting_down:
        return JSONResponse(
            status_code=503,
            content=error_payload("shutting_down", "the server is shutting down"),
            headers={"retry-after": "10"},
        )
    try:
        state.rate_limiter.check(client_key(request, state.settings))
    except Overloaded as exc:
        return JSONResponse(
            status_code=429,
            content=error_payload("rate_limited", str(exc)),
            headers={"retry-after": str(exc.retry_after)},
        )
    return None


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: Request, body: CompletionRequest):
    state = state_of(request)
    backend = _require_backend(state)
    if isinstance(backend, JSONResponse):
        return backend
    rejected = _admit(state, request)
    if rejected is not None:
        return rejected

    prompt, max_new_bytes, truncated = _resolve_request(state, body)
    timeout = state.settings.request_timeout
    try:
        async with state.gate:
            completion = await asyncio.wait_for(
                asyncio.to_thread(
                    backend.generate,
                    prompt,
                    max_new_bytes,
                    body.temperature,
                    body.top_k,
                    body.seed,
                    timeout,
                ),
                timeout=timeout + 5.0,
            )
    except Overloaded as exc:
        return JSONResponse(
            status_code=429,
            content=error_payload("overloaded", str(exc)),
            headers={"retry-after": str(exc.retry_after)},
        )
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content=error_payload("timeout", f"generation exceeded {timeout:.0f}s"),
        )
    except Exception:
        log.exception(
            "generation failed", extra={"request_id": getattr(request.state, "request_id", None)}
        )
        return JSONResponse(
            status_code=500,
            content=error_payload("generation_failed", "the model failed to produce a completion"),
        )

    info = backend.info()
    return CompletionResponse(
        text=completion.text,
        prompt_bytes=completion.prompt_bytes,
        generated_bytes=completion.generated_bytes,
        duration_s=round(completion.duration_s, 4),
        bytes_per_second=round(completion.bytes_per_second, 2),
        model=info.get("model_name", "braid"),
        backend=info.get("backend", state.settings.backend),
        seed=completion.seed,
        finish_reason=completion.finish_reason,
        truncated_prompt=truncated,
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _ndjson(event: str, data: dict[str, Any]) -> str:
    return json.dumps({"event": event, **data}) + "\n"


@app.post("/v1/completions/stream")
async def completions_stream(
    request: Request,
    body: CompletionRequest,
    format: str = Query("sse", pattern="^(sse|ndjson)$"),
):
    state = state_of(request)
    backend = _require_backend(state)
    if isinstance(backend, JSONResponse):
        return backend
    rejected = _admit(state, request)
    if rejected is not None:
        return rejected

    prompt, max_new_bytes, truncated = _resolve_request(state, body)
    encode = _sse if format == "sse" else _ndjson
    media = "text/event-stream" if format == "sse" else "application/x-ndjson"
    timeout = state.settings.request_timeout
    info = backend.info()

    async def produce() -> AsyncIterator[str]:
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=256)
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        produced = 0

        def worker() -> None:
            try:
                for chunk in backend.stream(
                    prompt, max_new_bytes, body.temperature, body.top_k, body.seed, timeout
                ):
                    asyncio.run_coroutine_threadsafe(queue.put(("chunk", chunk)), loop).result()
                asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop).result()
            except Exception as exc:  # surfaced as an error event, not a traceback
                log.exception("stream failed")
                asyncio.run_coroutine_threadsafe(
                    queue.put(("error", type(exc).__name__)), loop
                ).result()

        try:
            async with state.gate:
                yield encode(
                    "start",
                    {
                        "model": info.get("model_name", "braid"),
                        "backend": info.get("backend", state.settings.backend),
                        "max_new_bytes": max_new_bytes,
                        "truncated_prompt": truncated,
                        "seed": body.seed if body.seed is not None else -1,
                    },
                )
                task = loop.run_in_executor(None, worker)
                while True:
                    try:
                        kind, payload = await asyncio.wait_for(queue.get(), timeout=timeout + 5.0)
                    except asyncio.TimeoutError:
                        yield encode("error", {"error": "timeout"})
                        break
                    if kind == "chunk":
                        produced += len(payload.encode("utf-8"))
                        yield encode("chunk", {"text": payload})
                        if await request.is_disconnected():
                            break
                        continue
                    if kind == "error":
                        yield encode("error", {"error": "generation_failed"})
                        break
                    elapsed = time.perf_counter() - started
                    yield encode(
                        "done",
                        {
                            "generated_bytes": produced,
                            "duration_s": round(elapsed, 4),
                            "bytes_per_second": round(produced / elapsed, 2) if elapsed else 0,
                            "finish_reason": "length" if produced >= max_new_bytes else "stopped",
                        },
                    )
                    break
                await asyncio.wrap_future(task) if hasattr(task, "result") else None
        except Overloaded as exc:
            yield encode("error", {"error": "overloaded", "detail": str(exc)})

    return StreamingResponse(
        produce(),
        media_type=media,
        headers={
            "cache-control": "no-cache, no-transform",
            "x-accel-buffering": "no",  # nginx: do not buffer the stream
            "connection": "keep-alive",
        },
    )


# --------------------------------------------------------------- benchmarks


@app.get("/v1/benchmarks")
async def benchmarks() -> JSONResponse:
    path = DATA_DIR / "benchmarks.json"
    if not path.exists():  # pragma: no cover - shipped with the repo
        return JSONResponse(
            status_code=404, content=error_payload("not_found", "no benchmark data is published")
        )
    return JSONResponse(content=json.loads(path.read_text()))


# ---------------------------------------------------------------- static UI


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/benchmarks", include_in_schema=False)
async def benchmarks_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "benchmarks.html")


@app.get("/robots.txt", include_in_schema=False)
async def robots() -> PlainTextResponse:
    return PlainTextResponse("User-agent: *\nDisallow: /v1/\n")


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(404)
async def not_found(request: Request, exc: Exception) -> Response:
    if request.url.path.startswith("/v1/") or request.url.path.startswith("/api/"):
        return JSONResponse(status_code=404, content=error_payload("not_found", "unknown endpoint"))
    return FileResponse(STATIC_DIR / "index.html", status_code=404)


def run() -> None:  # pragma: no cover - production entry point
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=os.environ.get("BRAID_HOST", "0.0.0.0"),
        port=int(os.environ.get("BRAID_PORT", "8000")),
        log_config=None,
        access_log=False,
        timeout_graceful_shutdown=30,
        workers=1,  # one process owns the model
        proxy_headers=settings.trust_forwarded_for,
    )


if __name__ == "__main__":  # pragma: no cover
    run()
