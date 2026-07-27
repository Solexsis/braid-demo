"""Request/response models. Validation lives here so handlers stay boring."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(default="", description="UTF-8 text to continue")
    max_new_bytes: int | None = Field(
        default=None,
        ge=1,
        description="bytes to generate; server clamps to its configured maximum",
    )
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0, le=256)
    seed: int | None = Field(
        default=None,
        ge=0,
        le=2**53 - 1,
        description="omit for a fresh sample, set for a reproducible one",
    )


class CompletionResponse(BaseModel):
    text: str
    prompt_bytes: int
    generated_bytes: int
    duration_s: float
    bytes_per_second: float
    model: str
    backend: str
    seed: int
    finish_reason: str
    truncated_prompt: bool = False


class ModelResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    backend: str
    parameters: int
    architecture: str
    architecture_summary: str
    context_limit_bytes: int
    max_prompt_bytes: int
    max_new_bytes: int
    checkpoint_version: str | None = None
    checkpoint_id: str | None = None
    runtime_version: str
    device: str
    precision: str | None = None
    research_preview: bool = True
    preset: str | None = None
    training_bytes: int | None = None
    dataset: str | None = None
    license: str | None = None
    known_limitations: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    backend: str
    uptime_s: float
    in_flight: int
    version: str


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
    request_id: str | None = None


def error_payload(
    error: str, detail: str | None = None, request_id: str | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {"error": error}
    if detail:
        payload["detail"] = detail
    if request_id:
        payload["request_id"] = request_id
    return payload
