"""API contract tests. All of these run against the mock backend on CPU."""

from __future__ import annotations

import json

import pytest

# ------------------------------------------------------------------- health


def test_health_reports_a_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["backend"] == "mock"
    assert body["uptime_s"] >= 0
    assert body["in_flight"] == 0
    assert body["version"]


def test_healthz_alias(client):
    assert client.get("/healthz").status_code == 200


# -------------------------------------------------------------------- model


def test_model_metadata_is_public_and_complete(client):
    body = client.get("/v1/model").json()
    for key in (
        "model_name",
        "backend",
        "parameters",
        "architecture",
        "architecture_summary",
        "context_limit_bytes",
        "max_prompt_bytes",
        "max_new_bytes",
        "runtime_version",
        "device",
        "research_preview",
        "known_limitations",
    ):
        assert key in body, key
    assert body["research_preview"] is True
    assert body["backend"] == "mock"


def test_mock_backend_declares_itself(client):
    body = client.get("/v1/model").json()
    assert any("mock" in item.lower() for item in body["known_limitations"])


# -------------------------------------------------------------- completions


def test_completion_returns_the_documented_fields(client):
    response = client.post(
        "/v1/completions",
        json={
            "prompt": "Hello",
            "max_new_bytes": 64,
            "temperature": 0.8,
            "top_k": 50,
            "seed": 1234,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"]
    assert body["prompt_bytes"] == 5
    assert 0 < body["generated_bytes"] <= 64
    assert body["duration_s"] >= 0
    assert body["bytes_per_second"] >= 0
    assert body["model"]
    assert body["seed"] == 1234
    assert body["backend"] == "mock"
    assert body["finish_reason"] in ("length", "stopped")


def test_completion_is_deterministic_for_a_fixed_seed(client):
    payload = {"prompt": "Hello", "max_new_bytes": 48, "seed": 7}
    first = client.post("/v1/completions", json=payload).json()["text"]
    second = client.post("/v1/completions", json=payload).json()["text"]
    assert first == second
    other = client.post("/v1/completions", json={**payload, "seed": 8}).json()["text"]
    assert other != first


def test_completion_respects_the_byte_budget(client):
    body = client.post("/v1/completions", json={"prompt": "x", "max_new_bytes": 24}).json()
    assert len(body["text"].encode("utf-8")) <= 24


def test_defaults_apply_when_fields_are_omitted(client):
    body = client.post("/v1/completions", json={"prompt": "hi"}).json()
    assert body["generated_bytes"] > 0


# --------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": "hi", "temperature": 0},
        {"prompt": "hi", "temperature": 5},
        {"prompt": "hi", "top_k": -1},
        {"prompt": "hi", "top_k": 9999},
        {"prompt": "hi", "max_new_bytes": 0},
        {"prompt": "hi", "seed": -5},
        {"prompt": "hi", "unexpected_field": True},
        {"prompt": 12},
    ],
)
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/v1/completions", json=payload).status_code == 422


def test_oversized_prompt_is_truncated_not_rejected(make_client):
    with make_client(BRAID_MAX_PROMPT_BYTES=64) as client:
        body = client.post(
            "/v1/completions",
            json={
                "prompt": "z" * 500,
                "max_new_bytes": 16,
            },
        ).json()
        assert body["truncated_prompt"] is True
        assert body["prompt_bytes"] <= 64


def test_max_new_bytes_is_clamped_to_the_server_limit(make_client):
    with make_client(BRAID_MAX_NEW_BYTES=32) as client:
        body = client.post("/v1/completions", json={"prompt": "hi", "max_new_bytes": 100000}).json()
        assert body["generated_bytes"] <= 32
        assert client.get("/v1/model").json()["max_new_bytes"] == 32


# ------------------------------------------------------------------ streaming


def _sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        event = "message"
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if data:
            events.append((event, json.loads("\n".join(data))))
    return events


def test_streaming_emits_start_chunks_and_done(client):
    with client.stream(
        "POST",
        "/v1/completions/stream",
        json={
            "prompt": "Hello",
            "max_new_bytes": 64,
            "seed": 3,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    events = _sse_events(body)
    kinds = [kind for kind, _ in events]
    assert kinds[0] == "start"
    assert kinds[-1] == "done"
    assert "chunk" in kinds
    done = events[-1][1]
    assert done["generated_bytes"] > 0
    assert done["duration_s"] >= 0


def test_streamed_text_equals_non_streamed_text(client):
    payload = {"prompt": "A strange machine", "max_new_bytes": 96, "seed": 11}
    whole = client.post("/v1/completions", json=payload).json()["text"]
    with client.stream("POST", "/v1/completions/stream", json=payload) as response:
        body = "".join(response.iter_text())
    streamed = "".join(data["text"] for kind, data in _sse_events(body) if kind == "chunk")
    assert streamed == whole


def test_ndjson_streaming_format(client):
    with client.stream(
        "POST",
        "/v1/completions/stream?format=ndjson",
        json={
            "prompt": "Hi",
            "max_new_bytes": 32,
            "seed": 4,
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/x-ndjson")
        body = "".join(response.iter_text())
    records = [json.loads(line) for line in body.splitlines() if line.strip()]
    assert records[0]["event"] == "start"
    assert records[-1]["event"] == "done"


def test_streaming_rejects_an_unknown_format(client):
    response = client.post("/v1/completions/stream?format=xml", json={"prompt": "hi"})
    assert response.status_code == 422


# ----------------------------------------------------------------- limits


def test_rate_limiting_returns_429_with_retry_after(make_client):
    with make_client(BRAID_RATE_LIMIT_REQUESTS=2, BRAID_RATE_LIMIT_WINDOW=60) as client:
        payload = {"prompt": "hi", "max_new_bytes": 8}
        assert client.post("/v1/completions", json=payload).status_code == 200
        assert client.post("/v1/completions", json=payload).status_code == 200
        blocked = client.post("/v1/completions", json=payload)
        assert blocked.status_code == 429
        assert blocked.headers.get("retry-after")
        assert blocked.json()["error"] == "rate_limited"


def test_rate_limiting_can_be_disabled(make_client):
    with make_client(BRAID_RATE_LIMIT_REQUESTS=0) as client:
        for _ in range(5):
            assert (
                client.post(
                    "/v1/completions", json={"prompt": "hi", "max_new_bytes": 8}
                ).status_code
                == 200
            )


# ------------------------------------------------------------- benchmarks


def test_benchmarks_endpoint_serves_the_checked_in_data(client):
    body = client.get("/v1/benchmarks").json()
    assert body["schema_version"] == 1
    assert body["sections"]
    assert body["global_caveats"]


def test_benchmark_pending_rows_carry_no_numbers(client):
    body = client.get("/v1/benchmarks").json()
    pending = [s for s in body["sections"] if s["status"] == "pending"]
    assert pending, "the 300M section must be present and marked pending"
    for section in pending:
        for row in section["rows"]:
            assert row["val_bits_per_byte"] is None
            assert row["train_bytes_per_second"] is None


# ------------------------------------------------------------------ pages


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Braid" in response.text
    assert "Research preview" in response.text
    assert "Solexsis" in response.text


def test_benchmarks_page_is_served(client):
    response = client.get("/benchmarks")
    assert response.status_code == 200
    assert "Measured results" in response.text


def test_static_assets_are_served(client):
    for path in ("/static/style.css", "/static/app.js", "/static/benchmarks.js"):
        assert client.get(path).status_code == 200, path


def test_robots_disallows_the_api(client):
    assert "Disallow: /v1/" in client.get("/robots.txt").text


def test_unknown_api_path_returns_json_404(client):
    response = client.get("/v1/nope")
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


# ------------------------------------------------------------ error hygiene


def test_errors_never_expose_internals(client, monkeypatch):
    from app import main

    state = main.app.state.braid

    class Exploding:
        name = "mock"
        ready = True

        def info(self):
            return {"model_name": "braid-mock", "backend": "mock"}

        def generate(self, *args, **kwargs):
            raise RuntimeError("/secret/path/to/weights.pt exploded")

        def stream(self, *args, **kwargs):
            raise RuntimeError("/secret/path/to/weights.pt exploded")

    monkeypatch.setattr(state, "backend", Exploding())
    response = client.post("/v1/completions", json={"prompt": "hi"})
    assert response.status_code == 500
    body = response.text
    assert "secret" not in body
    assert "Traceback" not in body
    assert response.json()["error"] == "generation_failed"


def test_request_id_is_echoed(client):
    response = client.get("/health", headers={"x-request-id": "abc123"})
    assert response.headers["x-request-id"] == "abc123"
