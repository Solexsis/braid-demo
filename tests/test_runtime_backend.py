"""Optional integration tests against the REAL runtime wheel.

Skipped entirely when `braid-runtime` is not installed, so public CI — which has
no access to the private repository — stays green. Install the wheel locally to
run them:

    uv pip install wheels/braid_runtime-0.1.0-py3-none-any.whl
    uv run pytest tests/test_runtime_backend.py -q
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("idklm.runtime", reason="braid-runtime wheel is not installed")

from idklm.package import build_package  # noqa: E402


@pytest.fixture(scope="module")
def tiny_package(tmp_path_factory):
    root = tmp_path_factory.mktemp("model")
    return build_package(root / "braid-ci-tiny", preset="ci-tiny", random_init=True, seed=1337)


@pytest.fixture
def runtime_client(make_client, tiny_package):
    with make_client(
        BRAID_BACKEND="runtime",
        BRAID_MODEL_PATH=str(tiny_package),
        BRAID_DEVICE="cpu",
        BRAID_PRESET="ci-tiny",
        BRAID_MAX_NEW_BYTES="128",
        BRAID_RATE_LIMIT_REQUESTS="0",
    ) as client:
        yield client


def test_real_model_loads_and_reports_itself(runtime_client):
    health = runtime_client.get("/health").json()
    assert health["model_loaded"] is True
    assert health["backend"] == "runtime"

    info = runtime_client.get("/v1/model").json()
    assert info["backend"] == "runtime"
    assert "HourglassBraid" in info["architecture"]
    assert info["parameters"] > 0
    assert info["preset"] == "ci-tiny"
    assert info["research_preview"] is True


def test_real_generation_is_deterministic(runtime_client):
    payload = {"prompt": "Hello", "max_new_bytes": 48, "seed": 1234}
    first = runtime_client.post("/v1/completions", json=payload).json()
    second = runtime_client.post("/v1/completions", json=payload).json()
    assert first["text"] == second["text"]
    assert first["generated_bytes"] == 48
    other = runtime_client.post("/v1/completions", json={**payload, "seed": 4321}).json()
    assert other["text"] != first["text"]


def test_real_streaming_matches_non_streaming(runtime_client):
    payload = {"prompt": "Hello", "max_new_bytes": 64, "seed": 5}
    whole = runtime_client.post("/v1/completions", json=payload).json()["text"]
    with runtime_client.stream("POST", "/v1/completions/stream", json=payload) as response:
        body = "".join(response.iter_text())
    streamed = "".join(
        json.loads(line[5:])["text"]
        for line in body.splitlines()
        if line.startswith("data:") and "text" in json.loads(line[5:])
    )
    assert streamed == whole


def test_real_output_is_always_valid_unicode(runtime_client):
    # Random-init weights emit near-uniform bytes: the worst case for UTF-8
    # assembly. The response must still be valid JSON with valid text.
    for seed in range(3):
        body = runtime_client.post(
            "/v1/completions", json={"prompt": "", "max_new_bytes": 96, "seed": seed}
        ).json()
        body["text"].encode("utf-8")


def test_preset_mismatch_refuses_to_start(make_client, tiny_package):
    with make_client(
        BRAID_BACKEND="runtime",
        BRAID_MODEL_PATH=str(tiny_package),
        BRAID_DEVICE="cpu",
        BRAID_PRESET="default-300m",
    ) as client:
        health = client.get("/health").json()
        assert health["model_loaded"] is False
        assert client.get("/v1/model").status_code == 503


def test_runtime_version_mismatch_refuses_to_start(make_client, tiny_package):
    with make_client(
        BRAID_BACKEND="runtime",
        BRAID_MODEL_PATH=str(tiny_package),
        BRAID_DEVICE="cpu",
        BRAID_RUNTIME_VERSION="9.9.9",
    ) as client:
        assert client.get("/health").json()["model_loaded"] is False


def test_corrupt_package_refuses_to_start(make_client, tmp_path):
    from idklm.package import build_package as build

    package = build(tmp_path / "pkg", preset="ci-tiny", random_init=True)
    blob = package / "model.bin"
    data = bytearray(blob.read_bytes())
    data[0] ^= 0xFF
    blob.write_bytes(bytes(data))
    with make_client(
        BRAID_BACKEND="runtime",
        BRAID_MODEL_PATH=str(package),
        BRAID_DEVICE="cpu",
    ) as client:
        assert client.get("/health").json()["model_loaded"] is False
        error = client.get("/v1/model").json()
        assert error["error"] == "model_unavailable"
        assert "checksum" in error["detail"].lower()
