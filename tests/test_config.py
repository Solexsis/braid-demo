from __future__ import annotations

import pytest

from app.config import load_settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    import os

    for key in list(os.environ):
        if key.startswith("BRAID_"):
            monkeypatch.delenv(key, raising=False)


def test_defaults_are_mock_and_safe():
    settings = load_settings()
    assert settings.backend == "mock"
    assert settings.is_mock
    assert settings.max_prompt_bytes == 4096
    assert settings.max_new_bytes == 1024
    assert settings.max_concurrent == 1
    assert settings.model_path == "" and settings.model_url == ""
    assert settings.trust_forwarded_for is False


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("BRAID_BACKEND", "gpt5")
    with pytest.raises(ValueError, match="BRAID_BACKEND"):
        load_settings()


def test_runtime_backend_requires_a_model_source(monkeypatch):
    monkeypatch.setenv("BRAID_BACKEND", "runtime")
    with pytest.raises(ValueError, match="BRAID_MODEL_PATH or BRAID_MODEL_URL"):
        load_settings()


def test_model_url_requires_a_checksum(monkeypatch):
    monkeypatch.setenv("BRAID_BACKEND", "runtime")
    monkeypatch.setenv("BRAID_MODEL_URL", "https://example.invalid/model.tar.gz")
    with pytest.raises(ValueError, match="BRAID_MODEL_SHA256"):
        load_settings()


def test_checksum_must_look_like_sha256(monkeypatch):
    monkeypatch.setenv("BRAID_BACKEND", "runtime")
    monkeypatch.setenv("BRAID_MODEL_URL", "https://example.invalid/model.tar.gz")
    monkeypatch.setenv("BRAID_MODEL_SHA256", "deadbeef")
    with pytest.raises(ValueError, match="64-character hex"):
        load_settings()


@pytest.mark.parametrize(
    "name,value",
    [
        ("BRAID_MAX_PROMPT_BYTES", "not-a-number"),
        ("BRAID_MAX_NEW_BYTES", "0"),
        ("BRAID_MAX_CONCURRENT", "-1"),
        ("BRAID_REQUEST_TIMEOUT", "0"),
    ],
)
def test_bad_numeric_settings_are_rejected(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        load_settings()


def test_default_output_is_clamped_to_the_ceiling(monkeypatch):
    monkeypatch.setenv("BRAID_MAX_NEW_BYTES", "64")
    settings = load_settings()
    assert settings.default_new_bytes == 64
