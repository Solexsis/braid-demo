from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def make_client(monkeypatch):
    """Build a TestClient with a fresh, explicit environment.

    Settings are cached, so every test that changes an environment variable must
    force a reload; doing that here keeps the tests independent of ordering.
    """

    def factory(**env: str):
        for key in list(os.environ):
            if key.startswith("BRAID_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("BRAID_BACKEND", env.pop("BRAID_BACKEND", "mock"))
        for key, value in env.items():
            monkeypatch.setenv(key, str(value))

        from app import config, main

        config.get_settings(refresh=True)
        return TestClient(main.app)

    return factory


@pytest.fixture
def client(make_client):
    with make_client() as instance:
        yield instance
