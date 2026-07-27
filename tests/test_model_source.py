"""Model resolution must fail closed on a bad checksum and cache on a good one."""

from __future__ import annotations

import json
import tarfile

import pytest

from app import model_source
from app.model_source import ModelSourceError, resolve_model, sha256_file


def _fake_package(root):
    package = root / "braid-model"
    package.mkdir(parents=True)
    (package / "manifest.json").write_text(
        json.dumps(
            {
                "package_format": "braid-model-package",
                "package_version": 1,
            }
        )
    )
    (package / "model.bin").write_bytes(b"weights")
    return package


def _tarball(tmp_path):
    package = _fake_package(tmp_path / "src")
    archive = tmp_path / "model.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(package, arcname="braid-model")
    return archive


def test_local_directory_is_used_as_is(tmp_path):
    package = _fake_package(tmp_path)
    assert resolve_model(model_path=str(package)) == package


def test_local_parent_directory_is_searched_one_level(tmp_path):
    package = _fake_package(tmp_path / "outer")
    assert resolve_model(model_path=str(tmp_path / "outer")) == package


def test_missing_local_path_is_reported(tmp_path):
    with pytest.raises(ModelSourceError, match="does not exist"):
        resolve_model(model_path=str(tmp_path / "absent"))


def test_directory_without_manifest_is_rejected(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ModelSourceError, match="no manifest.json"):
        resolve_model(model_path=str(tmp_path / "empty"))


def test_download_verifies_before_loading(tmp_path, monkeypatch):
    archive = _tarball(tmp_path)
    digest = sha256_file(archive)
    calls = []

    def fake_download(url, destination, timeout=300.0):
        calls.append(url)
        destination.write_bytes(archive.read_bytes())

    monkeypatch.setattr(model_source, "_download", fake_download)
    cache = tmp_path / "cache"
    resolved = resolve_model(
        model_url="https://example.invalid/m.tar.gz",
        model_sha256=digest,
        cache_dir=cache,
    )
    assert (resolved / "manifest.json").exists()
    assert len(calls) == 1

    # A second resolution must reuse the cache, not re-download.
    again = resolve_model(
        model_url="https://example.invalid/m.tar.gz",
        model_sha256=digest,
        cache_dir=cache,
    )
    assert again == resolved
    assert len(calls) == 1


def test_checksum_mismatch_fails_closed(tmp_path, monkeypatch):
    archive = _tarball(tmp_path)

    def fake_download(url, destination, timeout=300.0):
        destination.write_bytes(archive.read_bytes())

    monkeypatch.setattr(model_source, "_download", fake_download)
    with pytest.raises(ModelSourceError, match="SHA-256 mismatch"):
        resolve_model(
            model_url="https://example.invalid/m.tar.gz",
            model_sha256="0" * 64,
            cache_dir=tmp_path / "cache",
        )
    # Nothing usable may be left behind.
    cache = tmp_path / "cache"
    assert (
        not any(p.is_dir() and (p / "manifest.json").exists() for p in cache.iterdir())
        if cache.exists()
        else True
    )


def test_download_failure_is_wrapped(tmp_path, monkeypatch):
    def boom(url, destination, timeout=300.0):
        raise OSError("connection refused")

    monkeypatch.setattr(model_source, "_download", boom)
    with pytest.raises(ModelSourceError, match="download failed"):
        resolve_model(
            model_url="https://example.invalid/m.tar.gz",
            model_sha256="a" * 64,
            cache_dir=tmp_path / "cache",
        )


def test_path_traversal_in_an_archive_is_refused(tmp_path, monkeypatch):
    evil = tmp_path / "evil.tar.gz"
    payload = tmp_path / "payload"
    payload.write_text("pwned")
    with tarfile.open(evil, "w:gz") as tar:
        tar.add(payload, arcname="../escaped")
    digest = sha256_file(evil)

    def fake_download(url, destination, timeout=300.0):
        destination.write_bytes(evil.read_bytes())

    monkeypatch.setattr(model_source, "_download", fake_download)
    with pytest.raises(ModelSourceError, match="path traversal"):
        resolve_model(
            model_url="https://example.invalid/evil.tar.gz",
            model_sha256=digest,
            cache_dir=tmp_path / "cache",
        )


def test_unsupported_archive_format_is_reported(tmp_path, monkeypatch):
    blob = tmp_path / "model.bin"
    blob.write_bytes(b"not an archive at all")
    digest = sha256_file(blob)

    def fake_download(url, destination, timeout=300.0):
        destination.write_bytes(blob.read_bytes())

    monkeypatch.setattr(model_source, "_download", fake_download)
    with pytest.raises(ModelSourceError, match="unsupported archive format"):
        resolve_model(
            model_url="https://example.invalid/m.bin",
            model_sha256=digest,
            cache_dir=tmp_path / "cache",
        )
