"""Resolve a model package: a local path, or a verified one-time download.

The rules, in order of importance:

1. A downloaded artifact is **verified before it is used**, never after.
2. A checksum mismatch fails closed — the bad file is deleted and the process
   refuses to start rather than serving something nobody authorised.
3. A valid cached copy is not re-downloaded on restart.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger("braid.model_source")

CHUNK = 1 << 20
STAMP = ".braid-source.json"


class ModelSourceError(RuntimeError):
    """The model could not be resolved, downloaded, or verified."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_package_root(directory: Path) -> Path:
    """A package is the directory containing `manifest.json`.

    Archives usually expand into a single top-level folder, so look one level
    down before giving up.
    """
    if (directory / "manifest.json").exists():
        return directory
    children = [child for child in directory.iterdir() if child.is_dir()]
    for child in children:
        if (child / "manifest.json").exists():
            return child
    raise ModelSourceError(
        f"{directory}: no manifest.json found — is this a `braid package` output?"
    )


def _extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                target = (destination / member.name).resolve()
                if not str(target).startswith(str(destination.resolve())):
                    raise ModelSourceError(
                        f"{archive}: refusing path traversal entry {member.name!r}"
                    )
            try:
                # Python >= 3.12 wants an explicit extraction filter; `data`
                # strips ownership/permission surprises. Older runtimes fall
                # back to the validated-members path above.
                tar.extractall(destination, filter="data")
            except TypeError:  # pragma: no cover - Python < 3.11.4
                tar.extractall(destination)  # noqa: S202 - members validated above
        return
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                target = (destination / name).resolve()
                if not str(target).startswith(str(destination.resolve())):
                    raise ModelSourceError(f"{archive}: refusing path traversal entry {name!r}")
            zf.extractall(destination)
        return
    raise ModelSourceError(f"{archive}: unsupported archive format (expected .tar.gz or .zip)")


def _cached_dir(cache: Path, digest: str) -> Path:
    return cache / f"braid-{digest[:16]}"


def _stamp_is_valid(directory: Path, digest: str) -> bool:
    stamp = directory / STAMP
    if not stamp.exists() or not (directory / "manifest.json").exists():
        return False
    try:
        return json.loads(stamp.read_text()).get("sha256") == digest
    except (OSError, json.JSONDecodeError):
        return False


def _download(url: str, destination: Path, timeout: float = 300.0) -> None:
    log.info("downloading model package", extra={"url": url})
    request = urllib.request.Request(url, headers={"User-Agent": "braid-demo/0.1"})
    with (
        urllib.request.urlopen(request, timeout=timeout) as response,
        destination.open("wb") as out,
    ):
        shutil.copyfileobj(response, out, CHUNK)


def resolve_model(
    *,
    model_path: str = "",
    model_url: str = "",
    model_sha256: str = "",
    cache_dir: Path = Path("/var/cache/braid"),
) -> Path:
    """Return a local directory containing a Braid model package."""
    if model_path:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise ModelSourceError(f"BRAID_MODEL_PATH does not exist: {path}")
        if path.is_file():
            # A raw .pt checkpoint is allowed for local development; the runtime
            # validates it on load.
            return path
        return _find_package_root(path)

    if not model_url:
        raise ModelSourceError("no BRAID_MODEL_PATH and no BRAID_MODEL_URL")
    if not model_sha256:
        raise ModelSourceError("BRAID_MODEL_URL requires BRAID_MODEL_SHA256")

    digest = model_sha256.lower()
    cache_dir = Path(cache_dir).expanduser()
    target = _cached_dir(cache_dir, digest)
    if _stamp_is_valid(target, digest):
        log.info("using cached model package", extra={"path": str(target)})
        return _find_package_root(target)

    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir) as scratch:
        scratch_path = Path(scratch)
        archive = scratch_path / "model-package.bin"
        try:
            _download(model_url, archive)
        except OSError as exc:
            raise ModelSourceError(f"download failed: {exc}") from exc

        actual = sha256_file(archive)
        if actual != digest:
            archive.unlink(missing_ok=True)
            raise ModelSourceError(
                "SHA-256 mismatch for the downloaded model package "
                f"(expected {digest[:16]}..., got {actual[:16]}...). "
                "Refusing to load it."
            )

        unpacked = scratch_path / "unpacked"
        _extract(archive, unpacked)
        root = _find_package_root(unpacked)
        (root / STAMP).write_text(json.dumps({"sha256": digest, "url": model_url}))

        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(root), str(target))
    log.info("model package ready", extra={"path": str(target)})
    return target
