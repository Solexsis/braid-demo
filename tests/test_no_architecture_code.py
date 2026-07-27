"""This repository must never grow a copy of the Braid architecture.

The whole point of the split is that there is exactly one implementation, in the
private research repository, consumed through a versioned wheel. A copied model
file here would rot immediately and silently. So: assert its absence.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [
    path for path in ROOT.rglob("*.py") if ".venv" not in path.parts and "build" not in path.parts
]

FORBIDDEN_IMPORTS = (
    r"^\s*import\s+torch\b",
    r"^\s*from\s+torch\b",
    r"^\s*import\s+numpy\b",
)

ARCHITECTURE_SYMBOLS = (
    "HourglassBraid(",
    "HourglassConfig(",
    "class ConvGLU",
    "class Attention",
    "nn.Module",
    "torch.nn",
    "scaled_dot_product_attention",
)

WEIGHT_SUFFIXES = (".pt", ".ckpt", ".bin", ".safetensors", ".pth", ".onnx")


def test_no_direct_tensor_framework_imports():
    """Only the adapter may touch the runtime, and it does so lazily by name."""
    offenders = []
    for path in SOURCES:
        if path.name.startswith("test_"):
            continue
        text = path.read_text()
        for pattern in FORBIDDEN_IMPORTS:
            if re.search(pattern, text, re.MULTILINE):
                offenders.append(f"{path.relative_to(ROOT)}: {pattern}")
    assert not offenders, (
        "the demo must not import a tensor framework directly; go through "
        f"idklm.runtime: {offenders}"
    )


def test_no_architecture_definitions():
    offenders = []
    for path in SOURCES:
        if path.name.startswith("test_"):
            continue
        text = path.read_text()
        for symbol in ARCHITECTURE_SYMBOLS:
            if symbol in text:
                offenders.append(f"{path.relative_to(ROOT)}: {symbol}")
    assert not offenders, f"architecture code leaked into the demo: {offenders}"


def test_no_training_entry_points():
    for path in SOURCES:
        if path.name.startswith("test_"):
            continue
        text = path.read_text().lower()
        assert "loss.backward" not in text, path
        assert "optimizer.step" not in text, path


def test_no_weights_are_committed():
    tracked = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and ".venv" not in path.parts
        and path.suffix in WEIGHT_SUFFIXES
    ]
    assert not tracked, f"weight-like files present: {tracked}"


def test_no_env_file_is_committed():
    leaked = [
        path
        for path in ROOT.rglob(".env*")
        if path.is_file() and path.name != ".env.example" and ".venv" not in path.parts
    ]
    assert not leaked, f"environment files present: {leaked}"


def test_runtime_backend_imports_the_wheel_not_a_copy():
    text = (ROOT / "app" / "backends.py").read_text()
    assert "from idklm.runtime import" in text
    assert "BraidRuntime.load" in text
