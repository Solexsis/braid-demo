"""Braid on HF ZeroGPU: a text-continuation demo over the public 300m model.

The Space holds no model code. At startup it downloads the model repo, which
carries the weights (`config.json` + `model.safetensors`) and the matching
`braid-lite` wheel (`runtime/`), installs that wheel and serves through it.
Pinning the inference code to the model repo's revision means the weights and
the code that reads them can never drift apart.

ZeroGPU specifics (https://huggingface.co/docs/hub/spaces-zerogpu):
* `spaces` must be imported before anything initialises CUDA.
* The model goes onto `cuda` at module level (ZeroGPU emulates CUDA outside
  `@spaces.GPU`; real GPU work only happens inside the decorated function).
* `torch.compile` is unsupported; braid-lite never compiles anything.
"""

from __future__ import annotations

import spaces  # noqa: I001 -- must precede torch's CUDA initialisation

import glob
import importlib
import os
import subprocess
import sys
import time

import gradio as gr
from huggingface_hub import snapshot_download

MODEL_REPO = os.environ.get("BRAID_MODEL_REPO", "Solexsis/braid-300m")
MAX_NEW_BYTES = int(os.environ.get("BRAID_MAX_NEW_BYTES", "1024"))

# HF_TOKEN is only needed while the model repo is private.
PACKAGE_DIR = snapshot_download(MODEL_REPO, token=os.environ.get("HF_TOKEN") or None)


def _install_runtime() -> None:
    """Install the braid-lite wheel shipped inside the model repo, once."""
    try:
        importlib.import_module("braid_lite")
        return
    except ImportError:
        pass
    wheels = sorted(glob.glob(os.path.join(PACKAGE_DIR, "runtime", "braid_lite-*.whl")))
    if not wheels:
        raise RuntimeError(f"{MODEL_REPO} has no runtime/braid_lite-*.whl")
    # --no-deps: torch and numpy come from requirements.txt, pinned to what
    # ZeroGPU supports; the wheel must not pull a different torch.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", wheels[-1]])
    importlib.invalidate_caches()


_install_runtime()
import braid_lite  # noqa: E402

MODEL = braid_lite.load(PACKAGE_DIR, device="cuda")


def _gpu_seconds(prompt, max_new_bytes, temperature, top_k, seed):
    """Reserve GPU time in proportion to the request (shorter = better queue spot).

    Budget: a fixed allowance for the prompt prefill and graph capture, plus a
    per-byte rate about 4x under the measured ~210-240 B/s decode.
    """
    return min(60, int(10 + int(max_new_bytes) / 60))


@spaces.GPU(duration=_gpu_seconds)
def generate(prompt, max_new_bytes, temperature, top_k, seed):
    if not prompt:
        raise gr.Error("Write the start of something for Braid to continue.")
    seed = None if seed is None or int(seed) < 0 else int(seed)
    started = time.perf_counter()
    text = ""
    for chunk in MODEL.generate_stream(
        prompt, max_new_bytes=int(max_new_bytes), temperature=float(temperature),
        top_k=int(top_k), seed=seed,
    ):
        text += chunk
        yield text, ""
    elapsed = time.perf_counter() - started
    produced = len(text.encode("utf-8"))
    yield text, (
        f"{produced} bytes in {elapsed:.1f} s ({produced / max(elapsed, 1e-9):.0f} B/s, "
        f"including prefill) · seed {seed if seed is not None else 'random'}"
    )


def _read(name: str) -> str:
    path = os.path.join(os.path.dirname(__file__), name)
    with open(path, encoding="utf-8") as handle:
        return handle.read()


EXAMPLES = [
    ["The history of the printing press"],
    ["def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n"],
    ["Chapter 1\n\nThe lighthouse keeper had not spoken to anyone in"],
    ["Abstract. We study the problem of"],
]

with gr.Blocks(title="Braid — tokenizer-free byte-level LM") as demo:
    gr.Markdown(_read("intro.md"))
    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="Prompt", lines=6, placeholder="Write the start of something...")
            with gr.Row():
                run = gr.Button("Continue", variant="primary")
                stop = gr.Button("Stop")
        with gr.Column(scale=2):
            max_new = gr.Slider(16, MAX_NEW_BYTES, value=300, step=16, label="New bytes")
            temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
            top_k = gr.Slider(1, 256, value=50, step=1, label="Top-k (1 = greedy)")
            seed = gr.Number(value=1234, precision=0, label="Seed (-1 = random)")
    output = gr.Textbox(label="Continuation", lines=12, interactive=False)
    stats = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=[prompt])
    gr.Markdown(_read("about.md"))
    gr.Markdown(
        f"Model `{MODEL.manifest.get('model_name')}` · {MODEL.parameters:,} parameters "
        f"at inference · braid-lite {braid_lite.__version__}"
    )

    job = run.click(generate, [prompt, max_new, temperature, top_k, seed], [output, stats])
    prompt.submit(generate, [prompt, max_new, temperature, top_k, seed], [output, stats])
    stop.click(None, cancels=[job])

demo.queue(default_concurrency_limit=2).launch()
