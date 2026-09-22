"""Braid on HF ZeroGPU: a text-continuation demo over the public 300m package.

The Space holds no model code. At startup it downloads the model repo, which
carries both the verified weight package (repo root) and the matching runtime
wheel (`runtime/`), installs that wheel, and serves through
`idklm.runtime.BraidRuntime` like every other Braid host. Pinning the runtime
to the model repo's revision means the weights and the code that reads them
can never drift apart.

ZeroGPU specifics (https://huggingface.co/docs/hub/spaces-zerogpu):
* `spaces` must be imported before anything initialises CUDA.
* The model goes onto `cuda` at module level (ZeroGPU emulates CUDA outside
  `@spaces.GPU`; real GPU work only happens inside the decorated function).
* `torch.compile` is unsupported, hence `kernels="portable"`: no FlexAttention
  and no Triton ConvGLU, same weights and same math.
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

MODEL_REPO = os.environ.get("BRAID_MODEL_REPO", "Solenopsisbot/braid-300m")
MAX_NEW_BYTES = int(os.environ.get("BRAID_MAX_NEW_BYTES", "1024"))
# CUDA-graph decode is the fast single-stream path on CUDA; the switch exists
# in case graph capture ever misbehaves inside a ZeroGPU worker.
CUDA_GRAPHS = os.environ.get("BRAID_CUDA_GRAPHS", "1") != "0"

# HF_TOKEN is only needed while the model repo is private.
PACKAGE_DIR = snapshot_download(MODEL_REPO, token=os.environ.get("HF_TOKEN") or None)


def _install_runtime() -> None:
    """Install the runtime wheel shipped inside the model repo, once."""
    try:
        importlib.import_module("idklm.runtime")
        return
    except ImportError:
        pass
    wheels = sorted(glob.glob(os.path.join(PACKAGE_DIR, "runtime", "braid_runtime-*.whl")))
    if not wheels:
        raise RuntimeError(f"{MODEL_REPO} has no runtime/braid_runtime-*.whl")
    # --no-deps: torch and numpy come from requirements.txt, pinned to what
    # ZeroGPU supports; the wheel must not pull a different torch.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", wheels[-1]])
    importlib.invalidate_caches()


_install_runtime()
from idklm.runtime import BraidRuntime  # noqa: E402

RUNTIME = BraidRuntime.load(
    PACKAGE_DIR, device="cuda", kernels="portable", max_new_bytes_limit=MAX_NEW_BYTES,
)
if not CUDA_GRAPHS:
    RUNTIME._cuda_graph_decode = False  # no public toggle; the Space's escape hatch
INFO = RUNTIME.model_info()


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
    for chunk in RUNTIME.generate_stream(
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
        f"Model `{INFO.get('model_name')}` · {INFO.get('parameters', 0):,} parameters · "
        f"runtime {INFO.get('runtime_version')}"
    )

    job = run.click(generate, [prompt, max_new, temperature, top_k, seed], [output, stats])
    prompt.submit(generate, [prompt, max_new, temperature, top_k, seed], [output, stats])
    stop.click(None, cancels=[job])

demo.queue(default_concurrency_limit=2).launch()
