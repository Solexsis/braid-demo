# Braid — public research preview

**Braid** is a language model that is not a transformer and has no tokenizer.
This repository is the **public demo**: a small FastAPI service and a
dependency-free web UI that talk to a versioned Braid inference runtime.

> **Research preview.** Braid is not production-ready, not instruction-tuned,
> not aligned, and not generally intelligent. At equal training data a matched
> dense transformer still reaches slightly lower loss. What Braid buys is
> systems cost — see [the benchmarks](#benchmarks).

---

## What Braid is

```
bytes ─▶ stem (gated causal conv)
       ─▶ ↓4 ─▶ ↓4                      causal strided downsample, one-position shift
            ══ dense attention @ 1/16 rate, bounded window ══
       ─▶ ↑4 + gated skip ─▶ ↑4 + gated skip
       ─▶ 256-byte sliding-window attention
       ─▶ byte logits  (+ an exact 4-byte suffix-copy branch)
```

* **Tokenizer-free.** Input and output are raw bytes (vocabulary 256). No
  vocabulary file, no merge table, no tokenizer version to keep in sync.
* **Hierarchical, not isotropic.** A transformer runs every layer at full
  resolution. Braid folds the sequence 16× through a causal U-Net so quadratic
  attention runs on 1/16 of the positions, and spends the savings on depth over
  cheap local mixers.
* **Causal by construction.** Each pooled sequence is shifted one position, so a
  coarse slot only ever summarises *completed* earlier groups. The prediction at
  byte *t* depends only on bytes ≤ *t*, and that is an enforced, tested
  invariant — as is "cached incremental decoding equals the parallel forward".
* **Bounded decode state.** The coarse attention window is bounded, so the
  decoder's state is constant in context length rather than growing forever.

Architecture research and training live in a **separate private repository**
(`idk-lm`). This repository contains no model code, no training code and no
weights.

## What is in this repository

```
app/
  main.py          FastAPI app: /health, /v1/model, /v1/completions[/stream]
  backends.py      mock backend + a thin adapter over the private runtime
  model_source.py  resolve a model package: local path, or verified download
  config.py        environment-driven settings with validation
  limits.py        concurrency admission + per-IP rate limiting
  schemas.py       request/response models
  static/          the web UI (vanilla HTML/CSS/JS, no build step)
  data/            benchmarks.json, exported from the private repo
tests/             API, config, limits, model source, and a guard test that
                   fails if architecture code or weights ever appear here
Dockerfile         weights-free image
```

## Quick start (mock mode — no private access needed)

```bash
uv venv --python 3.12
uv sync --extra dev
BRAID_BACKEND=mock uv run uvicorn app.main:app --reload --port 8000
# open http://127.0.0.1:8000
```

The mock backend emits deterministic placeholder text so the UI, the streaming
protocol and the tests all work without the model. The API and the page both say
so explicitly; there is no mode in which the demo pretends mock output is model
output.

Run the tests:

```bash
uv run pytest -q
uv run ruff check .
```

## Real-runtime mode

You need two artifacts from the private `idk-lm` repository:

1. **the runtime wheel** — `braid_runtime-0.1.0-py3-none-any.whl` (inference
   code + canonical presets; no training data),
2. **a model package** — a `braid package` output directory containing
   `manifest.json`, `model.json`, `model.bin`, `SHA256SUMS`, `MODEL_CARD.md`.

```bash
# 1. install the runtime
uv pip install wheels/braid_runtime-0.1.0-py3-none-any.whl

# 2. point the demo at a packaged model
export BRAID_BACKEND=runtime
export BRAID_MODEL_PATH=/models/braid-default-300m
export BRAID_DEVICE=auto
export BRAID_PRESET=default-300m        # optional: refuse a mismatched checkpoint
uv run uvicorn app.main:app --port 8000
```

Or fetch it once over HTTP, verified:

```bash
export BRAID_BACKEND=runtime
export BRAID_MODEL_URL=https://models.example.com/braid-default-300m.tar.gz
export BRAID_MODEL_SHA256=<64 hex chars>
export BRAID_MODEL_CACHE=/var/cache/braid
```

The download is verified **before** it is loaded, a mismatch fails closed, and a
valid cached copy is not re-downloaded on restart.

### Producing those artifacts (in the private repo)

```bash
# runtime wheel
uv run python scripts/build_runtime_wheel.py --version 0.1.0 --out dist

# model package from a trained checkpoint
uv run braid package \
  --checkpoint checkpoints/braid-default-300m.pt \
  --preset default-300m \
  --output dist/braid-default-300m \
  --dtype bfloat16 \
  --checkpoint-version 1.0.0 \
  --training-bytes 40000000000 \
  --dataset "FineWeb-Edu raw bytes, 0x00 document separators" \
  --license "CC-BY-4.0"

uv run braid inspect dist/braid-default-300m

# publishable tarball + checksum for BRAID_MODEL_URL / BRAID_MODEL_SHA256
tar -czf braid-default-300m.tar.gz -C dist braid-default-300m
shasum -a 256 braid-default-300m.tar.gz
```

See `docs/runtime-distribution.md` in the private repository for how a
deployment pipeline authenticates to it while this repository stays public.

## Environment variables

| variable | default | meaning |
|---|---|---|
| `BRAID_BACKEND` | `mock` | `mock` or `runtime` |
| `BRAID_MODEL_PATH` | — | local package directory (or a `.pt` for dev) |
| `BRAID_MODEL_URL` | — | download the package from here, once |
| `BRAID_MODEL_SHA256` | — | **required** with `BRAID_MODEL_URL`; fails closed |
| `BRAID_MODEL_CACHE` | `/var/cache/braid` | where verified downloads are cached |
| `BRAID_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `cuda:0` / `mps` |
| `BRAID_PRESET` | — | refuse to start unless the checkpoint matches it |
| `BRAID_RUNTIME_VERSION` | — | refuse to start on a wheel version mismatch |
| `BRAID_MAX_PROMPT_BYTES` | `4096` | prompts above this are truncated |
| `BRAID_MAX_NEW_BYTES` | `1024` | hard ceiling on generated bytes |
| `BRAID_DEFAULT_NEW_BYTES` | `256` | used when the request omits it |
| `BRAID_MAX_CONCURRENT` | `1` | generations at once (one model, one GPU) |
| `BRAID_QUEUE_LIMIT` | `8` | waiting requests before returning 429 |
| `BRAID_REQUEST_TIMEOUT` | `60` | seconds before a generation is cut off |
| `BRAID_RATE_LIMIT_REQUESTS` | `20` | per-IP window count; `0` disables |
| `BRAID_RATE_LIMIT_WINDOW` | `60` | window length in seconds |
| `BRAID_TRUST_FORWARDED_FOR` | `false` | honour `X-Forwarded-For` (proxy only) |
| `BRAID_LOG_LEVEL` | `INFO` | structured JSON logs on stdout |

Full annotated list: [`.env.example`](.env.example).

## API

### `GET /health`

```json
{"status":"ok","model_loaded":true,"backend":"runtime","uptime_s":42.1,"in_flight":0,"version":"0.1.0"}
```

### `GET /v1/model`

```bash
curl -s localhost:8000/v1/model | jq
```

```json
{
  "model_name": "braid-default-300m",
  "backend": "runtime",
  "parameters": 295082179,
  "architecture": "byte-level HourglassBraid (tokenizer-free)",
  "architecture_summary": "byte hourglass, factors 4x4 (1/16 coarse rate); width 1344 x 21 heads; ...",
  "context_limit_bytes": 8192,
  "max_prompt_bytes": 4096,
  "max_new_bytes": 1024,
  "checkpoint_version": "1.0.0",
  "runtime_version": "0.1.0",
  "device": "cuda",
  "research_preview": true
}
```

### `POST /v1/completions`

```bash
curl -s localhost:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"prompt":"A strange machine woke beneath the city.",
       "max_new_bytes":256,"temperature":0.8,"top_k":50,"seed":1234}' | jq
```

```json
{
  "text": "...",
  "prompt_bytes": 41,
  "generated_bytes": 256,
  "duration_s": 1.83,
  "bytes_per_second": 139.9,
  "model": "braid-default-300m",
  "backend": "runtime",
  "seed": 1234,
  "finish_reason": "length",
  "truncated_prompt": false
}
```

`seed` is optional; supplying it makes the completion reproducible.

### `POST /v1/completions/stream`

Server-sent events by default, newline-delimited JSON with `?format=ndjson`.

```bash
curl -N localhost:8000/v1/completions/stream \
  -H 'content-type: application/json' \
  -d '{"prompt":"Hello","max_new_bytes":128,"seed":7}'
```

```
event: start
data: {"model":"braid-default-300m","backend":"runtime","max_new_bytes":128,"truncated_prompt":false,"seed":7}

event: chunk
data: {"text":" world"}

event: done
data: {"generated_bytes":128,"duration_s":0.91,"bytes_per_second":140.6,"finish_reason":"length"}
```

Streamed and non-streamed output are byte-identical for the same request; that
is a tested invariant on both sides of the boundary.

### `GET /v1/benchmarks`

The checked-in measured results, exactly as exported from the private repo.

### Error shape

```json
{"error": "rate_limited", "detail": "rate limit exceeded (20 requests per 60s)"}
```

| status | `error` | when |
|---|---|---|
| 422 | (FastAPI validation) | malformed request body |
| 429 | `rate_limited` / `overloaded` | per-IP limit, or the queue is full |
| 500 | `generation_failed` / `internal_error` | anything unexpected |
| 503 | `model_unavailable` / `shutting_down` | model not loaded, or draining |
| 504 | `timeout` | generation exceeded `BRAID_REQUEST_TIMEOUT` |

Internal exceptions are logged with a request id and never returned to the
client.

## Docker

```bash
# mock (fully public build)
docker build -t braid-demo:0.1.0 .
docker run --rm -p 8000:8000 braid-demo:0.1.0

# with the real runtime and a mounted model package
cp ../idk-lm/dist/braid_runtime-0.1.0-py3-none-any.whl wheels/
docker build --build-arg INSTALL_RUNTIME=1 --build-arg RUNTIME_VERSION=0.1.0 \
  -t braid-demo:0.1.0-runtime .

docker run -d --name braid-demo -p 8000:8000 \
  -e BRAID_BACKEND=runtime \
  -e BRAID_MODEL_PATH=/models/braid-default-300m \
  -e BRAID_DEVICE=cpu \
  -v /srv/models/braid-default-300m:/models/braid-default-300m:ro \
  -v braid-model-cache:/var/cache/braid \
  --restart unless-stopped \
  braid-demo:0.1.0-runtime
```

Development compose: `docker compose -f docker-compose.dev.yml up --build`.

**Pin your tags.** The examples above deliberately never use `latest`.

### GPU deployment

```bash
docker build --build-arg INSTALL_RUNTIME=1 \
  --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 \
  -t braid-demo:0.1.0-cuda .

docker run -d --gpus '"device=0"' -p 8000:8000 \
  -e BRAID_BACKEND=runtime -e BRAID_DEVICE=cuda \
  -e BRAID_MODEL_PATH=/models/braid-default-300m \
  -e BRAID_MAX_CONCURRENT=1 \
  -v /srv/models/braid-default-300m:/models/braid-default-300m:ro \
  braid-demo:0.1.0-cuda
```

One process owns the model, so scale with replicas behind the proxy, not with
`--workers`. Braid's decode state is bounded, so VRAM does not grow with context
— but a 300M bf16 checkpoint still needs roughly 0.6 GB of weights plus
activations.

### Reverse proxy notes

Streaming needs buffering disabled. For nginx:

```nginx
location / {
    proxy_pass              http://127.0.0.1:8000;
    proxy_http_version      1.1;
    proxy_set_header        X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_buffering         off;      # required for SSE
    proxy_cache             off;
    proxy_read_timeout      120s;
    chunked_transfer_encoding on;
}

limit_req_zone $binary_remote_addr zone=braid:10m rate=20r/m;
location /v1/completions { limit_req zone=braid burst=5 nodelay; ... }
```

If the proxy owns rate limiting, set `BRAID_RATE_LIMIT_REQUESTS=0` and
`BRAID_TRUST_FORWARDED_FOR=true`. Only trust forwarded headers behind a proxy
you control.

## Benchmarks

[`/benchmarks`](http://127.0.0.1:8000/benchmarks) renders
`app/data/benchmarks.json`, which is exported from the private repository by
`scripts/export_demo_benchmarks.py`. Every value on that page was measured;
rows with no measurement are marked *pending* and carry no numbers.

Current state, honestly:

* **Measured:** matched ~30M-parameter FineWeb runs on an RTX 3090 (Braid's full
  default recipe vs dense MHA/GQA transformers vs flat Braid variants), the
  mechanism ablations behind the recipe, and context-scaling throughput.
* **Not measured:** anything at 100M or 300M with the current recipe. Those
  presets construct and their parameter counts are exact, but no checkpoint has
  been trained, so the page shows *awaiting checkpoint* rather than a projection.
* **Not claimed:** that Braid beats a transformer on quality. At equal bytes the
  dense MHA baseline is ahead by ~0.019 bits/byte at 30M.

## Current limitations

* Research preview. Output is frequently wrong, repetitive, or meaningless.
* No instruction tuning, no RLHF, no alignment, no safety filtering, no
  retrieval, no tool use.
* Byte-level: the model can emit invalid UTF-8. The runtime buffers partial
  sequences and renders unrecoverable bytes as U+FFFD rather than failing.
* Bounded coarse attention means this is **not** a long-context model.
* One generation at a time per process; heavy traffic gets a 429, not a queue
  that silently grows.
* The demo does not persist prompts or completions. It also does not moderate
  them — do not deploy it publicly without a moderation layer if that matters
  to you.

## Security notes

* No secrets in this repository. `.env` is git-ignored; `.env.example` contains
  only placeholders.
* No weights in this repository or in the default image.
* Downloaded model packages are SHA-256 verified before use and fail closed.
* Archive extraction rejects path-traversal entries.
* Errors are sanitised: no stack traces, no filesystem paths, no internal
  hostnames in any response body.
* Structured logs record method, path, status, duration and a request id. They
  do **not** record prompts or completions.
* `robots.txt` disallows `/v1/`.
* Report vulnerabilities per [`SECURITY.md`](SECURITY.md).

## Checkpoint provenance

Every model package carries a `manifest.json` recording the preset and its
version, the full resolved architecture configuration, exact parameter count,
weight dtype, source commit SHA of the private repository, build timestamp,
training byte count, dataset description, licence, known limitations, the
runtime version range it requires, and SHA-256 hashes of every file.
`braid inspect <package>` re-verifies all of it, and `/v1/model` surfaces the
public subset. A package that fails any check does not load.

## Licence

Code in this repository: Apache-2.0 (see [`LICENSE`](LICENSE)).

Model weights are distributed separately and carry their own licence, recorded
in each package's `manifest.json` and `MODEL_CARD.md`.

## Related

* [`MODEL_CARD.md`](MODEL_CARD.md) — what the model is, and is not, for.
* [`CONTRIBUTING.md`](CONTRIBUTING.md) — what belongs here and what does not.
* [`SECURITY.md`](SECURITY.md) — reporting a vulnerability.
* Solexsis Research — <https://solexsis.com>
