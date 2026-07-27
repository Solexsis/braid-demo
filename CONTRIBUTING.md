# Contributing

Thanks for looking. This repository is the **public demo** for Braid: an HTTP
API, a web UI, and the deployment plumbing around a versioned inference runtime.

## What belongs here

* API behaviour: endpoints, validation, limits, error shapes, streaming.
* The web interface: HTML, CSS, vanilla JS. No framework, no build step.
* Deployment: Dockerfile, compose, proxy notes, health checks.
* Documentation and the benchmark page renderer.
* Tests for all of the above.

## What does not belong here

* **Model architecture.** Not a layer, not a block, not a tensor operation.
  There is exactly one implementation and it lives in the private research
  repository; a copy here would drift silently. `tests/test_no_architecture_code.py`
  enforces this and will fail your PR.
* **Training code.** Same reason.
* **Model weights.** Ever. They are distributed as signed, checksummed packages.
* **A direct `torch` or `numpy` import.** Go through `idklm.runtime`.
* **Benchmark numbers typed by hand.** `app/data/benchmarks.json` is generated
  by the exporter in the private repository from tracked measurement CSVs. If a
  number is wrong, fix the measurement, not the JSON.
* **Claims the code did not measure.** No "faster than GPT-x", no projected
  300M results, no rounding a 0.019 bits/byte deficit out of existence.

If you need a runtime capability that does not exist yet (a sampler option, a
new metric in `model_info()`), open an issue describing the *interface* you
want. It will be implemented in `idk-lm` and arrive in a new wheel version.

## Development

```bash
uv venv --python 3.12
uv sync --extra dev

BRAID_BACKEND=mock uv run uvicorn app.main:app --reload --port 8000

uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Everything must pass in **mock mode**. CI has no access to the private runtime
or to any checkpoint, and that is deliberate — the public test suite must never
depend on private artifacts.

## Pull requests

* One logical change per PR, with tests.
* Keep the API backwards compatible, or bump the path (`/v2/...`) and say why.
* Update `README.md` when you change an environment variable, an endpoint, or
  a response field. An undocumented knob is a bug.
* If you touch streaming, prove that streamed and non-streamed output still
  match; there is a test for it, keep it honest.
* Do not weaken an existing test to make a change pass. If a test is wrong,
  explain why in the PR and fix it deliberately.

## Style

* Python: 4-space indent, type hints, `from __future__ import annotations`,
  comments that explain *why*. `ruff` settings live in `pyproject.toml`.
* JavaScript: tabs, `"use strict"`, no dependencies, no bundler.
* Commit messages: lowercase scope prefix (`api:`, `ui:`, `docker:`, `docs:`),
  a short body for anything non-obvious.

## Reporting problems

* Bugs and feature requests: GitHub issues.
* Security vulnerabilities: **not** GitHub issues — see [`SECURITY.md`](SECURITY.md).
* Bad model output: that is a documented limitation of a research preview, not
  a bug. Interesting failure modes are still welcome as issues.
