# Security policy

## Reporting a vulnerability

Email **security@solexsis.ai** with:

* what you found and where (endpoint, file, or configuration),
* how to reproduce it,
* what an attacker could do with it.

Please do **not** open a public issue for a vulnerability. We will acknowledge
within 5 working days and aim to have a fix or a mitigation within 30 days for
anything exploitable against a default deployment.

Do not run automated scanners or load tests against a hosted instance. If you
need to test aggressively, run the demo locally — `BRAID_BACKEND=mock` needs no
credentials and no model.

## Scope

In scope:

* the HTTP API and the web UI in this repository,
* model package resolution, download, and verification,
* the shipped `Dockerfile` and compose file, in their documented configurations.

Out of scope:

* model output quality. A language model producing false, offensive, or
  nonsensical text is a documented limitation, not a vulnerability.
* prompt injection or jailbreaking. There is no system prompt and no privileged
  context to escape; the model has no tools and no data access.
* denial of service by simply sending many requests. Rate limiting and
  concurrency admission are configurable and documented; tune them for your
  deployment.
* the private research repository, which is not published here.

## Security properties this repository maintains

These are asserted by tests, not just by intention:

* **No weights and no secrets are committed.** `tests/test_no_architecture_code.py`
  fails if a weight-like file or a non-example `.env` appears, and CI repeats the
  check against the git index.
* **Downloads fail closed.** A model fetched via `BRAID_MODEL_URL` is SHA-256
  verified *before* it is loaded; a mismatch deletes the artifact and refuses to
  start. `BRAID_MODEL_SHA256` is mandatory whenever `BRAID_MODEL_URL` is set.
* **Archives cannot escape their directory.** Tar and zip members with paths
  resolving outside the destination are rejected before extraction.
* **Packages are validated.** Format, package version, file presence, checksums,
  weight dtype, architecture options and the required runtime range are all
  checked; anything unexpected refuses to load rather than loading partially.
* **Errors do not leak internals.** No stack trace, filesystem path, or internal
  hostname appears in a response body. Unhandled exceptions become a generic
  500 with a request id that correlates to a server-side log line.
* **Logs do not contain user content.** Structured logs record method, path,
  status, duration and request id — not prompts and not completions.
* **The container runs as a non-root user** (uid 10001) and holds no checkpoint
  by default.

## Deployment hardening

* Terminate TLS at a reverse proxy; the app speaks plain HTTP.
* Set `BRAID_TRUST_FORWARDED_FOR=true` **only** behind a proxy you control.
  Otherwise clients can spoof `X-Forwarded-For` and defeat the rate limiter.
* Prefer proxy-level rate limiting (`BRAID_RATE_LIMIT_REQUESTS=0` in the app).
* Mount model packages read-only.
* Pin image tags and runtime versions. Never deploy `latest`.
* Set a memory limit on the container; a 300M bf16 checkpoint plus activations
  is the floor, not the ceiling.
* The demo has no authentication. If it should not be public, put it behind one.
