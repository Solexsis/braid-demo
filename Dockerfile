# Braid demo — API + static UI.
#
# The image contains NO model weights by default. Supply them at run time with
# BRAID_MODEL_PATH (mounted volume) or BRAID_MODEL_URL + BRAID_MODEL_SHA256.
#
# Two build modes:
#
#   mock (default, no private access needed):
#     docker build -t braid-demo:0.1.0 .
#
#   runtime (needs the private wheel present at wheels/):
#     cp ../idk-lm/dist/braid_runtime-0.1.0-py3-none-any.whl wheels/
#     docker build --build-arg INSTALL_RUNTIME=1 -t braid-demo:0.1.0-runtime .
#
# Pin the tag in production. Never deploy `latest`.

FROM python:3.12-slim AS base

ARG RUNTIME_VERSION=0.1.0
ARG INSTALL_RUNTIME=0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BRAID_BACKEND=mock \
    BRAID_MODEL_CACHE=/var/cache/braid \
    BRAID_HOST=0.0.0.0 \
    BRAID_PORT=8000

RUN useradd --create-home --uid 10001 braid \
    && mkdir -p /var/cache/braid \
    && chown -R braid:braid /var/cache/braid

WORKDIR /srv/braid-demo

COPY pyproject.toml README.md ./
COPY app ./app

RUN pip install --no-cache-dir .

# Optional: install the private inference runtime wheel. `wheels/` holds only a
# .gitkeep in git, so the default build stays fully public and reproducible.
COPY wheels ./wheels
RUN if [ "$INSTALL_RUNTIME" = "1" ]; then \
        pip install --no-cache-dir --extra-index-url "${TORCH_INDEX}" \
            "./wheels/braid_runtime-${RUNTIME_VERSION}-py3-none-any.whl"; \
    else \
        echo "building without the Braid runtime wheel (mock backend only)"; \
    fi \
    && rm -rf ./wheels

USER braid
EXPOSE 8000
VOLUME ["/var/cache/braid"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys,json; \
r=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4); \
sys.exit(0 if json.load(r)['model_loaded'] else 1)"

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", \
     "--timeout-graceful-shutdown", "30", \
     "--no-access-log"]
