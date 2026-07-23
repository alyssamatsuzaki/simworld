ARG BASE=python:3.11-slim
FROM ${BASE} AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    MPLBACKEND=Agg \
    JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    REGWORLD_ARTIFACT_ROOT=/work/artifacts \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git curl graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# non-root
RUN useradd -m -u 1000 rw
WORKDIR /work
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
# All extras: the smoke CMD runs the bayes/causal/rl/opt stages and the app
# (Streamlit) dashboard, and `make smoke` inside the container runs `pytest -m
# slow` (dev), so a --no-dev core-only sync cannot run the image's own workload.
RUN uv sync --frozen --all-extras

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY Makefile ./
RUN chown -R rw:rw /work
USER rw

ENTRYPOINT ["uv", "run"]
CMD ["python", "scripts/run_pipeline.py", "profile=smoke"]
