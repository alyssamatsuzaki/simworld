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
RUN uv sync --frozen --no-dev

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY Makefile ./
RUN chown -R rw:rw /work
USER rw

ENTRYPOINT ["uv", "run"]
CMD ["python", "scripts/run_pipeline.py", "profile=smoke"]
