# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

# Stage 2: Runtime

FROM python:3.11-slim AS runtime

WORKDIR /app

RUN groupadd --system appuser \
    && useradd --system --gid appuser appuser


COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

COPY src ./src

COPY models ./models

# Make src importable
ENV PYTHONPATH="/app/src"

# Switch away from root
USER appuser

# FastAPI port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=10s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# Start FastAPI
CMD ["uvicorn", "prodml.api.main:app", "--host", "0.0.0.0", "--port", "8000"]