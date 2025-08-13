# ---------- base ----------
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface

# System dependencies (ssl, locales, and build basics if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---------- deps ----------
FROM base AS deps
WORKDIR /app

COPY requirements-prod.txt /app/requirements-prod.txt
RUN python -m pip install -r requirements-prod.txt

# ---------- app ----------
FROM base AS runtime
WORKDIR /app

# Repeat Python deps layer from "deps"
COPY --from=deps /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy project
COPY . /app

# Default model directory (override at runtime with -e MODEL_DIR=/path/in/container)
ENV MODEL_DIR="runs/distilbert_auto/final_model" \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fs http://localhost:${PORT}/health || exit 1

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
