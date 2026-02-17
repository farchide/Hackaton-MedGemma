# ============================================================
# Digital Twin Tumor Response Assessment -- Multi-stage Docker
# ============================================================
# Stage 1: Builder -- install dependencies and generate demo data
# Stage 2: Runtime -- slim image with only what's needed
# ============================================================

# ---- Stage 1: Builder ----
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        gfortran \
        libopenblas-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python dependencies first (layer caching)
COPY pyproject.toml ./
COPY src/ ./src/
COPY scripts/ ./scripts/

RUN pip install --no-cache-dir --prefix=/install . \
    && pip install --no-cache-dir --prefix=/install pandas

# Generate demo data during build so it's baked into the image
ENV PYTHONPATH=/install/lib/python3.12/site-packages:/build/src
RUN mkdir -p /build/.cache \
    && python -c "\
from digital_twin_tumor.data.synthetic import generate_all_demo_data; \
generate_all_demo_data(db_path='/build/.cache/demo.db', seed=42, verbose=True)"


# ---- Stage 2: Runtime ----
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 dttuser \
    && useradd --uid 1000 --gid dttuser --create-home dttuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source code
COPY --chown=dttuser:dttuser src/ ./src/
COPY --chown=dttuser:dttuser pyproject.toml ./
COPY --chown=dttuser:dttuser scripts/ ./scripts/

# Copy pre-generated demo database
COPY --from=builder --chown=dttuser:dttuser /build/.cache/demo.db ./.cache/demo.db

# Set environment
ENV PYTHONPATH=/app/src \
    DTT_DEMO_DB=/app/.cache/demo.db \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

EXPOSE 7860

# Healthcheck -- Gradio serves on /
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

USER dttuser

CMD ["python", "-m", "digital_twin_tumor", "--port", "7860"]
