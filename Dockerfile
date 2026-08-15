# ShelfSight AI — backend (FastAPI + OpenCV + PyTorch, CPU)
#
# Two stages so the runtime image does not carry build toolchains. Python 3.10
# to match the project's supported floor (the code deliberately avoids 3.11-only
# syntax) and the interpreter the client already runs locally.
FROM python:3.10-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build-only toolchain. `pycocotools` compiles C extensions; without gcc the
# ML requirements fail late and confusingly.
RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt requirements-ml.txt ./

# CPU-only torch from the dedicated index: the default wheel pulls ~2.5 GB of
# CUDA libraries that are dead weight in a CPU container.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && /opt/venv/bin/pip install -r requirements-ml.txt


FROM python:3.10-slim AS runtime

# System libraries OpenCV links against at runtime. `libgl1-mesa-glx` is the one
# everybody hits: without it `import cv2` dies with
# "libGL.so.1: cannot open shared object file", which looks like a Python
# problem and is not. glib/gomp are needed by opencv and torch respectively.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    # Everything mutable lives under /app/runtime, which compose mounts as a
    # volume — so weights, uploads and the database survive `docker compose down`.
    DATABASE_URL=sqlite:////app/runtime/shelfsight.db \
    DETECTION_WEIGHTS=/app/runtime/weights/shelfsight-sku110k.pt \
    FRESHNESS_WEIGHTS=/app/runtime/weights/freshness_mobilenetv2_clean.pt \
    UPLOAD_DIR=/app/runtime/uploads \
    REPORTS_DIR=/app/runtime/reports \
    OLLAMA_BASE_URL=http://ollama:11434 \
    # EasyOCR and torch.hub cache into the volume; otherwise every container
    # restart re-downloads ~100 MB of OCR models.
    EASYOCR_MODULE_PATH=/app/runtime/easyocr \
    TORCH_HOME=/app/runtime/torch

WORKDIR /app

COPY app ./app
COPY models ./models
COPY evaluation ./evaluation
COPY tools ./tools
COPY data ./data
COPY pyproject.toml README.md ./

RUN mkdir -p /app/runtime/weights /app/runtime/uploads /app/runtime/reports \
    && useradd --create-home --uid 10001 shelfsight \
    && chown -R shelfsight:shelfsight /app
USER shelfsight

EXPOSE 8000

# Hits the real dependency graph (DB + services), not just the socket.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
