# Parakeet TDT 0.6B v3 - CPU-only OpenAI-compatible transcription service
FROM python:3.11.18-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash parakeet
WORKDIR /app

# requirements.txt is a fully pinned lock; it carries the PyTorch CPU index
# so torch resolves to the CPU-only wheel.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY parakeet_service/ parakeet_service/

RUN mkdir -p models && chown parakeet:parakeet models
USER parakeet

ENV HF_HOME=/app/models \
    HF_HUB_CACHE=/app/models \
    HF_HUB_DISABLE_SYMLINKS_WARNING=true \
    PYTHONUNBUFFERED=1

EXPOSE 5092

# The first start downloads the model (~700 MB), hence the long start period.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5092/health')" || exit 1

CMD ["python", "server.py"]
