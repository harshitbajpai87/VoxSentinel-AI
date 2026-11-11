# VoxSentinel AI — full backend image for Render (or any Docker host).
# Same stack as deploy/hf/Dockerfile (ffmpeg, libsndfile, torch/speechbrain,
# faster-whisper) but built with the repository root as build context, so
# Render can point at this file directly:
#   dockerfilePath: ./deploy/render/Dockerfile
#   dockerContext:  .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    VOXSENTINEL_ENV=production \
    VOXSENTINEL_DB_URL=sqlite:////app/data/voxsentinel.db \
    PORT=10000

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
 && apt-get clean

WORKDIR /app

# CPU-only torch keeps the image near 1.5 GB instead of pulling CUDA wheels.
COPY backend/requirements*.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Application code plus the trained artifacts that ship in the repo.
COPY backend/ ./

RUN mkdir -p /app/data /app/reports /app/.cache/huggingface

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000} --proxy-headers --forwarded-allow-ips=*"]
