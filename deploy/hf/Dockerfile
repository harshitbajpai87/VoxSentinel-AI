# VoxSentinel AI — full backend for a Docker host (Hugging Face Spaces, Render,
# Fly.io, Railway, a VPS). This is the deployment that can actually listen to a
# call: it has ffmpeg, libsndfile, torch/speechbrain (speaker verification) and
# faster-whisper (ASR), none of which fit inside a Vercel function.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    VOXSENTINEL_ENV=production \
    VOXSENTINEL_DB_URL=sqlite:////app/data/voxsentinel.db \
    PORT=7860

# ffmpeg: M4A/AAC/WebM/Opus fallback decoding. libsndfile: WAV/FLAC/OGG/MP3.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
 && apt-get clean

WORKDIR /app

# CPU-only torch keeps the image around 1.5 GB instead of pulling CUDA wheels.
COPY requirements.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Application code, trained artifacts and data files.
COPY . .

RUN mkdir -p /app/data /app/reports /app/.cache/huggingface \
 && useradd -m -u 1000 user \
 && chown -R user:user /app
USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips=*"]
