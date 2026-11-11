---
title: VoxSentinel AI API
emoji: 🎙️
colorFrom: indigo
colorTo: slate
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Voice-authenticity and scam-call analysis backend
---

# VoxSentinel AI — backend

FastAPI service behind the VoxSentinel AI frontend: voice-authenticity detection,
speaker verification, multilingual transcription and family-emergency scam analysis.

* `GET /health` — liveness and model readiness
* `GET /api/v1/models` — model registry
* `GET /api/v1/protection/capabilities` — what this instance can actually do
* `POST /api/v1/protection/analyze` — analyse a call recording (consent required)
* `GET /api/v1/protection/analyses` — analysis history

Interactive docs at `/docs`. CORS is restricted to the origins listed in
`VOXSENTINEL_ALLOWED_ORIGINS`, so add the frontend origin there.

The first transcription downloads the Whisper model (default `tiny`) into
`HF_HOME`, so expect a slower first request.
