# Deployment Guide

## 1. Requirements
- Python 3.11+, Node 20+, ~2 GB RAM for inference (CPU is fine)
- Inference models in `backend/models/` (bundled where size permits, otherwise fetched by name)
- Optional Groq key (`PROTECTION_LLM_PROVIDER=groq`) or Gemini key; without a key the app runs
  rules-only and labels it

## 2. Backend
```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # fill in keys locally; never commit
set -a && source .env && set +a
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Health: `curl localhost:8000/health` → `{"status":"ok","model_ready":true,...}`

## 3. Frontend
```bash
cd frontend
npm ci
npm run build
npm run preview -- --port 5173
```
Point the frontend at the backend origin (`VITE_API_BASE_URL`) before building.

## 4. Environment variables
| Name | Purpose | Required |
|------|---------|----------|
| `PROTECTION_LLM_PROVIDER` | `groq` (default) or `gemini` | no |
| `GROQ_API_KEY` | contextual classification | for the LLM stage |
| `GEMINI_API_KEY` | legacy provider | optional |
| `PROTECTION_REASONER` | `enabled` turns on the second-pass review | no |
| `DETECTOR_MODEL_DIR`, `SPEAKER_MODEL_DIR` | model paths | if not default |
| `CORS_ORIGINS` | allowed frontend origins | recommended |

## 5. Verify
```bash
python3 backend/tests/api_checks.py http://<host>:8000   # expect 18/18
cd e2e && node verify.mjs                                # expect 31/31
```

## 6. Operations
- Groq free tier throttles quickly under repeated analysis; the app degrades to `rules_only` and records
  `quota_exhausted: true`. Budget for it or self-host a reasoning model.
- Add authentication before public exposure.
- Use a log driver in production rather than `/tmp` files.
