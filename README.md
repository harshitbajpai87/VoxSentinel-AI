# VoxSentinel AI

> Current repair-release status: read [START_HERE.md](START_HERE.md) and
> [docs/REPAIR_VERIFICATION.md](docs/REPAIR_VERIFICATION.md) first. Historical
> completion tables and stored metrics below were supplied in the original
> archive and are NOT fresh verification. This release does not guarantee
> 100% accuracy or complete production readiness.

**Voice-Cloning Impersonation Detection and Risk-Aware Prevention System**

| | |
|---|---|
| Context | AI Foundations CA project |
| Hackathon | Smart India Hackathon 2026 |
| Problem ID | SIH26104 |
| Team | Harshit Bajpai (12509899), Oisiky Sinha (12501033) |

> **Ethical stance:** VoxSentinel AI never claims an audio clip is definitively
> genuine or fake on the basis of a model prediction. Every result carries
> uncertainty information and a recommendation for human verification.

## Status (built incrementally, honestly)

| Phase | Scope | Status |
|---|---|---|
| 1 | Architecture, repo, API contract | Done |
| 2 | Frontend design system + navigation (9 pages) | Done |
| 3 | Audio upload, validation, real waveform + playback | Done |
| 4 | FastAPI backend, database, honest 503 model gating | Done |
| 5 | Detection model | Done — production Candidate B (enhanced features + HistGradientBoosting, sigmoid-calibrated), version `0.3.202610241603`, trained on a balanced corpus built from five real audio datasets (see `docs/MODEL_CARD.md`, `docs/REAL_TRAINING_RESULTS.md`) |
| 6 | Risk policy + speaker verification | Done — risk policy active; ECAPA-TDNN speaker verification calibrated on real LibriSpeech pairs (held-out EER 0.025, ROC-AUC 0.9994; see `docs/MODEL_CARD.md`) |
| 7 | History, reports, evaluation | Done — filterable history, CSV export, async evaluation jobs, per-window scores and explainability evidence |
| 8 | Security hardening + tests | Done — audit log, retention policy, upload hardening; backend 173 tests + frontend 73 tests green, ruff clean, tsc clean, production build green |
| 9 | GitHub collaboration + deployment | Partial — repo ready (clean .gitignore, no secrets); deployment not started |
| 10 | Acceptance checks | Pending — live end-to-end checks passed on localhost; SIH demo checklist remains |

## Uploads, decoding and limits (2026-11-11)

All four audio workflows share one ingress layer: 25 MB per file (26 214 400
bytes, streaming, aborted at the limit), container sniffing instead of trusting
filenames, and an ffmpeg fallback for M4A/AAC/WebM/Opus. ffmpeg and ffprobe must
be installed by the operator (`pip install ffmpeg-python` is not enough). Full
matrix, environment variables, `/health` contract and every error code:
[`docs/UPLOAD_LIMITS_AND_DECODERS.md`](./docs/UPLOAD_LIMITS_AND_DECODERS.md).
Repair report: [`docs/BACKEND_REPAIR_2026_11_11.md`](./docs/BACKEND_REPAIR_2026_11_11.md).
Model reliability and validation limits: [`docs/MODEL_RELIABILITY_REPAIRS_2026_11_11.md`](./docs/MODEL_RELIABILITY_REPAIRS_2026_11_11.md).

## Repository layout

```
voxsentinel-ai/
├── frontend/          # React + Vite + TypeScript + Tailwind
├── backend/           # FastAPI + SQLAlchemy
│   ├── app/           # main, api routes, core, schemas, services, models, db
│   ├── scripts/       # training, multi-dataset pipeline, evaluation, holdout eval
│   └── tests/         # 174 pytest tests (173 passing, 1 environment-dependent skip)
├── evaluation/        # measured evidence: eval + holdout reports, manifests, audit
├── notebooks/         # multidataset_training.ipynb (Colab reproduction workflow)
├── docs/              # ARCHITECTURE, API_CONTRACT, MODEL_CARD, REAL_TRAINING_RESULTS,
│                      # MULTIDATASET_UPGRADE, DATASET_GUIDE, DEPLOYMENT_WINDOWS, TEAM_PLAN
├── scripts/windows/   # PowerShell setup/start scripts
├── FILE_MANIFEST.txt  # relative paths + sizes of every packaged file
└── README.md
```

## Quickstart

**Windows (PowerShell):** run the ready-made setup scripts once, then the two
start scripts (full guide: `docs/DEPLOYMENT_WINDOWS.md`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\setup-backend.ps1
powershell -ExecutionPolicy Bypass -File scripts\windows\setup-frontend.ps1
# then, in two terminals:
powershell -ExecutionPolicy Bypass -File scripts\windows\start-backend.ps1
powershell -ExecutionPolicy Bypass -File scripts\windows\start-frontend.ps1
```

**Linux / macOS** — Backend (terminal 1):

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend (terminal 2):

```bash
cd frontend
npm ci
npm run dev
```

Open <http://localhost:5173>. The API contract is versioned and shared:
`docs/API_CONTRACT.md` is the single source of truth for both sides.

> **Honest model status (production Candidate B, v0.3.202610241603):**
> trained on real data from five audio datasets (Speech Commands, Synthetic
> Speech Commands, FoR for-norm, audio-deepfake-detection, human-vs-AI
> Indonesian voices; 35,631 training clips, class-balanced, leakage-safe
> speaker/recording splits, sha256 duplicate merging). On the held-out
> **seen-generator** eval partition: F1(synthetic) 0.960, FPR 0.054,
> ROC-AUC 0.993. **These are seen-generator numbers, NOT generalization
> performance.** On an unseen-generator holdout (xTTS, FlashSpeech, VoiceBox,
> NaturalSpeech3 — 854 synthetic clips), overall recall is only **5.3%**
> (xTTS 0.8%): the model largely misses modern TTS engines it was not
> trained on. The score is sigmoid-calibrated on validation data. Without
> the artifact, the API answers `503 model_unavailable` and nothing is
> fabricated. Full tables: `docs/REAL_TRAINING_RESULTS.md`.

### Training and evaluation (backend/)

The current production model was trained with the multi-dataset pipeline:

```bash
# 1. Download the five real datasets (Kaggle credentials required for four;
#    Speech Commands downloads directly). Writes data/<source> dirs + registry.
KAGGLE_USERNAME=... KAGGLE_KEY=... python scripts/multisource_download.py --data-dir ../data

# 2. Inspect sources and build the leakage-safe unified manifest
#    (stable per-file ids, global LibriSpeech speaker namespace, generator-complete
#     holdout, sha256 duplicate clustering; caps at 8000 per class per source):
python scripts/build_multisource_manifest.py --data-dir ../data \
    --output-manifest ../data/multidataset/unified_manifest.jsonl \
    --output-training-manifest ../data/multidataset/training_manifest.jsonl \
    --max-per-class-per-source 8000 \
    --holdout-generators "xTTS,FlashSpeech,VoiceBox,NaturalSpeech3"

# 3. Train the baseline (MFCC+LR) and candidates (enhanced features)
python scripts/train_detector.py --manifest ../data/multidataset/training_manifest.jsonl \
    --dataset-dir ../data --model-dir ../data/models/baseline_multidataset --seed 42
python scripts/train_candidate.py --candidate enhanced-hgb \
    --manifest ../data/multidataset/training_manifest.jsonl --dataset-dir ../data \
    --model-dir ../data/models/candidate_enhanced_hgb --seed 42 --calibrate

# 4. Evaluate: seen-generator eval partition + unseen-generator holdout
python scripts/evaluate_detector.py --artifact <artifact> \
    --manifest ../data/multidataset/training_manifest.jsonl --partition eval --report-dir <dir>
python scripts/eval_holdout_generators.py --artifact <artifact> \
    --manifest ../data/multidataset/holdout_manifest.jsonl --dataset-dir ../data --report-dir <dir>

# 5. Promote: copy the chosen artifact to models/detector.joblib and restart
#    the API (auto-loads it; /health flips to model_ready: true).
```

Reproduction guide (incl. Google Colab): `notebooks/multidataset_training.ipynb`
and `docs/MULTIDATASET_UPGRADE.md`. The original single-corpus baseline
workflow (`prepare_dataset.py`, librispeech_piper preset) remains available
for reproducing the archived v0.2.0 model.

Datasets (`data/`), model artifacts (`models/`) and generated reports
(`reports/`) are git-ignored. Never commit them without license review.

## Quickstart (frontend only)

```bash
cd frontend
npm ci
cp .env.example .env.local   # optional: point to your backend
npm run dev
```

Verify the build and tests:

```bash
npm run build    # type-check + production build
npm run smoke    # server-renders every page and checks expected content
npm test         # unit + component tests (validation, waveform, API, workflow)
```

The app runs without the backend; every backend-dependent panel shows an
explicit "backend offline" state instead of fake data.

### Analyze Audio workflow

On the Analyze Audio page you can drag & drop or browse for WAV/MP3/FLAC/M4A
files (up to 25 MB). The file is decoded locally with the Web Audio API and
validated: empty, oversized, unsupported, corrupted, silent and too-long files
are rejected with friendly messages; short, very quiet or clipped recordings
get quality warnings. A canvas waveform shows the real sample amplitudes with
play/pause, current time and click-to-seek. Nothing is uploaded until you tick
the consent box and press "Start analysis"; the backend independently
re-validates everything. All input limits are configurable constants in
`frontend/src/lib/audio/validation.ts`.

See `docs/ARCHITECTURE.md` for the system design, `docs/REAL_TRAINING_RESULTS.md`
for every measured training/evaluation number, and `docs/DEPLOYMENT_WINDOWS.md`
for the step-by-step localhost deployment guide.

## Deployment

The project deploys as a split stack: a static React build on Vercel and the
FastAPI service on Render. They are independent deployments that meet over HTTP.

### Frontend — Vercel

| | |
|---|---|
| Platform | Vercel |
| Live URL | <https://voxs-two.vercel.app/> |
| Build | `npm run build` in `frontend/` (Vite static output in `frontend/dist`) |
| Runtime | Static hosting, no server-side rendering |

This is the deployed frontend. It is a pure client application: it holds no API
keys and talks to the backend only through the normalized API client in
`frontend/src/lib/api.ts`.

### Backend — Render

| | |
|---|---|
| Platform | Render |
| Framework | FastAPI, Python 3.11 |
| Deployment method | Docker |
| Dockerfile | `deploy/render/Dockerfile` |
| Health endpoint | `/health` |
| API documentation | `/docs` (OpenAPI UI, `/openapi.json`) |
| Public URL | <https://voxsentinel-backend-lo9k.onrender.com> (verified 2026-11-15: `/health` returned 200) |

The Dockerfile is the same stack used for any container host: `python:3.11-slim`
with the CPU torch wheel, speechbrain, faster-whisper and the trained artifacts
copied from `backend/`. It listens on `$PORT` (Render injects 10000) and ships a
`HEALTHCHECK` that polls `/health`.

The service runs on the free instance type, which sleeps after about 15 minutes
without traffic. The next request then pays a 60–90 s cold start, and the
frontend reports "API waking" while it finishes instead of claiming the backend
is down. An always-on instance type removes this behaviour; a periodic request
to `/health` only masks it.

### Frontend ↔ backend integration

- The frontend reads its API base from the **`VITE_API_BASE_URL`** environment
  variable. On Vercel, set it in Project → Settings → Environment Variables for
  the Production (and Preview, if used) targets, and it must contain the verified
  Render backend URL, for example
  `VITE_API_BASE_URL=https://voxsentinel-backend-lo9k.onrender.com`.
- Vite inlines `VITE_*` values at build time, so **changing the variable requires
  a new frontend deployment**. An edited variable alone changes nothing until the
  site is redeployed.
- CORS must allow the deployed frontend origin. The backend's allowlist comes
  from `VOXSENTINEL_ALLOWED_ORIGINS` (comma-separated, for example
  `https://voxs-two.vercel.app`); a missing origin surfaces in the browser as a
  CORS error rather than as an HTTP status.
- With `VITE_API_BASE_URL` unset, the client falls back to the page's own origin,
  which is only correct when frontend and backend are served from the same host.

### Render configuration

| Setting | Value |
|---|---|
| Repository root | Docker build context (`.`), not `backend/` |
| Dockerfile path | `deploy/render/Dockerfile` |
| Port | `$PORT` (10000) |
| Persistent disk | Mount at `/app/data` when using the default SQLite database |

- **Build context is the repository root.** The image copies `backend/` from that
  context, so pointing Render at `backend/` or at another Dockerfile path
  produces a build that cannot find the application or its artifacts.
- **FFmpeg and libsndfile are installed inside the image** (`ffmpeg`,
  `libsndfile1`), which is what M4A/AAC/WebM/Opus decoding and the audio readers
  rely on. Do not expect a bare Python environment to decode those formats.
- **The image defaults to SQLite at `/app/data/voxsentinel.db`
  (`VOXSENTINEL_DB_URL=sqlite:////app/data/voxsentinel.db`).** A persistent disk
  mounted at `/app/data` is therefore required for analyses, audit records and
  family voice profiles to survive a deploy or restart; without it the container
  filesystem is rebuilt and the history is lost. If you point
  `VOXSENTINEL_DB_URL` at a managed Postgres instance instead, as the current
  deployment does, the disk is optional.

### Verifying a deployment

```bash
curl -fsS https://<backend-host>/health    # 200 + JSON status
open  https://<backend-host>/docs          # OpenAPI UI renders
```

A 200 from `/health` proves the process is up and serving; it does **not** prove
every model is loaded and ready. Read the payload for
`backend_ready`, `model_ready`, `speaker_model_ready` and the per-service block
(for example `services.transcription`, `services.contextual_ai`): the API gates
each feature honestly and reports the ones it cannot serve, so a green health
check can still coexist with "assistant not configured" or a speaker backend
running as the MFCC fallback.

Troubleshooting:

- **Wrong API URL** — the UI shows "API offline" or an empty history while
  `/health` works when opened directly. Check the built bundle's
  `VITE_API_BASE_URL` and redeploy the frontend after fixing it.
- **CORS errors** — the browser console shows a blocked request and the network
  tab shows no status. Add the exact frontend origin to
  `VOXSENTINEL_ALLOWED_ORIGINS`, including scheme, and no trailing slash.
- **Backend unavailable** — 502/503 immediately after a quiet period is the free
  plan starting the container, not a crash. Re-check `/health` after 60–90 s.
  If it persists, inspect the service's logs for an exit or out-of-memory kill;
  the protected pipelines are the memory-heavy paths.
- **Data disappeared after a deploy** — the SQLite database was written outside
  a mounted disk. Attach the disk at `/app/data` or use a managed database.

## v14 — Groq contextual provider (verified live)

- **Provider**: Groq (`api.groq.com/openai/v1`) via a provider-agnostic adapter.
  Default model `openai/gpt-oss-120b`. `PROTECTION_LLM_PROVIDER` selects
  `groq | xai | mistral | gemini | ollama`; with no explicit choice the first
  configured provider wins.
- **Grok is not Groq**: xAI keys (`xai-...`) belong to `api.x.ai`; Groq keys
  (`gsk_...`) belong to `api.groq.com`. Both adapters now exist and are tested.
- **Live proof**: a ransom recording was transcribed, checked, rule-scored and
  contextually classified as **LIKELY_SCAM / CRITICAL / KIDNAPPING_OR_RANSOM**
  with 3 verified evidence quotes in **2.06 s**. Legitimate calls returned
  LIKELY_LEGITIMATE / LOW.
- **Robustness**: flat model output (`evidence_quotes`, `risk`, `FULL` quality) is
  normalised onto the canonical schema; free-text categories resolve by keyword
  ("Kidnap/Extortion" → `KIDNAPPING_OR_RANSOM`). Fabricated quotes are still
  dropped, and unknown classification/risk values still fail closed.
- **Tests**: 306 backend (2 skipped integration), 76 frontend, 0 type errors,
  clean lint and production build.

## v15 — Mistral AI and Ollama providers, plus training tooling

- **Mistral AI** (`api.mistral.ai/v1`) joins the adapter set. Default model
  `mistral-small-latest`, `MISTRAL_API_KEY` is the only required setting,
  `PROTECTION_LLM_PROVIDER=mistral`. Same schema, same verbatim-quote check,
  same never-raise contract; status codes are provider-scoped
  (`mistral_auth_failed`, `mistral_rate_limited`, …).
- **Ollama** (local) is supported for offline deployments: no API key, loopback
  `http` allowed and remote hosts forced to `https`, one retry instead of two for
  cold model loads, plus a `health()` probe of `/api/tags`. Off by default and
  enabled only by `VOXSENTINEL_OLLAMA_ENABLED=true` or explicit selection.
- **Training pipeline**, honestly scoped: `scripts/build_scam_finetune_dataset.py`
  turns labelled transcripts into schema-validated chat-format JSONL with
  verbatim evidence quotes; `scripts/train_mistral_finetune.py` validates and
  (only with `--submit`) starts a hosted Mistral fine-tuning job;
  `scripts/create_ollama_model.py` builds a local Ollama model from the
  production system prompt plus labelled examples, optionally importing a LoRA
  adapter. The bundled label set is a development regression set: the builders
  warn below 50 (dataset) and 100 (training) examples, so a real corpus is
  required before claiming a tuned model.
- **Docs**: `docs/MISTRAL_OLLAMA.md`.
- **Tests**: 434 backend (3 skipped), all offline via mocked transports.

## Documentation

- [docs/UNIFIED_PIPELINE.md](docs/UNIFIED_PIPELINE.md) - the single-call
  `POST /api/analyze` contract, every model it can use, how to enable each one,
  and its honest limits.
