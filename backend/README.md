# VoxSentinel AI — FastAPI Backend

The Python half of VoxSentinel AI. It implements the REST contract in
`docs/API_CONTRACT.md` (v1), which the React frontend already speaks.

> **Honest status (production model v0.3.202610241603, Candidate B):** the
> detector architecture is complete — shared feature extraction, a
> multi-dataset training pipeline with leakage-safe splits, a strict artifact
> loader and full serving integration. The installed production artifact is a
> calibrated HistGradientBoosting model over enhanced audio features (MFCC +
> delta + spectral shape), trained on 35,631 real clips from five datasets.
> Measured results: on the seen-generator eval partition F1(synthetic) 0.960,
> FPR 0.054, ROC-AUC 0.993; on the unseen-generator holdout (xTTS,
> FlashSpeech, VoiceBox, NaturalSpeech3) overall synthetic recall is only
> 5.3% (xTTS 0.8%) — **the model largely misses TTS engines it was not
> trained on; seen-generator scores are not generalization performance.**
> Full tables: `docs/REAL_TRAINING_RESULTS.md`, `docs/MODEL_CARD.md`. If the
> artifact is absent, `POST /api/v1/analyze` honestly answers **503
> `model_unavailable`** and stores nothing.

## Requirements

- Python 3.11+
- pip

## Setup

```bash
cd backend
python -m venv .venv
```

Activate the virtual environment:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install:

```bash
python -m pip install -r requirements.txt          # run the API
python -m pip install -r requirements-dev.txt      # + tests & lint (recommended)
```

Optional configuration — copy the example and edit if you need non-default
values (the defaults are safe for local development):

```bash
cp .env.example .env
```

Never commit `.env`. It is git-ignored.

## Run

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Then:

- API docs (Swagger UI): <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>
- Models: <http://127.0.0.1:8000/api/v1/models>

`/health` reports API/database readiness and **separate** model readiness:

```json
{
  "status": "ok",
  "backend_ready": true,
  "model_ready": false,
  "speaker_model_ready": false,
  "version": "0.1.0",
  "mode": "baseline"
}
```

`model_ready: false` is the truthful Phase 4 state.

## Run with the frontend

In one terminal:

```bash
cd backend && uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

In another:

```bash
cd frontend && npm install && npm run dev
```

Open <http://localhost:5173>. The frontend calls the backend at
`VITE_API_BASE_URL` (default `http://localhost:8000`). With no model
installed, the Analyze page stays in honest "model not ready" state.

## Tests and lint

```bash
python -m pytest tests/ -q     # 174 collected: 173 pass, 1 environment-dependent skip
ruff check app tests           # lint
```

The test suite includes a `MockVoiceDetector` **test double** — a clearly
labelled fake used to exercise the pipeline end to end. It is never installed
in the application, and its outputs are never real predictions.

## Training, evaluation and artifacts (Phase 5)

```bash
# current multi-dataset pipeline (five real datasets; full commands in the top-level
# README "Training and evaluation" section and notebooks/multidataset_training.ipynb):
python scripts/multisource_download.py --data-dir ../data          # needs KAGGLE_USERNAME/KAGGLE_KEY
python scripts/build_multisource_manifest.py --data-dir ../data \
    --output-manifest ../data/multidataset/unified_manifest.jsonl \
    --output-training-manifest ../data/multidataset/training_manifest.jsonl \
    --max-per-class-per-source 8000 --holdout-generators "xTTS,FlashSpeech,VoiceBox,NaturalSpeech3"
python scripts/train_candidate.py --candidate enhanced-hgb --manifest ../data/multidataset/training_manifest.jsonl \
    --dataset-dir ../data --model-dir ../data/models/candidate_enhanced_hgb --seed 42 --calibrate
python scripts/evaluate_detector.py --artifact models/detector.joblib \
    --manifest ../data/multidataset/training_manifest.jsonl --partition eval --report-dir reports
python scripts/eval_holdout_generators.py --artifact models/detector.joblib \
    --manifest ../data/multidataset/holdout_manifest.jsonl --dataset-dir ../data --report-dir reports/holdout

# legacy single-corpus workflow (reproduces the archived v0.2.0 model;
# docs/DATASET_GUIDE.md section 2 has the full build commands):
python scripts/train_detector.py --manifest data/librispeech_piper_manifest.jsonl --model-dir models --seed 42 --version 0.2.0 --leakage-exempt-sources piper_tts
```

- Real datasets: `docs/DATASET_GUIDE.md` (ASVspoof 2019 LA preset included).
- The API auto-loads `models/detector.joblib` at startup, but only after strict
  validation; any defect → honest `model_unavailable`, never a half-model.
- `models/`, `data/`, `reports/` are git-ignored. The artifact is a pickle:
  load only artifacts this project's trainer produced (see the security notes
  in `app/services/model_loader.py`).
- Training never runs at API startup; it is always an explicit command.

## Supported audio formats

Decoding uses `soundfile` (bundled libsndfile). On the reference build
(lsndfile 1.2.2) the backend decodes **WAV, FLAC, OGG and MP3**. **M4A is not
decodable** by libsndfile and is rejected with `415` and a clear message —
even though the frontend allows previewing M4A in the browser. The runtime
capability is discoverable via `soundfile.available_formats()`.

## Security properties (Phase 4 scope)

- Server-side validation of size, extension, content sniffing, decodability,
  duration, silence and sample values — the browser's checks are never trusted.
- Uploaded audio is decoded **in memory**; it is never written to disk, never
  executed, and never sent to any external service.
- Filenames are untrusted: reduced to a display label, never used as paths.
- CORS allows only configured origins; no wildcard with credentials.
- All errors use the contract shape `{"error": {"code", "message"}}` with no
  tracebacks, paths or secrets. Internal details go to server logs only.
- Raw audio is never stored; history keeps metadata only.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `VOXSENTINEL_ENV` | `development` | Environment label |
| `VOXSENTINEL_API_VERSION` | `0.1.0` | Version reported by `/health` |
| `VOXSENTINEL_APP_MODE` | `baseline` | Label only; never a readiness claim |
| `VOXSENTINEL_DB_URL` | `sqlite:///./voxsentinel.db` | Database URL |
| `VOXSENTINEL_ALLOWED_ORIGINS` | `http://localhost:5173,...` | CORS origins, comma-separated |
| `VOXSENTINEL_MAX_UPLOAD_BYTES` | `26214400` | Upload cap (25 MB) |
| `VOXSENTINEL_MIN_DURATION_S` | `2` | Below this: quality warning |
| `VOXSENTINEL_MAX_DURATION_S` | `600` | Above this: rejected |
| `VOXSENTINEL_DETECTOR_MODEL_PATH` | `models/detector.joblib` | Detector artifact path (currently Candidate B v0.3.202610241603; previous baseline archived under `models/archive/`) |

## Deployment notes

Works on standard Python hosts (e.g. Render):

- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`
- Set `VOXSENTINEL_ALLOWED_ORIGINS` to your frontend origin.
- **Persistence warning:** on ephemeral/free hosting the filesystem is wiped
  on redeploy, taking the SQLite file with it. Point `VOXSENTINEL_DB_URL` at a
  managed database for anything long-lived.
- Debug mode is never enabled by the app; Uvicorn's `--reload` flag is for
  development only.

## Layout

```
backend/app/
├── main.py                    App factory, CORS, lifespan
├── api/routes.py              All endpoints (thin layer)
├── core/config.py             Settings from environment variables
├── core/errors.py             Contract-shaped error handling
├── schemas/analysis.py        Pydantic models = API contract
├── services/audio_validation.py   Server-side validation + decoding
├── services/detector.py       Detector interface + honest Null implementation
├── services/speaker.py        Speaker-verification interface
├── services/risk_engine.py    Documented (provisional) risk policy
├── services/history.py        Analysis repository
├── models/analysis_record.py  SQLAlchemy ORM
└── db/session.py              Engine, sessions, health ping
```
