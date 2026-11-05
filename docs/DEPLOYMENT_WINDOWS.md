# Localhost Deployment Guide (Windows)

VoxSentinel AI runs fully on your own machine: a FastAPI backend on
`http://127.0.0.1:8000` and a React frontend on `http://localhost:5173`.
No cloud services, no accounts, no telemetry. This guide assumes Windows 10/11
with PowerShell.

## 0. Prerequisites

| Tool | Version | Check | Install |
|---|---|---|---|
| Python | 3.11+ | `python --version` | `winget install Python.Python.3.12` |
| Node.js | 18+ | `node -v` | `winget install OpenJS.NodeJS.LTS` |
| ffmpeg | any | `ffmpeg -version` | `winget install Gyan.FFmpeg` (optional) |

ffmpeg is **optional**: it only enables M4A/AAC/WebM/Opus uploads. WAV, FLAC,
OGG and MP3 work without it.

## 1. Unzip

Right-click the ZIP in Explorer and "Extract All", or:

```powershell
Expand-Archive VoxSentinel_AI_Final_v0.3.202610241603.zip -DestinationPath C:\projects
cd C:\projects\voxsentinel-ai
```

## 2. One-time setup (two commands)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\setup-backend.ps1
powershell -ExecutionPolicy Bypass -File scripts\windows\setup-frontend.ps1
```

These create `backend\.venv`, install the pinned Python requirements, and run
`npm install` for the frontend. No secrets are involved: configuration lives in
`.env.example` templates only.

## 3. Run (two terminals)

```powershell
# Terminal 1 - backend on http://127.0.0.1:8000
powershell -ExecutionPolicy Bypass -File scripts\windows\start-backend.ps1

# Terminal 2 - frontend on http://localhost:5173
powershell -ExecutionPolicy Bypass -File scripts\windows\start-frontend.ps1
```

Open <http://localhost:5173>. The Settings page shows the live backend
connection state.

## 4. AI models — what ships and what to get

| Artifact | Ships in ZIP? | Size | How to get it | Without it |
|---|---|---|---|---|
| `backend/models/detector.joblib` (**Candidate B v0.3.202610241603**, calibrated) | Yes | 0.9 MB | already included | — |
| `backend/models/archive/detector_v0.2.0_20261024.joblib` (previous baseline) | Yes | 3 KB | already included | — |
| `backend/models/speaker_thresholds.json` (calibrated 0.3448) | Yes | <1 KB | already included | — |
| `backend/models/speaker_ecapa/` (ECAPA-TDNN embeddings) | **No** | ~84 MB | **included in the release archive** (`models/speaker_ecapa/`); if missing, fetch once: `python scripts/download_speaker_model.py` | without it that page/endpoint reports `unavailable` honestly; everything else works |
| raw audio datasets (five sources, ~29.6 GB) | **No** | ~29.6 GB | see the top-level README "Training and evaluation" section — only needed to retrain/re-evaluate | analysis of your own audio works without it |

> The shipped detector is the real multi-dataset model, verified loadable
> through the production loader. Its measured scope is documented in
> `docs/MODEL_CARD.md` and `docs/REAL_TRAINING_RESULTS.md`: strong on the
> generators seen in training (F1 0.960) but weak on unseen TTS engines
> (holdout recall 5.3%, xTTS 0.8%) — treat every verdict as decision support
> requiring human verification, exactly as the UI states.

The ECAPA download happens automatically the first time you open Speaker
Verification (or call `POST /api/v1/speaker-verify`) with internet access. It is
cached in `backend/models/speaker_ecapa/` and never leaves your machine.

If you intentionally remove `detector.joblib`, the API stays honest:
`GET /health` reports `model_ready: false` and `/api/v1/analyze` returns
`503 model_unavailable` — the UI shows a clear offline state, never fake data.

## 5. Verify the install (optional)

```powershell
# Backend: 174 collected (173 pass, one live-ECAPA test skips until the model is downloaded)
cd backend ; .venv\Scripts\python.exe -m pytest tests -q
# Frontend: 73 tests, then production build and SSR smoke
cd ..\frontend ; npm test ; npm run build ; npm run smoke
```

## 6. Troubleshooting

- **"python not found" during setup** — reopen PowerShell after installing, or
  use the full path to `python.exe` (App Store aliases can shadow it).
- **Backend offline in the UI** — check terminal 1 for errors; the Settings
  page's base URL must match the backend port (default `http://localhost:8000`).
- **415 on M4A upload** — install ffmpeg (step 0) and restart the backend.
- **Scripts blocked ("running scripts is disabled")** — that is why the commands
  pass `-ExecutionPolicy Bypass`; it applies only to that one invocation.
- **Port already in use** — stop the old process or change the port in
  `start-backend.ps1` / the Settings page URL.

## 7. Privacy on localhost

All audio, history and the SQLite database stay in `backend/` on your machine.
Temporary audio is deleted after each analysis; the retention policy and audit
log are visible under Security & Audit. Nothing is uploaded anywhere else.
