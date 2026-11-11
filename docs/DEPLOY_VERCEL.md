# Deploying VoxSentinel AI on Vercel

Live deployment: <https://voxsentinel-ai-iota.vercel.app>

## What Vercel can and cannot run

Vercel executes Python as short-lived serverless functions with a 250 MB unzipped
bundle limit and no system packages. This project's audio stack does not fit:

| Component | Requirement | Fits Vercel? |
|---|---|---|
| rule engine, risk scoring, schemas, history | pure Python + scikit-learn (~40 MB) | **yes** |
| synthetic-voice detector | `models/detector.joblib` (0.9 MB) | **yes** (bundled via `includeFiles`) |
| speaker verification | torch CPU (~200 MB) + ECAPA weights (85 MB) | no |
| ASR | faster-whisper / ctranslate2 + model weights | no |
| audio decoding | libsndfile / ffmpeg binaries | no |

### Cloud fallbacks that make it usable anyway

* **Transcription** — `app/services/vach_asr.py` sends the clip to Vach
  (Gnani) at `POST https://api.vachana.ai/stt/v3` and reports engine
  `gnani_asr`, so `/capabilities` never claims a local model ran. This is the
  supported transcription path on Vercel: it needs no local model, no system
  packages, and only the backend secret `API_KEY`. Verified on this host with
  English, Hindi and Hinglish clips (see `docs/VACH_INTEGRATION.md`).
  * Audio is capped at 30 s per provider request, so longer calls are chunked at
    pauses and reassembled; each chunk reports its own status.
  * `language_code` defaults to `indic-auto` (the provider's auto-detect).
    `auto` and `hi-en` are rejected by the provider, so do not use them.
  * The Gemini audio path (`app/protection/transcription_remote.py`) still
    exists as an alternative, but it is no longer what runs here.
* **Decoding** — `soundfile` 0.14 ships its own libsndfile, so WAV/FLAC/OGG/MP3
  decode with no system packages. M4A/AAC/WebM still need ffmpeg and answer 415.
* **Contextual scam analysis** — the Gemini classifier runs with `GEMINI_MODEL`
  set to a model that is not rate-limited (`gemini-3.5-flash` here).

Verified end to end on the live URL: an 11 s speech clip returns a real
transcript, language, scam-content analysis, contextual reasoning and a risk
category.

So without those keys the deployment is a **frontend + metadata API**:

* the full React UI is served statically from `frontend/dist`
* `/health`, `/api/v1/models`, `/api/v1/protection/capabilities`, history and
  admin endpoints answer from the real FastAPI app
* `capabilities` reports `transcription.available: false` and
  `speaker_verifier_ready: false` instead of pretending: the heavy imports are
  blocked on purpose in `api/index.py`, so the honesty checks in the app see the
  truth
* uploading audio requires `API_KEY` (the Vach speech credential) — without it
  transcription is genuinely unavailable and the upload is refused with a clear
  message instead of pretending, while text-only analysis keeps working
* the deepfake detector and family-voice comparison stay unavailable: they need
  librosa → numba/llvmlite and torch/speechbrain, roughly 100 MB and 285 MB
  unzipped respectively, on top of numpy/scipy/scikit-learn already in the
  bundle. Vercel's 250 MB function limit rules both out

For the complete pipeline (deepfake detector + speaker verification), run
`backend/` on a container host using `deploy/hf/Dockerfile`. The frontend then
points at it through Settings → backend URL, or the `VITE_API_BASE_URL` build
variable. No code change is needed: `frontend/src/lib/api.ts` defaults to the
page's own origin in a deployed build and to `http://localhost:8000` in local dev.

Host notes (checked 6 Nov 2026): Hugging Face now refuses Docker Spaces on the
free `cpu-basic` plan ("requires a PRO subscription"), so `deploy/hf/publish.sh`
works only on a PRO account. Render's free web service (512 MB) is below what
torch + Whisper need; a 2 GB container is the realistic minimum.

## Files that make this work

* `vercel.json` — static build of `frontend/`, SPA fallback, security headers,
  and the function config (`includeFiles: backend/models/**`, 60 s, 1 GB)
* `requirements.txt` (repo root) — slim serverless dependency set
* `api/index.py` — ASGI entry that sets serverless-safe env (SQLite in `/tmp`),
  blocks torch/speechbrain/whisper imports, then exposes the real `create_app()`

## Redeploy

```bash
npm i -g vercel            # once
vercel login
cd <repo root>
vercel deploy --prod -e VOXSENTINEL_ALLOWED_ORIGINS=https://voxsentinel-ai-iota.vercel.app
```

Or with a token, no login prompt:

```bash
VERCEL_TOKEN=... vercel deploy --prod --yes --token "$VERCEL_TOKEN" \
  -e VOXSENTINEL_ALLOWED_ORIGINS=https://voxsentinel-ai-iota.vercel.app
```

Project: `harshitbajpayi-8636/voxsentinel-ai`. The production alias stays
`voxsentinel-ai-iota.vercel.app` across deployments.

## Optional env vars for the hosted API

| Var | Effect on Vercel |
|---|---|
| `API_KEY` | **required** for transcription on this host. The one Vach (Gnani) secret: same value also enables text-to-speech. Aliases accepted in priority order: `VOXSENTINEL_VACH_API_KEY`, `VOXSENTINEL_GNANI_API_KEY`, `API_KEY`, `GNANI_API_KEY`. Set it as a Vercel secret for Production, Preview and Development |
| `VOXSENTINEL_VACH_STT_MAX_CHUNK_SECONDS` | chunk window for calls longer than the provider's 30 s cap (default 28) |
| `GEMINI_API_KEY` | optional; enables the contextual LLM classifier. Transcription no longer depends on it |
| `GEMINI_MODEL` | classifier model; `gemini-3.8-flash` is the code default but was answering 503 on 6 Nov 2026, so this deployment pins `gemini-3.5-flash` |
| `VOXSENTINEL_ASR_MODEL` / `VOXSENTINEL_ASR_FALLBACKS` | only used by the Gemini audio path, not by Vach |
| `MISTRAL_API_KEY` | same, via the Mistral adapter |
| `VOXSENTINEL_ALLOWED_ORIGINS` | CORS allowlist if you host the API and UI on different domains |
| `VOXSENTINEL_DB_URL` | defaults to `sqlite:////tmp/voxsentinel.db` here; `/tmp` is not persistent, so history does not survive a cold start. Use a hosted Postgres for a durable demo |

## Torch-free audio authenticity on this host

The trained detector and ECAPA speaker model cannot run here, so the deployment
uses two torch-free modules instead of reporting "unavailable":

* `app/services/acoustic_forensics.py` measures four physical properties and
  scores them: high-band energy ratio (a 16 kHz render leaves nothing above
  8.5 kHz), exact-zero runs between phrases, pitch-contour smoothness and
  loudness uniformity. The elevated band sits at 0.50 and the module reports
  "heuristic screen", never a calibrated probability.
* `app/services/speaker_mfcc.py` compares MFCC mean/std/delta statistics with
  cosine similarity. It reports a closeness value only: no threshold is
  calibrated for it, so no same/different-speaker verdict is produced.

Both are switched on by `VOXSENTINEL_FORENSICS_ENABLED` and
`VOXSENTINEL_MFCC_SPEAKER_ENABLED` (default `true`, set explicitly in
`api/index.py`). Set either to `false` to go back to the honest "unavailable"
state. A full container host still uses the trained detector and ECAPA.

## What the 2026-11-11 backend repair changes here

The shared ingestion layer is active on Vercel too, so failures now explain
themselves instead of surfacing as a bare 415/413:

| Upload | Result on Vercel | Why |
| --- | --- | --- |
| WAV / MP3 / FLAC / OGG | works (libsndfile ships in the wheel) | no system package needed |
| WebM-Opus, M4A/AAC (every browser/phone recording) | **503 `decoder_unavailable`**, message names the formats that do work | Vercel functions have no ffmpeg binary |
| anything over 4.5 MB | **413 `file_too_large`** stating the *effective* 4.3 MB ceiling | platform body cap; `VOXSENTINEL_HOSTING_BODY_LIMIT_BYTES` defaults to 4 500 000 when `VERCEL` is set |
| speaker verification | `unavailable`, no 500 | torch/ECAPA cannot fit the 250 MB bundle |
| scam protection transcription | stage reports `unavailable`, rule engine still scores | faster-whisper is not installed here |

`/health` reports all of this up front: `audio_decoder_ready`,
`effective_max_upload_bytes`, decodable extensions, and per-service availability.
Use `scripts/deploy_vercel.sh` (= frontend + slim API) for a demo URL, and a
container host for the full audio pipeline.

## Known limits of this deployment

1. The voice score is a heuristic screen, not the trained classifier, and
   speaker comparison gives closeness without a verdict. Full-fidelity
   detection needs the container build.
2. History is per-instance and ephemeral (`/tmp`).
3. Request bodies over **4.5 MB are rejected by the platform**, so the honest
   upload ceiling here is 4.5 MB, not 25 MB. The API reports the real figure as
   `effective_max_upload_bytes` in `/health` and in every 413 message, and
   `VOXSENTINEL_HOSTING_BODY_LIMIT_BYTES` defaults to 4 500 000 on Vercel. Use
   a container host (Render/Railway/Fly/Docker) for full-size uploads; see
   `UPLOAD_LIMITS_AND_DECODERS.md`.
3. Cold starts add ~1 s while the detector artifact loads.
4. No persistent file storage for family-voice profiles.

## Production environment variables (verified 7 Nov 2026)

| Var | Value | Why |
|---|---|---|
| `GEMINI_API_KEY` | secret | Without it the build falls back to local faster-whisper, which is absent here, so every upload returns an empty transcript |
| `GEMINI_MODEL` | `gemini-3.5-flash` | `gemini-3.8-flash` returns 503; the fallback list drops to `gemini-3.1-flash-lite` on 429 |
| `PROTECTION_LLM_PROVIDER` | `gemini` | Leaves the contextual classifier on a provider that actually works; an unset value picks the first configured key, which was an expired xAI key |

Environment changes only take effect on a new deployment (`vercel deploy --prod`).

## Live verification (7 Nov 2026)

| Input | Result |
|---|---|
| Rendered Hindi clip (27 s, MP3) | voice 0.65 / threshold 0.50 -> `SUSPECTED_SYNTHETIC`, risk `MEDIUM`, transcript 358 chars, no warnings |
| Real scam recording (WAV) | risk `CRITICAL`, `family_emergency_impersonation`, voice 0.05 (human-like) |
