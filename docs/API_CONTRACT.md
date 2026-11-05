# VoxSentinel API Contract (v1)

Base URL: `VITE_API_BASE_URL` (default `http://localhost:8000`).
All JSON. Errors use the shape in section 4.

## 1. Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Backend and model readiness |
| GET | `/api/v1/models` | Available detection and speaker models |
| POST | `/api/v1/analyze` | Analyze one audio file (multipart) |
| POST | `/api/v1/speaker-verify` | Compare reference vs. test audio (multipart) |
| GET | `/api/v1/analyses` | List analyses (filter, sort, paginate) |
| GET | `/api/v1/analyses/{analysis_id}` | One analysis |
| GET | `/api/v1/evaluation` | Evaluation summary and provenance |
| GET | `/api/v1/evaluation/metrics` | Measured metrics, or `evaluated: false` |
| POST | `/api/v1/evaluation/run` | Start an async evaluation job (202; poll) |
| GET | `/api/v1/evaluation/run/{job_id}` | Poll job status and progress |
| GET | `/api/v1/audit` | Filtered, paginated audit log |
| GET | `/api/v1/admin/retention` | Current retention policy and last cleanup |
| POST | `/api/v1/admin/retention/run?dry_run=true` | Preview (`true`, default) or execute (`false`) the cleanup |
| GET | `/api/v1/analyses/export.csv` | CSV export of history (same filters as list) |
| POST | `/api/v1/synthesize` | External speech generation (Gnani.ai/Vach TTS). Audio, not JSON; synthetic by definition |
| GET | `/api/v1/providers` | Honest capability report: local detector vs. external provider |
| POST | `/api/v1/voicechat` | One speech-to-speech turn via self-hosted NemotronLabs VoiceChat 11B |

## 2. Core types (mirrored in `frontend/src/lib/types.ts`)

### GET /health
```json
{
  "status": "ok | degraded | down",
  "backend_ready": true,
  "model_ready": false,
  "speaker_model_ready": false,
  "version": "0.1.0",
  "mode": "baseline | demo | production"
}
```
`version` is the **API version** (0.1.0). The detection **model** version
(e.g. `VoxSentinel MFCC Baseline` `0.2.0`, trained on the `librispeech_piper`
corpus — see `docs/MODEL_CARD.md`) is reported in every analysis response and
in `GET /api/v1/models`, not in `/health`. With the artifact removed,
`model_ready` flips to `false` and `/api/v1/analyze` answers 503
`model_unavailable` — verified behaviour, not a mock.
```

### POST /api/v1/analyze (multipart/form-data)
Fields: `file` (audio), `consent` (must be `true`).

Response 200 — the same `AnalysisRecord` schema served by the history
endpoints, so every analysis the frontend displays has one shape:

```json
{
  "id": "uuid",
  "created_at": "ISO-8601",
  "input": {
    "display_name": "string",
    "duration_s": 0.0,
    "sample_rate_hz": 0,
    "channels": 0,
    "quality_warnings": ["string"]
  },
  "model": { "name": "string", "version": "string", "mode": "baseline|demo|production" },
  "scores": {
    "raw_score": 0.0,
    "raw_score_definition": "string",
    "calibrated_probability": null,
    "uncertainty": null,
    "uncertainty_method": null
  },
  "decision": {
    "category": "low_observed_risk | elevated_risk | high_risk | inconclusive",
    "label": "string",
    "reasons": ["string"],
    "recommendations": ["string"]
  },
  "evidence": {
    "measured_features": [{ "name": "string", "value": 0.0, "unit": "string" }],
    "model_explanation": null,
    "limitations": ["string"]
  },
  "latency_ms": 0
}
```

### POST /api/v1/speaker-verify (multipart/form-data)
Fields: `reference`, `test`, `consent` (`true`), `reference_authorized` (`true`).
`consent=false` -> 400 `missing_consent`; `reference_authorized=false` -> 400
`missing_authorization` (both explicit gates, checked before any model call).
Response includes `status: "completed" | "unavailable"`, `similarity`, `threshold`,
`outcome`, `embedding_model`, `warnings`, `limitations`. If the speaker model is
not loaded, `status` is `"unavailable"` and `similarity` is `null`.

### GET /api/v1/evaluation/metrics
If no evaluation has been run: `{ "evaluated": false }`. Otherwise the measured
metrics plus `dataset`, `model_version`, `evaluated_at`, `positive_class`, and
`n_samples`.

### POST /api/v1/synthesize (application/json)

Authorized external speech generation (Phase 6, Gnani.ai/Vach Timbre TTS).
Disabled by default; returns `503 provider_disabled` until the deployment
enables it and configures a backend-only key (see `docs/GNANI_PROVIDER.md`).

```json
{
  "text": "Text to synthesize (1..N chars; N defaults to 1000)",
  "voice": "Nalini",
  "model": "timbre-v2.5 | timbre-v3.0 (default timbre-v2.5)",
  "language": "auto | hi-IN | en-IN | ... (default auto)",
  "speed": 1.0,
  "container": "wav | raw | mp3 | ogg | mulaw | alaw (default wav)",
  "sample_rate": 48000
}
```

Response `200`: raw audio bytes (`audio/wav` etc.) with provenance headers
`X-Synthetic-Content: true`, `X-Content-Provenance: synthetic`,
`X-Provider: gnani`, `X-Provider-Model`, `X-Provider-Voice`,
`X-Provider-Language`. The response is provider output, NOT a VoxSentinel
detection result. Errors: `422` validation, `429` rate limited (local or
provider), `502` provider error, `503 provider_disabled | provider_auth_failed
| provider_unavailable`.

### GET /api/v1/providers

```json
{
  "detection": { "...": "local detector ModelInfo" },
  "detection_note": "Detection is performed by VoxSentinel's local classifier. ...",
  "generations": [
    {
      "provider": "gnani",
      "api_name": "Gnani Speech API (Vach)",
      "generation_enabled": false,
      "generation_supported": true,
      "detection_supported": false,
      "note": "The provider's documented API offers speech synthesis only. ..."
    },
    {
      "provider": "voicechat",
      "api_name": "NVIDIA NemotronLabs VoiceChat 11B",
      "generation_enabled": false,
      "generation_supported": true,
      "detection_supported": false,
      "note": "Self-hosted full-duplex speech-to-speech model (requires an NVIDIA GPU deployment of the official inference container). ..."
    }
  ]
}
```

Every listed provider reports `detection_supported: false`; detection is
always the local classifier.

### POST /api/v1/voicechat (multipart/form-data)

One conversational speech-to-speech turn via the self-hosted NVIDIA
NemotronLabs VoiceChat 11B container (Phase 6C). Disabled by default; returns
`503 provider_disabled` until an administrator enables it and configures the
WebSocket URL of a GPU deployment (see `docs/VOICECHAT_PROVIDER.md`).

Fields: `file` (user speech, same accepted formats/limits as /analyze),
optional `instructions` (system prompt for the agent).

Response `200` (the agent's audio is SYNTHETIC by construction):

```json
{
  "synthetic": true,
  "provider": "voicechat",
  "model": "nemotron-voicechat",
  "generator_id": "voicechat:nemotron-voicechat",
  "audio_b64": "<base64-encoded WAV, 24 kHz mono>",
  "audio_format": "wav",
  "sample_rate": 24000,
  "channels": 1,
  "agent_transcript": "What the agent said",
  "user_transcript": "What the user said"
}
```

Errors: `413/415/422` input problems (same semantics as /analyze), `429`
rate limited, `502` provider error/protocol violation, `503` disabled or
unavailable, `504` turn timed out / connection lost.

## 3. Status codes

| Code | When |
|---|---|
| 200 | Success |
| 400 | Malformed request, missing consent |
| 413 | File too large |
| 415 | Unsupported media type |
| 422 | Validation failure, silent or corrupted or too-short audio |
| 429 | Rate limited |
| 502 | Speech provider internal error (synthesis only) |
| 503 | Model unavailable; or synthesis provider disabled / unavailable / auth failed |
| 504 | Voice-chat provider turn timed out or connection lost |

## 4. Error shape

```json
{ "error": { "code": "file_too_large", "message": "Safe, user-facing text" } }
```

Messages never contain stack traces, file system paths, or secrets.

## 5. Change log
- **Phase 6C (voice-chat provider):** added `POST /api/v1/voicechat` — one
  speech-to-speech turn through a SELF-HOSTED deployment of NVIDIA
  NemotronLabs VoiceChat 11B (official realtime container; OpenAI-Realtime-
  compatible WebSocket protocol verified 2026-10-25). Requires an NVIDIA GPU
  server; disabled by default; agent audio is always labelled synthetic.
  `GET /api/v1/providers` now returns a `generations` list (gnani +
  voicechat), each reporting `detection_supported: false`. Detection remains
  exclusively local. See `docs/VOICECHAT_PROVIDER.md`.

- **Phase 6 (provider integration):** added `POST /api/v1/synthesize` and
  `GET /api/v1/providers` (external speech generation via the officially
  documented Gnani.ai/Vach TTS REST API; base `https://api.vachana.ai`,
  `X-API-Key-ID` auth). Generation is disabled by default, backend-key-only,
  rate limited, and always labelled synthetic. The provider offers NO
  detection API (verified 2026-10-25); the local detector remains the sole
  detection source. See `docs/GNANI_PROVIDER.md` for the verification record.
- **Phase 7 (full-system):** new endpoints above. `GET /api/v1/analyses`
  supports `category`, `min_score`, `max_score`, `limit`, `offset`, `sort`
  filters; `export.csv` streams the same filtered set. `GET /api/v1/audit`
  supports `event_type`, `action`, `outcome`, `limit`, `offset`, `before_id`.
  Every request is logged with a request id (`X-Request-Id` header). The
  speaker pipeline is now real: ECAPA-TDNN embeddings with a threshold
  calibrated on labelled LibriSpeech pairs (see `docs/MODEL_CARD.md`).
  Upload support now includes M4A/AAC/WebM/Opus via ffmpeg
  (`KNOWN_UNSUPPORTED_EXTENSIONS` is empty; when ffmpeg is missing those
  formats are rejected with 415 explaining that ffmpeg must be installed).

- **Phase 3 (unified record shape):** the analyze response now uses the same
  `AnalysisRecord` schema as the history endpoints. The earlier draft used
  `analysis_id` for the analyze response while list/get records used `id`;
  the contract now consistently uses `id`, mirrored in
  `frontend/src/lib/types.ts` (`AnalysisRecord`, with optional `evidence`).

- **Phase 4 (implementation notes):** the contract is implemented as written.
  `GET /api/v1/models` returns `{"models": [...]}` with one entry per
  pipeline (`kind`: `detection` | `speaker`), `loaded: false` and version
  `n/a` while no model is installed. Supported upload formats on the reference
  build (libsndfile 1.2.2): WAV, FLAC, OGG, MP3; M4A is rejected with `415`
  (the frontend preview still plays it locally). `/health` reports
  `backend_ready` (API + database) separately from `model_ready`.
- **Phase 5:** schema unchanged. `evidence.model_explanation` may now be a
  non-null string when the loaded detector supports an explanation (the MFCC
  logistic baseline reports its top standardized feature contributions).
  Scores from a loaded baseline are uncalibrated (`calibrated_probability`
  stays `null`); inference failures answer 503 `model_unavailable` and store
  nothing.
