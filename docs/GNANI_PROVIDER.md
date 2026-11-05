# Gnani.ai / Vach Provider — Verified Integration Record (Phase 6)

Status: **API contract verified against official documentation on 2026-10-25.**
No live request has been made yet: the deployment platform's secret store is
not currently accessible to this backend (see "Secret handling" below).

## 1. Provider identification

| Item | Value | Source |
| - | - | - |
| Product | Gnani Speech API ("Vach" platform) | docs.gnani.ai |
| Official documentation | https://docs.gnani.ai (Mintlify-hosted; page footer: "Last verified: 5 August 2026") | fetched 2026-10-25 |
| Base URL | `https://api.vachana.ai` | docs → Gnani APIs → Introduction → Base URL |
| API keys | Generated at https://app.gnani.ai/voice | docs → Authentication |
| Legacy gRPC service (`asr.gnani.ai`) | A DIFFERENT, older API (token/accesskey/cert per email). NOT used by VoxSentinel. | github.com/gnani-ai/API-service |

The base URL was taken from the official documentation only. Nothing was
inferred from the API key's format or prefix, and no request has been sent to
any unverified host.

## 2. Verified authentication scheme

- Header on every request: `X-API-Key-ID: <api_key>`
- Missing or invalid keys return `401 Unauthorized` (documented).
- The official Python SDK (`pip install gnani-vachana`) reads the key from the
  `GNANI_API_KEY` environment variable; our backend accepts that name as a
  fallback (`VOXSENTINEL_GNANI_API_KEY` takes precedence).

## 3. Verified endpoints (documented catalog)

| Endpoint | Method + path | Purpose | Used by VoxSentinel |
| - | - | - | - |
| STT REST | `POST /stt/v3` (multipart) | Transcribe audio ≤ 60 s (WAV/MP3/OGG/FLAC/AAC/M4A, 8–44.1 kHz) | Not exposed (no product need) |
| STT Realtime | `wss://api.vachana.ai/stt/v3/stream` | Live transcription | Not exposed |
| STT Batch | Batch Jobs API | Long/bulk transcription | Not exposed |
| **TTS REST** | **`POST /api/v1/tts/inference`** | Synchronous text-to-speech, full audio in one response | **YES — `/api/v1/synthesize`** |
| TTS Streaming | `POST /api/v1/tts/sse` | SSE chunks | Not exposed |
| TTS Realtime | `wss://api.vachana.ai/api/v1/tts` | WebSocket TTS | Not exposed |
| Voice cloning | Embeddings + REST/SSE/WS | Cloned-voice TTS | Not exposed (deliberate: cloning training data is out of scope) |

**Only the TTS REST endpoint is wired into the API** (`/api/v1/synthesize` →
`app/services/gnani_client.py`). The other endpoints are verified as
documented but intentionally not exposed, keeping the attack surface small.

## 4. Verified TTS request/response schema

`POST https://api.vachana.ai/api/v1/tts/inference`
Headers: `Content-Type: application/json`, `X-API-Key-ID: <key>`

```json
{
  "text": "...",                    // required, string
  "model": "timbre-v2.5",           // required: "timbre-v2.5" | "timbre-v3.0"
  "voice": "Nalini",                // required: voice from the model's catalog
  "language": "hi-IN",              // required: auto, hi-IN, en-IN, ta-IN, te-IN,
                                    //   kn-IN, ml-IN, mr-IN, pa-IN, bn-IN, gu-IN, hi-en
  "speed": 1.0,                     // optional: 0.85–1.15
  "audio_config": {
    "sample_rate": 48000,           // 8000|16000|22050|24000|44100|48000
    "num_channels": 1,
    "sample_width": 2,
    "encoding": "linear_pcm",       // linear_pcm|pcm_s16le|pcm_mulaw|pcm_alaw|oggopus
    "container": "wav"              // wav|raw|mp3|ogg|mulaw|alaw
  }
}
```

- `200 OK` returns the full audio as binary in the requested container
  (`audio/wav` for wav, `audio/mpeg` for mp3, `audio/ogg`, `audio/basic` for
  mulaw, `application/octet-stream` for raw, `audio/alaw` for alaw).
- `encoding` is omitted when `container=mp3` (documented).
- `mulaw`/`alaw` force 8000 Hz (documented).
- Error statuses (documented): `400` bad request / unsupported voice,
  `429` rate limit exceeded, `500` internal error (retry with backoff),
  `503` temporarily unavailable.

These facts are encoded as validated constants in
`app/services/gnani_client.py`; requests are validated locally before they
are sent, and every provider status maps to a typed exception
(`GnaniAuthError`, `GnaniBadRequestError`, `GnaniRateLimitError`,
`GnaniProviderError`, `GnaniUnavailableError`, plus timeout/connection/
malformed-response classes).

## 5. Not documented / not published by the provider

Stated honestly; do not guess these:

- **Exact rate-limit quotas** for the Vach REST API are not published (the
  docs only specify `429` semantics). The legacy gRPC service documented
  250 requests / 5 concurrent, but that is a different API. VoxSentinel
  therefore treats 429 reactively (mapped to our own 429) and applies its own
  conservative in-app limits (`VOXSENTINEL_GNANI_RATE_LIMIT_PER_MINUTE=10`,
  `VOXSENTINEL_GNANI_DAILY_QUOTA=200`).
- **Maximum TTS text length** is not published. VoxSentinel enforces a local
  ceiling (`VOXSENTINEL_GNANI_MAX_TEXT_CHARS=1000`).
- Provider response headers for request IDs on the TTS REST path are not
  documented, so no request-ID correlation is claimed.

## 6. Detection capability — verified NOT offered

The complete documented catalog (docs.gnani.ai/llms.txt) covers
speech-to-text, text-to-speech, voice cloning, the Agent Builder platform,
and the Agent Builder Platform API. **No synthetic-speech detection API
exists in the official documentation.** Accordingly:

- VoxSentinel does NOT call, and does not claim to have, a provider-side
  detector. Detection remains exclusively the local classifier.
- `GET /api/v1/providers` reports this honestly:
  `generation_supported: true`, `detection_supported: false`.
- Provider results (synthesized audio) and local classifier results are
  separate domains: generation responses are labelled `X-Synthetic-Content:
  true` / `X-Content-Provenance: synthetic`, and are never mixed into the
  evaluation corpus.

If the provider later documents a detection API, it must be integrated as a
separately configured inference provider with its own response-schema
validation and attribution — never as a silent replacement of the local
detector.

## 7. Secret handling

- The API key is read ONLY from backend environment variables
  (`VOXSENTINEL_GNANI_API_KEY`, fallback `GNANI_API_KEY`), never from Vite
  `VITE_*` variables, never from browser-reachable code, and never from Git.
- `.env.example` contains placeholders only.
- The key is never logged, never echoed in error messages, and never returned
  in API responses (verified by `tests/test_synthesize_api.py::test_no_secret_in_any_response`
  and the contract-shape checks in `tests/test_security_contract.py`).
- Authorization headers and generated audio are not logged; audit events
  record only lengths, container, and provenance metadata.

### Platform secret-store finding (blocker for live calls)

The API key was saved in the development platform's secret settings, but that
store is **not injected into this backend's environment** — it holds only the
platform's own variables (verified 2026-10-25: app id, API URL, service
token; no provider key). A secret saved in the platform's settings is
available to that platform's runtime, not automatically to a standalone
FastAPI process. Before enabling external requests, the deployment
environment where FastAPI actually runs must export
`VOXSENTINEL_GNANI_API_KEY` (or `GNANI_API_KEY`) and set
`VOXSENTINEL_GNANI_ENABLED=true`. Until then the API refuses generation with
an honest `503 provider_disabled` and no network call is attempted.

## 8. How generation is exposed

`POST /api/v1/synthesize` (see docs/API_CONTRACT.md):
- Disabled by default (`VOXSENTINEL_GNANI_ENABLED=false`) and additionally
  requires a non-empty key; either missing → `503 provider_disabled`.
- Validates text (1..VOXSENTINEL_GNANI_MAX_TEXT_CHARS chars), model, voice,
  language, speed (0.85–1.15), container, sample rate locally.
- HTTPS only, bounded timeouts (connect 5 s, read 30 s, write 10 s).
- Per-client sliding-window rate limit + daily quota (in-memory).
- Optional storage (`VOXSENTINEL_GNANI_STORE_GENERATED=false` by default):
  audio is written only when explicitly enabled, always with a
  `provenance.jsonl` manifest row labelling it synthetic with generator,
  model, voice, and language.
- No temporary files are created for the request path; audio stays in memory
  unless storage is explicitly enabled.

## 9. Training workflow (explicit, never automatic)

Generated samples may become training data ONLY through an explicit,
reproducible, human-run pipeline:

1. Collect `data/generated/gnani/provenance.jsonl` (written only when
   storage is enabled).
2. Run `scripts/build_generated_manifest.py` with `--confirm-usage-rights`
   to append synthetic rows to a `train`/`validation` manifest. The script
   refuses to run without the flag, rejects rows with missing provenance, and
   can NEVER target the `eval` partition.
3. Train a candidate with `scripts/train_candidate.py` (existing pipeline);
   the production model is replaced only after the candidate is separately
   evaluated and explicitly approved.

No API request can trigger training. A generation call never retrains or
overwrites the production model.

## 10. Verification checklist (2026-10-25)

- [x] Official documentation located and read (docs.gnani.ai; docs index /llms.txt)
- [x] Base URL, auth scheme, TTS request/response schema verified from docs
- [x] Provider does NOT offer detection (verified against full docs index)
- [x] Adapter implemented only for the verified TTS REST contract
- [x] Unit tests with mocked HTTP (16 tests) and endpoint tests with an
      injected test-double client (15 tests) — no real key, no network
- [x] Backend suite: 206 passed, 1 skipped; Ruff clean
- [ ] Live request against the provider — blocked on key availability in the
      backend environment (section 7)
