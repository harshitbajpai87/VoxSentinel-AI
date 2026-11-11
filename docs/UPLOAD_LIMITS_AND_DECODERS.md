# Upload limits, audio decoding and error codes

Everything about "the file is too large", "this format is not supported" and
"the decoder is missing" in one place. Every audio endpoint follows the same
rules; there is exactly one implementation (`backend/app/services/ingestion.py`)
and no per-route overrides.

## The one policy

| Rule | Value | Where it is enforced |
| --- | --- | --- |
| Maximum size **per uploaded file** | 25 MB (26 214 400 bytes) | `VOXSENTINEL_MAX_UPLOAD_BYTES` → `Settings.max_upload_bytes`, applied by the streaming reader |
| Effective ceiling | `min(25 MB, hosting body limit)` | `Settings.effective_upload_limit_bytes()`, reported in `/health` |
| Hosting body cap | unset locally, **4.5 MB on Vercel by default** | `VOXSENTINEL_HOSTING_BODY_LIMIT_BYTES` |
| Multipart overhead | **not** counted against the file | only the file's own bytes are summed |
| Minimum duration | 2 s | accepted below it, but a quality warning is returned |
| Maximum duration | 600 s | `VOXSENTINEL_MAX_DURATION_S` |
| Internal format after decoding | mono PCM float32, 16 kHz | `audio_validation.decode_*` |

The reader streams the upload in 1 MB chunks and **stops as soon as the ceiling
is crossed**, so a 500 MB body is never buffered in memory. Temporary files are
removed on success and on failure.

### Why the ceiling can be lower than 25 MB

A reverse proxy or hosting platform can reject a request before the application
ever sees it. That limit cannot be raised from Python. The app therefore:

1. reads the platform limit from `VOXSENTINEL_HOSTING_BODY_LIMIT_BYTES`,
2. reports the resulting effective ceiling in `/health`
   (`effective_max_upload_bytes`) and in every 413 message,
3. never claims 25 MB when the deployment can only accept 4.5 MB.

Set that variable to match your infrastructure:

```bash
# Render / Railway / Fly / Docker behind nginx: usually no body cap
VOXSENTINEL_HOSTING_BODY_LIMIT_BYTES=

# Vercel (serverless) rejects bodies over 4.5 MB; this is the default there
VOXSENTINEL_HOSTING_BODY_LIMIT_BYTES=4500000

# nginx as a front proxy
client_max_body_size 26m;              # nginx.conf, above the app's 25 MB
```

## Audio decoding matrix

The server detects the real container from the file's bytes, not from the
filename or the declared `Content-Type`. The filename extension is only used to
reject obvious non-audio and to report a useful name.

| Format | libsndfile | ffmpeg fallback | Notes |
| --- | --- | --- | --- |
| WAV / WAVE / RF64 | yes | yes | always available |
| MP3 | yes (1.2+) | yes | |
| FLAC | yes | yes | |
| OGG / OGA (Vorbis) | yes | yes | |
| OPUS / WebM-Opus | no | **yes** | what Chromium/Firefox `MediaRecorder` produces |
| M4A / MP4 / AAC | no | **yes** | what Safari/iOS `MediaRecorder` and phone recorders produce |
| AIFF, AU, CAF, W64 | yes | yes | |

Decoded audio is resampled to the model's 16 kHz mono and the first 4 s are used
for detection; the full clip is kept for the window/segment scores.

## Installing ffmpeg (required for M4A/AAC/WebM/Opus)

`pip install ffmpeg-python` does **not** install the ffmpeg executable. Install
the binaries with the OS package manager.

```bash
# Debian / Ubuntu (Docker, Render, Railway, Fly)
apt-get update && apt-get install -y --no-install-recommends ffmpeg

# Alpine
apk add --no-cache ffmpeg

# macOS (local development)
brew install ffmpeg

# Windows (local development)
winget install Gyan.FFmpeg      # or: choco install ffmpeg
```

Dockerfile example:

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
```

If the binaries live somewhere unusual, point at them explicitly:

```bash
VOXSENTINEL_FFMPEG_BINARY=/usr/local/bin/ffmpeg
VOXSENTINEL_FFPROBE_BINARY=/usr/local/bin/ffprobe
```

### Startup and health diagnostics

The app logs both decoders at startup:

```
audio decoders | ffmpeg=True (ffmpeg version 6.1.1 ...) | ffprobe=True | libsndfile=1.2.2
upload policy  | configured=25 MB | effective=25 MB | hosting_body_limit=none
```

`GET /health` reports the same facts so the UI can tell the difference between
"backend offline" and "this server cannot decode M4A":

```json
{
  "status": "ok",
  "backend_ready": true,
  "audio_decoder_ready": true,
  "max_upload_bytes": 26214400,
  "effective_max_upload_bytes": 26214400,
  "audio": {
    "decoders": {
      "ffmpeg": true,
      "ffprobe": true,
      "ffmpeg_version": "ffmpeg version 6.1.1",
      "libsndfile": "1.2.2",
      "m4a_aac_webm_opus": true,
      "notes": []
    },
    "decodable_extensions": ["aiff", "flac", "m4a", "mp3", "ogg", "opus", "wav", "webm"],
    "max_upload_bytes": 26214400,
    "effective_max_upload_bytes": 26214400,
    "hosting_body_limit_bytes": null
  },
  "services": {
    "detector": true,
    "speaker_verifier": true,
    "transcription": true,
    "contextual_ai": false
  },
  "environment": { "platform": "Linux", "python": "3.11.9" }
}
```

Missing ffmpeg is **not** an outage: `/health` stays `ok` with
`audio_decoder_ready: false`, and only the formats that need it are refused.

## Error contract

Every rejection uses the same envelope:

```json
{ "error": { "code": "file_too_large", "message": "The audio file is 26 MB, above this deployment's 25 MB upload limit." } }
```

| Status | `code` | Meaning | What the user should do |
| --- | --- | --- | --- |
| 413 | `file_too_large` | file exceeds the **effective** ceiling; the message names the file, its size and the real limit | shorten the clip or raise the limit |
| 415 | `unsupported_media_type` | the bytes are not audio, or the container cannot be decoded by this deployment | convert to WAV/MP3/FLAC/OGG |
| 422 | `invalid_audio` | empty, truncated, corrupted, silent, or outside the duration window | re-record / re-export |
| 422 | `validation_error` | a required field is missing (the message names it) | fill it in |
| 400 | `missing_consent` / `missing_authorization` | consent checkbox or reference authorization not given | confirm consent |
| 503 | `decoder_unavailable` | this deployment has no ffmpeg, so M4A/AAC/WebM/Opus cannot be decoded; the message lists the formats that still work | install ffmpeg (above) or upload WAV/MP3/FLAC/OGG |
| 503 | `model_unavailable` | the detector or speaker model is not installed | install the model |
| 500 | `internal_error` | unexpected server error; the traceback stays in the server log, correlated by `X-Request-ID` | report with the request id |

Speaker verification validates **each recording independently**, so the message
names the offending file (`The reference file is 26 MB ...`). Two files of 20 MB
each are fine: the combined 40 MB body is not what the limit measures.

## Live recording

`MediaRecorder` output is accepted as-is:

* `audio/webm;codecs=opus` (Chromium, Firefox)
* `audio/webm`, `audio/ogg;codecs=opus` (older Firefox)
* `audio/mp4` (Safari)

The upload may arrive with a `blob` filename and no extension at all: the
container is identified from the signature bytes. Nothing is transcoded on the
server beyond normalising to 16 kHz mono for the model.

The frontend behaves as follows (`prepareUpload` in `frontend/src/lib/api.ts`):

1. ask `/health` what this deployment decodes and what its real ceiling is,
2. if the format is decodable → upload the **original** bytes (no re-encoding,
   no size explosion),
3. if not → convert in the browser to 16 kHz mono WAV and check the size again
   **before** uploading, so an oversized conversion fails with a clear message
   instead of an opaque 413.

## Quick self-check

```bash
curl -s localhost:8000/health | jq '.audio_decoder_ready, .effective_max_upload_bytes'

# 5.5 MB must be accepted
curl -s -o /dev/null -w '%{http_code}\n' -F "file=@clip.wav" -F consent=true \
  localhost:8000/api/v1/analyze

# 26 MB must be 413 with the real limit in the message
curl -s -F "file=@huge.wav" -F consent=true localhost:8000/api/v1/analyze | jq .error
```
