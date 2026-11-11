# Backend repair & stabilisation — 11 November 2026

Scope: the four reported failures (live recording, analyze audio, speaker
verification, scam protection), plus backend availability/API connectivity.
No model architecture, weights, thresholds, embedding dimension or endpoint
contract was changed. No new backend framework or service was introduced.

## Reported symptoms → actual causes

| Reported | Actual root cause | Fix |
| --- | --- | --- |
| Live recording: `Backend returned 415` | `MediaRecorder` uploads `audio/webm;codecs=opus` (Chromium/Firefox) or `audio/mp4` (Safari). The endpoint trusted the extension/declared type and only decoded with libsndfile, which cannot read WebM/Opus/AAC. | One shared ingestion layer: signature sniffing + libsndfile with an ffmpeg fallback for WebM/Opus/M4A/AAC, normalised to 16 kHz mono. A `blob` upload with no filename works too. |
| Analyze audio: 5.5 MB rejected as too large | The frontend converted every upload to 16 kHz mono WAV *before* sending. A compressed 5.5 MB M4A expands to well over the limit as PCM, so the server legitimately rejected the derived file. A second, unbounded `file.read()` in the protection route also allowed huge bodies into RAM. | Upload the **original** when the deployment can decode it; convert only as a fallback and re-check the size first. One streaming reader (1 MB chunks, abort at the ceiling) shared by all audio routes. |
| Speaker verification: "ffmpeg is not installed" | ffmpeg detection existed but the decoder path did not fall back to it reliably, and the message was static. A missing torch/ECAPA stack produced an unhandled exception → HTTP 500. | ffmpeg/ffprobe are detected once, reported at startup and in `/health`, and used as the decoding fallback. A missing embedding stack returns the documented `unavailable` result (HTTP 200, `similarity: null`) instead of a 500. |
| Speaker verification: HTTP 413 | The **combined** multipart body (two files) was compared against the per-file limit. | Each recording is size-checked and decoded independently; errors name the offending file (`The reference file is …`). Two 20 MB files (40 MB body) are accepted. |
| Protection: "Backend not running / Analysis failed / 413" | Same upload-size and decoding faults, plus buffering the whole body and no stage-level error reporting. | Same ingestion layer, plus a stage-level report so a broken ASR or LLM degrades one stage instead of failing the analysis. |

## What changed, file by file

Backend (no architecture change):

* `app/services/ingestion.py` — **new**, the single audio ingress: bounded
  streaming read, real size ceiling, container sniffing, decoder selection,
  honest error mapping, and one structured log line per upload.
* `app/services/audio_validation.py` — decoding split into libsndfile and an
  ffmpeg fallback with decoder capability probing; unchanged duration/silence
  rules and unchanged feature contract.
* `app/services/ffmpeg_decoder.py` — binary/version detection, cached, with an
  explicit configurable path.
* `app/api/routes.py` — all four audio endpoints routed through ingestion;
  speaker comparison validates each file independently.
* `app/core/config.py` — `max_upload_bytes` (25 MB), `hosting_body_limit_bytes`,
  `effective_upload_limit_bytes()`, `ffmpeg_binary`/`ffprobe_binary`.
* `app/core/errors.py` — 422 messages now name the missing/invalid fields
  (values are still never echoed).
* `app/main.py` — startup diagnostics for decoders and the upload policy;
  `X-Request-ID` on every response for log correlation.
* `app/api/routes.py::health` + `app/schemas/analysis.py` — additive health
  block: decoders, decodable extensions, configured/effective limit, hosting
  cap, optional services, environment.

Frontend (behaviour only, no design change):

* `src/lib/api.ts` — `getUploadPolicy()` caches `/health`; `prepareUpload()`
  sends the original when possible, converts otherwise, re-checks the size;
  API messages are surfaced verbatim; speaker verification labels which file
  failed.
* `src/lib/types.ts` — additive, all-optional health fields.
* `src/pages/AnalyzeAudioPage.tsx` — drops the unconditional conversion.

## Verification performed

Real HTTP requests against a locally running uvicorn with the real detector
(`20261107_hi`) and real ECAPA-TDNN. Speech fixtures were synthesised with
ffmpeg's `flite` filter (two distinct voices); they exercise decoding and
plumbing, **not** detection accuracy.

| Check | Result |
| --- | --- |
| WAV / M4A-AAC / WebM-Opus / `blob`-filename uploads | 200, decoded to 16 kHz mono, window evidence present |
| 5.5 MB and 24 MB uploads | 200 (previously 413) |
| 26 MB upload | 413 `file_too_large`, message states 26 MB vs 25 MB |
| Truncated WebM | 422 `invalid_audio`, "corrupted, truncated or use an unsupported codec" |
| Speaker verification, same voice (WAV vs M4A) | 200, `completed`, similarity 0.995, `likely_same_speaker` |
| Speaker verification, different voices | 200, `completed`, similarity 0.109, `likely_different_speaker` |
| Speaker verification, 24 MB reference | 200 `completed` |
| Speaker verification, 26 MB reference | 413 naming the reference file |
| Speaker verification without authorization | 400 `missing_authorization` |
| Protection, real Whisper transcription | 200; stages completed, risk computed, transcript returned |
| Protection, 24 MB / 26 MB / non-audio | 200 / 413 / 415 |
| `/health` | decoders, effective limit, per-service availability, no secrets |

Automated suites:

| Suite | Result |
| --- | --- |
| Backend `pytest` | **750 passed, 5 skipped** (was 686) |
| Frontend `vitest` | **104 passed, 14 files** (was 91/13) |
| `tsc --noEmit` + `vite build` | clean, 414.6 kB / 132.6 kB gzip |

New backend test files: `test_upload_limits.py` (size matrix 1/5.5/10/20/24/25
MB, 26 MB, hosting cap, streaming abort), `test_live_recording_api.py`
(WebM/MP4/M4A/blob/no-extension/truncated/provisional), 
`test_decoders_and_health.py` (health contract, CORS, `X-Request-ID`, error
envelope, missing-ffmpeg behaviour), `test_speaker_verify_uploads.py`
(per-file 413, M4A, consent/authorization, no-500 guarantee),
`test_protection_uploads.py` (size limits, failed ASR, failing LLM, evidence
keys). New frontend file: `src/lib/upload.test.ts`.

## Unchanged limitations (still true)

* Detection performance is unchanged: seen-generator F1 0.960 with 5.4% FPR,
  but unseen-generator recall is 5.3%. Do not quote this repair as an accuracy
  improvement.
* No application authentication or per-user ownership. Consent flags are not
  authentication. Do not expose history, audit or family-enrolment endpoints to
  untrusted users.
* SQLite on Vercel `/tmp` is ephemeral, and Vercel rejects bodies over 4.5 MB,
  so the effective ceiling there is 4.5 MB unless a container host is used.
* The scam module's LLM second pass needs a provider key; without one the stage
  reports `unavailable` and the rule engine result stands alone.
* ffmpeg must be installed by the operator. `pip install ffmpeg-python` does not
  install the executable. See `UPLOAD_LIMITS_AND_DECODERS.md`.
* Test fixtures are generated/synthesised. They prove software behaviour, not
  speech-detection performance.
