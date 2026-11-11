# Vach (Gnani) speech integration

One backend credential drives both halves of the voice pipeline:

| Direction | Endpoint | Used by |
| --- | --- | --- |
| Speech → text | `POST https://api.vachana.ai/stt/v3` | Family scam protection (transcript → scam classification) |
| Text → speech | `POST https://api.vachana.ai/api/v1/tts/inference` | Scam-audio generation for testing and demos |

Verified against the live provider on 2026-11-11. Nothing in this document is
inferred from the key format; every claim below was observed in a real response.

## Verified contract

```
POST https://api.vachana.ai/stt/v3
X-API-Key-ID: <key>
Content-Type: multipart/form-data

audio_file     <wav, 16-bit mono PCM>
language_code  indic-auto        # or en-IN, hi-IN, ta-IN, ...
```

Success:

```json
{"success": true, "transcript": "...", "model": "prisma-v3.0",
 "detected_language_code": "en-IN", "request_id": "..."}
```

Failure:

```json
{"success": false, "error": {"type": "AUDIO_DURATION_EXCEEDED",
                             "message": "..."}}
```

Facts that shaped the implementation:

- **30-second audio cap.** Anything longer fails with
  `AUDIO_DURATION_EXCEEDED`, so the client splits long calls at pauses and
  rejoins the text. Verified with a 40 s and a 75 s input.
- **Language codes are a closed set:** `as-IN, bn-IN, en-IN, gu-IN, hi-IN,
  indic-auto, kn-IN, ml-IN, mr-IN, pa-IN, ta-IN, te-IN`. `auto` and `hi-en` are
  rejected with HTTP 400, so `indic-auto` is the auto-detect option.
- **`success: false` can arrive with HTTP 200.** A 200 alone is not success.
- The detected language comes back as `detected_language_code`, and Hindi comes
  back in Devanagari (verified with a ransom sample).

## Credential handling

The key is resolved once, in `app/core/config.py::_resolve_vach_api_key`, and is
read from the backend environment only. First non-empty wins:

1. `VOXSENTINEL_VACH_API_KEY`
2. `VOXSENTINEL_GNANI_API_KEY` (explicit legacy name)
3. `API_KEY` (the shared deployment secret)
4. `GNANI_API_KEY` (the provider SDK's documented variable)

Consequences:

- Setting **only** `API_KEY` on the platform is enough to turn on
  transcription *and* text-to-speech.
- Text-to-speech auto-enables when a key is present, so dropping in the
  credential lights up both services. Set `VOXSENTINEL_GNANI_ENABLED=false` to
  keep TTS off while STT stays on.
- A key pasted with wrapping whitespace is normalised; that is the usual cause
  of a spurious 401.

### What is never exposed

The key is never logged, never placed in an exception message, never returned in
a status payload, and never put in a `VITE_*` variable. `describe()` reports
`configured: true/false` only. Verified by asserting the key is absent from
`GET /api/capabilities`, `GET /api/v1/protection/capabilities`, a full
`POST /api/v1/protection/analyze` response, and every error path in
`tests/test_vach_asr.py`.

## Where the code lives

| File | Role |
| --- | --- |
| `app/services/vach_asr.py` | ASR client, chunk planner, WAV encoding, typed errors |
| `app/protection/asr_engines.py` | `GnaniAsrEngine` adapter + engine chain ordering |
| `app/core/config.py` | Single credential resolution and STT settings |
| `tests/test_vach_asr.py` | 20 tests, fully offline via `httpx.MockTransport` |
| `tests/test_risk_calibration.py` | 17 acceptance tests for the decision policy |

### Engine order

Accuracy first: the large local model, then the hosted Indic model, then
progressively smaller local models.

```
whisper_large_turbo → gnani_asr → faster_whisper_large
  → whisper_small → indic_hindi → faster_whisper
```

On a host without `torch`/`transformers` (including serverless deployments) the
local engines report `available: false` with the real reason, and `gnani_asr`
serves the request. The engine name stays `gnani_asr` so existing saved
analyses, capability payloads and dashboards keep resolving.

## Chunking behaviour

`MAX_CHUNK_SECONDS = 28.0`. The provider's hard limit is 30 s; the margin
absorbs boundary rounding.

Before a hard cut, the planner looks back up to 2 s for the quietest 20 ms frame.
If that frame is under 35% of the chunk's average energy, the cut moves there, so
a word is not sliced in half. If there is no clear pause, it cuts at the ideal
boundary rather than inventing one.

Each chunk reports its own status. A partial failure produces a transcript
marked as partial in `limitations`, never a silently short one.

## Calibration for this use case

The decision policy was calibrated against this project's own labeled
transcripts, because the failure that matters for a family product is twofold: a
missed ransom call, and a false alarm on an ordinary call with a parent (which
teaches people to ignore the warning).

Measured after tuning:

| Dataset | Cases | Accuracy | Recall | Precision |
| --- | --- | --- | --- | --- |
| `tests/data/scam_eval_v2.jsonl` (tuning set) | 54 | 100% | 100% | 100% |
| `data/finetune/valid.jsonl` (held out) | 39 | 100% | 100% | 100% |
| `data/finetune/train.jsonl` (held out) | 192 | 100% | 100% | 100% |

Before calibration the tuning set scored 83% with 5 missed scams and 4 false
alarms.

Three policy changes produced that:

1. **A single ordinary word no longer raises an alarm.** A transcript scoring
   1–2/10 stays LOW; "I sent my son money for books" was previously MEDIUM.
   From 3/10 there is at least one corroborated pattern, which is MEDIUM.
2. **A concrete fraud script cannot fall through.** A transcript matching a real
   fraud category (upfront fee, courier/customs pretext, remote access,
   job/prize lure) at 4–6/10 is MEDIUM even without the coercion flags.
3. **The report describes what was actually said.** Generic money and threat
   phrases had been filed under `ransom_or_kidnapping`, so a plain "send money
   or else" call was described as a kidnapping claim. They now live in
   `emergency_money_request` and `threats_or_coercion`. A custody claim requires
   an actual custody phrase, in Devanagari or romanised Hinglish ("Beti hamare
   paas hai" — previously a real miss).

### Honest limits

- These numbers come from this project's own small labeled sets, mostly
  text-written rather than recorded. They are an acceptance floor for
  regression, not a claim about accuracy on noisy telephone audio.
- The rule baseline is deterministic keyword and pattern matching. It is
  explainable and offline, but it cannot understand novel social engineering.
  The contextual LLM layer remains the primary signal when configured.
- Automatic transcription can mishear names, amounts and accents. Every
  transcript carries that warning and the evidence quotes shown to the user are
  copied from the transcript, not paraphrased.

## Reproducing

```bash
cd backend
pytest tests/test_vach_asr.py tests/test_risk_calibration.py -q   # new
pytest tests -q --ignore=tests/test_audio_quality.py \
                 --ignore=tests/test_dataset_pipeline.py          # full suite
```

Live check against the provider, with a key configured:

```bash
curl -s -X POST https://api.vachana.ai/stt/v3 \
  -H "X-API-Key-ID: $API_KEY" \
  -F "audio_file=@sample.wav;type=audio/wav" \
  -F "language_code=indic-auto"
```

Known pre-existing suite failures are environmental: `test_features*`,
`test_model_loader.py`, `test_train_candidate.py`, `test_training_pipeline.py`
and `test_detector_ready.py` need `scikit-learn`, `torch` and `soundfile`,
which are not installed in a bare environment. They fail identically on the
untouched original.
