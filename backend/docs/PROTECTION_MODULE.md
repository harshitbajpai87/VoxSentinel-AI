# Family Emergency & Scam Protection

Additive module. Nothing in the existing analysis pipeline, detector, models,
database tables or API contracts was modified.

## How it works

`POST /api/v1/protection/analyze` runs, in order:

| Stage | Implementation | Notes |
|---|---|---|
| Audio validation | existing `validate_and_decode` | same limits as the core app |
| Transcription | faster-whisper, local CPU | multilingual incl. Hindi / Hinglish |
| Language ID | whisper | probability reported only when the model provides it |
| Voice authenticity | existing detector, unchanged | same thresholds as the core app |
| Family voice comparison | existing ECAPA verifier | only when an enrolled profile is selected |
| Scam content analysis | rule engine (`app/protection/rules.py`) | EN + Hindi + Hinglish patterns, combination logic, discussion/awareness damping |
| Risk assessment | `app/protection/risk.py` | transparent rules; CRITICAL only for strong combinations |
| Report | JSON export endpoint | transcript + evidence + reasoning + limitations |

## API

- `GET  /api/v1/protection/capabilities`
- `POST /api/v1/protection/analyze` (multipart: `file`, `consent=true`, optional `profile_id`)
- `GET  /api/v1/protection/analyses` · `GET/DELETE /api/v1/protection/analyses/{id}`
- `GET  /api/v1/protection/analyses/{id}/report` (JSON download)
- `GET/POST /api/v1/protection/family-voices` · `DELETE /api/v1/protection/family-voices/{id}`

## Configuration

- `VOXSENTINEL_WHISPER_MODEL` (default `tiny`; use `small`/`medium` for better Hindi).
- First run downloads the Whisper model from Hugging Face (~75 MB for tiny).

## Honest limitations

- The scam classifier is a deterministic contextual rule engine, not an LLM.
  It is explainable and fast but cannot understand arbitrary paraphrases the way
  a language model can. An LLM hook can be added behind the same interface.
- A suspicious transcript does not prove a kidnapping or fraud occurred.
- Speaker similarity is raw cosine; it cannot prove identity.
- Whisper `tiny` is fast but has a meaningful word error rate on noisy phone
  audio; use a larger model for production accuracy.
- No telephone-codec-specific fine-tuning has been done.

## Privacy

- Audio is processed in-process and is not sent to any external service.
- Reference recordings are stored under `data/family_voices/` and deleted with
  the profile. Transcripts follow the project's retention policy.

---

# Gemini contextual classification (v12)

## Role in the pipeline

Gemini is the **primary contextual classifier**. The keyword rule engine is kept
as an independent baseline/safety layer and as the fallback when Gemini is
unavailable. The synthetic-voice detector and family-voice comparison are
separate components and are never merged into a single score.

```
audio → validate/decode → Whisper transcript → [Gemini contextual analysis]
                                            → [rule baseline]
                                            → [synthetic-voice detector]
                                            → [optional family-voice comparison]
                                            → deterministic risk policy → report
```

Only the **transcript** is sent to Gemini. Raw audio never leaves the server.

## Configuration (server-side only)

```ini
GEMINI_API_KEY=...          # never a VITE_* variable, never committed
GEMINI_MODEL=gemini-3.8-flash
GEMINI_TIMEOUT_S=60
GEMINI_MAX_TRANSCRIPT_CHARS=6000
```

`GET /api/v1/protection/capabilities` reports `configured`, `model` and `role`
without exposing the key.

## Live verification status (30 Oct 2026)

Verified against the real API with a valid key:

| Check | Result |
|---|---|
| Auth + endpoint reachability | **PASS** — `200 OK`, model replied `OK` |
| Model availability | `gemini-2.5-flash` **retired** for new keys (404, API instructs `gemini-3.8-flash`); `gemini-3.8-flash` **works** |
| Full scenario classification | **BLOCKED** — free-tier daily quota (20 req/day) exhausted mid-test |
| Fallback behaviour under quota exhaustion | **PASS** — `gemini_rate_limited`, clear warning, rule result served, no fabricated Gemini output |

**Not verified live:** end-to-end scenario accuracy on real Gemini responses.
All 26 Gemini unit tests run against a mocked transport, and one integration test
proves the full API path with a mocked provider.

Retry hint when quota is exhausted is returned by the provider (`retry in ~9h50m`
in this run). For production use, enable billing on the Gemini API key or raise the
project quota.

## Cost and latency control

- Transcript capped at `GEMINI_MAX_TRANSCRIPT_CHARS` (default 6000).
- Single request per analysis; **no** second validation model.
- Retries only on `429/500/502/503/504`, at most two, with backoff.
- Never retried: `400/401/403/404`.
- First call can be slow (cold start ~30-60s). The UI allows 240s for this module.

## Prompt-injection posture

- Fixed server-side system instruction; the transcript is wrapped in
  `<transcript>` delimiters.
- The model has no tools, no file or database access, and cannot trigger
  payments, calls or account changes.
- Output is validated by a strict Pydantic schema; evidence quotes are checked
  against the transcript and discarded when not found.
- Personal data (long digit runs) is redacted before the request is sent.

## Schema tolerance (v12.1)

Provider casing and taxonomy drift are normalised before validation: enum-like
fields are compared case-insensitively, unknown `scam_categories` map to
`OTHER_SUSPICIOUS_ACTIVITY`, and each remap is reported in `status.notes`.
`classification` and `risk_level` remain strict — an unrecognised core label
fails validation rather than being guessed.

---

# Provider switch: Grok (xAI) replaces Gemini (v14)

## What changed

The contextual classification layer is now a **provider-neutral stage** with two
adapters. Grok is the default; Gemini remains selectable.

| | Grok (default) | Gemini (optional) |
|---|---|---|
| Module | `app/protection/grok.py` | `app/protection/gemini.py` |
| Endpoint | `POST https://api.x.ai/v1/chat/completions` | `POST .../v1beta/models/{model}:generateContent` |
| Auth | `Authorization: Bearer $XAI_API_KEY` | `x-goog-api-key` header |
| Default model | `grok-4.7` | `gemini-3.8-flash` |
| Env vars | `XAI_API_KEY`, `XAI_MODEL`, `XAI_TIMEOUT_S`, `XAI_MAX_TRANSCRIPT_CHARS` | `GEMINI_API_KEY`, `GEMINI_MODEL`, ... |
| Select with | `PROTECTION_LLM_PROVIDER=xai` (default) | `PROTECTION_LLM_PROVIDER=gemini` |

Both adapters share exactly the same guarantees, because they share the same
schema and validators:

- Strict Pydantic schema (`GeminiResult`), enum-checked, length-bounded.
- Every `evidence_quote` must appear in the transcript, otherwise it is dropped
  and the count is reported.
- Personal identifiers (long digit runs) are redacted before the request leaves
  the server.
- Transcripts are wrapped in `<transcript>` tags and treated as data; fixed
  server-side system instructions cannot be overridden by caller speech.
- Never raises: returns `(result | None, status)` so the rule baseline always runs.
- Retries only on `429/5xx`, with backoff; `400/401/403/404` fail immediately.
- API keys never appear in logs, responses, exports or status payloads.

## API compatibility

The report keeps the historical `ai.gemini` key as an alias of `ai.llm` so
records written before the switch still render. New fields:

```json
"ai": {
  "llm": { "provider": "xai", "provider_label": "Grok (xAI)", "status": "completed", ... },
  "gemini": { ...same object... },
  "provider": "xai",
  "provider_label": "Grok (xAI)",
  "role": "primary_contextual_classifier"
}
```

The pipeline stage is now named `contextual_ai_analysis` (was
`gemini_contextual_analysis`). The UI labels the section from `provider_label`,
so it reads "Grok (xAI) contextual analysis" today and still shows "Gemini" on
older saved analyses.

## Setup

1. Get a key at `console.x.ai`.
2. Set `XAI_API_KEY` in the **backend** environment (never in the frontend).
3. Optionally set `XAI_MODEL` (e.g. a cheaper/faster Grok variant).
4. Restart the API. `GET /api/v1/protection/capabilities` reports
   `"provider": "xai", "configured": true` when the key is loaded.
5. Without a key the module still works: every report says which provider was
   unavailable and marks the result as rule-based.

## Live verification status

- **Verified:** endpoint, auth header shape, JSON-mode request body, and error
  mapping are exercised by 15 mocked tests; the 404 path is validated against a
  real "model unavailable" response shape.
- **Not verified:** live Grok classification accuracy. No `XAI_API_KEY` was
  configured when this build was produced, so the opt-in live test is skipped.
  Run `XAI_API_KEY=... pytest tests/test_grok.py -k live -q` to verify.

---

# Live provider validation (v14, October 30 2026)

## Important: Grok (xAI) vs Groq

They are **different providers with confusingly similar names**:

| | Grok | Groq |
|---|---|---|
| Company | xAI | Groq, Inc. |
| Host | `api.x.ai` | `api.groq.com/openai/v1` |
| Key prefix | `xai-...` | `gsk_...` |
| Models | grok-4.x family | open models (GPT-OSS, Qwen, Llama, Kimi) |

The key supplied for this build was a **`gsk_` Groq key**. xAI rejected it with
`400 invalid-argument: Incorrect API key provided`. Rather than stop there, a
**Groq adapter** was added on the same provider-neutral path, and the live
pipeline was validated against it end to end.

## Supported providers

| Provider | Env | Default model | Status |
|---|---|---|---|
| `groq` | `GROQ_API_KEY` | `openai/gpt-oss-120b` | **verified live** |
| `xai` | `XAI_API_KEY` | `grok-4.7` | adapter ready, no xAI key present |
| `gemini` | `GEMINI_API_KEY` | `gemini-3.8-flash` | legacy, still selectable |

`PROTECTION_LLM_PROVIDER` picks one explicitly; if unset, the first provider with
a configured key is used, so dropping in a key is enough to enable it.

## Live results (real API calls)

Text classification through the Groq adapter:

| Scenario | Classification | Risk | Latency |
|---|---|---|---|
| Hinglish ransom demand | LIKELY_SCAM | CRITICAL | 1.7 s |
| Hinglish bank/OTP request | LIKELY_SCAM | HIGH | 1.7 s |
| Normal family check-in | LIKELY_LEGITIMATE | LOW | 1.0 s |
| Routine bill confirmation | LIKELY_LEGITIMATE | LOW | 1.0 s |
| Educational scam discussion | EDUCATIONAL_OR_DISCUSSION | LOW | 0.8 s |

Full pipeline through `POST /api/v1/protection/analyze` with a real ransom
recording: transcription, speaker check, rule baseline, Groq contextual pass and
risk fusion produced **LIKELY_SCAM / CRITICAL / KIDNAPPING_OR_RANSOM** with three
verified indicators in **2.06 s** total LLM time, model reported as
`openai/gpt-oss-120b`.

## Robustness added for open models

Open models do not always respect the requested schema. Two defensive layers were
added, both fact-preserving:

1. **Structural normalisation.** Flat shapes are mapped onto the canonical schema:
   `evidence_quotes` → `indicators`, `risk` → `risk_level`, `FULL/GOOD/CLEAR` →
   `ADEQUATE`, thin indicators get neutral provenance text. Nothing is invented:
   quotes are kept verbatim and still verified against the transcript.
2. **Free-text category mapping.** Labels like "Kidnap/Extortion" resolve to the
   taxonomy (`KIDNAPPING_OR_RANSOM`) by keyword, with deterministic priority
   tie-breaks. Ambiguous labels still fall back to `OTHER_SUSPICIOUS_ACTIVITY`.

Unknown `classification` / `risk_level` values are still rejected, so the risk
policy can never be softened by a malformed model answer.
