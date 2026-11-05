# Gemini Contextual Scam Classification

Additive upgrade to the Family Emergency & Scam Protection module. The synthetic
speech detector, Whisper transcription, database schema, existing routes and the
rule engine are unchanged in behaviour.

## Architecture

```
audio ─► validate/decode ─► Whisper transcript ─► ┌─ Gemini (primary)     ─┐
                                                 ├─ rule baseline (support)├─► risk policy ─► API/UI/JSON
                                                 └─ speaker compare (opt)  ┘
```

Gemini receives ONLY the redacted transcript (long digit runs such as phone or
account numbers are replaced with `[number]`). Audio is never sent. Whisper runs
locally, so audio does not leave the machine; the transcript text does leave it
when Gemini is configured — the UI states this clearly.

## Configuration

Backend environment only (never `VITE_*`):

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Server-side key. Unset → rule-based fallback, shown honestly. |
| `GEMINI_MODEL` | Model id, default `gemini-2.5-flash`. |
| `GEMINI_TIMEOUT_S` | Per-request timeout, default 20 s. |
| `GEMINI_MAX_TRANSCRIPT_CHARS` | Transcript cap, default 6000. |

Verify: `GET /api/v1/protection/capabilities` → `gemini.configured`.

## Safety properties

- Fixed server-side system instruction; the transcript is untrusted data inside
  `<transcript>` delimiters. Injected instructions are analysed, never followed.
- Strict Pydantic schema (enums, lengths, list caps). Invalid output → failure,
  never a partial guess.
- Every `evidence_quote` must appear in the transcript or it is discarded and
  counted in `ai.gemini.dropped_quotes`.
- No numeric probabilities are accepted from the model.
- Retries: transient only (429/5xx, 3 attempts). 401/403/400 fail fast.
- Keys are never logged, returned, or exposed in the bundle.

## Risk policy (deterministic)

| Gemini classification | Result |
|---|---|
| LIKELY_SCAM + CRITICAL, or ransom/family-emergency category | CRITICAL |
| LIKELY_SCAM | HIGH |
| SUSPICIOUS + credential/ransom category | HIGH |
| SUSPICIOUS | MEDIUM |
| EDUCATIONAL_OR_DISCUSSION | LOW (MEDIUM if the voice detector suspects synthesis) |
| INCONCLUSIVE / LIKELY_LEGITIMATE | rules decide; voice suspicion can still escalate |

When Gemini is unavailable the rules decide and the response says so
(`ai.role = unavailable_fallback_to_rules`, plus a stage entry and warning).

## Testing

```bash
cd backend
python -m pytest tests/ -q                    # 278 tests, live Gemini skipped
python -m pytest tests/test_gemini.py -q      # Gemini unit + pipeline tests
python -m scripts.eval_scam_rules             # rule-baseline metrics (offline)
VOXSENTINEL_LIVE_GEMINI=1 GEMINI_API_KEY=... python -m pytest tests/test_gemini.py -q
```

The live test is skipped without a key. No metrics are reported for Gemini
until a live key is configured — offline results cover the rule baseline only.
