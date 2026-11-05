# Security and Privacy Report

## Secrets
- Provider keys exist only as environment variables (`GROQ_API_KEY`, `GEMINI_API_KEY`, …).
- `/api/v1/protection/capabilities` never returns key material (automated check in
  `backend/tests/api_checks.py`).
- The release ZIP is scanned for key patterns before distribution; the result is recorded in
  `FINAL_VALIDATION_REPORT.md`.

## Upload handling
- Explicit consent flag required (422 without it).
- Content type and size validated before decode; rejected uploads are not persisted.
- Transcript, evidence and scores are stored; source audio is not retained after processing.

## Access model
- Single-tenant local deployment: **no end-user authentication**. Do not expose the API publicly
  without a reverse proxy with auth.
- CORS: development origins allowed, unknown origins rejected (verified).

## Family voice profiles
- Reference audio stored only on explicit enrolment with consent.
- Similarity is **evidence, never proof**; ECAPA is not spoof-robust, so a cloned voice can score close
  to an enrolled relative (an impostor TTS clip scored −0.048, i.e. clearly different, in testing).
- Removing a profile deletes its reference.

## Known risks (stated, not hidden)
1. No API authentication — add one before public exposure.
2. Transcripts are sent to the LLM provider; their retention policy applies.
3. The rule engine is evadable by paraphrase; it is an explainable baseline, not a guarantee.
4. ASR error can flip a verdict (`ASR_LIMITATIONS_AND_LANGUAGE_SUPPORT.md`).
