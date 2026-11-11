# Backend repair & stabilisation (2026-11-11)

Backend-focused release: the four reported workspace failures were diagnosed to
root cause, fixed, and verified end-to-end with real audio and the real models.
No model architecture, weights, thresholds or endpoint contract changed.

## Fixed
- **Live Recording (415).** Browser `audio/webm;codecs=opus` (Chromium/Firefox)
  and `audio/mp4` (Safari) are decoded server-side via an ffmpeg fallback; a
  `blob` upload with no filename works via container sniffing.
- **Analyze Audio (413 at 5.5 MB).** Uploads send the original file when the
  deployment can decode it; browser conversion is now only a fallback and the
  converted size is re-checked first. 5.5 MB and 24 MB are accepted.
- **One real ceiling.** 25 MB (26 214 400 bytes) per file, enforced by a
  streaming reader that aborts at the limit, shared by all four workflows.
  26 MB returns 413 stating the actual limit; multipart overhead is not counted.
- **Speaker Verification.** M4A/AAC decodes; a missing embedding stack returns
  `unavailable` (HTTP 200) instead of HTTP 500; the combined-body 413 is gone —
  each recording is validated independently and errors name the file.
- **Scam Protection.** Same upload path, stage-level statuses, honest
  degradation when ASR or the LLM provider fails.
- **Diagnostics.** ffmpeg/ffprobe detection at startup, structured per-request
  logging, `X-Request-ID`, and an extended `/health` (decoders, configured and
  effective upload limit, optional services, environment) that never leaks
  secrets.

## Verified
- Live, against real models: WAV/M4A/WebM/`blob` accepted; 5.5/24 MB accepted;
  26 MB → 413 with the true limit; ECAPA same-speaker 0.995 `likely_same_speaker`
  vs different-speaker 0.109 `likely_different_speaker`; real Whisper transcript;
  24 MB call recording accepted.
- Backend `pytest`: **750 passed, 5 skipped**. Frontend `vitest`: **104 passed**.
  Typecheck and production build clean (414.6 kB / 132.6 kB gzip).

## Still true, still disclosed
Unseen-generator recall remains 5.3%; no application authentication; Vercel's
`/tmp` SQLite is ephemeral and the platform caps bodies at 4.5 MB; the LLM
second pass needs a provider key; ffmpeg must be installed by the operator.

Details: `docs/BACKEND_REPAIR_2026_11_11.md` and
`docs/UPLOAD_LIMITS_AND_DECODERS.md`.

# VoxSentinel AI — Final Submission (2026-10-30)

## Verified state
- Backend: 370 passed, 2 skipped (live Gemini/Groq network tests)
- Frontend: 76/76 vitest, TypeScript clean, production build OK
- Browser E2E: 31/31 · API/security checks: 18/18 · Ruff clean
- 15 live audio scenarios executed; full table in `docs/FINAL_VALIDATION_REPORT.md`

## Contents
`backend/ frontend/ e2e/ docs/ training/ evaluation/ scripts/` plus `.env.example`.

## Excluded deliberately
`backend/data/` (datasets), `backend/*.db` (local analysis history), `node_modules/`, `dist/`,
virtualenvs, caches, logs, and all real credentials.

## Known limitation
`faster-whisper:tiny` mangles code-mixed Hinglish, which can hide a ransom script from the classifier.
Documented in `docs/ASR_LIMITATIONS_AND_LANGUAGE_SUPPORT.md`. No model was trained by this project;
Colab configurations are prepared but not executed.

## UI finalisation (2026-10-31)
Frontend-only changes, verified with TypeScript, 76/76 vitest, production build and 32/32 browser checks:
- Navigation: Model Evaluation removed from the UI; Family Emergency & Scam Protection moved last.
- Analyze: "How scores are reported" and "Consent & privacy" panels removed; raw score shown to 5 decimals;
  score descriptions, model/mode pills and the analysis identifier no longer displayed.
- Protection: detected language shown by full name; transcript and risk-assessment sections removed; Limitations retained.
- Live Recording: segment tag removed. Topbar: "API online" pill removed.
- No backend file was modified (`git status` shows frontend paths only).
