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
