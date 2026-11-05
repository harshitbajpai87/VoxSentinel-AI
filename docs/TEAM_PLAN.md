# Team Plan

Each teammate commits from their **own** GitHub account with their own verified
email. No shared accounts, no rewritten history, no spoofed authorship.

| Area | Owner |
|---|---|
| Frontend integration, upload workflow, dashboard, results UI, frontend tests | Harshit Bajpai |
| Python backend, preprocessing, model integration, evaluation pipeline, backend tests | Oisiky Sinha |
| Architecture review, integration testing, README | Both |

## Branches
`main`, `feat/frontend-dashboard`, `feat/audio-upload`, `feat/backend-api`,
`feat/model-pipeline`, `feat/evaluation`.

## Workflow
1. Open a GitHub issue per module.
2. Branch from `main`, commit small, open a pull request.
3. The other teammate reviews before merge.
4. Resolve conflicts carefully; never force-push `main`.

Configure Git once per machine:
```bash
git config --global user.name  "Your Real Name"
git config --global user.email "the-email-on-your-GitHub-account"
```


## Phase 4 assignments

These are suggested work assignments for reviewing, testing and committing the
Phase 4 backend — not a record of who did what. Each teammate commits their
own reviewed work under their own account.

**Oisiky Sinha — backend foundation**
- FastAPI application factory, configuration and CORS
- Health/readiness and models endpoints
- Database setup (SQLAlchemy + SQLite) and analysis persistence
- Error handling and audio-validation service
- Backend README

**Harshit Bajpai — integration and verification**
- API contract review against `frontend/src/lib/api.ts`
- Frontend integration checks (health, model readiness, upload errors)
- Frontend and integration test runs
- Security review of upload handling and CORS
- Documentation and deployment verification


## Phase 5 assignments (suggested split; not a record of completed work)

**Oisiky Sinha — dataset preparation, features, training**
- `scripts/prepare_dataset.py`, `scripts/make_fixture_dataset.py`
- `app/services/features.py` (shared MFCC path), `app/services/manifest.py`
- `scripts/train_detector.py` and reproducibility metadata

**Harshit Bajpai — loading, integration, tests, evaluation, docs**
- `app/services/model_loader.py` validation rules
- Serving integration (`detector.py` TrainedVoiceDetector, routes)
- `scripts/evaluate_detector.py` metrics + report
- docs/MODEL_CARD.md, docs/DATASET_GUIDE.md, test coverage

## Phase 5–6 wrap-up note (2026-10-22)

Model v0.2.0 was trained on the open `librispeech_piper` corpus (LibriSpeech
real speech vs Piper VITS neural TTS; build commands in
`docs/DATASET_GUIDE.md` §2). Held-out eval (unseen TTS voice `en_US-amy-medium`
AND speaker-disjoint test-clean readers, n=160): F1 0.9317, ROC-AUC 0.9386,
EER 0.0750. An EER implementation bug (argmin tie-breaking on flat ROC
segments) was found by the new regression tests, fixed in
`app/services/metrics.py`, and both reports regenerated with the corrected
math — the old EER 0.05 must not be quoted. Scope: results describe this
specific human-vs-Piper-TTS setup only, not general voice-clone detection.
Speaker verification remains honestly unavailable pending an embedding model.

## Full-system wrap-up note (2026-10-23)

Speaker verification is now real: SpeechBrain ECAPA-TDNN embeddings
(`models/speaker_ecapa`, lazy ~83 MB download) with a decision threshold
calibrated on 80 labelled LibriSpeech dev-clean pairs and evaluated on 80
held-out test-clean pairs (EER 0.025, ROC-AUC 0.9994, FAR/FRR 0.025; see
`docs/MODEL_CARD.md`). The live server had been running on a placeholder
threshold artifact (`eer_threshold: 0.5`, marked TEST FIXTURE); regenerating
`models/speaker_thresholds.json` with `scripts/evaluate_speaker.py` fixed it —
the calibrated 0.3448 threshold now ships and is what `/api/v1/speaker-verify`
uses. Phase 7/8 features are live: filterable history + CSV export, async
evaluation jobs, per-window scores with explainability, audit log with request
ids, retention policy (dry-run + execute; `days <= 0` = keep forever by
design). Backend: 146 tests green, ruff clean. Frontend: 73 tests green,
build clean, 9/9 SSR routes (run `npm run smoke` with tsx; vite-node loads
duplicate react-router instances and fails). Remaining: deployment (Phase 9)
and the SIH demo acceptance checklist (Phase 10).
