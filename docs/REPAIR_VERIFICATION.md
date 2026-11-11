# Repair verification: SIH26104

## Release verdict

Partially functional, ML-integrated research application with repaired inference
and risk behavior. Not all objectives are complete. This is not a 100%-accuracy
release. No deployment was changed: the current request was ZIP-only.

## Executed verification

| Check | Result | Evidence |
|---|---|---|
| Frontend component/page tests | 87 passed, 12 files | verification/frontend-tests.log |
| Frontend typecheck + production bundle | Passed | verification/frontend-build.log |
| Broad backend regression | 671 passed, 7 skipped | verification/backend-broad.log |
| Remaining training/evaluation regressions, separate run | 13 passed | verification/backend-training.log |
| Numerical feature/resampling tests | 11 passed | verification/parity-tests.log |
| Included trained artifact load | Loaded, version 20261107_hi | verification/model-runtime-check.json |
| Local vs slim feature/model scoring, generated signals at 6 rates | Max score delta 0.0 | verification/runtime-parity.json |
| Speech dataset preflight | Failed honestly: 0 available rows / 1762 references | verification/dataset-readiness.json |

The backend was verified in two non-overlapping runs: 671 passed / 7 skipped
in the broad run, then all 13 tests from test_evaluation_calibration_wording.py,
test_train_candidate.py and test_training_pipeline.py passed in isolation.
Combined result: 684 passed, 7 skipped. A single combined-process full run was
not completed reliably; no root cause was established. Do not present it as a
single full-suite pass. The broad count includes the 11 parity tests and other
focused repairs; do not add those counts again. Skips include optional external
integrations, real recording fixtures and Windows symlink permissions; see logs.
Most tests use mocks or generated non-speech fixtures. They establish software
contracts, not speech detection performance. The training regressions execute
real training/evaluation code on generated test fixtures, not real speech.

## Repairs

- Removed missing-detector substitution with a heuristic authenticity verdict.
  Optional acoustic forensics is now separately labelled evidence.
- Added margin-based uncertainty: 1 - 2*abs(p - 0.5). It measures boundary
  ambiguity, not calibrated error likelihood. Existing inconclusive thresholds
  can now use the active model's non-null output.
- Added training-feature-distance warnings, explicitly heuristic and not a
  validated OOD detector.
- Added overlapping windows including short clips and trailing audio. Timestamp
  ends reflect real audio duration. Whole-clip official scoring still follows
  the original first-model-segment semantics; windows are separate evidence.
- Fixed explanation callbacks receiving unscaled instead of standardized data.
- Added actual grouped feature-replacement sensitivity for nonlinear models,
  using fitted training means. At most five extra scoring calls. This is not
  causal proof, an additive attribution guarantee or a localization ground truth.
- Corrected calibrated vs uncalibrated score wording.
- Replaced approximate inference resampling with soxr HQ and explicit dependency
  errors. Corrected STFT/cumulative-energy precision to match librosa.
- Centralized risk finalization across contextual early returns. Synthetic voice
  may impose a MEDIUM verification warning, but does not prove fraud and cannot
  independently cause HIGH/CRITICAL. Content/voice conflicts are explained.
- Fixed protection service's nonexistent assessment.actions attribute.
- Added read-only dataset/split/content-hash checks with safe-path validation.
- Separated core, full audio and test dependencies. Repaired the original
  torch/torchaudio version mismatch to 2.11.0/2.11.0; full requirements resolve
  in a dry run. Heavy runtime installation/inference was NOT verified here.
- Windows backend setup fails on dependency errors and offers -FullAudio rather
  than forcing large optional downloads. Existing UI/navigation retained.

## Before / after

| Item | Supplied code/artifact | Repair release |
|---|---|---|
| Active model weights | 20261107_hi | Preserved, not retrained |
| Speech accuracy | Historical reports only | Not independently measured |
| Active detector uncertainty | None | Explicit margin ambiguity |
| Missing trained model | Could substitute acoustic heuristic | Detection unavailable; forensics separate |
| Nonlinear feature attribution | None | Measured group replacement sensitivity |
| Short/trailing window evidence | Missing | Scored and timestamped |
| Runtime resampling | Approximate slim fallback | soxr HQ matching training path |
| Contextual early-return conflicts | Could omit voice warning | Centralized independent reasoning |
| Full optional dependencies | Mismatched torch/torchaudio pins | Matching pins, resolver checked only |

## Module acceptance

| Module | Current assessment | What remains |
|---|---|---|
| 1 Audio processing | Software regression coverage; partial acceptance | Real device/codec/noise matrix and deployment tests |
| 2 Feature extraction | Numerical paths tested across six rates | Representative speech parity/robustness corpus |
| 3 Synthetic detection | Actual supplied model loads and infers | Independent representative performance; unseen attacks |
| 4 Speaker verification | Existing ECAPA integration preserved | Full runtime + authorized same/different/noisy speech evaluation |
| 5 Uncertainty | Margin and heuristic OOD logic tested | Empirical calibration/OOD validation |
| 6 XAI | Measured sensitivity + windows implemented | Human validation, persistence of window evidence in history |
| 7 Family safety | Broad content/protection regression tests | Real Hindi/Hinglish ASR recordings and contextual-provider evaluation |
| 8 Integrated engine | Independence/conflict regressions pass | Real multi-signal audio scenarios |
| 9 Risk policy | Software policy tested | Empirical false-alarm analysis |
| 10 Training | Scripts preserved; preflight added | Licensed audio acquisition, relocatable data, representative retraining |
| 11 Evaluation | Metric code and fixture tests exist | Reproducible speech evaluation and combined-process test diagnosis |
| 12 Responsible AI | Consent/warnings retained | Authenticated privacy controls, retention review |
| 13 UI | 87 tests + build pass | Exhaustive browser/mobile interaction sweep |
| 14 Backend/database | Broad API/persistence tests pass | User authentication/ownership, persistent production database |
| 15 End-to-end | Generated fixtures and contract tests | All real speech scenarios and production verification |

`OBJECTIVE_INVENTORY.json` lists 206 original numbered objectives. Its conservative
statuses distinguish targeted code verification from full objective acceptance.
No trustworthy global implementation percentage can be calculated from this run;
unit-test pass rates are not objective completion rates.

## Dataset and accuracy limits

The all_hi manifest contains 1762 references: 1344 point to missing relative audio
and 418 use external/unsafe absolute paths. No referenced audio was hashed. Other
historical training manifests likewise reference absent external corpora (see
DATASET_READINESS.md). The all_hi training metadata contains 1098 real vs 120
synthetic rows and no Hindi synthetic training rows. These are metadata counts,
not proof that this exact artifact used those rows, and not a quality estimate.

No new speech training, AASIST/WavLM comparison, speaker FAR/FRR/EER measurement,
Hindi/Hinglish accuracy measurement, or before/after speech accuracy claim is
made. The original models and historical reports remain unchanged. Generated
signal tests and tone/noise fixture training are not genuine-vs-synthetic speech
benchmarks. A new release metric requires licensed accessible audio, reliable
labels, speaker/generator-disjoint holdouts and fresh evaluation.

## Remaining critical work

1. Install and validate full ECAPA/Whisper runtime; configure contextual provider
   securely or provide a validated local model. Measure Hindi/Hinglish ASR.
2. Acquire licensed representative speech, validate splits and training provenance,
   retrain candidates and evaluate on untouched attacks/devices/languages.
3. Implement authenticated sessions and per-user access control. Current history,
   audit, family enrollment and administrative APIs are not safe to expose to
   untrusted users. Consent flags are not authentication.
4. Use durable production storage: Vercel /tmp SQLite is ephemeral. Complete
   deployment checks with real audio on an authorized inference host.
5. Resolve the combined-process test-run inconsistency and finish exhaustive UI/device
   acceptance. Validate OOD, calibration, multiple speakers, phone-channel errors,
   and critical-content handling with human-reviewed test recordings.

No production URL is claimed as validated by this release. No secrets or new
remote-service credentials are supplied. Original database and model bytes are
preserved; packaging integrity and changed-file evidence are included separately.
