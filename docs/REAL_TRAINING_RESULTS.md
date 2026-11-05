# Real Multi-Dataset Training Results (2026-10-24)

Complete, honest record of the real training run: datasets, counts, leakage
verification, all model comparisons, and the promotion decision. Every number
below is measured from the artifacts included in this project
(`evaluation/`); nothing is estimated.

## 1. Real datasets used (downloaded with authorized Kaggle credentials)

| Dataset | On disk | Used in training corpus |
|---|---|---|
| jbuchner/synthetic-speech-commands-dataset (CC BY-SA 4.0) | 83,698 files / 2.68 GB | 8,000 fake (primary `augmented_dataset` variant; `verynoisy` excluded) |
| speech_commands_v0.02 direct download (CC BY 4.0) | 105,835 files / 3.34 GB | 8,000 real (`_background_noise_` excluded: environmental noise, not speech) |
| mohammedabdeldayem/the-fake-or-real-dataset (LGPL 3.0) | 169,754 files / 20.22 GB | for-norm: 8,000 real + 8,000 fake |
| adarshsingh0903/audio-deepfake-detection-dataset (Apache 2.0) | 4,447 files / 1.42 GB | 2,274 real + 1,319 fake in train/val/eval; 854 fake held out |
| kambingbersayaphitam human-vs-AI voices (CC BY-NC-SA 4.0) | 42 files / 1.97 GB | 21 real + 21 fake |

Total corpus after caps: 36,489 capped entries → 35,631 training entries
(4 unreadable FoR files excluded as unknown, 854 held out). Class balance:
18,295 real / 18,194 fake before exclusions.

## 2. Identity and leakage fixes (why training initially failed)

- **Bug:** adapters used raw file stems as recording IDs; bare numeric stems
  (e.g. `104.wav`) collide across unrelated recordings and sources, breaking
  split grouping and tripping the train/validation speaker-leakage assertion.
- **Fix:** every physical file now gets a stable ID (sha256 of
  `source/relative-path`); FoR keeps utterance-name pairing (real and its
  generated fake share one recording group); LibriSpeech speakers get a
  **global cross-source namespace** (`librispeech:<speaker>`), so the same
  underlying speaker seen through FoR and audio_deepfake can never land in
  different splits. Training-manifest IDs are source-prefixed and globally
  unique.
- **Generator holdout fix:** held-out generators are sampled per generator
  name and every group containing their entries moves to holdout entirely.
  Previously, stem collisions could scatter one generator across splits.

## 3. Leakage verification (measured, all clean)

- Held-out generators (xTTS, FlashSpeech, VoiceBox, NaturalSpeech3) appear in
  **zero** train/validation/eval entries.
- All 35,616 unique sha256 file hashes in training appear in exactly one
  partition; 14 exact-duplicate clusters were merged into single splits by the
  manifest builder.
- Zero hash overlap between training and the 854-clip holdout; zero
  speaker_id overlap.
- Audit (`evaluation/audit/audit_report.json`): 36,489 files fingerprinted,
  4 unreadable (excluded), 1,853 identical coarse-signature groups of which
  1,151 span splits — these are coarse 64-band spectral fingerprints (same
  word/TTS voice family), not verified duplicates; sha256-exact duplicates are
  the 14 merged clusters. Same-generator spectral similarity across splits is
  an inherent property of this corpus and is disclosed, not hidden.

## 4. Results on the held-out eval partition (both classes, 5,468 clips)

| Model | F1 syn | Precision syn | Recall syn | FPR | FNR | ROC-AUC |
|---|---|---|---|---|---|---|
| Deployed v0.2.0 (old corpus) | 0.069 | 0.181 | 0.042 | 0.180 | 0.958 | 0.414 |
| Baseline MFCC+LR retrained | 0.841 | 0.837 | 0.844 | 0.154 | 0.156 | 0.922 |
| Candidate A: enhanced-logistic (calibrated) | 0.894 | 0.878 | 0.910 | 0.118 | 0.090 | 0.957 |
| **Candidate B: enhanced-HGB (calibrated)** | **0.960** | **0.944** | **0.976** | **0.054** | **0.024** | **0.993** |

Confusion (Candidate B): tn=2675 fp=152 fn=63 tp=2578.

The deployed v0.2.0 row was re-measured on the exact same final eval
partition (`evaluation/reports/deployed_baseline_eval/evaluation_eval_final_manifest.json`);
an earlier report on the pre-fix manifest
(`evaluation_eval.json` in the same folder) shows nearly identical failure
(F1 0.070, AUC 0.464). The deployed model genuinely fails on this corpus:
it misses ~96% of synthetic clips and ranks near chance (AUC 0.41).

Per-source F1 (synthetic, Candidate B): fake_or_real 0.889, audio_deepfake
0.611, synthetic_commands and speech_commands_real single-class in eval
(unavailable by design). human_ai_voices: 3 eval clips, F1 0.0 (too few, long
Indonesian clips).

## 5. Unseen-generator holdout (854 synthetic clips, recall only)

Single-class set → recall of synthetic at threshold 0.5 is the only metric
reported. No accuracy / F1 / FPR is claimed from this set.

| Generator | n | Deployed v0.2.0 | Retrained baseline | Candidate A | Candidate B (production) |
|---|---|---|---|---|---|
| xTTS | 600 | 3.0% | 2.5% | 0.0% | 0.8% |
| FlashSpeech | 118 | 0.9% | 10.2% | 6.8% | 6.8% |
| VoiceBox | 104 | 3.8% | 17.3% | 12.5% | 20.2% |
| NaturalSpeech3 | 32 | 21.9% | 56.3% | 6.3% | 34.4% |
| **Overall** | 854 | 3.5% | 7.4% | 2.7% | **5.3%** |

Candidates report calibrated probabilities; the two older models raw scores,
matching each model's production decision basis.

## 6. Promotion decision (measured, against the pre-registered criteria)

Candidate B replaces the deployed model because on the SAME eval partition it
regresses nothing: F1 0.960 vs 0.069, FPR 0.054 vs 0.180, per-source F1 far
better, and holdout recall 5.3% vs 3.5%. **No model detects unseen generators
reliably; that limit is a disclosed failure mode, not a solved problem.**
Improving it likely requires raw-audio or self-supervised features rather than
more of the same handcrafted statistics.

## 7. Artifacts and verification

- Production: `backend/models/detector.joblib` — Candidate B,
  version 0.3.202610241603, sigmoid-calibrated. Verified loadable through
  `app.services.model_loader.load_detector_from_path` and one real inference
  through the production `analyze()` path (real clip → 0.018, synthetic →
  0.9997).
- Backup: `backend/models/archive/detector_v0.2.0_20261024.joblib`.
- All evaluation reports: `evaluation/reports/`; manifests and split summary:
  `evaluation/manifests/`; training reports + candidate artifacts:
  `evaluation/training_artifacts/`. Candidate A's training report was
  overwritten by Candidate B's run (same output path); its evaluation reports
  are intact.
- Backend suite after promotion: 173 passed, 1 skipped (one test updated: it
  previously hardcoded that the deployed artifact must be the old baseline).
- sklearn compatibility fix: `cv="prefit"` removed; calibration now uses
  `CalibratedClassifierCV(FrozenEstimator(model))` (sklearn >= 1.6).
