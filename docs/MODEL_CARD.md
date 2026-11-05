# Model Card — VoxSentinel Synthetic-Speech Detector

**Production model:** VoxSentinel Enhanced MFCC Candidate (HistGradientBoosting
over enhanced features), version **0.3.202610241603**, mode `candidate`.
**Artifact:** `backend/models/detector.joblib`
**Previous model:** `backend/models/archive/detector_v0.2.0_20261024.joblib`
(MFCC+LogisticRegression baseline, trained on librispeech_piper only).

> **STATUS: promoted 2026-10-24 after real multi-dataset training.** The
> production model was trained on a balanced 35,631-clip corpus drawn from five
> real audio datasets (Speech Commands, Synthetic Speech Commands, FoR
> for-norm, audio-deepfake-detection, human-vs-AI Indonesian voices), with
> leakage-safe speaker/recording-group splits, sha256 duplicate merging, and
> four named TTS generators (xTTS, FlashSpeech, VoiceBox, NaturalSpeech3)
> reserved entirely for an unseen-generator holdout. Measured results below are
> the only claims made; the score is **sigmoid-calibrated on validation data**,
> and `calibrated_probability` is reported only because calibration was
> actually performed. Removing `models/detector.joblib` makes the API answer
> `503 model_unavailable` — nothing is fabricated when the artifact is absent.
>
> **Known hard limit (measured):** unseen TTS generators defeat this model at
> the 0.5 threshold (5.3% recall on the 854-clip holdout; xTTS 0.8%). Any use
> must assume modern unseen generators are largely missed.

## Intended use

- Decision support for triaging audio that *may* be synthetic, in research and
  demonstration contexts.
- Producing a calibrated score plus a risk category with explicit
  recommendations for secondary, independent human verification.
- Running inside the VoxSentinel API, which pairs the score with input-quality
  findings, limitations and abstention rules.

## Out-of-scope uses

- Any autonomous accept/reject decision on identity, payments or access.
- Evidence in legal, employment or immigration contexts.
- Speaker identification or similarity claims.
- Any claim that a recording is definitively genuine or synthetic.

## Method

1. Input: mono float32 audio, resampled to 16 kHz, first 4 s (zero-padded if
   shorter).
2. Features (91 dims): 20 MFCC mean+std (40), delta-MFCC mean+std (40),
   spectral shape: centroid/rolloff/flatness/band-energy mean+std (11).
3. `StandardScaler` fit on training data only; `HistGradientBoostingClassifier`
   (200 iters, lr 0.1, seed 42).
4. Sigmoid calibration fit on the validation split only
   (`CalibratedClassifierCV(FrozenEstimator(model))`). The eval partition was
   never used for training, calibration or threshold selection.
5. Explanation: top standardized feature contributions, dispatched by artifact
   kind in the detector service.

## Training data (real, downloaded 2026-10-24)

| Source | Used (train+val+eval) | Notes |
|---|---|---|
| speech_commands_real (v0.02) | 8,000 real | genuine volunteer speech, speaker-grouped |
| synthetic_commands (jbuchner) | 8,000 fake | synthetic-only TTS corpus, primary variant |
| fake_or_real (for-norm) | 8,000 real + 8,000 fake | official splits honored, LibriSpeech speakers grouped globally |
| audio_deepfake (adarshsingh0903) | 2,274 real + 1,319 fake (train/val/eval) | per-generator provenance; 854 fake held out |
| human_ai_voices (Indonesian) | 21 real + 21 fake | long clips, first 4 s used |

Caps: 8,000 per class per source (no dataset dominates). 4 unreadable FoR
files excluded as `unknown`. sha256 exact-duplicate clusters merged into one
split (14 clusters). Verified: no held-out generator, file hash, or speaker id
appears in more than one split.

## Measured results (v0.3.202610241603)

**Held-out eval partition** (5,468 clips: 2,827 real / 2,641 synthetic, both
classes, generators seen in training):

- F1(synthetic) **0.960**, precision 0.944, recall 0.976
- FPR **0.054**, FNR 0.024, ROC-AUC **0.993**
- Confusion: tn=2675 fp=152 fn=63 tp=2578

**Unseen-generator holdout** (854 synthetic clips; synthetic recall only —
single-class set, no accuracy/F1/FPR claimed): overall **5.3%**
(xTTS 0.8%, FlashSpeech 6.8%, VoiceBox 20.2%, NaturalSpeech3 34.4%).

**Comparison on the same eval partition:** deployed old v0.2.0 F1 0.069 /
FPR 18.0% / AUC 0.414 (fails on this corpus); retrained MFCC+LR F1 0.841 /
FPR 15.4% / AUC 0.922; enhanced-logistic (Candidate A) F1 0.894 / FPR 11.8% /
AUC 0.957.

## Evaluation protocol

- Group-aware splits: a speaker/recording group (and any sha256 duplicate
  cluster) lands in exactly one of train / validation / eval / holdout.
- The eval partition is untouched by training, calibration and selection.
- Positive class `synthetic`. FPR = real clips flagged synthetic; FNR =
  synthetic clips passed as real. Threshold 0.5 (documented in every report).
- Reports: `evaluation/reports/*` (per-source and per-generator breakdowns);
  reproduction workflow in `notebooks/multidataset_training.ipynb` and
  `docs/MULTIDATASET_UPGRADE.md`; full result tables in
  `docs/REAL_TRAINING_RESULTS.md`.

## Limitations

- Unseen generators are largely missed (see measured holdout recall above).
- Only the first 4 s of each clip feed the model.
- Non-commercial CC BY-NC-SA source (human_ai_voices) restricts product use.
- Decision thresholds (elevated >= 0.35, high >= 0.70) are provisional defaults,
  not validated operating points.
