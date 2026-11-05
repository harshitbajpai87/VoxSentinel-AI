# Multi-Dataset Training Upgrade (v0.3.0)

Honest record of the multi-dataset training upgrade: what was verified, what
was built, what actually ran, and what still needs a real training
environment.

> **Bottom line:** the full pipeline is implemented and validated end-to-end
> on generated fixtures (manifest building, duplicate detection, splitting,
> baseline + candidate training, held-out evaluation, serving). The real
> Kaggle datasets were **verified as accessible** but were **not downloaded
> in the build environment** (no Kaggle credentials; ~26 GB total). No real
> training run has happened yet, so **no real-data metrics exist and none
> are claimed**. The production baseline model is unchanged. Follow
> `notebooks/multidataset_training.ipynb` to produce the real numbers and the
> promotion decision.

## 1. Audit of the existing project (preserved contracts)

| Component | State before upgrade | Effect of upgrade |
|---|---|---|
| API (`/analyze`, `/models`, `/history`, evaluation, speaker, audit routes) | Stable contract (docs/API_CONTRACT.md) | **Unchanged.** No route, schema or frontend change. |
| Features (`app/services/features.py`) | MFCC mean/std, 40 dims | **Unchanged** (baseline artifact untouched). |
| Detector + model loader | `kind == "mfcc_baseline"` only | Extended: kinds now `mfcc_baseline` + `mfcc_enhanced`; older artifacts still load. |
| Training scripts | `train_detector.py`, `prepare_dataset.py` | Kept; new scripts added alongside. |
| Production model | `models/detector.joblib` (MFCC+LR) | **Still the production model.** Candidate NOT promoted. |
| Tests | 145 passed, 1 skipped | 171 passed, 1 skipped after upgrade (26 new). |
| Frontend | 73 tests, tsc clean, vite build | Unchanged; still green after upgrade. |

## 2. Dataset verification (measured 2026-10-23)

Accessibility verified live against Kaggle's public API. The Speech
Synthesis link in the request is a **search page**, not a dataset, and was
skipped as a download source.

| # | Dataset (Kaggle ref) | Verified size | License | Status in build env |
|---|---|---|---|---|
| 1 | `jbuchner/synthetic-speech-commands-dataset` | 2.68 GB | CC BY-SA 4.0 | accessible, **not downloaded** (no credentials) |
| 2 | `kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices` | 1.97 GB | CC BY-NC-SA 4.0 (non-commercial) | accessible, not downloaded |
| 3 | (search page `tags=16720-Speech+Synthesis`) | n/a | n/a | skipped: not a dataset identifier |
| 4 | `mohammedabdeldayem/the-fake-or-real-dataset` | 20.2 GB | LGPL 3.0 | accessible, not downloaded |
| 5 | `adarshsingh0903/audio-deepfake-detection-dataset` | 1.42 GB | Apache 2.0 | accessible, not downloaded |
| + | `speech_commands_v0.02` (direct URL, verified HTTP 200) | 2.43 GB | CC BY 4.0 | verified, not downloaded |

Structural findings (from each dataset's own published description):

- **(1) is synthetic-only** (synthetic counterparts to real Speech Commands).
  Per the upgrade plan it is supplemented with **Speech Commands v0.02**,
  the authentic volunteer recordings. The manifest builder refuses to train
  synthetic-vs-nothing.
- **(2)** is 42 long WAVs (21 human, 21 AI), Indonesian, 4-9 min each.
  Labels come from its class folders; only the first 4 s of each clip feed
  the current models (see Limitations).
- **(4) FoR** has four published versions; the pipeline defaults to
  **for-norm** (balanced, normalized), configurable via `--for-version`.
- **(5)** description self-contradicts on sample rate (16 vs 24 kHz); the
  inspector measures the real rates rather than trusting either claim.

## 3. What was built

| File | Purpose |
|---|---|
| `app/services/multisource.py` | Unified manifest schema (`file_path, label(real/fake/unknown), source_dataset, speaker_id, recording_id, generator_id, language, split`), label mapping with `unknown` for anything unverified, sha256 duplicate clustering, deterministic group-aware splitting with unseen-generator holdout, conversion to the training manifest. |
| `app/services/features_enhanced.py` | Enhanced features (MFCC + delta-MFCC + spectral shape, 91 dims), same determinism/NaN rules as the baseline path. |
| `scripts/multisource_download.py` | Credential-safe Kaggle + direct downloads; writes `dataset_registry.json` with status/license/size; honest `verified_accessible_not_downloaded` when unauthenticated; `--dry-run`. |
| `scripts/multisource_inspect.py` | Per-dataset inspection: sample rates, channels, durations, exact duplicates (sha256), near-duplicate candidates (spectral fingerprint). |
| `scripts/build_multisource_manifest.py` | Per-source adapters (verified structure only), unified manifest, split summary, training manifest (unknown/holdout excluded). |
| `scripts/train_candidate.py` | Candidate training (enhanced-logistic / enhanced-hgb), optional sigmoid calibration, per-source/per-generator validation breakdown, provenance-rich report. |
| `notebooks/multidataset_training.ipynb` | The full real workflow in Colab: credentials via Colab Secrets, downloads, inspection, manifest, baseline + candidate training, held-out comparison, promotion. |

Modified (backward compatible):

- `app/services/manifest.py`: `ManifestEntry` gains optional `generator_id`
  and `language` (defaults `None`; legacy manifests unchanged).
- `app/services/model_loader.py`: kind-to-config mapping; validated
  CalibratedClassifierCV artifacts; explanation dispatch by kind.
- `app/services/detector.py`: feature extraction dispatched by artifact
  kind; `calibrated_probability` reported only when the artifact is actually
  calibrated (otherwise stays `null`).
- `scripts/evaluate_detector.py`: adds `breakdown.by_source` and
  `breakdown.by_generator` to evaluation reports (single-class groups get an
  explicit "not computable" note, never a fabricated number).

## 4. Leakage protection (verified by tests)

1. Splits are assigned to **groups** (speaker or recording), never single
   files; all of a speaker's clips land in one split.
2. **Exact duplicate clusters** (sha256, across all sources) are merged into
   one split unit. The end-to-end fixture run planted a byte-identical file
   across two sources; the pipeline detected it and kept both copies in the
   same split.
3. **Unseen generators** can be reserved as an independent `holdout` split
   (`--holdout-generator-fraction`, default 0.2); holdout generators never
   appear in train/validation/eval.
4. The **eval partition is never touched by training scripts**; model
   selection happens on validation only.
5. `unknown` labels are excluded from supervised training, never guessed.

## 5. Fixture validation (what actually ran here)

The build environment has no Kaggle credentials and ~26 GB of downloads are
out of scope, so the pipeline was validated on a generated fixture corpus
mimicking all five source layouts (101 WAVs, real+fake classes, one planted
cross-source duplicate). **Fixture metrics are trivially perfect (the
classes are synthetic signals) and are NOT synthetic-speech detection
performance.** The run proved mechanics:

- `multisource_download.py --dry-run`: all 5 sources verified accessible.
- `multisource_inspect.py`: measured rates/durations, duplicate detection OK.
- `build_multisource_manifest.py`: 101 unified entries, 13 duplicate
  clusters found and merged, group-aware splits, conversion report correct.
- `train_detector.py` (baseline) and `train_candidate.py` (enhanced-logistic):
  both trained from the same manifest.
- `evaluate_detector.py` on the reserved eval partition: both artifacts load,
  per-source breakdowns produced, scores finite and in range.
- The candidate artifact was loaded through `model_loader` and served through
  the same detector path used by `/analyze` (covered by
  `tests/test_train_candidate.py`).

## 6. Verification results in the build environment

- Backend: `pytest` **171 passed, 1 skipped** (was 145 + 1; zero regressions).
- Backend lint: `ruff check .` **All checks passed**.
- Frontend: `tsc --noEmit` clean; `vitest` **73 tests passed**; production
  `vite build` succeeded.
- `.gitignore` already excludes datasets, `*.joblib` artifacts and reports;
  Kaggle credentials are only ever read from env/`kaggle.json`, never stored.

## 7. Real training run (produces the real numbers)

Run `notebooks/multidataset_training.ipynb` in Colab (Kaggle API token in
Colab Secrets). It produces, from the real data: the dataset registry,
per-source inspection reports, the unified manifest with real class counts,
baseline + candidate training reports, and **held-out eval metrics with
per-source and per-generator breakdowns**.

## 8. Promotion criteria (do not skip)

The production model is replaced **only if**, on the SAME held-out eval
partition, the candidate:

- F1 (synthetic) at least the baseline value (no regression);
- FPR (real clips flagged synthetic) no worse than baseline + 0.02;
- no single source degrades catastrophically (per-source F1 >= baseline - 0.05);
- per-source AND per-generator numbers reported, plus the unseen-generator
  holdout result if a holdout exists.

Promotion = copy `models_candidate/detector.joblib` to
`backend/models/detector.joblib` and update `docs/MODEL_CARD.md` with real
measured results. No code change is required.

## 9. Known limitations (disclosed, not hidden)

- **Only the first 4 s** of each clip feed the models (fixed-segment design).
  The 4-9-minute Indonesian clips are under-used; windowed training is a
  future option.
- **FoR generator metadata** is not published per file in an obvious layout;
  without generator ids the unseen-generator holdout is empty for that
  source. FoR's official partitioning is still honored.
- **Dataset (2) is non-commercial** (CC BY-NC-SA); mind this for product use.
- The pretrained anti-spoofing model comparison (e.g. SSL-based fine-tuned
  wav2vec2) is **not included**: it needs GPU-scale training and was out of
  scope for this round.
- `soundfile` cannot decode MP3; FoR's mp3 parts would be recorded as
  unreadable by the inspector, never silently dropped.
