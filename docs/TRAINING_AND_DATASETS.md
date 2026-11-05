# Training & Dataset Documentation (reproducible configurations)

> **No model in this repository has been trained by this project.** These are prepared,
> reproducible configurations. Anything listed as INTEGRATED is inference-only.

## 0. Layout
```
training/
  colab_synthetic_speech_detection.ipynb   # wav2vec2 / AASIST fine-tune
  colab_hindi_asr_eval.ipynb               # Hindi + Hinglish ASR benchmark
  colab_scam_text_classification.ipynb     # MuRIL / IndicBERT classifier
  colab_speaker_verification_eval.ipynb    # ECAPA / WavLM EER evaluation
  datasets/manifest.py                     # download + manifest + split generator
  datasets/manifests/*.csv                 # produced manifests (real counts)
  requirements-colab.txt
```

## 1. Synthetic speech detection
- **Data:** ASVspoof 2019 LA (train/dev/eval, bona fide vs spoof), WaveFake (bonus generalisation), plus
  the project's own TTS clips as an *out-of-domain probe* (never as training data).
- **Preprocess:** mono 16 kHz, peak-normalise to -1 dBFS, 4 s random crop for train, full file for eval.
- **Splits:** official ASVspoof partitions (already speaker- and attack-disjoint). No re-splitting.
- **Leakage guards:** dedupe by SHA-256 of waveform bytes; assert no speaker id in both train and eval;
  assert no identical transcript across splits (checked by the manifest script).
- **Model:** `wav2vec2-base` + mean-pool + linear head (baseline), or AASIST if GPU ≥ 16 GB.
- **Config:** AdamW 3e-4, cosine, 10% warmup, batch 8×4 s, 6 epochs, bf16 on A100/T4-fp16.
- **Metrics:** EER (primary), min-tDCF, AUC; report per-codec and per-degradation (the project's quality
  buckets: clean/degraded/insufficient) — the last one matters most for real call audio.
- **Export:** `model.save_pretrained()` → ONNX/CTranslate2 optional; integrate by dropping the folder in
  `models/` and pointing `DETECTOR_MODEL_DIR` at it. Must keep the existing threshold policy unchanged.

## 2. Hindi / multilingual ASR evaluation
- **Data:** Common Voice 17 (hi, en-IN) test split, FLEURS hi, Kathbath sample, plus internal Hinglish clips.
- **Preprocess:** 16 kHz mono; keep natural pauses.
- **Splits:** speaker-disjoint by `client_id` (CV) / `speaker_id` (FLEURS); eval-only benchmark, no training.
- **Metrics:** WER + CER after Devanagari normalisation (NFC, strip danda) and a documented Hinglish
  normalisation (romanised variants mapped to a canonical spelling list); also report `language_probability`.
- **Leakage guard:** no clip may appear in both the prompt set and the eval set (hash check).

## 3. Scam text classification
- **Data:** internal consented transcript corpus (required for hi/Hinglish voice fraud);
  public SMS/email spam corpora only as an English sanity check.
- **Splits:** stratified 70/15/15 by *scenario family* (ransom, OTP, banking, benign, awareness) so the
  same scenario template never appears in two splits.
- **Metrics:** macro-F1, per-class recall (ransom recall is the safety-critical one), calibration (ECE),
  plus a "strong-warning suppression" rate: fraction of true ransom cases where any stage-2 override fired.
- **Guard:** the reviewer may never turn a ransom/fraud label into benign without quoted evidence.

## 4. Speaker verification evaluation
- **Data:** VoxCeleb1 test pairs, CN-Celeb, plus enrolled-profile clips from the app.
- **Protocol:** verification trials (same/different), thresholds swept for EER; **impostor** condition
  includes TTS/cloned voices to show that similarity is not spoof-proof.
- **Metrics:** EER, minDCF, and score distribution overlap; report the observed 0.71 vs 0.51 separation
  as an operating point, not a guarantee.

## 5. Reproducibility rules used everywhere
1. Every manifest row: `path, sha256, duration_s, speaker_id, label, split, source, license`.
2. Split assignment is deterministic (`sha256(path + seed)`), never random at runtime.
3. Duplicate detection by content hash before splitting; leakage assertions run in CI.
4. Dataset licences recorded per row; NC datasets never enter a shipped model.
