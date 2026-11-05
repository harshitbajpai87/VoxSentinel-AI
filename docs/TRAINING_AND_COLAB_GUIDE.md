# Training & Colab Guide

> **No model has been trained by this project.** These are prepared, reproducible configurations.
> Integrated models (wav2vec2 deepfake detector, ECAPA encoder, faster-whisper ASR, Groq LLM) are used
> for inference only.

## Files
```
training/
  colab_synthetic_speech_detection.ipynb
  colab_hindi_asr_eval.ipynb
  colab_scam_text_classification.ipynb
  colab_speaker_verification_eval.ipynb
  requirements-colab.txt
  datasets/manifest.py + manifests/
```

## Run
1. Colab → Runtime → Change runtime type → T4 GPU (detector) or CPU (ASR eval).
2. `!pip install -r training/requirements-colab.txt`
3. Mount Drive for approval-gated data (ASVspoof).
4. Run cells top to bottom; each section prints what it did.

## Configurations
| Task | Model | Data | Splits | Metrics | Export |
|------|-------|------|--------|---------|--------|
| Synthetic speech | `facebook/wav2vec2-base` + head (AASIST if ≥16 GB) | ASVspoof 2019 LA, WaveFake OOD | official protocol | EER, min-tDCF, per-quality-bucket EER | `save_pretrained()` → `backend/models/` |
| Hindi ASR | `faster-whisper` tiny/small/medium, IndicWhisper | CV17 hi, FLEURS hi, internal Hinglish | speaker-disjoint eval only | WER, CER, language_probability | evaluation only |
| Scam text | `google/muril-base-cased` | consented internal transcripts (required) | 70/15/15 stratified by scenario template | macro-F1, ransom recall, ECE, suppression rate | `save_pretrained()` |
| Speaker verification | ECAPA / WavLM-SV | VoxCeleb1 trials, enrolled clips | official trials | EER, minDCF | evaluation only |

Detector defaults: AdamW 3e-4, cosine, 10% warmup, batch 8×4 s, 6 epochs, bf16/fp16.

## Non-negotiable rules
- Never change `app/services/risk_engine.py` thresholds to improve apparent accuracy.
- Quality handling lives in `app/protection/quality.py` and only *qualifies* a verdict, preserving the
  raw score for audit.
- After any model swap: backend suite + `tests/test_audio_quality.py` + the 15-scenario harness.

## Status
Configurations prepared and documented. **Not executed.** No training run happened, and no new weights
are shipped.
