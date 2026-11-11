# Unified protection pipeline (`POST /api/analyze`)

One request returns the whole assessment: transcription, speaker verification,
acoustic forensics, scam-text risk, LLM threat reasoning, risk badges and a
structured threat explanation. The older `POST /api/v1/protection/analyze`
endpoint still works and is unchanged, so existing clients keep running.

```
POST /api/analyze           multipart: file=<audio>, consent=true, profile_id?, language?
POST /api/analyze           application/json: {"text": "...", "consent": true, "language": "hi"}
GET  /api/capabilities      which model families are live on this host right now
```

Text input runs every content layer (rules, TF-IDF, MuRIL/DeBERTa ensemble, LLM
reasoning, risk) and returns `skipped` with a reason for the two audio-only
layers instead of inventing a verdict.

## Response shape (schema 1.0)

```jsonc
{
  "schema_version": "1.0",
  "input_kind": "audio",
  "transcription":        { "status": "...", "text": "...", "language": "hi", "engine": "whisper_small",
                            "engines": [ { "engine": "indic_hindi", "status": "completed" } ],
                            "limitations": ["..."] },
  "speaker_verification": { "status": "completed", "outcome": "likely_same_speaker",
                            "match": true, "similarity": 0.81, "threshold": 0.62 },
  "acoustic_forensics":   { "status": "completed",
                            "detector":    { "score": 0.12, "calibrated_probability": 0.12 },
                            "handcrafted": { "score": 0.31, "signals": ["..."] },
                            "wav2vec2_head": { "status": "unavailable", "detail": "..." },
                            "fusion":      { "score": 0.18, "verdict": "not_elevated",
                                             "components_used": ["detector", "handcrafted"],
                                             "components_missing": ["wav2vec2_head"] } },
  "scam_risk":            { "score": 0.93, "primary_category": "family_emergency_impersonation",
                            "transformer_ensemble": { "status": "completed", "score": 0.977 } },
  "llm_reasoning":        { "provider": "ollama", "model": "voxisentinel-mistral",
                            "fallback_used": false, "severity": "HIGH", "indicators": [...] },
  "risk_badges":          [ { "id": "risk", "label": "High risk", "tone": "critical" } ],
  "explanation":          { "verdict": "HIGH", "evidence": ["..."], "why_not_conclusive": ["..."],
                            "recommended_actions": ["..."], "model_trail": { } }
}
```

Errors use the project-wide envelope: `{"error": {"code": "...", "message": "..."}}`
with `invalid_request`, `consent_required`, `invalid_audio`, `file_too_large`,
`unsupported_media_type`, `decoder_unavailable`, `model_unavailable`.

## Model table (what is actually wired)

| Layer | Module | Model / artifact | State |
|---|---|---|---|
| Multilingual ASR | `app/protection/asr_engines.py` (`WhisperSmallEngine`) | `openai/whisper-small` | live, local weights in `models/hf/whisper-small` |
| Hindi ASR | `asr_engines.py` (`IndicHindiEngine`) | `ai4bharat/indicwav2vec-hindi` | gated HF repo (needs acceptance); reports the 401 verbatim |
| ASR fallback | `asr_engines.py` (`FasterWhisperEngine`, `GnaniAsrEngine`) | faster-whisper, Gnani | used only when the primary engine is unavailable |
| Speaker biometrics | `models/speaker_ecapa/`, `app/services/speaker*.py` | `speechbrain/spkrec-ecapa-voxceleb` (192-d, 16 kHz mono) | live |
| Acoustic forensics | `app/services/acoustic_forensics.py`, `forensics_embeddings.py` | `detector.joblib` + handcrafted cues + `facebook/wav2vec2-large-xlsr-53` | detector + cues live; XLSR embedding branch live when the checkpoint is present, head artifact needs labelled data |
| Scam text | `app/protection/text_ensemble.py` | `google/muril-base-cased` + `microsoft/deberta-v3-small` heads over the TF-IDF classifier | live (`models/text_heads/*.joblib`) |
| Threat reasoning | `app/protection/llm_router.py`, `ollama.py`, `mistral.py` | local `voxisentinel-mistral` via Ollama → `mistral-small-latest` | live; fallback engages on connection error, timeout, or schema violation |

Fusion weights live in `app/services/forensics_embeddings.py:FUSION_WEIGHTS`
(detector 0.5, handcrafted 0.2, wav2vec2 head 0.3) and are renormalised over the
components that actually ran, so a missing branch cannot silently zero the score.

## Turning each model on

Everything degrades to an explicit `unavailable` + reason, never to a fake score.

```bash
# 1. Python deps for the optional tiers (heavy: torch, transformers)
pip install -r backend/requirements.txt
pip install -r backend/requirements-models.txt

# 2. Download the checkpoints into backend/models/hf (skips gated repos with a reason)
python backend/scripts/download_hf_models.py --all
python backend/scripts/download_hf_models.py --only whisper-small
python backend/scripts/download_hf_models.py --only indicwav2vec-hindi   # needs an HF token + accepted terms

# 3. Rebuild the MuRIL/DeBERTa heads from the local corpus
python backend/scripts/train_transformer_text_heads.py

# 4. Local Mistral, then create the custom model
ollama pull mistral:7b-instruct-v0.3
python backend/scripts/create_ollama_model.py        # writes voxisentinel-mistral from Modelfile
ollama serve
```

`GET /api/capabilities` is the source of truth: it lists the ASR engines with
their availability, the speaker model, the forensics fusion weights and each
text branch's status, so the dashboard shows the same truth the pipeline uses.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `VOXSENTINEL_ASR_ENGINE` | `auto` | Preferred ASR engine (`auto`, `whisper_small`, `indic_hindi`, `faster_whisper`, `gnani`) |
| `VOXSENTINEL_ASR_FALLBACKS` | on | Allow the next available engine when the preferred one is missing |
| `VOXSENTINEL_WHISPER_MODEL` | `small` | faster-whisper size when that engine is selected |
| `HF_HOME` / `VOXSENTINEL_HF_DOWNLOAD` | `backend/models/hf` | Where Hugging Face weights are looked up / downloaded |
| `VOXSENTINEL_FORENSICS_ENABLED` | `true` | Enable the fused acoustic-forensics layer |
| `VOXSENTINEL_FORENSICS_HEAD` | empty | Path to a trained XLSR classification head; absent means the branch reports `unavailable` |
| `VOXSENTINEL_FORENSICS_THRESHOLD` | `0.5` | Fusion threshold for `elevated` |
| `VOXSENTINEL_TEXT_ENSEMBLE` | `true` | Enable the MuRIL/DeBERTa ensemble |
| `VOXSENTINEL_TEXT_ENSEMBLE_THRESHOLD` | `0.55` | Ensemble decision threshold |
| `VOXSENTINEL_TEXT_HEADS_DIR` | `backend/models/text_heads` | Trained head artifacts |
| `VOXSENTINEL_OLLAMA_ENABLED` | auto | Force the local reasoning provider on/off |
| `VOXSENTINEL_OLLAMA_URL` / `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `VOXSENTINEL_OLLAMA_MODEL` | `voxisentinel-mistral` | Local model name |
| `VOXSENTINEL_LLM_FALLBACK` | `mistral` | Fallback provider (`mistral`, `groq`, `gemini`, `off`) |
| `VOXSENTINEL_LLM_PRIMARY_COOLDOWN_S` | `60` | Skip the local model this many seconds after it fails |
| `MISTRAL_API_KEY` | empty | Enables `mistral-small-latest` as the cloud fallback |
| `HF_TOKEN` | empty | Required for the gated `indicwav2vec-hindi` download |
| `GNANI_ASR_ENDPOINT`, `GNANI_API_KEY` | empty | Remote ASR fallback |

## Measured behaviour on this machine (CPU only)

- `openai/whisper-small`: 16.9 s for the 16.9 s English fixture clip, exact transcript match.
- MuRIL + DeBERTa heads: held-out F1 **0.932** each on the local corpus; on the
  live path a scam sentence scores **0.977** and an ordinary one **0.0007**.
- `detector.joblib` (v20261107_hi): loads and infers in ~90 ms.
- `indicwav2vec-hindi`: download refused with **HTTP 401** (gated) and that reason
  is what `/api/capabilities` and the response show.

## Honest limitations

1. **Hindi ASR is not running yet.** The Indic checkpoint is gated; until a token
   with accepted terms is supplied, Hindi/Hinglish audio is transcribed by
   whisper-small, which handles Hinglish but is weaker on Devanagari.
2. **The XLSR forensics head is not trained.** The embedding extractor loads, but
   without a labelled artifact the fused score uses detector + handcrafted cues
   and says so (`components_missing: ["wav2vec2_head"]`). Fusion is not a
   deepfake guarantee: the detector's unseen-generator recall remains 5.3%.
3. **Speaker verification needs a reference.** With no family voice profile the
   result is `outcome: "skipped"`, never a guessed match.
4. **LLM reasoning is advisory.** It can restate a false premise in the
   transcript; the rule/TF-IDF/transformer scores and the detector stand on their
   own and the explanation lists what it does not prove.
5. **Serverless hosts** (Vercel) cannot run torch/transformers: expect
   `503 decoder_unavailable` for browser WebM, and `unavailable` for every heavy
   model. Use a container host (Render/Fly/HF Spaces) for the full pipeline.
