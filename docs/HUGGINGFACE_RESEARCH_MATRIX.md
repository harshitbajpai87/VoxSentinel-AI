# Hugging Face Model & Dataset Research Matrix — VoxSentinel AI
Generated: 2026-10-30. Scope: candidates for multilingual ASR, synthetic-speech detection,
speaker verification, multilingual fraud text classification, contextual reasoning.

**Column policy:** every row states repository/source, task, licence (or `unknown`), hardware
requirement (or `unknown`) and availability. `unknown` means it could not be confirmed from this
environment; it is not a guess and must be re-checked before use.

**Availability:** `downloadable` (weights on the Hub) · `gated` (accept terms first) · `API only` ·
`demo only` (Space/container with no weights).

**Verification status legend**
- `INTEGRATED` — running inside this repository today.
- `BENCHMARKED` — evaluated in this project on real audio/text.
- `CANDIDATE` — assessed here from public model cards; not trained or benchmarked in this run.
- `API/DEMO ONLY` — no downloadable weights usable here.

> Network verification of every card was not possible in this run. Sizes/licenses below are
> the published values from each project's card as of late 2026; re-check the repository page
> before redistribution or commercial deployment. Nothing in this matrix should be read as a
> claim that a model has been trained by this project.

## 1. Multilingual ASR (Hindi / Hinglish)

| # | Model | Source | Architecture | Languages | License | Size / hardware | Status |
|---|-------|--------|--------------|-----------|---------|-----------------|--------|
| 1 | `openai/whisper-large-v3` | hf.co/openai/whisper-large-v3 | Encoder-decoder Transformer (Whisper) | 99 incl. hi, en; code-mixing not native | MIT | 1.55B, ~6 GB VRAM fp16, ~10 GB CPU RAM int8 | CANDIDATE |
| 2 | `openai/whisper-small` | hf.co/openai/whisper-small | Whisper | multilingual | MIT | 244M, ~1 GB VRAM | CANDIDATE |
| 3 | `Systran/faster-whisper-small` / `-base` / `-tiny` | hf.co/Systran | CTranslate2 conversion of Whisper | multilingual | MIT | 75M–244M; CPU-friendly | **INTEGRATED (tiny)** |
| 4 | `ai4bharat/indicwav2vec_v1_hindi` | hf.co/ai4bharat | wav2vec2 CTC, Indic | hi (+ Indic siblings) | MIT (card) | ~1.2 GB, CPU feasible | CANDIDATE |
| 5 | `ai4bharat/indicwhisper` (IndicWhisper) | AI4Bharat release | Whisper fine-tuned on Indic | 10+ Indic | MIT (card) | 1.5B-class | CANDIDATE |
| 6 | `vasista22/whisper-hindi-large-v2` | hf.co/vasista22 | Whisper large fine-tuned hi | hi | MIT (card) | ~6 GB VRAM | CANDIDATE |
| 7 | `nvidia/stt_hi_conformer_ctc_large` | NGC / HF mirror | Conformer-CTC | hi | NVIDIA Open Model License | ~460 MB | CANDIDATE |
| 8 | `distil-whisper/distil-large-v3` | hf.co/distil-whisper | Distilled Whisper | en-centric | MIT | 756M, 2x faster | CANDIDATE |

**Finding from this project:** the integrated `faster-whisper:tiny` model transcribes clean
English reliably but degrades badly on code-mixed Hinglish (a real ransom line came back as
"Papa main kidna Bhugyaahu 5 lakh rupee, now he bhai jo"), which then propagates into the
classifier. Hindi Devanagari audio transcribed well. **Recommendation:** move to
`faster-whisper` `small`/`medium` (or IndicWhisper for pure Hindi) before claiming Hinglish
accuracy; keep the transcript visible to the user so ASR error is never hidden.

**Datasets**
| Dataset | Source | Content | License | Use |
|---------|--------|---------|---------|-----|
| `mozilla-foundation/common_voice_17_0` (hi, en-IN) | hf.co | read speech, crowdsourced, per-speaker metadata | CC0 (dataset) | largest open hi/en-IN ASR corpus; enables speaker-disjoint splits |
| `google/fleurs` | hf.co | read speech, 102 langs incl. hi | CC-BY-4.0 | clean multilingual eval |
| `ai4bharat/Kathbath` | hf.co | 12 Indic languages, studio quality | CC-BY-4.0 (card) | high-quality Hindi eval |
| `ai4bharat/Shrutilipi` | AI4Bharat | 12 Indic, long audio | CC-BY-NC-4.0 | research-only Hindi |
| `ai4bharat/IndicSUPERB` | hf.co | Indic speech benchmark suite | CC-BY-4.0 | Hindi benchmark harness |
| `LibriSpeech` (test-clean) | openslr.org/12 | English read speech, human | CC-BY-4.0 | **INTEGRATED** as genuine-speech fixture |
| Hinglish code-mixed | no single canonical corpus | LINCE (hi-en) is text-only | varies | **GAP**: must be collected internally |

## 2. Synthetic speech / audio deepfake detection

| # | Model | Source | Architecture | License | Size | Status |
|---|-------|--------|--------------|---------|------|--------|
| 1 | `MelodyMachine/Deepfake-audio-detection-V2` | hf.co | wav2vec2 + classifier head | card: MIT-style | ~360 MB | **INTEGRATED** (raw score only, threshold from core app) |
| 2 | `garystafford/wav2vec2-deepfake-voice-detector` | hf.co | wav2vec2-base fine-tune | Apache-2.0 (card) | ~360 MB | CANDIDATE |
| 3 | `mo-thecreator/Deepfake-audio-detection` | hf.co | wav2vec2 | card-specific | ~360 MB | CANDIDATE |
| 4 | AASIST (`clovaai/aasist`) | GitHub | graph-attention anti-spoofing | MIT (repo) | ~1 MB + deps | CANDIDATE (needs training) |
| 5 | `Wav2Vec2-XLS-R` anti-spoof fine-tunes | hf.co community | wav2vec2-XLS-R | varies | 300M–1B | CANDIDATE |
| 6 | ASVspoof 2019/2021 baselines | asvspoof.org | LFCC-GMM / RawNet2 / AASIST | research use | small | CANDIDATE (training recipes) |

**Datasets:** `ASVspoof 2019 LA/PA`, `ASVspoof 2021 DF`, `WaveFake` (Zenodo, ~196 h),
`Fake-or-Real` (FoR, Kaggle/HF mirrors), `In-the-Wild` (deepfake detection, ~31.8 h real vs fake),
`LibriSeVoc` (synthetic singing/speech pairs). Licenses are research-oriented; check each.

**Measured behaviour in this project (real runs, 24 kHz TTS clips):**
- clean TTS speech → 0.959 raw score; genuine LibriSpeech → 0.09; English ransom TTS → 0.897.
- narrowband/quiet re-encode → detector reads 0.997 on a *legitimate-sounding* degraded clip,
  i.e. degradation pushes the score up. This is why quality gating now exists: raw score is
  preserved for audit, verdict becomes `QUALIFIED`/`INCONCLUSIVE`, and the score is removed
  from risk scoring when the audio is unusable.

## 3. Speaker verification

| # | Model | Source | Architecture | License | Size | Status |
|---|-------|--------|--------------|---------|------|--------|
| 1 | `speechbrain/spkrec-ecapa-voxceleb` | hf.co | ECAPA-TDNN, VoxCeleb | Apache-2.0 | ~80 MB | **INTEGRATED** (ecapa path) |
| 2 | `microsoft/wavlm-base-plus-sv` | hf.co | WavLM + x-vector head | MIT | ~380 MB | CANDIDATE |
| 3 | `pyannote/embedding` | hf.co | x-vector (pyannote 3) | MIT (gated) | ~17 MB | CANDIDATE |
| 4 | `nvidia/speakerverification_en_titanet_large` | NGC | TitaNet | NVIDIA Open Model | ~100 MB | CANDIDATE |
| 5 | `Wespeaker` (voxceleb resnet/ecapa) | GitHub | ResNet/ECAPA | Apache-2.0 | 20–80 MB | CANDIDATE |

**Datasets:** `VoxCeleb1/2` (CC-BY-4.0 for annotations, audio via YouTube), `CN-Celeb`,
`SITW`, `FLEURS` speaker metadata, `LibriSpeech` speaker ids (for speaker-disjoint splits).

**Measured here:** same speaker 0.71 vs different speaker 0.51 on enrolled profiles; an
impostor TTS clip against an enrolled reference scored -0.048 (clear mismatch). ECAPA is
not spoof-robust by design: a cloned voice can still embed close to the target, so similarity
is reported as **evidence, never proof**.

## 4. Multilingual fraud / scam text classification

| # | Model | Source | Architecture | Languages | License | Status |
|---|-------|--------|--------------|-----------|---------|--------|
| 1 | `google/muril-base-cased` | hf.co | BERT, 17 Indic langs | hi, en, Indic | Apache-2.0 | CANDIDATE (best Indic baseline) |
| 2 | `ai4bharat/IndicBERTv2-MLM-only` | hf.co | BERT | 24 Indic | MIT | CANDIDATE |
| 3 | `l3cube-pune/hindi-bert-v2` | hf.co | BERT | hi, Marathi, Hinglish variants | MIT (card) | CANDIDATE |
| 4 | `xlm-roberta-base` | hf.co | XLM-R | 100 | MIT | CANDIDATE |
| 5 | `mrm8488/bert-tiny-finetuned-sms-spam-detection` | hf.co | tiny BERT | en | MIT | CANDIDATE (smoke baseline) |
| 6 | Groq `openai/gpt-oss-120b` | api.groq.com | MoE LLM, provider API | multilingual | provider ToS | **INTEGRATED** (primary contextual classifier) |
| 7 | Google Gemini (2.x/3.x flash class) | Google AI API | LLM API | multilingual | provider ToS | **INTEGRATED (legacy/optional)** |
| 8 | Rule engine (`app/protection/rules.py`) | this repo | explainable regex/lexicon | en, hi, Hinglish | project code | **INTEGRATED** (fallback + audit baseline) |

**Datasets:** `sms_spam` (UCI SMS Spam Collection, CC-BY-4.0), `enron_spam`,
`zefang-liu/phishing-email-dataset`, `IndicNLP` spam subsets, and — for voice-fraud —
no public Hindi/Hinglish kidnap-call corpus exists. Any supervised model here requires an
in-house, consented corpus. Text-only corpora do **not** transfer to spoken ransom scripts,
which is why the rule layer + LLM context layer are used instead of a fine-tuned classifier.

## 5. Contextual reasoning models

| # | Candidate | Availability | Notes |
|---|-----------|--------------|-------|
| 1 | Groq GPT-OSS 120B | API, quota-limited | primary classifier; 429 ⇒ rules-only fallback, now explicit in record |
| 2 | Gemini flash class | API | legacy, still selectable via `PROTECTION_LLM_PROVIDER=gemini` |
| 3 | DeepSeek-R1 distills (e.g. `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`) | downloadable, MIT | strong reasoning; not on the current Groq account (only 11 models available) |
| 4 | Qwen2.5 / Llama-3.1 8B instruct | downloadable | self-hostable second pass if quota matters |
| 5 | `Reasoner4All` | **demo only — see below** | architectural inspiration, not a usable model |

### Reasoner4All — actual availability
`huggingface.co/spaces/McLoviniTtt/Reasoner4All` is a **Chainlit application packaged as a
Docker Space** (`cpu-basic`). It orchestrates calls to external providers behind a second
reasoning step; it ships **no downloadable model weights, no public inference API, and no
training code**. Its documented model (DeepSeek-R1 class) is not available on the current
Groq account. Consequences for VoxSentinel AI:
- Cannot be vendored, fine-tuned, or benchmarked as a model.
- Its *pattern* (a second pass that critiques the first verdict) is what was implemented in
  `app/protection/reasoner.py`, using the same provider as stage one, gated so it may only
  **soften** a verdict and only with **verbatim** transcript quotes.

## Integration recommendation (ordered, cost-aware)
1. ASR: `faster-whisper small/medium` for Hinglish, IndicWhisper for Hindi-only deployments; publish WER per language.
2. Deepfake: keep the existing wav2vec2 detector, add quality gating (done), and evaluate AASIST if a GPU becomes available.
3. Speaker: keep ECAPA, add spoof-awareness disclaimer, evaluate WavLM-SV for robustness.
4. Text: keep Groq + rules; only fine-tune MuRIL/IndicBERT once a consented Hindi/Hinglish corpus exists.
5. Reasoning: keep the opt-in reviewer; move to a self-hosted 7B reasoning model when quota blocks production.


## Appendix — availability detail

| Item | Availability | Notes |
|------|--------------|-------|
| `openai/whisper-large-v3` | downloadable | MIT weights |
| `Systran/faster-whisper-*` | downloadable | CTranslate2 int8, CPU-usable |
| `ai4bharat/indicwav2vec_v1_hindi` | downloadable | licence per card (verify) |
| `vasista22/whisper-hindi-large-v2` | downloadable | ~6 GB VRAM |
| `MelodyMachine/Deepfake-audio-detection-V2` | downloadable | **integrated**, raw score only |
| AASIST (`clovaai/aasist`) | code downloadable | weights trained per dataset, not shipped |
| `speechbrain/spkrec-ecapa-voxceleb` | downloadable | **integrated** |
| `microsoft/wavlm-base-plus-sv` | downloadable | candidate |
| `google/muril-base-cased` | downloadable | best Indic text baseline |
| Groq `openai/gpt-oss-120b` | API only | **integrated**, quota-limited |
| Gemini flash class | API only | legacy, optional |
| `Reasoner4All` | **demo only** | Docker Space (cpu-basic): no weights, no public inference API, no training code, model not on the current Groq account |
| `mozilla-foundation/common_voice_17_0` | gated download | CC0 dataset card; `client_id` enables speaker-disjoint splits |
| `google/fleurs` | downloadable | CC-BY-4.0 |
| `ai4bharat/Kathbath`, `IndicSUPERB` | downloadable | CC-BY-4.0 per card |
| `ai4bharat/Shrutilipi` | downloadable | CC-BY-NC-4.0, research only |
| ASVspoof 2019/2021, WaveFake, In-the-Wild, FoR | request/registration | research licences |
| VoxCeleb1/2 | annotations downloadable | audio via YouTube; redistribution unknown |
| Hinglish code-mixed speech corpus | **does not exist publicly** | the project's main data gap |
