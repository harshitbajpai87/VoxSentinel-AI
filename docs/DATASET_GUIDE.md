# Dataset Guide — Getting Real Data for VoxSentinel

> The repository ships **no** audio dataset. Everything under
> `VOXSENTINEL_DATASET_DIR` (default `data/`) is git-ignored. Never commit
> datasets, private recordings, or model artifacts without license review.

## 1. Recommended dataset: ASVspoof 2019 Logical Access (LA)

**What it is:** the standard benchmark for synthetic-speech (spoofing)
detection: real human speech plus spoofs produced by 17+ synthesis methods
(vocoders, neural TTS, voice conversion), with speaker-disjoint train/dev/eval
partitions.

**Why it fits our problem (SIH 2026 SIH26104):** logical-access attacks are
exactly "audio presented as genuine" — no physical-channel tricks — and the
official partitions already prevent speaker leakage.

**Access requirements (verify current terms yourself):**

- Register at <https://asvspoof.org> (the organizers, RedTeam Lab @ NUS and
  partners) and accept their license/EULA.
- ASVspoof 2015 and 2017 editions have their own (different) terms; the 2019 LA
  corpus is the one referenced by `scripts/prepare_dataset.py`.
- The corpus is large (tens of GB). Download it over a stable connection to
  `data/asvspoof2019_la/` and verify any checksums the organizers provide.

We do not redistribute the dataset, cannot download it for you, and cannot
quote exact file counts — check the official release notes.

## 2. The librispeech_piper corpus (open license; used for model v0.2.0)

When ASVspoof is not yet available, VoxSentinel can train on a **legitimately
accessible, fully open-licensed** corpus built from two public sources:

- **Real speech:** [LibriSpeech](https://www.openslr.org/12) dev-clean +
  test-clean (OpenSLR SLR12, CC BY 4.0) — genuine human-read English
  audiobooks, 16 kHz FLAC.
- **Synthetic speech:** [Piper](https://github.com/rhasspy/piper) VITS neural
  TTS (open models from <https://huggingface.co/rhasspy/piper-voices>),
  synthesised locally from LibriSpeech transcripts.

**Scope and honesty:** Piper TTS is text-to-speech, **not voice cloning**, and
the eval results in `docs/MODEL_CARD.md` apply only to this human-vs-Piper
setup. They do not demonstrate detection of arbitrary cloning systems. Use it
as a reproducible interim corpus; still register for ASVSpoof 2019 LA for the
real benchmark.

**Partition design (leakage-resistant):**

| Partition | Real | Synthetic | Notes |
|---|---|---|---|
| train | 240 | 240 | dev-clean readers 1–30; Piper voice `en_US-lessac-medium` |
| validation | 60 | 60 | dev-clean readers 31–40; same lessac voice (generator overlap train/val is declared via `--leakage-exempt-sources piper_tts`, mirroring ASVspoof's train/dev attack sharing) |
| eval (held-out) | 80 | 80 | test-clean readers (speaker-disjoint from dev-clean, verified); **unseen** voice `en_US-amy-medium` |

Human speakers are disjoint across ALL partitions (verified at manifest build
and again during training: train∩eval and validation∩eval overlaps are
refused). The training script additionally refuses ANY eval speaker present in
train or validation — no exemptions allowed there.

### Build it (all commands from `backend/`)

```bash
# 1. Real speech (≈680 MB total)
mkdir -p data/librispeech
curl -L -o data/librispeech/dev-clean.tar.gz  https://www.openslr.org/resources/12/dev-clean.tar.gz
curl -L -o data/librispeech/test-clean.tar.gz https://www.openslr.org/resources/12/test-clean.tar.gz
tar xzf data/librispeech/dev-clean.tar.gz  -C data    # -> data/LibriSpeech/dev-clean
tar xzf data/librispeech/test-clean.tar.gz -C data    # -> data/LibriSpeech/test-clean

# 2. TTS voices (one for train/val, a DIFFERENT one held out for eval)
mkdir -p data/tts_models && cd data/tts_models
curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
cd ../..

# 3. Synthesise the spoof half (deterministic: seed 42, disjoint text slices)
mkdir -p data/spoofs && rm -f data/spoofs/index.jsonl
python scripts/build_tts_spoofs.py --librispeech data/LibriSpeech/dev-clean   --voice data/tts_models/en_US-lessac-medium.onnx --voice-id en_US-lessac-medium   --out-dir data/spoofs --partition train      --count 240 --offset 0 --index data/spoofs/index.jsonl
python scripts/build_tts_spoofs.py --librispeech data/LibriSpeech/dev-clean   --voice data/tts_models/en_US-lessac-medium.onnx --voice-id en_US-lessac-medium   --out-dir data/spoofs --partition validation --count 60  --offset 240 --index data/spoofs/index.jsonl
python scripts/build_tts_spoofs.py --librispeech data/LibriSpeech/test-clean   --voice data/tts_models/en_US-amy-medium.onnx   --voice-id en_US-amy-medium   --out-dir data/spoofs --partition eval         --count 80  --offset 0 --index data/spoofs/index.jsonl

# 4. Manifest with provenance, class checks and reader-disjointness verification
python scripts/prepare_dataset.py --preset librispeech_piper --dataset-dir data   --spoof-index data/spoofs/index.jsonl --output data/librispeech_piper_manifest.jsonl
```

Download sizes: dev-clean 337 MB, test-clean 347 MB, each voice ≈63 MB.
Synthesis of 380 clips takes a few minutes on a laptop CPU.

## 2b. Speaker-verification calibration data (used for the ECAPA threshold)

The speaker pipeline uses the same LibriSpeech download (SLR12, CC BY 4.0)
but a different labelling scheme: readers and chapters provide ground-truth
identities and recording sessions. `scripts/evaluate_speaker.py` builds 80
labelled pairs from dev-clean (calibration) and 80 from test-clean (held-out;
speakers disjoint from calibration). Same-speaker pairs always use different
chapters of the same reader — never the same file twice.

```bash
# from backend/ ; requires data/LibriSpeech/{dev-clean,test-clean} and the
# one-time ~83 MB ECAPA model download (models/speaker_ecapa)
# python scripts/download_speaker_model.py   # explicit, never automatic
python scripts/evaluate_speaker.py --librispeech data/LibriSpeech
# writes models/speaker_thresholds.json + reports/speaker_eval.json
```

Results are documented in `docs/MODEL_CARD.md` (held-out EER 0.025,
ROC-AUC 0.9994, threshold 0.3448 cosine).

## 3. Preparing the ASVspoof manifest

Expected layout (the official release structure under your download root):

```
data/asvspoof2019_la/
└── LA/
    ├── ASVspoof2019_LA_train/flac/*.flac
    ├── ASVspoof2019_LA_dev/flac/*.flac
    ├── ASVspoof2019_LA_eval/flac/*.flac
    └── ASVspoof2019_LA_cm_protocols/
        ├── ASVspoof2019.LA.cm.train.trn.txt
        ├── ASVspoof2019.LA.cm.dev.trl.txt
        └── ASVspoof2019.LA.cm.eval.trl.txt
```

Build and validate the manifest:

```bash
cd backend
python scripts/prepare_dataset.py --preset asvspoof2019_la \
    --dataset-dir data/asvspoof2019_la \
    --output data/asvspoof2019_la/manifest.jsonl
```

The script maps `bonafide`→`real`, `spoof`→`synthetic`, `dev`→our
`validation` partition, and keeps `eval` untouched for final held-out
evaluation. Missing files or unknown labels are **rejected and counted**, never
guessed. A summary JSON is written next to the manifest.

## 4. Alternative: WaveFake

**What it is:** real (LJSpeech) + spoofed speech from six neural vocoders,
hosted on GitHub with a license — check the repository's LICENSE file and the
paper's terms before use.

**Caution:** it is tens of GB, the spoof set covers only vocoder-based
attacks, and it is not a drop-in for our manifest preset. You would need to
write the manifest yourself in the JSONL format below, or add a preset to
`prepare_dataset.py`.

## 5. Manifest format (dataset-agnostic)

One JSON object per line; paths relative to the manifest's directory:

```json
{"path": "audio/clip-0001.wav", "label": "real", "partition": "train",
 "speaker_id": "spk-01", "source": "my-dataset"}
```

- `label`: `real` / `synthetic` (aliases: `bonafide`, `spoof`, `genuine`,
  `fake`, `natural` are normalized; anything else is rejected).
- `partition`: `train`, `validation`, or `eval`.
- `speaker_id`: optional but strongly recommended — the trainer refuses
  speaker overlap between train and validation.
- Keep a `FIXTURE_NOTICE`/README with your dataset describing source, version,
  license and any preprocessing you performed. **Do not** put private
  recordings in a public repository.

## 6. Smoke fixture (no dataset needed)

```bash
cd backend
python scripts/make_fixture_dataset.py --out-dir data/fixture_smoke --per-class-per-partition 8
python scripts/train_detector.py --manifest data/fixture_smoke/manifest.jsonl --model-dir models --seed 42
python scripts/evaluate_detector.py --artifact models/detector.joblib --manifest data/fixture_smoke/manifest.jsonl --partition eval --report-dir reports
```

The fixture is **generated tones and noise, not speech**. It exists to verify
the pipeline mechanics end to end. Metrics from it are never real-world
detection performance, and every artifact it produces should be deleted before
training on real data (`rm models/detector.joblib`).
