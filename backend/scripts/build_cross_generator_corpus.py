"""Build a generator-held-out audio corpus + manifests for detector v4.

The point of this corpus is GENERALIZATION measurement. The shipped v0.3
artifact was trained and evaluated on in-distribution data only (F1 0.96 on
5468 clips) and scores at chance on unseen edge-tts audio. This builder
constructs a corpus where:

  * the EVAL partition contains TTS voices that appear in NO other partition
    (two Hindi voices, one unseen en-IN voice, two unseen English locales) and
    LibriSpeech speakers that appear in no other partition;
  * the VALIDATION partition uses its own unseen voices/speakers, so model
    selection cannot peek at eval;
  * both classes are exposed to four channel conditions (clean, narrowband
    8 kHz, mp3 64 kbps, light noise at ~15 dB SNR) so a classifier cannot win
    by learning "lossless FLAC = real, mp3 = synthetic" — the shortcut that
    inflates in-distribution scores and dies on real deployment audio.

Outputs (paths are relative to --dataset-dir):
  real_clean/<spk>/<id>.flac        hardlink to the LibriSpeech original
  real_8k/<spk>/<id>.wav            8 kHz mono PCM16 (telephony narrowband)
  real_mp3/<spk>/<id>.mp3           mono mp3 64 kbps @ 16 kHz
  real_noisy/<spk>/<id>.wav         16 kHz, white+babble noise at ~15 dB SNR
  tts_clean|8k|mp3|noisy/...        same four conditions for edge-tts clips
  (manifests land in the repo, not here, so they stay reviewable)

Manifest schema is the EXISTING one (app/services/manifest.py): path, label,
partition, speaker_id, source, generator_id, language. Channel condition is not
expressible in that schema, so it is recorded in a sidecar CSV
(manifests/clip_meta.csv) keyed by manifest path — see build_metadata_sidecar.

Usage:
    python scripts/build_cross_generator_corpus.py \
        --librispeech-dir /tmp/av/LibriSpeech/dev-clean \
        --tts-index /tmp/av/generator_v4/tts_index.json \
        --dataset-dir /tmp/av/generator_v4 \
        --manifest-dir data/cross_generator_v4/manifests \
        --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CONDITIONS = ("clean", "narrowband8k", "mp3_64k", "noisy_15db")
TARGET_SR = 16000
CLIP_SECONDS_CAP = 10.0
NOISE_SNR_DB = 15.0

# Real: how many clips to take per speaker (LibriSpeech speakers hold 36..~110).
REAL_CLIPS_PER_SPEAKER = 30
# Synthetic: clips per voice per split (set by gen_cross_generator_tts.py).
TTS_VOICE_SPLITS = ("train", "validation", "eval")


def _run(cmd: list[str]) -> None:
    """Run ffmpeg; raise with stderr on failure (never silently keep bad audio)."""
    proc = subprocess.run(cmd, capture_output=True, text=True)  # noqa: S603
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-800:]}")


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1).astype(np.float32) if data.shape[1] > 1 else data.reshape(-1)
    return mono, int(sr)


def _resample(mono: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return mono
    import librosa  # noqa: PLC0415

    return librosa.resample(mono, orig_sr=sr, target_sr=target).astype(np.float32)


def _babble_bank(noise_files: list[Path], seed: int, seconds: float) -> list[np.ndarray]:
    """Deterministic babble sources: 16 kHz mono, peak-normalised."""
    rng = np.random.default_rng(seed)
    bank: list[np.ndarray] = []
    for path in noise_files:
        mono, sr = _read_mono(path)
        mono = _resample(mono, sr, TARGET_SR)
        if mono.size < TARGET_SR:
            continue
        start = int(rng.integers(0, max(1, mono.size - int(seconds * TARGET_SR))))
        chunk = mono[start : start + int(seconds * TARGET_SR)]
        peak = float(np.max(np.abs(chunk))) or 1.0
        bank.append(chunk / peak)
    if not bank:
        raise RuntimeError("No usable babble noise files were found.")
    return bank


def _mix_noise(mono: np.ndarray, sr: int, noise_bank: list[np.ndarray],
               index: int, seed: int) -> np.ndarray:
    """White + babble mixture at a fixed SNR relative to active-speech RMS."""
    rng = np.random.default_rng(seed + index)
    mono = _resample(mono, sr, TARGET_SR)
    frames = int(0.025 * TARGET_SR)
    usable = (mono.size // frames) * frames
    if usable >= frames:
        energy = np.square(mono[:usable].reshape(-1, frames)).mean(axis=1)
        active = energy[energy > energy.max() * 0.01]
        speech_rms = float(np.sqrt(active.mean())) if active.size else float(np.sqrt(energy.mean()))
    else:
        speech_rms = float(np.sqrt(np.mean(np.square(mono))))
    speech_rms = max(speech_rms, 1e-4)
    noise_rms = speech_rms / (10 ** (NOISE_SNR_DB / 20.0))

    white = rng.standard_normal(mono.size).astype(np.float32)
    babble = np.zeros(mono.size, dtype=np.float32)
    picks = [(index + k) % len(noise_bank) for k in range(3)]
    for n, pick in enumerate(picks):
        source = noise_bank[pick]
        offset = int(rng.integers(0, max(1, source.size)))
        tail = np.roll(source, offset)[: mono.size]
        if tail.size < mono.size:
            tail = np.pad(tail, (0, mono.size - tail.size))
        babble += tail / max(1.0, float(np.max(np.abs(tail))) or 1.0) / 3.0
    combined = 0.5 * white / (np.std(white) or 1.0) + 0.5 * babble
    combined = combined / (np.std(combined) or 1.0) * noise_rms
    mixed = mono + combined.astype(np.float32)
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.99:  # avoid clipping the mixture, not the speech itself
        mixed = mixed / peak * 0.99
    return mixed.astype(np.float32)


def _make_variant(src: Path, condition: str, dest: Path, noise_bank: list[np.ndarray],
                  index: int, seed: int, source_sr: int | None = None) -> None:
    """Produce one channel variant of `src` at `dest`, truncated to the cap."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if condition == "clean":
        suffix = dest.suffix.lower()
        if suffix == ".flac":
            # Lossless original: hardlink keeps the exact bytes, zero copies.
            if dest.exists():
                dest.unlink()
            os.link(src, dest)
            return
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", f"{CLIP_SECONDS_CAP}",
              "-ac", "1", "-ar", str(source_sr or 24000), "-c:a", "libmp3lame",
              "-b:a", "48k", str(dest)])
        return
    if condition == "narrowband8k":
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", f"{CLIP_SECONDS_CAP}",
              "-ac", "1", "-ar", "8000", "-c:a", "pcm_s16le", str(dest)])
        return
    if condition == "mp3_64k":
        _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", f"{CLIP_SECONDS_CAP}",
              "-ac", "1", "-ar", "16000", "-c:a", "libmp3lame", "-b:a", "64k", str(dest)])
        return
    if condition == "noisy_15db":
        mono, sr = _read_mono(src)
        mono = mono[: int(CLIP_SECONDS_CAP * sr)]
        mixed = _mix_noise(mono, sr, noise_bank, index, seed)
        sf.write(str(dest), mixed, TARGET_SR, subtype="PCM_16")
        return
    raise ValueError(f"Unknown condition {condition!r}")


def _choose_speakers(speakers: list[str], seed: int) -> dict[str, list[str]]:
    """Speaker-disjoint splits: 28 train / 5 validation / 5 eval / 2 noise-only."""
    rng = random.Random(seed)
    ordered = sorted(speakers)
    rng.shuffle(ordered)
    return {
        "train": ordered[:28],
        "validation": ordered[28:33],
        "eval": ordered[33:38],
        "noise_only": ordered[38:],
    }


def build(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir)
    manifest_dir = Path(args.manifest_dir)
    librispeech = Path(args.librispeech_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    flac_files = sorted(librispeech.glob("*/*/*.flac"))
    if not flac_files:
        print(f"ERROR: no LibriSpeech flac files under {librispeech}", file=sys.stderr)
        return 2
    by_speaker: dict[str, list[Path]] = {}
    for path in flac_files:
        by_speaker.setdefault(path.parent.parent.name, []).append(path)
    print(f"LibriSpeech: {len(flac_files)} clips, {len(by_speaker)} speakers")

    splits = _choose_speakers(list(by_speaker), args.seed)
    print("Speakers:", {k: len(v) for k, v in splits.items()})

    # Babble pool: speakers excluded from every labelled split.
    noise_files = [p for spk in splits["noise_only"] for p in by_speaker[spk]][:40]
    noise_bank = _babble_bank(noise_files, args.seed, CLIP_SECONDS_CAP)
    print(f"Babble bank: {len(noise_bank)} sources from noise-only speakers "
          f"{splits['noise_only']}")

    tts = json.loads(Path(args.tts_index).read_text(encoding="utf-8"))
    tts_clips = [c for c in tts["clips"] if Path(c["path"]).is_file()
                 and Path(c["path"]).stat().st_size > 0]
    tts_by_voice: dict[str, list[dict]] = {}
    for clip in tts_clips:
        tts_by_voice.setdefault(clip["voice"], []).append(clip)
    print(f"edge-tts: {len(tts_clips)} clips, {len(tts_by_voice)} voices")

    entries: list[dict[str, object]] = []
    meta: list[dict[str, object]] = []
    counter = 0
    for split, speakers in (("train", splits["train"]), ("validation", splits["validation"]),
                            ("eval", splits["eval"])):
        rng = random.Random(args.seed + hash(split) % 1000)
        for speaker in speakers:
            clips = sorted(by_speaker[speaker])
            rng.shuffle(clips)
            chosen = clips[:REAL_CLIPS_PER_SPEAKER]
            for i, src in enumerate(chosen):
                condition = CONDITIONS[i % len(CONDITIONS)]
                kind_dir = f"real_{condition.replace('narrowband8k', '8k').replace('mp3_64k', 'mp3').replace('noisy_15db', 'noisy')}"
                rel = Path(kind_dir) / speaker / f"{src.stem}.{ 'flac' if condition == 'clean' else ('mp3' if condition in ('mp3_64k',) else 'wav')}"
                dest = dataset_dir / rel
                _make_variant(src, condition, dest, noise_bank, counter, args.seed,
                              source_sr=16000 if condition == "clean" else None)
                counter += 1
                entries.append({
                    "path": rel.as_posix(), "label": "real", "partition": split,
                    "speaker_id": f"librispeech:{speaker}", "source": "librispeech",
                    "language": "en",
                })
                meta.append({"path": rel.as_posix(), "label": "real", "partition": split,
                             "source_id": f"librispeech:{speaker}", "condition": condition,
                             "generator_id": "", "language": "en",
                             "held_out_group": speaker})

    for voice, clips in sorted(tts_by_voice.items()):
        split = clips[0]["split"]
        for i, clip in enumerate(sorted(clips, key=lambda c: c["path"])):
            condition = CONDITIONS[i % len(CONDITIONS)]
            kind = condition.replace("narrowband8k", "8k").replace("mp3_64k", "mp3").replace("noisy_15db", "noisy")
            suffix = "mp3" if condition in ("clean", "mp3_64k") else "wav"
            rel = Path(f"tts_{kind}") / voice / f"{Path(clip['path']).stem}.{suffix}"
            dest = dataset_dir / rel
            _make_variant(Path(clip["path"]), condition, dest, noise_bank, counter, args.seed)
            counter += 1
            entries.append({
                "path": rel.as_posix(), "label": "synthetic", "partition": split,
                "speaker_id": f"edge-tts:{voice}", "source": "edge_tts",
                "generator_id": f"edge-tts:{voice}",
                "language": clip["language"],
            })
            meta.append({"path": rel.as_posix(), "label": "synthetic", "partition": split,
                         "source_id": f"edge-tts:{voice}", "condition": condition,
                         "generator_id": f"edge-tts:{voice}", "language": clip["language"],
                         "held_out_group": voice})

    for name, partition in (("manifest.jsonl", None), ("train.jsonl", "train"),
                            ("valid.jsonl", "validation"), ("eval.jsonl", "eval")):
        rows = [e for e in entries if partition is None or e["partition"] == partition]
        with (manifest_dir / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    with (manifest_dir / "clip_meta.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(meta[0].keys()))
        writer.writeheader()
        writer.writerows(meta)

    summary: dict[str, object] = {}
    for split in ("train", "validation", "eval"):
        subset = [e for e in entries if e["partition"] == split]
        subset_meta = [m for m in meta if m["partition"] == split]
        summary[split] = {
            "n": len(subset),
            "real": sum(1 for e in subset if e["label"] == "real"),
            "synthetic": sum(1 for e in subset if e["label"] == "synthetic"),
            "speakers_real": sorted({e["speaker_id"] for e in subset if e["label"] == "real"}),
            "voices_synthetic": sorted({e["generator_id"] for e in subset
                                        if e["label"] == "synthetic" and e["generator_id"]}),
            "conditions": {c: sum(1 for m in subset_meta if m["condition"] == c)
                           for c in CONDITIONS},
            "by_condition_label": {
                f"{c}/{lab}": sum(1 for m in subset_meta
                                  if m["condition"] == c and m["label"] == lab)
                for c in CONDITIONS for lab in ("real", "synthetic")
            },
        }
    (manifest_dir / "corpus_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Manifests written to {manifest_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build generator-held-out corpus for detector v4.")
    parser.add_argument("--librispeech-dir", default="/tmp/av/LibriSpeech/dev-clean")
    parser.add_argument("--tts-index", default="/tmp/av/generator_v4/tts_index.json")
    parser.add_argument("--dataset-dir", default="/tmp/av/generator_v4")
    parser.add_argument("--manifest-dir", default="data/cross_generator_v4/manifests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
