"""Generate synthetic-speech samples with a real neural TTS (Piper/VITS).

This builds the "spoof" half of the librispeech_piper corpus: REAL neural-TTS
audio (not the smoke fixture) synthesised from LibriSpeech transcripts. The
texts come from open-licensed audiobooks; the synthesised audio is produced
locally by an open Piper voice model.

Honest scope: this is TTS synthesis, NOT voice cloning of a specific person,
and it covers only the synthesis voices actually used (documented in the
index). Any detector trained on it can only claim performance against this
corpus and these voices.

Usage (three invocations — one per partition):

    python scripts/build_tts_spoofs.py --librispeech data/LibriSpeech/dev-clean \
        --voice data/tts_models/en_US-lessac-medium.onnx --voice-id en_US-lessac-medium \
        --out-dir data/spoofs --partition train --count 240 --offset 0 --index data/spoofs/index.jsonl

    ... same with --partition validation --count 60 --offset 240
    (eval uses an UNSEEN voice, e.g. en_US-amy-medium, and test-clean texts)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect_transcripts(subset_dir: Path) -> list[tuple[str, str]]:
    """All (utterance_id, text) pairs from a LibriSpeech subset, sorted."""
    pairs: list[tuple[str, str]] = []
    for trans_file in sorted(subset_dir.rglob("*.trans.txt")):
        for line in trans_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            utt_id, _, text = line.partition(" ")
            if utt_id and text:
                pairs.append((utt_id, text))
    pairs.sort()
    return pairs


def select_transcripts(
    pairs: list[tuple[str, str]],
    count: int,
    offset: int,
    seed: int,
    min_words: int,
    max_words: int,
) -> list[tuple[str, str]]:
    """Deterministic selection: word-count filter, seeded shuffle, then slice."""
    import random

    filtered = [p for p in pairs if min_words <= len(p[1].split()) <= max_words]
    if len(filtered) < count + offset:
        raise SystemExit(
            f"ERROR: only {len(filtered)} suitable transcripts; need {count + offset}."
        )
    order = list(range(len(filtered)))
    random.Random(seed).shuffle(order)
    chosen_idx = order[offset : offset + count]
    return [filtered[i] for i in sorted(chosen_idx)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--librispeech", required=True, help="LibriSpeech subset dir (for texts)")
    parser.add_argument("--voice", required=True, help="Path to a Piper .onnx voice model")
    parser.add_argument("--voice-id", required=True, help="Voice identifier, e.g. en_US-lessac-medium")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--partition", required=True, choices=["train", "validation", "eval"])
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0, help="Skip this many in the deterministic order")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index", required=True, help="Index JSONL to append entries to")
    parser.add_argument("--min-words", type=int, default=8)
    parser.add_argument("--max-words", type=int, default=40)
    args = parser.parse_args()

    subset_dir = Path(args.librispeech)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_transcripts(subset_dir)
    chosen = select_transcripts(
        pairs, args.count, args.offset, args.seed, args.min_words, args.max_words
    )
    print(f"Selected {len(chosen)} transcripts from {subset_dir} for partition '{args.partition}'.")

    import wave

    from piper.voice import PiperVoice

    print(f"Loading voice {args.voice_id} ...")
    voice = PiperVoice.load(args.voice)

    index_path = Path(args.index)
    written = 0
    with index_path.open("a", encoding="utf-8") as index_handle:
        for i, (utt_id, text) in enumerate(chosen):
            name = f"{args.voice_id}_{args.partition}_{i:04d}.wav"
            audio_path = out_dir / name
            with wave.open(str(audio_path), "wb") as wav_handle:
                voice.synthesize_wav(text, wav_handle)
            index_handle.write(
                json.dumps(
                    {
                        "path": f"spoofs/{name}",
                        "label": "synthetic",
                        "partition": args.partition,
                        "speaker_id": f"piper:{args.voice_id}",
                        "source": "piper_tts",
                        "voice_id": args.voice_id,
                        "text_origin_uttid": utt_id,
                    }
                )
                + "\n"
            )
            written += 1
            if written % 20 == 0:
                print(f"  synthesized {written}/{len(chosen)}")

    print(f"Wrote {written} synthetic samples to {out_dir}")
    print(f"Index entries appended to {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
