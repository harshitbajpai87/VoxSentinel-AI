"""Build a validated dataset manifest from a real, downloaded dataset.

Currently supported preset:

  asvspoof2019_la — ASVspoof 2019 Logical Access (LA) corpus.
    Expected layout under --dataset-dir (the official release structure):
        <dir>/LA/ASVspoof2019_LA_train/flac/*.flac
        <dir>/LA/ASVspoof2019_LA_dev/flac/*.flac
        <dir>/LA/ASVspoof2019_LA_eval/flac/*.flac
        <dir>/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{train,dev,eval}.*.txt
    Protocol lines look like:
        LA_0079 LA_E_0898165 - bonafide     (train/dev)
        LA_0079 LA_E_4904197 - spoof A07     (spoof attacks carry a method tag)
    The dev partition is mapped to our "validation" name; the official eval
    partition to "eval" and is reserved for FINAL held-out evaluation only.

The script verifies every referenced file exists and every label is known.
Missing files and unknown labels are REJECTED (never guessed) and reported.
Audio files themselves are never modified or copied.

See docs/DATASET_GUIDE.md for how to legitimately obtain the dataset.

Usage:
    python scripts/prepare_dataset.py --preset asvspoof2019_la \\
        --dataset-dir data/asvspoof2019_la --output data/asvspoof2019_la_manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.manifest import (  # noqa: E402
    ManifestError,
    normalize_label,
)

AUDIO_EXTENSIONS = (".flac", ".wav", ".ogg")

PROTOCOL_MAP = {
    "train": ("ASVspoof2019.LA.cm.train.trn.txt", "train"),
    "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", "validation"),
    "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "eval"),
}
# NOTE: the official dev protocol file is ASVspoof2019.LA.cm.dev.trl.txt
# (trl = trial list). Keep both spellings tolerated:
PROTOCOL_MAP["dev"] = ("ASVspoof2019.LA.cm.dev.trl.txt", "validation")

PARTITION_DIRS = {
    "train": ("ASVspoof2019_LA_train", "train"),
    "dev": ("ASVspoof2019_LA_dev", "validation"),
    "eval": ("ASVspoof2019_LA_eval", "eval"),
}


def find_protocol_file(protocols_dir: Path, name: str) -> Path | None:
    for candidate in protocols_dir.glob("*.txt"):
        if candidate.name == name:
            return candidate
    return None


def find_audio_file(audio_dir: Path, utterance_id: str) -> Path | None:
    for ext in AUDIO_EXTENSIONS:
        candidate = audio_dir / f"{utterance_id}{ext}"
        if candidate.is_file():
            return candidate
    return None


def build_asvspoof2019_la(dataset_dir: Path) -> tuple[list[dict[str, object]], list[str]]:
    la_dir = dataset_dir / "LA"
    protocols_dir = la_dir / "ASVspoof2019_LA_cm_protocols"
    if not protocols_dir.is_dir():
        raise ManifestError(
            f"Expected official layout: {protocols_dir} not found. "
            "Check docs/DATASET_GUIDE.md for the expected directory structure."
        )

    entries: list[dict[str, object]] = []
    rejected: list[str] = []

    for key, (protocol_name, partition) in PROTOCOL_MAP.items():
        protocol_file = find_protocol_file(protocols_dir, protocol_name)
        if protocol_file is None:
            rejected.append(f"protocol file not found: {protocol_name}")
            continue
        audio_dir_name, _ = PARTITION_DIRS[key]
        audio_dir = la_dir / audio_dir_name / "flac"
        if not audio_dir.is_dir():
            rejected.append(f"audio directory not found: {audio_dir}")
            continue

        with protocol_file.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                parts = line.split()
                if len(parts) < 5:
                    rejected.append(f"{protocol_file.name}:{line_no}: malformed line")
                    continue
                speaker_id, utterance_id, _tag, raw_label = parts[0], parts[1], parts[2], parts[3]
                try:
                    label = normalize_label(raw_label)
                except ManifestError:
                    rejected.append(
                        f"{protocol_file.name}:{line_no}: unknown label {raw_label!r}"
                    )
                    continue
                audio_file = find_audio_file(audio_dir, utterance_id)
                if audio_file is None:
                    rejected.append(f"{protocol_file.name}:{line_no}: missing audio {utterance_id}")
                    continue
                entries.append(
                    {
                        "path": str(audio_file.relative_to(dataset_dir)),
                        "label": label,
                        "partition": partition,
                        "speaker_id": speaker_id,
                        "source": "asvspoof2019_la",
                    }
                )

    return entries, rejected


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a validated dataset manifest.")
    parser.add_argument("--preset", choices=["asvspoof2019_la", "librispeech_piper"], required=True)
    parser.add_argument("--dataset-dir", required=True, help="Root of the downloaded dataset")
    parser.add_argument("--output", required=True, help="Manifest JSONL path to write")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any protocol entry was rejected (default: warn)",
    )
    # librispeech_piper preset:
    parser.add_argument("--spoof-index", default=None, help="Spoof index JSONL (piper TTS)")
    parser.add_argument("--train-readers", type=int, default=30)
    parser.add_argument("--val-readers", type=int, default=10)
    parser.add_argument("--per-reader-train", type=int, default=8)
    parser.add_argument("--per-reader-val", type=int, default=6)
    parser.add_argument("--per-reader-eval", type=int, default=2)
    args = parser.parse_args()

    if args.preset == "librispeech_piper":
        return main_librispeech_piper(args)

    dataset_dir = Path(args.dataset_dir)
    try:
        if args.preset == "asvspoof2019_la":
            entries, rejected = build_asvspoof2019_la(dataset_dir)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not entries:
        print("ERROR: no valid entries produced; nothing to write.", file=sys.stderr)
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")

    counts = {
        partition: {
            "real": sum(1 for e in entries if e["partition"] == partition and e["label"] == "real"),
            "synthetic": sum(
                1 for e in entries if e["partition"] == partition and e["label"] == "synthetic"
            ),
        }
        for partition in ("train", "validation", "eval")
    }
    summary = {
        "preset": args.preset,
        "dataset_dir": str(dataset_dir),
        "manifest": str(output_path),
        "counts": counts,
        "total_entries": len(entries),
        "rejected": len(rejected),
        "rejected_examples": rejected[:20],
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(entries)} entries to {output_path}")
    print(f"Summary: {summary_path}")
    for partition, partition_counts in counts.items():
        print(
            f"  {partition}: real={partition_counts['real']} "
            f"synthetic={partition_counts['synthetic']}"
        )
    if rejected:
        print(f"  rejected entries: {len(rejected)} (first: {rejected[:3]})")
        if args.strict:
            print("Failing due to --strict and rejections.", file=sys.stderr)
            return 1
    return 0




# ---------------------------------------------------------------------------
# Preset: librispeech_piper — a real, open-licensed speech-spoofing corpus
# built from LibriSpeech (real human speech, OpenSLR SLR12, CC BY 4.0) and
# Piper neural-TTS output (spoofs). See docs/DATASET_GUIDE.md.
# ---------------------------------------------------------------------------


def build_librispeech_piper(
    dataset_dir: Path,
    train_readers: int,
    val_readers: int,
    per_reader_train: int,
    per_reader_val: int,
    per_reader_eval: int,
    spoof_index: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    dev_dir = dataset_dir / "LibriSpeech" / "dev-clean"
    test_dir = dataset_dir / "LibriSpeech" / "test-clean"
    for required in (dev_dir, test_dir, spoof_index):
        if not required.exists():
            raise SystemExit(f"ERROR: expected path not found: {required}")

    dev_readers = sorted((p.name for p in dev_dir.iterdir() if p.is_dir()), key=int)
    test_readers = sorted((p.name for p in test_dir.iterdir() if p.is_dir()), key=int)
    shared = set(dev_readers) & set(test_readers)
    if shared:
        raise SystemExit(f"ERROR: reader overlap between dev-clean and test-clean: {sorted(shared)}")
    if train_readers + val_readers > len(dev_readers):
        raise SystemExit(
            f"ERROR: dev-clean has {len(dev_readers)} readers; split needs {train_readers + val_readers}."
        )

    entries: list[dict[str, object]] = []

    def add_real(reader: str, flac: Path, partition: str) -> None:
        entries.append(
            {
                "path": str(flac.relative_to(dataset_dir)),
                "label": "real",
                "partition": partition,
                "speaker_id": reader,
                "source": "librispeech",
            }
        )

    for i, reader in enumerate(dev_readers):
        partition = "train" if i < train_readers else "validation"
        limit = per_reader_train if partition == "train" else per_reader_val
        if partition == "validation" and i >= train_readers + val_readers:
            continue
        flacs = sorted((dev_dir / reader).rglob("*.flac"))[:limit]
        if not flacs:
            raise SystemExit(f"ERROR: no chapters found for reader {reader}")
        for flac in flacs:
            add_real(reader, flac, partition)

    for reader in test_readers:
        for flac in sorted((test_dir / reader).rglob("*.flac"))[:per_reader_eval]:
            add_real(reader, flac, "eval")

    spoofs: list[dict[str, object]] = []
    with spoof_index.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("label") != "synthetic" or record.get("partition") not in (
                "train",
                "validation",
                "eval",
            ):
                raise SystemExit(f"ERROR: invalid spoof index line {line_no}: {line[:80]}")
            spoofs.append(record)
    entries.extend(spoofs)

    voices = sorted({r.get("voice_id", "unknown") for r in spoofs})
    provenance = {
        "real": "LibriSpeech dev-clean/test-clean (OpenSLR SLR12, CC BY 4.0)",
        "synthetic": f"Piper VITS TTS, voices: {voices} (see docs/DATASET_GUIDE.md)",
        "split": (
            f"dev-clean readers 1-{train_readers} -> train, "
            f"{train_readers + 1}-{train_readers + val_readers} -> validation, "
            "test-clean readers -> eval (speaker-disjoint by construction, verified)"
        ),
    }
    return entries, provenance


def main_librispeech_piper(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset_dir)
    entries, provenance = build_librispeech_piper(
        dataset_dir,
        args.train_readers,
        args.val_readers,
        args.per_reader_train,
        args.per_reader_val,
        args.per_reader_eval,
        Path(args.spoof_index),
    )

    import collections

    counts = collections.Counter((e["partition"], e["label"]) for e in entries)
    for partition in ("train", "validation", "eval"):
        for label in ("real", "synthetic"):
            if counts.get((partition, label), 0) == 0:
                print(f"ERROR: partition {partition} missing class {label}.", file=sys.stderr)
                return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")

    summary = {
        "preset": "librispeech_piper",
        "dataset_dir": str(dataset_dir),
        "manifest": str(output_path),
        "provenance": provenance,
        "counts": {
            partition: {
                "real": counts.get((partition, "real"), 0),
                "synthetic": counts.get((partition, "synthetic"), 0),
            }
            for partition in ("train", "validation", "eval")
        },
        "total_entries": len(entries),
        "parameters": {
            "train_readers": args.train_readers,
            "val_readers": args.val_readers,
            "per_reader_train": args.per_reader_train,
            "per_reader_val": args.per_reader_val,
            "per_reader_eval": args.per_reader_eval,
        },
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(entries)} entries to {output_path}")
    for partition in ("train", "validation", "eval"):
        c = summary["counts"][partition]
        print(f"  {partition}: real={c['real']} synthetic={c['synthetic']}")
    print(f"Summary: {summary_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
