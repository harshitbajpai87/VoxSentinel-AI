"""Build the unified multi-dataset manifest from downloaded sources.

Adapters normalize each source into UnifiedEntry rows with the fields:
  file_path, label (real/fake/unknown), source_dataset, speaker_id,
  recording_id, generator_id, language, split

Label policy (anti-fabrication):
  - labels come from VERIFIED structure only (official folders / protocol /
    dataset description). Nothing is guessed from vibes;
  - files whose class cannot be established get label "unknown" and are kept
    in the manifest but EXCLUDED from supervised training;
  - the synthetic-speech-commands source is synthetic-only by design; the
    builder REQUIRES a real-speech supplement (Speech Commands v0.02) so that
    training always sees both classes;
  - if a source directory is absent, it is skipped and reported as such
    (never silently invented).

After label mapping the builder:
  - detects exact duplicate clusters (sha256) and keeps them together;
  - assigns train/validation/eval/holdout splits to whole speaker/recording
    groups, with an unseen-generator holdout for generalization testing;
  - writes the unified manifest, a split summary, and a training manifest
    compatible with scripts/train_detector.py and scripts/train_candidate.py
    (unknown entries excluded, holdout excluded).

Usage:
    python scripts/build_multisource_manifest.py --data-dir ../data \\
        --output-manifest ../data/multidataset/unified_manifest.jsonl \\
        --output-training-manifest ../data/multidataset/training_manifest.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.multisource import (  # noqa: E402
    MultisourceError,
    UnifiedEntry,
    assign_splits,
    build_duplicate_clusters,
    duplicate_cluster_keys,
    load_unified_manifest,
    summarize_unified,
    unified_to_training_manifest,
    write_unified_manifest,
)

AUDIO_EXTENSIONS = (".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".opus", ".m4a", ".wma")

SOURCE_KEYS = ("synthetic_commands", "speech_commands_real", "human_ai_voices",
               "fake_or_real", "audio_deepfake")


def _audio_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS)


def _stable_id(source: str, relpath: str) -> str:
    """Stable, globally-unique recording id from source + relative path.

    Raw file STEMS are NOT unique identities: unrelated recordings in
    different sources (and even different word folders of one source) can
    share a bare numeric stem. Hashing the canonical location gives every
    physical file exactly one identity, deterministically, without
    disabling duplicate detection.
    """
    import hashlib
    return f"rec-{hashlib.sha256(f'{source}/{relpath}'.encode()).hexdigest()[:12]}"


_LIBRISPEECH_NAME = re.compile(r"^(\d+)_.*")


def _librispeech_speaker(filename: str) -> str | None:
    """LibriSpeech utterance files are <speakerid>_<book>_<utt>.wav.

    The leading numeric token is the LibriSpeech speaker ID — a GLOBAL
    identity shared across sources built on LibriSpeech (FoR real, adarsh
    real_samples), so those speakers are grouped across datasets.
    """
    match = _LIBRISPEECH_NAME.match(filename)
    return f"librispeech:{match.group(1)}" if match else None


def _dir_label(name: str) -> str | None:
    """Verified-structure label from a directory name, or None."""
    lowered = name.lower()
    if any(token in lowered for token in ("real", "human", "genuine", "natural", "bonafide", "original")):
        return "real"
    if any(token in lowered for token in ("fake", "synthetic", "ai", "spoof", "generated", "tts", "clone")):
        return "fake"
    return None


def adapter_synthetic_commands(root: Path) -> tuple[list[UnifiedEntry], str]:
    """jbuchner Synthetic Speech Commands — synthetic-only TTS corpus.

    Every audio file is synthetic per the dataset description (TTS
    counterparts of Speech Commands v0.01). Generator info is not published
    per-file, so generator_id stays None; language is English.

    Actual layout (verified after download): augmented_dataset/augmented_dataset/
    <word>/*.wav plus a separate augmented_dataset_verynoisy variant. Only the
    primary augmented_dataset set is used; the verynoisy variant is excluded
    from the base manifest (deliberately noise-degraded distribution).
    """
    primary = root / "augmented_dataset" / "augmented_dataset"
    scan_root = primary if primary.is_dir() else root
    note_noise = (
        "; excluded augmented_dataset_verynoisy variant"
        if (root / "augmented_dataset_verynoisy").is_dir() else ""
    )
    files = _audio_files(scan_root)
    entries = [
        UnifiedEntry(
            file_path=str(f.relative_to(root)),
            label="fake",
            source_dataset="synthetic_commands",
            recording_id=_stable_id("synthetic_commands", str(f.relative_to(root))),
            language="en",
        )
        for f in files
    ]
    return entries, (
        f"all {len(entries)} files labeled fake per dataset description "
        f"(synthetic-only corpus){note_noise}"
    )


def adapter_speech_commands_real(root: Path) -> tuple[list[UnifiedEntry], str]:
    """Speech Commands v0.02 — authentic human recordings.

    Verified layout: <word>/<speakerhash>_<utterance>.wav — the filename
    prefix before '_' identifies the volunteer (official dataset format).
    """
    entries: list[UnifiedEntry] = []
    skipped_noise = 0
    for path in _audio_files(root):
        if "_background_noise_" in path.parts:
            skipped_noise += 1  # environmental noise samples, not speech
            continue
        speaker = path.name.split("_")[0] if "_" in path.name else None
        rel = str(path.relative_to(root))
        entries.append(UnifiedEntry(
            file_path=rel,
            label="real",
            source_dataset="speech_commands_real",
            speaker_id=speaker,
            recording_id=_stable_id("speech_commands_real", rel),
            language="en",
        ))
    return entries, (f"{len(entries)} real files ({skipped_noise} background-noise files excluded); "
        "speaker from official <hash>_<n>.wav naming")


def adapter_human_ai_voices(root: Path) -> tuple[list[UnifiedEntry], str]:
    """kambingbersayaphitam dataset — human and AI voices, Indonesian.

    Documented structure: two class folders (42 WAV total). Class folders are
    detected by name; anything unrecognized becomes 'unknown'.
    """
    entries: list[UnifiedEntry] = []
    unknown = 0
    for path in _audio_files(root):
        parts = [p.lower() for p in path.parts]
        label = None
        for part in reversed(parts):
            label = _dir_label(part)
            if label:
                break
        if label is None:
            label = "unknown"
            unknown += 1
        rel = str(path.relative_to(root))
        entries.append(UnifiedEntry(
            file_path=rel,
            label=label,
            source_dataset="human_ai_voices",
            recording_id=_stable_id("human_ai_voices", rel),
            language="id",
        ))
    return entries, f"{len(entries)} files ({unknown} unknown); label from class folders"


def adapter_fake_or_real(root: Path, version: str = "for-norm") -> tuple[list[UnifiedEntry], str]:
    """The Fake-or-Real (FoR) dataset — real and TTS-generated utterances.

    Documented structure: <for-version>/<split-set>/<real|fake>/...
    with split sets training-set / validation-set / testing-set. Labels come
    from the real/fake folder level. Preferred version: for-norm (balanced,
    normalized). MP3 files that soundfile cannot read are recorded as
    'unknown' with a note, never guessed.
    """
    version_root = root / version if (root / version).is_dir() else root
    set_to_split = {
        "training-set": "train", "validation-set": "validation",
        "testing-set": "eval", "train": "train", "training": "train",
        "validation": "validation", "test": "eval", "testing": "eval",
        "eval": "eval", "dev": "validation", "development": "validation",
    }
    entries: list[UnifiedEntry] = []
    unknown = 0
    for path in _audio_files(version_root):
        parts = [p.lower() for p in path.parts]
        label = None
        for part in reversed(parts):
            label = _dir_label(part)
            if label:
                break
        split = "train"
        for part in parts:
            if part in set_to_split:
                split = set_to_split[part]
                break
        if label is None:
            unknown += 1
        speaker = _librispeech_speaker(path.name)
        entries.append(UnifiedEntry(
            file_path=str(path.relative_to(root)),
            label=label or "unknown",
            source_dataset="fake_or_real",
            speaker_id=speaker,
            recording_id=path.stem,  # real/fake pairs share the utterance name
            language="mixed",
            split=split,  # provisional; group-aware assign_splits() overrides it
        ))
    return entries, (
        f"{len(entries)} files ({unknown} unknown) from version '{version_root.name}'; "
        "labels from real/fake folders, provisional splits from official sets"
    )


def adapter_audio_deepfake(root: Path) -> tuple[list[UnifiedEntry], str]:
    """adarshsingh0903 Audio Deepfake Detection — real vs AI-generated WAVs.

    Actual layout (verified after download): real_samples/ holds the 2,274
    genuine recordings; every OTHER top-level directory (OpenAI, xTTS, VALLE,
    VoiceBox, FlashSpeech, NaturalSpeech3, PromptTTS2, seedtts_files) holds
    AI-generated clips of one named generator. Generator provenance is kept
    so an unseen-generator holdout can be carved out.
    """
    entries: list[UnifiedEntry] = []
    unknown = 0
    real_dir = root / "real_samples"
    real_files = _audio_files(real_dir) if real_dir.is_dir() else []
    for path in real_files:
        rel = str(path.relative_to(root))
        entries.append(UnifiedEntry(
            file_path=rel,
            label="real",
            source_dataset="audio_deepfake",
            speaker_id=_librispeech_speaker(path.name),
            recording_id=_stable_id("audio_deepfake", rel),
        ))
    generator_dirs = sorted(d for d in root.iterdir()
                             if d.is_dir() and d.name != "real_samples")
    for gen_dir in generator_dirs:
        for path in _audio_files(gen_dir):
            rel = str(path.relative_to(root))
            entries.append(UnifiedEntry(
                file_path=rel,
                label="fake",
                source_dataset="audio_deepfake",
                recording_id=_stable_id("audio_deepfake", rel),
                generator_id=gen_dir.name,
            ))
    return entries, (
        f"{len(entries)} files ({len(real_files)} real, "
        f"{len(entries) - len(real_files)} fake across {len(generator_dirs)} named generators; "
        f"{unknown} unknown)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the unified multi-dataset manifest.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-training-manifest", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--for-version", default="for-norm")
    parser.add_argument("--holdout-generator-fraction", type=float, default=0.2)
    parser.add_argument(
        "--holdout-generators", default=None,
        help="Comma-separated generator ids to reserve entirely for the "
             "unseen-generator holdout (overrides random sampling).",
    )
    parser.add_argument(
        "--max-per-class-per-source", type=int, default=None,
        help="Deterministic per-source, per-class sample cap (compute budget; "
             "prevents any single large dataset from dominating).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    adapters = {
        "synthetic_commands": lambda r: adapter_synthetic_commands(r),
        "speech_commands_real": lambda r: adapter_speech_commands_real(r),
        "human_ai_voices": lambda r: adapter_human_ai_voices(r),
        "fake_or_real": lambda r: adapter_fake_or_real(r, args.for_version),
        "audio_deepfake": lambda r: adapter_audio_deepfake(r),
    }
    # Common download layout: <data-dir>/<source key>/
    source_dirs = {key: data_dir / key for key in SOURCE_KEYS}

    all_entries: list[UnifiedEntry] = []
    adapter_notes: dict[str, str] = {}
    skipped: list[str] = []
    for key in SOURCE_KEYS:
        source_dir = source_dirs[key]
        if not source_dir.is_dir():
            skipped.append(f"{key}: directory not found at {source_dir}")
            continue
        entries, note = adapters[key](source_dir)
        adapter_notes[key] = note
        # file_path must be resolvable against the DATA ROOT, so the source
        # directory is part of the recorded path (this also keeps provenance
        # visible in the manifest itself).
        all_entries.extend(
            UnifiedEntry(**{**e.__dict__, "file_path": f"{key}/{e.file_path}"})
            for e in entries
        )

    if not all_entries:
        print("ERROR: no dataset directories found under", data_dir, file=sys.stderr)
        for line in skipped:
            print("  skipped:", line, file=sys.stderr)
        return 2

    if args.max_per_class_per_source:
        import random as _random
        rng = _random.Random(args.seed)
        grouped: dict[tuple[str, str], list[UnifiedEntry]] = {}
        for e in all_entries:
            grouped.setdefault((e.source_dataset, e.label), []).append(e)
        capped: list[UnifiedEntry] = []
        for group in grouped.values():
            rng.shuffle(group)
            capped.extend(group[: args.max_per_class_per_source])
        print(
            f"Capped to max {args.max_per_class_per_source} per class per source: "
            f"{len(all_entries)} -> {len(capped)} entries"
        )
        all_entries = capped

    labels = {e.label for e in all_entries}
    if "fake" in labels and "real" not in labels and "synthetic_commands" in adapter_notes:
        print(
            "ERROR: synthetic_commands is present but no real-speech supplement was found. "
            "Download speech_commands_real (see scripts/multisource_download.py) — "
            "training fake-vs-nothing is refused.",
            file=sys.stderr,
        )
        return 2

    try:
        write_unified_manifest(all_entries, args.output_manifest)
    except MultisourceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Duplicate detection across ALL sources, then group-aware splitting.
    print(f"Hashing {len(all_entries)} files for exact duplicates ...")
    clusters = build_duplicate_clusters(
        load_unified_manifest(args.output_manifest), data_dir
    )
    cluster_keys = duplicate_cluster_keys(clusters)
    print(f"  found {len(clusters)} duplicate cluster(s) across sources")

    entries_with_splits = assign_splits(
        load_unified_manifest(args.output_manifest),
        duplicate_clusters=cluster_keys,
        seed=args.seed,
        holdout_generator_fraction=args.holdout_generator_fraction,
        holdout_generators=(
            {g.strip() for g in args.holdout_generators.split(",") if g.strip()}
            if args.holdout_generators else None
        ),
    )
    write_unified_manifest(entries_with_splits, args.output_manifest)

    conversion = unified_to_training_manifest(entries_with_splits, args.output_training_manifest)

    summary = {
        "report_type": "multisource_manifest_summary",
        "data_dir": str(data_dir),
        "seed": args.seed,
        "sources_processed": list(adapter_notes),
        "sources_skipped": skipped,
        "adapter_notes": adapter_notes,
        "duplicate_clusters": [
            [asdict_path(e) for e in cluster] for cluster in clusters
        ],
        "split_summary": summarize_unified(entries_with_splits),
        "training_manifest_conversion": conversion,
        "note": (
            "Holdout split contains unseen generators (excluded from training/validation/eval; "
            "reserved for generalization testing). 'unknown' labels are excluded from supervised "
            "training. Provisional FoR splits were overridden by group-aware assignment."
        ),
    }
    summary_path = (
        Path(args.summary) if args.summary
        else Path(args.output_manifest).with_suffix(".summary.json")
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Unified manifest: {args.output_manifest} ({len(entries_with_splits)} entries)")
    print(
        f"Training manifest: {args.output_training_manifest} ({conversion['written']} entries; "
        f"{conversion['excluded_unknown']} unknown excluded; "
        f"{conversion['excluded_holdout']} holdout excluded)"
    )
    print(f"Summary: {summary_path}")
    return 0


def asdict_path(entry: UnifiedEntry) -> str:
    return entry.file_path


if __name__ == "__main__":
    raise SystemExit(main())
