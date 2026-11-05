"""Dataset manifest: the honest bridge between a real dataset and training.

A manifest is a JSONL file where each line describes ONE labelled audio file:

    {"path": "ASVspoof2019_LA_train/flac/LA_D_1000.flac",
     "label": "real", "partition": "train",
     "speaker_id": "LA_D_1000", "source": "asvspoof2019_la"}

Rules enforced everywhere in this project:
  - only two normalized labels exist: "real" and "synthetic";
  - missing, ambiguous or unknown labels are REJECTED, never guessed;
  - a manifest that cannot provide both classes for a partition fails loudly;
  - speaker ids, when present, must not overlap between train and validation
    (leakage protection).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

VALID_LABELS = ("real", "synthetic")
VALID_PARTITIONS = ("train", "validation", "eval")

_LABEL_ALIASES = {
    "real": "real",
    "bonafide": "real",
    "genuine": "real",
    "natural": "real",
    "synthetic": "synthetic",
    "spoof": "synthetic",
    "fake": "synthetic",
}


class ManifestError(RuntimeError):
    """Raised for invalid, incomplete or leaking manifests."""


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    label: str
    partition: str
    speaker_id: str | None
    source: str
    # Optional provenance (multi-dataset manifests); None in legacy manifests.
    generator_id: str | None = None
    language: str | None = None

    def resolved_path(self, dataset_dir: str | Path) -> Path:
        return Path(dataset_dir) / self.path


def normalize_label(raw: object) -> str:
    """Map a dataset's label vocabulary to our two classes, or reject."""
    if not isinstance(raw, str):
        raise ManifestError(f"Label must be a string, got {type(raw).__name__}.")
    key = raw.strip().lower()
    if key not in _LABEL_ALIASES:
        raise ManifestError(
            f"Unknown label {raw!r}. Valid: bonafide/real/genuine or spoof/synthetic/fake."
        )
    return _LABEL_ALIASES[key]


def _parse_entry(raw: dict[str, object], line_no: int) -> ManifestEntry:
    if not isinstance(raw, dict):
        raise ManifestError(f"Line {line_no}: manifest entry must be a JSON object.")
    path = raw.get("path")
    if not isinstance(path, str) or not path.strip():
        raise ManifestError(f"Line {line_no}: missing non-empty 'path'.")
    label = normalize_label(raw.get("label"))
    partition = raw.get("partition")
    if partition not in VALID_PARTITIONS:
        raise ManifestError(
            f"Line {line_no}: partition must be one of {VALID_PARTITIONS}, got {partition!r}."
        )
    speaker = raw.get("speaker_id")
    if speaker is not None and not isinstance(speaker, str):
        speaker = str(speaker)
    source = raw.get("source", "unknown")
    generator = raw.get("generator_id")
    language = raw.get("language")
    return ManifestEntry(
        path=path.strip(),
        label=label,
        partition=partition,
        speaker_id=speaker,
        source=str(source),
        generator_id=str(generator) if generator is not None else None,
        language=str(language) if language is not None else None,
    )


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    """Load and fully validate a manifest. Raises ManifestError on any defect."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ManifestError(f"Manifest file not found: {manifest_path}")

    entries: list[ManifestEntry] = []
    seen_paths: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"Line {line_no}: invalid JSON ({exc.msg}).") from exc
            entry = _parse_entry(raw, line_no)
            if entry.path in seen_paths:
                raise ManifestError(f"Line {line_no}: duplicate path {entry.path!r}.")
            seen_paths.add(entry.path)
            entries.append(entry)

    if not entries:
        raise ManifestError(f"Manifest {manifest_path} contains no entries.")
    return entries


def require_partition_classes(entries: list[ManifestEntry], partition: str) -> list[ManifestEntry]:
    """All entries of one partition; fails unless both classes are present."""
    subset = [entry for entry in entries if entry.partition == partition]
    if not subset:
        raise ManifestError(f"Manifest contains no entries for partition '{partition}'.")
    labels = {entry.label for entry in subset}
    missing = set(VALID_LABELS) - labels
    if missing:
        raise ManifestError(
            f"Partition '{partition}' is missing class(es) {sorted(missing)}; "
            "training/evaluation on a single class is not meaningful and is refused."
        )
    return subset


def assert_no_speaker_leakage(
    train: list[ManifestEntry],
    validation: list[ManifestEntry],
    exempt_sources: tuple[str, ...] = (),
) -> None:
    """Fail if any speaker id appears in both partitions (leakage protection).

    `exempt_sources` skips entries from named data sources. It exists for
    generator-based corpora (e.g. piper_tts), where — following the ASVspoof
    precedent of sharing attack types between train and dev — the SAME TTS
    voice is deliberately used in train and validation, while the eval
    partition uses an UNSEEN voice. Any exemption must be stated in the
    training report; human-speaker entries are never exempted implicitly.
    """
    train_speakers = {
        e.speaker_id for e in train if e.speaker_id and e.source not in exempt_sources
    }
    validation_speakers = {
        e.speaker_id
        for e in validation
        if e.speaker_id and e.source not in exempt_sources
    }
    overlap = sorted(train_speakers & validation_speakers)
    if overlap:
        raise ManifestError(
            f"Speaker leakage: ids {overlap[:5]} appear in both train and validation. "
            "Use speaker-disjoint splits (official ASVspoof partitions are disjoint by design)."
        )


def summarize(entries: list[ManifestEntry]) -> dict[str, object]:
    """Counts per partition and label — used in training reports."""
    summary: dict[str, dict[str, int]] = {}
    for partition in VALID_PARTITIONS:
        subset = [e for e in entries if e.partition == partition]
        summary[partition] = {
            "real": sum(1 for e in subset if e.label == "real"),
            "synthetic": sum(1 for e in subset if e.label == "synthetic"),
        }
    return {"counts": summary, "total": len(entries)}
