"""Unified multi-dataset manifest for multi-source training.

Layer ABOVE the existing training manifest (app/services/manifest.py). It
supports more sources, richer provenance, and a third label:

    {"file_path": "...", "label": "real|fake|unknown",
     "source_dataset": "...", "speaker_id": "...", "recording_id": "...",
     "generator_id": "...", "language": "...", "split": "train|validation|eval|holdout"}

Rules:
  - labels map to real / fake / unknown; unknown is recorded but EXCLUDED
    from supervised training (the training-manifest converter drops it);
  - provenance (source dataset, recording, generator, language) is preserved
    end-to-end so evaluation can break results down per source and per
    generator, and so leakage protection can group related recordings;
  - splits are assigned to GROUPS (speaker or recording), never to single
    files: related recordings, their synthetic versions, and duplicates land
    in the same split;
  - exact duplicate clusters (same audio hash) are detected across all
    sources and forced into the same split;
  - an unseen-generator holdout can be carved out for generalization testing.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

VALID_UNIFIED_LABELS = ("real", "fake", "unknown")
VALID_SPLITS = ("train", "validation", "eval", "holdout")

_LABEL_MAP = {
    # verified-authentic vocabulary
    "real": "real", "human": "real", "natural": "real", "genuine": "real",
    "bonafide": "real", "original": "real", "authentic": "real", "raw": "real",
    # synthetic vocabulary
    "fake": "fake", "synthetic": "fake", "spoof": "fake", "ai": "fake",
    "aigenerated": "fake", "generated": "fake", "clone": "fake", "cloned": "fake",
    "tts": "fake", "deepfake": "fake", "converted": "fake", "synth": "fake",
}


class MultisourceError(RuntimeError):
    """Raised for invalid unified manifests, labels or split assignments."""


@dataclass(frozen=True)
class UnifiedEntry:
    file_path: str
    label: str
    source_dataset: str
    speaker_id: str | None = None
    recording_id: str | None = None
    generator_id: str | None = None
    language: str | None = None
    split: str | None = None

    def group_key(self) -> str:
        """All recordings sharing this key must land in the same split.

        speaker_id values that carry the cross-source namespace prefix
        'librispeech:' identify the same underlying LibriSpeech speaker in
        DIFFERENT sources (FoR and audio_deepfake both use LibriSpeech), so
        they group globally — a speaker seen through one source must not
        appear in another source's split. Everything else is scoped to its
        source dataset to keep unrelated files with colliding names apart.
        """
        if self.speaker_id and self.speaker_id.startswith("librispeech:"):
            return f"speaker:{self.speaker_id}"
        for key in (self.speaker_id, self.recording_id):
            if key:
                return f"{self.source_dataset}:{key}"
        return f"{self.source_dataset}:{Path(self.file_path).name}"


def normalize_unified_label(raw: object) -> str:
    """Map a source label to real/fake; anything uncertain is 'unknown'."""
    if not isinstance(raw, str):
        return "unknown"
    key = raw.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if key in ("unknownlabel", "unknown", "unverified", ""):
        return "unknown"
    return _LABEL_MAP.get(key, "unknown")


def write_unified_manifest(entries: list[UnifiedEntry], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with output.open("w", encoding="utf-8") as handle:
        for entry in entries:
            if entry.file_path in seen:
                raise MultisourceError(f"Duplicate file_path in unified manifest: {entry.file_path}")
            seen.add(entry.file_path)
            if entry.label not in VALID_UNIFIED_LABELS:
                raise MultisourceError(f"Invalid unified label {entry.label!r}")
            handle.write(json.dumps(asdict(entry)) + "\n")


def load_unified_manifest(path: str | Path) -> list[UnifiedEntry]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise MultisourceError(f"Unified manifest not found: {manifest_path}")
    entries: list[UnifiedEntry] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MultisourceError(f"Line {line_no}: invalid JSON ({exc.msg}).") from exc
            if not isinstance(raw, dict) or "file_path" not in raw:
                raise MultisourceError(f"Line {line_no}: entry must be an object with file_path.")
            label = normalize_unified_label(raw.get("label"))
            split = raw.get("split")
            if split is not None and split not in VALID_SPLITS:
                raise MultisourceError(f"Line {line_no}: invalid split {split!r}.")
            entries.append(
                UnifiedEntry(
                    file_path=str(raw["file_path"]),
                    label=label,
                    source_dataset=str(raw.get("source_dataset", "unknown")),
                    speaker_id=raw.get("speaker_id"),
                    recording_id=raw.get("recording_id"),
                    generator_id=raw.get("generator_id"),
                    language=raw.get("language"),
                    split=split,
                )
            )
    if not entries:
        raise MultisourceError(f"Unified manifest {manifest_path} contains no entries.")
    return entries


def sha256_of_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_duplicate_clusters(
    entries: list[UnifiedEntry],
    dataset_dir: str | Path,
) -> list[list[UnifiedEntry]]:
    """Cluster entries whose underlying audio is byte-identical (sha256).

    Cross-source duplicates are the dangerous kind: the same recording sold
    as 'real' in one dataset and 'synthetic' in another silently leaks between
    splits. Returns a list of clusters with 2+ members.
    """
    by_hash: dict[str, list[UnifiedEntry]] = {}
    root = Path(dataset_dir)
    for entry in entries:
        path = root / entry.file_path
        if not path.is_file():
            continue  # missing files are reported by the inspector, not here
        by_hash.setdefault(sha256_of_file(path), []).append(entry)
    return [members for members in by_hash.values() if len(members) > 1]


def duplicate_cluster_keys(
    clusters: list[list[UnifiedEntry]],
) -> list[list[str]]:
    """Convert entry clusters into lists of group keys for assign_splits."""
    out: list[list[str]] = []
    for cluster in clusters:
        keys = list(dict.fromkeys(e.group_key() for e in cluster))
        if len(keys) > 1:
            out.append(keys)
    return out


def assign_splits(
    entries: list[UnifiedEntry],
    duplicate_clusters: list[list[str]] | None = None,
    seed: int = 42,
    validation_fraction: float = 0.15,
    eval_fraction: float = 0.15,
    holdout_generator_fraction: float = 0.2,
    holdout_generators: set[str] | None = None,
) -> list[UnifiedEntry]:
    """Deterministically assign train/validation/eval/holdout to whole groups.

    duplicate_clusters: lists of group keys whose recordings are
    byte-identical (from build_duplicate_clusters); each cluster is merged
    into one unit so duplicates always land in the same split.

    Guarantees (validated in tests):
      - every group (speaker/recording) and every duplicate cluster lands in
        exactly one split;
      - a share of unseen GENERATORS goes to the independent holdout split
        for generalization testing; holdout generators never appear in
        train, validation or eval;
      - unknown-label entries keep split=None (excluded from supervised work).
    """
    import random

    rng = random.Random(seed)
    known = [e for e in entries if e.label in ("real", "fake")]
    groups: dict[str, list[UnifiedEntry]] = {}
    for entry in known:
        groups.setdefault(entry.group_key(), []).append(entry)

    # Merge duplicate clusters into single split units (union-find).
    parents: dict[str, str] = {}
    for cluster in duplicate_clusters or []:
        keys = [find_root(parents, key) for key in cluster if key in groups]
        if len(keys) > 1:
            base = keys[0]
            for other in keys[1:]:
                parents[other] = base

    def root_of(key: str) -> str:
        return find_root(parents, key)

    # Unseen-generator holdout: reserve some synthetic-only generators
    # entirely. Sampling is per generator NAME (deterministic, sorted), and
    # every group containing an entry of a holdout generator moves to the
    # holdout split — so a held-out generator can never appear in train,
    # validation or eval, and paired source recordings travel with their
    # generated versions.
    keys_by_generator: dict[str, list[str]] = {}
    for key, members in groups.items():
        for member in members:
            if member.generator_id:
                keys_by_generator.setdefault(member.generator_id, []).append(key)
    holdout_roots: set[str] = set()
    generators = sorted(keys_by_generator)
    fake_only_generators = {
        g for g in generators
        if all(e.label == "fake" for key in keys_by_generator[g] for e in groups[key])
    }
    if holdout_generators is None:
        holdout_generators = {
            g for g in sorted(fake_only_generators)
            if len(generators) > 1 and rng.random() < holdout_generator_fraction
        }
    holdout_generators = {g for g in holdout_generators if g in generators}
    unknown_holdout = sorted(set(holdout_generators or set()) - set(generators))
    if unknown_holdout:
        raise MultisourceError(
            f"holdout generators not found in data: {unknown_holdout}"
        )
    for key, members in groups.items():
        if any(m.generator_id in holdout_generators for m in members):
            holdout_roots.add(root_of(key))

    all_roots = sorted({root_of(key) for key in groups})
    remaining = [r for r in all_roots if r not in holdout_roots]

    shuffled = list(remaining)
    rng.shuffle(shuffled)
    total = len(shuffled)
    n_validation = int(round(total * validation_fraction))
    n_eval = int(round(total * eval_fraction))
    validation_roots = set(shuffled[:n_validation])
    eval_roots = set(shuffled[n_validation : n_validation + n_eval])

    split_of: dict[str, str] = {}
    for root in shuffled:
        if root in validation_roots:
            split_of[root] = "validation"
        elif root in eval_roots:
            split_of[root] = "eval"
        else:
            split_of[root] = "train"

    assigned: list[UnifiedEntry] = []
    for entry in entries:
        if entry.label == "unknown":
            assigned.append(UnifiedEntry(**{**asdict(entry), "split": None}))
            continue
        root = root_of(entry.group_key())
        split = "holdout" if root in holdout_roots else split_of.get(root, "train")
        assigned.append(UnifiedEntry(**{**asdict(entry), "split": split}))
    return assigned


def find_root(parents: dict[str, str], key: str) -> str:
    while key in parents:
        key = parents[key]
    return key


def unified_to_training_manifest(
    entries: list[UnifiedEntry],
    path: str | Path,
    partition_map: dict[str, str] | None = None,
) -> dict:
    """Write a training manifest compatible with app.services.manifest.

    unknown labels are EXCLUDED (never guessed into a class). The holdout
    split maps to its own partition unless a mapping is supplied.
    Returns a summary of what was written and what was excluded.
    """
    partition_map = partition_map or {
        "train": "train", "validation": "validation", "eval": "eval",
    }
    written = 0
    excluded_unknown = 0
    excluded_holdout = 0
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for entry in entries:
            if entry.label == "unknown":
                excluded_unknown += 1
                continue
            partition = partition_map.get(entry.split or "")
            if partition is None:
                excluded_holdout += 1
                continue
            handle.write(json.dumps({
                "path": entry.file_path,
                "label": "real" if entry.label == "real" else "synthetic",
                "partition": partition,
                "speaker_id": entry.speaker_id
                if entry.speaker_id and entry.speaker_id.startswith("librispeech:")
                else f"{entry.source_dataset}:{entry.speaker_id or entry.recording_id}",
                "source": entry.source_dataset,
                "generator_id": entry.generator_id,
                "language": entry.language,
            }) + "\n")
            written += 1
    return {
        "written": written,
        "excluded_unknown": excluded_unknown,
        "excluded_holdout": excluded_holdout,
        "manifest": str(output),
    }


def summarize_unified(entries: list[UnifiedEntry]) -> dict:
    """Counts by source x label x split, for the split summary report."""
    summary: dict[str, dict[str, int]] = {}
    for entry in entries:
        key = f"{entry.source_dataset}|{entry.label}|{entry.split or 'none'}"
        summary[key] = summary.get(key, 0) + 1
    return {
        "total": len(entries),
        "by_source_label_split": dict(sorted(summary.items())),
    }
