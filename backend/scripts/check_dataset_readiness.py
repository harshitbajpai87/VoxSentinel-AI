"""Read-only manifest/audio readiness audit; standard library only, no model loading."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath

PARTITIONS = ("train", "validation", "eval")
SPLITS = {"train": "train", "validation": "validation", "val": "validation",
          "dev": "validation", "eval": "eval", "test": "eval", "holdout": "holdout"}
LABELS = {"real": "real", "bonafide": "real", "genuine": "real", "natural": "real",
          "synthetic": "synthetic", "spoof": "synthetic", "fake": "synthetic"}
UNKNOWN = {"", "unknown", "none", "null", "n/a", "na", "-"}


def known(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in UNKNOWN else text


def safe_audio_path(root: Path, raw: object, missing_prefixes: set[str] | None = None) -> Path:
    """Reject absolute, drive-relative, traversal and symlink escapes on any OS."""
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise ValueError("path must be a non-empty relative string without NUL")
    text = raw.replace("\\", "/")
    windows = PureWindowsPath(raw)
    relative = PurePosixPath(text)
    if windows.drive or windows.root or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("absolute paths and path traversal are not allowed")
    candidate = root / Path(*relative.parts)
    # Missing external corpora can have tens of thousands of rows. Once a
    # top-level component is absent there is no symlink below it to resolve.
    # Cache that snapshot fact rather than repeatedly resolving long Windows paths.
    if missing_prefixes is not None and relative.parts:
        first = relative.parts[0]
        if first in missing_prefixes:
            return candidate
        anchor = root / first
        if not anchor.exists() and not anchor.is_symlink():
            missing_prefixes.add(first)
            return candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes dataset root (including symlinks)") from exc
    return candidate


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def audit(manifests: list[Path], dataset_dir: Path, max_examples: int = 20) -> dict:
    """Stream rows and audio bytes, retaining only counters and identity/hash indexes.

    Manifest-supplied hashes are deliberately ignored. Counts describe manifest
    claims, not verified audio quality. Missing metadata fails closed.
    """
    root = dataset_dir.resolve()
    errors: Counter = Counter()
    examples: dict[str, list] = defaultdict(list)
    counts = {p: {"total": 0, "classes": {"real": 0, "synthetic": 0},
                  "languages": {}, "class_languages": {"real": {}, "synthetic": {}}}
              for p in (*PARTITIONS, "holdout")}
    speakers: dict[str, set] = defaultdict(set)
    generators: dict[str, set] = defaultdict(set)
    hashes: dict[str, set] = defaultdict(set)
    hash_examples: dict[str, list] = defaultdict(list)
    paths: set[tuple[str, str]] = set()
    hashed_paths: dict[Path, str] = {}
    missing_prefixes: set[str] = set()
    total_rows = valid_rows = existing = hashed_bytes = 0

    def error(code: str, detail: dict) -> None:
        errors[code] += 1
        if len(examples[code]) < max_examples:
            examples[code].append(detail)

    if not root.is_dir():
        error("missing_dataset_directory", {"path": str(root)})
    for manifest in manifests:
        try:
            handle = manifest.open("r", encoding="utf-8-sig", newline="")
        except (OSError, ValueError) as exc:
            error("manifest_unreadable", {"manifest": str(manifest), "error": str(exc)})
            continue
        with handle:
            try:
                if manifest.suffix.lower() == ".csv":
                    rows = enumerate(csv.DictReader(handle), start=2)
                else:
                    rows = enumerate(handle, start=1)
                for line_no, item in rows:
                    if isinstance(item, str) and not item.strip():
                        continue
                    total_rows += 1
                    location = {"manifest": str(manifest), "line": line_no}
                    try:
                        row = json.loads(item) if isinstance(item, str) else item
                        if not isinstance(row, dict):
                            raise ValueError("entry must be an object")
                        label = LABELS.get(str(row.get("label", "")).strip().lower())
                        partition = SPLITS.get(str(row.get("partition", row.get("split", ""))).lower())
                        if label is None or partition is None:
                            raise ValueError("unknown/missing label or partition/split")
                    except (ValueError, TypeError) as exc:
                        error("invalid_record", {**location, "error": str(exc)})
                        continue
                    valid_rows += 1
                    count = counts[partition]
                    language = known(row.get("language")) or "unknown"
                    count["total"] += 1
                    count["classes"][label] += 1
                    for language_counts in (count["languages"], count["class_languages"][label]):
                        language_counts[language] = language_counts.get(language, 0) + 1
                    speaker = known(row.get("speaker_id"))
                    generator = known(row.get("generator_id"))
                    if speaker is None:
                        error("missing_speaker_id", location)
                    elif ":rec-" in speaker:
                        error("recording_proxy_not_speaker", {**location, "speaker_id": speaker})
                        speakers[speaker].add(partition)
                    else:
                        speakers[speaker].add(partition)
                    if label == "synthetic":
                        if generator is None:
                            error("missing_generator_id", location)
                        else:
                            generators[generator].add(partition)
                    raw_path = row.get("path", row.get("file_path"))
                    try:
                        path = safe_audio_path(root, raw_path, missing_prefixes)
                    except (OSError, ValueError, RuntimeError) as exc:
                        error("unsafe_path", {**location, "path": raw_path, "error": str(exc)})
                        continue
                    key = (str(path), partition)
                    if key in paths:
                        error("duplicate_manifest_entry", {**location, "path": raw_path})
                    paths.add(key)
                    try:
                        relative_parts = path.relative_to(root).parts
                        prefix_missing = bool(relative_parts and relative_parts[0] in missing_prefixes)
                        if prefix_missing or not path.is_file():
                            error("missing_audio", {**location, "path": raw_path, "resolved_path": str(path)})
                            continue
                        existing += 1
                        if path not in hashed_paths:
                            digest, size = sha256_file(path)
                            hashed_paths[path] = digest
                            hashed_bytes += size
                        digest = hashed_paths[path]
                        hashes[digest].add(partition)
                        if len(hash_examples[digest]) < max_examples:
                            hash_examples[digest].append({**location, "path": raw_path, "partition": partition})
                    except OSError as exc:
                        error("audio_unreadable", {**location, "path": raw_path, "error": str(exc)})
            except (OSError, UnicodeError, csv.Error) as exc:
                error("manifest_unreadable", {"manifest": str(manifest), "error": str(exc)})
    for partition in PARTITIONS:
        for label in ("real", "synthetic"):
            if not counts[partition]["classes"][label]:
                error("missing_partition_class", {"partition": partition, "label": label})
    for code, index in (("speaker_overlap", speakers), ("generator_overlap", generators)):
        for identity, partitions in sorted(index.items()):
            if len(partitions) > 1:
                error(code, {"id": identity, "partitions": sorted(partitions)})
    for digest, partitions in sorted(hashes.items()):
        if len(partitions) > 1:
            error("content_overlap", {"sha256": digest, "partitions": sorted(partitions),
                                      "entries": hash_examples[digest]})
    if not total_rows:
        error("empty_manifest_set", {})
    return {
        "schema_version": 1, "ready": not bool(errors), "dataset_dir": str(root),
        "manifests": [str(p) for p in manifests], "total_rows": total_rows,
        "valid_rows": valid_rows, "existing_audio_rows": existing,
        "unique_files_hashed": len(hashed_paths), "bytes_hashed": hashed_bytes,
        "counts": counts, "error_counts": dict(sorted(errors.items())),
        "error_examples": dict(sorted(examples.items())), "max_examples_per_error": max_examples,
        "checks": {"content_hash": "sha256_streamed_actual_files_only",
                   "speaker_disjoint": "all_pairs_global_ids_no_source_exemptions",
                   "generator_disjoint": "all_pairs_synthetic_generator_ids",
                   "holdout": "separate_partition_checked_against_all_other_partitions"},
        "limitations": ["No audio decoding, speech validation, licensing verification, training or accuracy measurement.",
                       "Disjointness is only as reliable as manifest identities; missing/proxy identities fail closed.",
                       "Unknown languages are counted, never inferred; byte hashes do not detect re-encoded duplicates.",
                       "Snapshot check: inputs must not be modified concurrently."],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True,
                        help="JSONL or CSV; repeat only for non-overlapping parts of one corpus")
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Explicit root for relative audio paths (not the manifest directory)")
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args(argv)
    if args.max_examples < 0:
        parser.error("--max-examples must be non-negative")
    report = audit(args.manifest, args.dataset_dir, args.max_examples)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
