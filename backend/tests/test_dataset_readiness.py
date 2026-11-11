"""Tiny arbitrary bytes exercise audit mechanics, not speech/model performance."""
from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_dataset_readiness.py"
spec = importlib.util.spec_from_file_location("readiness", SCRIPT)
readiness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readiness)


def corpus(tmp_path):
    root = tmp_path / "audio"
    root.mkdir()
    rows = []
    for partition in ("train", "validation", "eval"):
        for label in ("real", "synthetic"):
            name = f"{partition}-{label}.wav"
            (root / name).write_bytes(name.encode())
            rows.append({"path": name, "partition": partition, "label": label,
                         "speaker_id": name, "generator_id": partition if label == "synthetic" else None,
                         "language": "hi" if partition == "eval" else "en"})
    return root, rows


def manifest(tmp_path, rows, name="manifest.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_ready_counts_and_read_only(tmp_path):
    root, rows = corpus(tmp_path)
    source = manifest(tmp_path, rows)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = readiness.audit([source], root)
    assert report["ready"]
    assert report["unique_files_hashed"] == 6
    assert report["counts"]["eval"]["class_languages"] == {"real": {"hi": 1}, "synthetic": {"hi": 1}}
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_content_speaker_and_generator_overlap(tmp_path):
    root, rows = corpus(tmp_path)
    (root / rows[2]["path"]).write_bytes((root / rows[0]["path"]).read_bytes())
    rows[2]["speaker_id"] = rows[0]["speaker_id"]
    rows[3]["generator_id"] = rows[1]["generator_id"]
    report = readiness.audit([manifest(tmp_path, rows)], root)
    for code in ("content_overlap", "speaker_overlap", "generator_overlap"):
        assert report["error_counts"][code] == 1
    assert not report["ready"]


def test_eval_overlap_is_not_skipped(tmp_path):
    root, rows = corpus(tmp_path)
    rows[4]["speaker_id"] = rows[0]["speaker_id"]
    rows[5]["generator_id"] = rows[1]["generator_id"]
    report = readiness.audit([manifest(tmp_path, rows)], root)
    assert report["error_examples"]["speaker_overlap"][0]["partitions"] == ["eval", "train"]
    assert report["error_counts"]["generator_overlap"] == 1


@pytest.mark.parametrize("bad", ["../outside.wav", "dir/../../outside.wav", "..\\outside.wav",
                                     "C:\\audio.wav", "C:audio.wav", "/tmp/audio.wav",
                                     "\\\\host\\share\\x.wav", ""])
def test_unsafe_paths_never_hashed(tmp_path, monkeypatch, bad):
    root, rows = corpus(tmp_path)
    for row in rows:
        row["path"] = bad
    def forbidden(*args, **kwargs):
        raise AssertionError("unsafe file must never be opened for hashing")
    monkeypatch.setattr(readiness, "sha256_file", forbidden)
    report = readiness.audit([manifest(tmp_path, rows)], root)
    assert report["error_counts"]["unsafe_path"] == 6
    assert report["unique_files_hashed"] == 0


def test_symlink_escape(tmp_path):
    root, rows = corpus(tmp_path)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside")
    try:
        (root / "escape.wav").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("OS does not permit symlink creation")
    rows[0]["path"] = "escape.wav"
    report = readiness.audit([manifest(tmp_path, rows)], root)
    assert report["error_counts"]["unsafe_path"] == 1


def test_absent_corpus_does_not_hash(tmp_path, monkeypatch):
    root, rows = corpus(tmp_path)
    for path in root.iterdir():
        path.unlink()
    def forbidden(*args, **kwargs):
        raise AssertionError("missing audio must not be hashed")
    monkeypatch.setattr(readiness, "sha256_file", forbidden)
    report = readiness.audit([manifest(tmp_path, rows)], root, max_examples=1)
    assert report["error_counts"]["missing_audio"] == 6
    assert len(report["error_examples"]["missing_audio"]) == 1
    assert report["bytes_hashed"] == 0


def test_missing_and_proxy_metadata_fail_closed(tmp_path):
    root, rows = corpus(tmp_path)
    rows[0]["speaker_id"] = None
    rows[1]["generator_id"] = "unknown"
    rows[2]["speaker_id"] = "source:rec-deadbeef"
    rows[4].pop("language")
    report = readiness.audit([manifest(tmp_path, rows)], root)
    assert report["error_counts"]["missing_speaker_id"] == 1
    assert report["error_counts"]["missing_generator_id"] == 1
    assert report["error_counts"]["recording_proxy_not_speaker"] == 1
    assert report["counts"]["eval"]["languages"]["unknown"] == 1
    assert not report["ready"]


def test_csv_and_unified_aliases(tmp_path):
    root, rows = corpus(tmp_path)
    for row in rows:
        row["file_path"] = row.pop("path")
        row["split"] = {"validation": "val", "eval": "test"}.get(row.pop("partition"), "train")
        row["label"] = "fake" if row["label"] == "synthetic" else "bonafide"
        row["sha256"] = "untrusted manifest hash"
    source = tmp_path / "manifest.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    assert readiness.audit([source], root)["ready"]


def test_holdout_checked_and_not_merged_with_eval(tmp_path):
    root, rows = corpus(tmp_path)
    extra = dict(rows[5], partition="holdout")
    rows.append(extra)
    report = readiness.audit([manifest(tmp_path, rows)], root)
    assert report["counts"]["eval"]["total"] == 2
    assert report["counts"]["holdout"]["total"] == 1
    assert report["error_counts"]["content_overlap"] == 1
    assert report["error_counts"]["generator_overlap"] == 1


def test_malformed_empty_and_missing_manifests(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('oops\n[]\n{"label":"unknown"}\n', encoding="utf-8")
    report = readiness.audit([bad, tmp_path / "missing.jsonl"], tmp_path)
    assert report["error_counts"]["invalid_record"] == 3
    assert report["error_counts"]["manifest_unreadable"] == 1
    assert report["error_counts"]["missing_partition_class"] == 6
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert readiness.audit([empty], tmp_path)["error_counts"]["empty_manifest_set"] == 1


def test_streamed_hash_matches_reference(tmp_path):
    import hashlib
    path = tmp_path / "tiny.wav"
    data = b"abcdefghijk" * 7
    path.write_bytes(data)
    assert readiness.sha256_file(path, chunk_size=3) == (hashlib.sha256(data).hexdigest(), len(data))


def test_cli_json_and_exit_status(tmp_path):
    root, rows = corpus(tmp_path)
    source = manifest(tmp_path, rows)
    command = [sys.executable, str(SCRIPT), "--manifest", str(source), "--dataset-dir", str(root)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ready"]
    (root / rows[0]["path"]).unlink()
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)["error_counts"]["missing_audio"] == 1
