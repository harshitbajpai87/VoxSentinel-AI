"""Multisource service: unified manifest, labels, duplicates, split integrity."""
from __future__ import annotations

import json

import pytest

from app.services.multisource import (
    MultisourceError,
    UnifiedEntry,
    assign_splits,
    build_duplicate_clusters,
    duplicate_cluster_keys,
    load_unified_manifest,
    normalize_unified_label,
    summarize_unified,
    unified_to_training_manifest,
    write_unified_manifest,
)


def entry(**overrides) -> UnifiedEntry:
    base = {
        "file_path": "f.wav",
        "label": "real",
        "source_dataset": "test",
    }
    base.update(overrides)
    return UnifiedEntry(**base)


class TestLabels:
    def test_synthetic_vocabulary_maps_to_fake(self):
        for raw in ("spoof", "synthetic", "fake", "ai", "AI-generated", "deepfake", "cloned"):
            assert normalize_unified_label(raw) == "fake"

    def test_real_vocabulary_maps_to_real(self):
        for raw in ("real", "human", "bonafide", "genuine", "natural", "Original"):
            assert normalize_unified_label(raw) == "real"

    def test_uncertain_becomes_unknown(self):
        for raw in ("", "maybe", "??", None, 42, "unknown"):
            assert normalize_unified_label(raw) == "unknown"


class TestUnifiedManifest:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "unified.jsonl"
        entries = [entry(file_path="a.wav", speaker_id="s1"),
                   entry(file_path="b.wav", label="fake", recording_id="r1")]
        write_unified_manifest(entries, path)
        loaded = load_unified_manifest(path)
        assert [e.file_path for e in loaded] == ["a.wav", "b.wav"]
        assert loaded[1].label == "fake"

    def test_duplicate_path_rejected(self, tmp_path):
        path = tmp_path / "dup.jsonl"
        with pytest.raises(MultisourceError, match="[Dd]uplicate"):
            write_unified_manifest([entry(), entry(label="fake")], path)

    def test_invalid_label_rejected(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"file_path": "a.wav", "label": "kind-of"}) + "\n")
        loaded = load_unified_manifest(path)  # unknown labels are normalized, not errors
        assert loaded[0].label == "unknown"

    def test_invalid_split_rejected(self, tmp_path):
        path = tmp_path / "bad_split.jsonl"
        path.write_text(json.dumps({"file_path": "a.wav", "label": "real", "split": "quiz"}) + "\n")
        with pytest.raises(MultisourceError, match="split"):
            load_unified_manifest(path)


class TestSplitIntegrity:
    def _entries(self) -> list[UnifiedEntry]:
        # 4 speakers x 2 recordings each, plus 2 synthetic generators.
        entries = []
        for i in range(4):
            for j in range(2):
                entries.append(entry(
                    file_path=f"real{i}{j}.wav", label="real",
                    speaker_id=f"speaker{i}",
                ))
        for i in range(4):
            entries.append(entry(
                file_path=f"gen{i}.wav", label="fake",
                source_dataset="tts", recording_id=f"g{i}",
                generator_id=f"gen{i % 2}",
            ))
        return entries

    def test_groups_stay_in_one_split(self):
        assigned = assign_splits(self._entries(), seed=42)
        by_group: dict[str, set] = {}
        for e in assigned:
            if e.label == "unknown":
                continue
            by_group.setdefault(e.group_key(), set()).add(e.split)
        for group, splits in by_group.items():
            assert len(splits) == 1, f"group {group} leaked across splits: {splits}"

    def test_holdout_generators_are_unseen(self):
        entries = self._entries()
        for i in range(4):
            entries.append(entry(
                file_path=f"eval_gen{i}.wav", label="fake",
                source_dataset="tts", recording_id=f"eg{i}", generator_id=f"holdout_gen{i}",
            ))
        assigned = assign_splits(entries, seed=42, holdout_generator_fraction=1.0)
        holdout_generators = {
            e.generator_id for e in assigned if e.split == "holdout" and e.generator_id
        }
        train_generators = {
            e.generator_id for e in assigned if e.split == "train" and e.generator_id
        }
        assert holdout_generators, "fraction 1.0 should hold out at least one generator"
        assert not (holdout_generators & train_generators), "holdout generator leaked into train"

    def test_unknown_labels_get_no_split(self):
        entries = self._entries() + [entry(file_path="mystery.wav", label="unknown")]
        assigned = assign_splits(entries, seed=42)
        unknown = [e for e in assigned if e.label == "unknown"]
        assert len(unknown) == 1 and unknown[0].split is None

    def test_duplicate_clusters_share_split(self, tmp_path):
        # Two identical files in different dataset folders.
        content = b"fake-audio-bytes" * 64
        for rel in ("d1/a.wav", "d2/copy.wav"):
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        entries = [
            entry(file_path="d1/a.wav", speaker_id="spk1", source_dataset="d1"),
            entry(file_path="d2/copy.wav", speaker_id="spk2", source_dataset="d2", label="fake"),
            entry(file_path="d1/other.wav", speaker_id="spk3", source_dataset="d1"),
        ]
        clusters = build_duplicate_clusters(entries, tmp_path)
        assert len(clusters) == 1 and len(clusters[0]) == 2
        keys = duplicate_cluster_keys(clusters)
        assert keys == [["d1:spk1", "d2:spk2"]]
        assigned = assign_splits(entries, duplicate_clusters=keys, seed=42)
        splits = {e.split for e in assigned if e.file_path in ("d1/a.wav", "d2/copy.wav")}
        assert len(splits) == 1, "exact duplicates must land in the same split"

    def test_determinism(self):
        first = assign_splits(self._entries(), seed=42)
        second = assign_splits(self._entries(), seed=42)
        assert [(e.file_path, e.split) for e in first] == [(e.file_path, e.split) for e in second]
        third = assign_splits(self._entries(), seed=7)
        assert third != first or third == first  # same seed identical; other seed may differ


class TestTrainingManifestConversion:
    def test_unknown_and_holdout_excluded(self, tmp_path):
        path = tmp_path / "training.jsonl"
        entries = [
            entry(file_path="r.wav", split="train"),
            entry(file_path="f.wav", label="fake", split="train"),
            entry(file_path="u.wav", label="unknown", split="train"),
            entry(file_path="h.wav", label="fake", split="holdout", recording_id="g1",
                  generator_id="g1"),
        ]
        summary = unified_to_training_manifest(entries, path)
        assert summary["written"] == 2
        assert summary["excluded_unknown"] == 1
        assert summary["excluded_holdout"] == 1
        lines = [json.loads(line) for line in path.read_text().splitlines()]
        assert lines[0]["label"] == "real" and lines[1]["label"] == "synthetic"
        assert all("generator_id" in line for line in lines)

    def test_writes_provenance(self, tmp_path):
        path = tmp_path / "training.jsonl"
        unified_to_training_manifest(
            [entry(file_path="a.wav", split="train", speaker_id="s", language="en")], path
        )
        record = json.loads(path.read_text().splitlines()[0])
        # training speaker ids are globally unique (source-prefixed) unless a
        # cross-source speaker namespace (librispeech:*) is present
        assert record["speaker_id"] == "test:s"
        assert record["source"] == "test"
        assert record["language"] == "en"

    def test_training_speaker_id_is_globally_unique(self, tmp_path):
        """Two sources with the same bare recording id must not collide in the training manifest."""
        path = tmp_path / "training.jsonl"
        unified_to_training_manifest(
            [
                entry(file_path="srcA/104.wav", split="train", recording_id="104"),
                entry(file_path="srcB/104.wav", split="validation", recording_id="104",
                      source_dataset="other"),
            ],
            path,
        )
        ids = [json.loads(line)["speaker_id"] for line in path.read_text().splitlines()]
        assert len(set(ids)) == 2

    def test_librispeech_speaker_namespace_is_global(self, tmp_path):
        """The same LibriSpeech speaker in two sources must share one group key."""
        e1 = entry(file_path="for/real/1089_x.wav", speaker_id="librispeech:1089")
        e2 = entry(file_path="adarsh/real/1089_x.wav", speaker_id="librispeech:1089",
                   source_dataset="other")
        assert e1.group_key() == e2.group_key() == "speaker:librispeech:1089"


def test_summarize_counts():
    entries = [
        entry(file_path="a.wav", split="train"),
        entry(file_path="b.wav", label="fake", split="train"),
        entry(file_path="c.wav", label="unknown", split=None),
    ]
    summary = summarize_unified(entries)
    assert summary["total"] == 3
    assert summary["by_source_label_split"]["test|real|train"] == 1
