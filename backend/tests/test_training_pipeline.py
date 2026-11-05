"""Training pipeline: validation failures and the smoke-training happy path.

The dataset here is the generated smoke fixture (tones vs noise). It verifies
PIPELINE MECHANICS only; its metrics are explicitly NOT synthetic-speech
detection performance.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.services.manifest import ManifestError, load_manifest


def write_manifest(path: Path, entries: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return path


def test_manifest_rejects_unknown_labels(tmp_path):
    manifest = write_manifest(
        tmp_path / "m.jsonl",
        [{"path": "a.wav", "label": "kind-of-real", "partition": "train", "speaker_id": "s1", "source": "t"}],
    )
    with pytest.raises(ManifestError, match="[Uu]nknown label"):
        load_manifest(manifest)


def test_manifest_rejects_missing_labels_and_partitions(tmp_path):
    manifest = write_manifest(
        tmp_path / "m.jsonl",
        [
            {"path": "a.wav", "partition": "train", "speaker_id": "s1", "source": "t"},
            {"path": "b.wav", "label": "real", "partition": "quiz", "speaker_id": "s1", "source": "t"},
        ],
    )
    with pytest.raises(ManifestError):
        load_manifest(manifest)


def test_manifest_rejects_duplicates_and_empty_files(tmp_path):
    duplicate = write_manifest(
        tmp_path / "dup.jsonl",
        [
            {"path": "a.wav", "label": "real", "partition": "train", "speaker_id": "s1", "source": "t"},
            {"path": "a.wav", "label": "synthetic", "partition": "train", "speaker_id": "s2", "source": "t"},
        ],
    )
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(duplicate)

    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "does-not-exist.jsonl")

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ManifestError, match="no entries"):
        load_manifest(empty)


def test_speaker_leakage_is_refused(tmp_path):
    from app.services.manifest import ManifestEntry, assert_no_speaker_leakage

    def entry(speaker, label, path, partition):
        return ManifestEntry(
            path=path, label=label, partition=partition, speaker_id=speaker, source="t"
        )

    train = [entry("LA_0079", "real", "a.wav", "train")]
    validation = [entry("LA_0079", "synthetic", "b.wav", "validation")]
    with pytest.raises(ManifestError, match="leakage"):
        assert_no_speaker_leakage(train, validation)

    # Disjoint speakers pass.
    assert_no_speaker_leakage(train, [entry("LA_0080", "synthetic", "b.wav", "validation")])


def test_fixture_training_smoke(trained_model_dir, fixture_dataset):
    """The real CLI trains, saves artifact + report. SMOKE FIXTURE results only."""
    artifact = trained_model_dir / "detector.joblib"
    # The script writes reports next to --model-dir's parent; re-locate robustly:
    candidates = list(trained_model_dir.parent.parent.rglob("training_report.json"))
    assert artifact.is_file()
    assert candidates, "training report missing"

    report = json.loads(candidates[0].read_text())
    assert report["report_type"] == "training_report"
    assert report["split"]["seed"] == 42
    assert report["split"]["train_n"] == 12
    assert report["split"]["validation_n"] == 12
    assert "scikit_learn" in report["environment"]
    assert report["validation_metrics"]["n"] == 12
    # Both fixture classes must be present in the report's dataset counts.
    counts = report["dataset"]["summary"]["counts"]
    assert counts["train"]["real"] == 6
    assert counts["train"]["synthetic"] == 6


def test_fixture_evaluation_smoke(evaluation_report):
    """The real CLI produces honest held-out metrics. SMOKE FIXTURE, not voice data."""
    assert evaluation_report["evaluated"] is True
    metrics = evaluation_report["metrics"]
    assert metrics["positive_class"] == "synthetic"
    assert metrics["confusion_matrix"]["tn"] + metrics["confusion_matrix"]["fp"] == 6
    assert metrics["confusion_matrix"]["tp"] + metrics["confusion_matrix"]["fn"] == 6
    assert evaluation_report["split"]["n"] == 12
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["eer"] <= 1.0
    assert "Held-out partition" in evaluation_report["split"]["note"]


def test_report_states_dataset_provenance(evaluation_report):
    assert evaluation_report["dataset"]["sources"] == ["fixture_smoke"]
    assert evaluation_report["dataset"]["class_distribution"] == {"real": 6, "synthetic": 6}
