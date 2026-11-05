"""Candidate training pipeline and enhanced-artifact serving integration.

The dataset is a small generated fixture (speech-like vs noise-modulated
signals). It verifies PIPELINE MECHANICS only: artifact structure, loader
acceptance, honest serving through the same detector interface, per-source
reporting, and the refusal paths. Fixture metrics are NOT real
synthetic-speech detection performance and are never reported as such.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]


def make_fixture_dataset(root: Path, n_per_class: int = 8, sr: int = 16000) -> Path:
    """Two sources x two classes of 1-second synthetic-fixture WAVs."""
    rng = np.random.default_rng(7)
    t = np.arange(sr) / sr
    for source in ("sourceA", "sourceB"):
        (root / source / "real").mkdir(parents=True, exist_ok=True)
        (root / source / "fake").mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            # 'real': modulated harmonic stack; 'fake': flattened buzz.
            envelope = 0.5 * (1 + np.sin(2 * np.pi * 4 * t))
            real = (
                0.4 * np.sin(2 * np.pi * 220 * t)
                + 0.2 * np.sin(2 * np.pi * 880 * t)
                + 0.01 * rng.standard_normal(t.size)
            ) * envelope
            buzz = 0.35 * np.sign(np.sin(2 * np.pi * 210 * t)) + 0.02 * rng.standard_normal(t.size)
            for label, data in (("real", real), ("fake", buzz)):
                wav_path = root / source / label / f"clip_{i}.wav"
                sf_write(str(wav_path), data.astype(np.float32), sr)
    return root


def sf_write(path: str, data: np.ndarray, sr: int) -> None:
    import soundfile as sf

    sf.write(path, data, sr, subtype="FLOAT")


def write_training_manifest(path: Path, root: Path) -> Path:
    entries = []
    for source in ("sourceA", "sourceB"):
        for label in ("real", "fake"):
            for clip in sorted((root / source / label).glob("*.wav")):
                partition = "train" if clip.stem.endswith(("0", "1", "2", "3", "4")) else "validation"
                entries.append({
                    "path": str(clip.relative_to(root)),
                    "label": label,
                    "partition": partition,
                    "speaker_id": f"{source}-{clip.stem}",
                    "source": source,
                    "generator_id": f"{source}-gen" if label == "fake" else None,
                    "language": "fixture",
                })
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def trained_candidate(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("fixture_dataset")
    make_fixture_dataset(root)
    manifest = write_training_manifest(root / "manifest.jsonl", root)
    model_dir = tmp_path_factory.mktemp("candidate_models")
    result = subprocess.run(
        [sys.executable, str(BACKEND_DIR / "scripts" / "train_candidate.py"),
         "--manifest", str(manifest), "--dataset-dir", str(root),
         "--model-dir", str(model_dir), "--seed", "42"],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr
    report_path = model_dir.parent / "reports" / "training_report_candidate.json"
    return {
        "artifact": model_dir / "detector.joblib",
        "report": json.loads(report_path.read_text()),
        "dataset_dir": root,
        "manifest": manifest,
    }


def test_candidate_artifact_loads_and_serves(trained_candidate):
    from app.services.model_loader import load_detector_from_path

    detector = load_detector_from_path(trained_candidate["artifact"])
    assert detector is not None, "enhanced artifact must pass loader validation"
    assert detector.is_ready()
    assert detector._feature_kind == "mfcc_enhanced"  # noqa: SLF001
    assert detector.describe().mode == "candidate"
    described = detector.describe()
    assert described.loaded and "Enhanced" in described.description

    # Serving path: same shared preprocessing, finite score in [0, 1].
    sr = 16000
    t = np.arange(sr) / sr
    sample = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    result = detector.analyze(sample, sr, 1)
    assert 0.0 <= result.raw_score <= 1.0 and np.isfinite(result.raw_score)
    # Artifact is not calibrated: calibrated_probability must stay None.
    assert result.calibrated_probability is None


def test_baseline_artifact_still_loads():
    from app.services.model_loader import load_detector_from_path

    models_dir = BACKEND_DIR / "models"
    artifact = models_dir / "detector.joblib"
    if not artifact.is_file():  # artifact is git-ignored in fresh clones
        pytest.skip("baseline artifact not present")
    detector = load_detector_from_path(artifact)
    assert detector is not None
    # the deployed artifact may be the original baseline or a promoted candidate
    assert detector._feature_kind in {"mfcc_baseline", "mfcc_enhanced"}  # noqa: SLF001


def test_training_report_provenance(trained_candidate):
    report = trained_candidate["report"]
    assert report["model_mode"] == "candidate"
    assert report["features"]["include_delta_mfcc"] is True
    assert set(report["dataset"]["sources"]) == {"sourceA", "sourceB"}
    breakdown = report["validation_breakdown"]
    assert set(breakdown["by_source"]) == {"sourceA", "sourceB"}
    assert breakdown["by_generator"], "generator breakdown must be present"
    metrics = report["validation_metrics"]
    assert metrics["n"] > 0
    assert "note" not in metrics  # not calibrated -> no calibration caveat


def test_refuses_single_class_manifest(tmp_path):
    manifest = tmp_path / "single.jsonl"
    make_fixture_dataset(tmp_path / "ds", n_per_class=4)
    # Manifest with only 'real' entries.
    entries = []
    for clip in sorted((tmp_path / "ds" / "sourceA" / "real").glob("*.wav")):
        entries.append({
            "path": str(clip.relative_to(tmp_path / "ds")), "label": "real",
            "partition": "train", "speaker_id": f"s{clip.stem}", "source": "sourceA",
        })
    manifest.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    result = subprocess.run(
        [sys.executable, str(BACKEND_DIR / "scripts" / "train_candidate.py"),
         "--manifest", str(manifest), "--dataset-dir", str(tmp_path / "ds"),
         "--model-dir", str(tmp_path / "models"), "--seed", "42"],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 2
    output = result.stdout + result.stderr
    assert "refused" in output.lower() or "refusing" in output.lower(), output
