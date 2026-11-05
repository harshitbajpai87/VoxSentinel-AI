"""Regression tests: evaluation reports must describe the score they used.

The production artifact (Candidate B, v0.3.202610241603) is sigmoid-calibrated.
evaluate_detector.py applies its 0.5 threshold to the SAME classifier output
the production analyze() path serves, so the report's threshold definition
must say "calibrated probability" for calibrated artifacts and "uncalibrated
classifier score" otherwise. These tests pin both the wording and the
score-path equivalence.
"""
import json
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

from tests.test_train_candidate import make_fixture_dataset  # noqa: E402


def _write_manifest_with_eval(path: Path, root: Path) -> Path:
    """Fixture manifest with train/validation/eval partitions (eval needed by the evaluator)."""
    entries = []
    for source in ("sourceA", "sourceB"):
        for label in ("real", "fake"):
            for clip in sorted((root / source / label).glob("*.wav")):
                idx = int(clip.stem.split("_")[-1])
                if idx < 5:
                    partition = "train"
                elif idx < 10:
                    partition = "validation"
                else:
                    partition = "eval"
                entries.append({
                    "path": str(clip.relative_to(root)),
                    "label": label,
                    "partition": partition,
                    "speaker_id": f"{source}-{clip.stem}",
                    "source": source,
                    "generator_id": f"{source}-gen" if label == "fake" else None,
                    "language": "fixture",
                })
    path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    return path


def _train(root: Path, model_dir: Path, calibrate: bool) -> Path:
    manifest = _write_manifest_with_eval(root / "manifest.jsonl", root)
    cmd = [
        sys.executable, str(BACKEND_DIR / "scripts" / "train_candidate.py"),
        "--manifest", str(manifest), "--dataset-dir", str(root),
        "--model-dir", str(model_dir), "--seed", "42",
    ]
    if calibrate:
        cmd.append("--calibrate")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    return manifest


def _evaluate(artifact: Path, manifest: Path, dataset_dir: Path, report_dir: Path) -> dict:
    result = subprocess.run(
        [sys.executable, str(BACKEND_DIR / "scripts" / "evaluate_detector.py"),
         "--artifact", str(artifact), "--manifest", str(manifest),
         "--dataset-dir", str(dataset_dir), "--partition", "eval",
         "--report-dir", str(report_dir)],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr
    return json.loads((report_dir / "evaluation_eval.json").read_text())


def test_report_wording_matches_calibration(tmp_path):
    """Calibrated artifacts get calibrated-probability wording; uncalibrated don't."""
    from app.services.model_loader import load_detector_from_path

    make_fixture_dataset(tmp_path / "ds", n_per_class=12)

    manifest_c = _train(tmp_path / "ds", tmp_path / "models_cal", calibrate=True)
    report_c = _evaluate(
        tmp_path / "models_cal" / "detector.joblib", manifest_c,
        tmp_path / "ds", tmp_path / "reports_cal",
    )
    detector_c = load_detector_from_path(tmp_path / "models_cal" / "detector.joblib")
    assert detector_c.calibrated is True
    threshold_c = report_c["metrics"]["definitions"]["threshold"]
    assert "calibrated probability" in threshold_c, threshold_c
    assert "uncalibrated" not in threshold_c, threshold_c

    manifest_u = _train(tmp_path / "ds", tmp_path / "models_uncal", calibrate=False)
    report_u = _evaluate(
        tmp_path / "models_uncal" / "detector.joblib", manifest_u,
        tmp_path / "ds", tmp_path / "reports_uncal",
    )
    detector_u = load_detector_from_path(tmp_path / "models_uncal" / "detector.joblib")
    assert detector_u.calibrated is False
    threshold_u = report_u["metrics"]["definitions"]["threshold"]
    assert "uncalibrated" in threshold_u, threshold_u
    assert "calibrated probability" not in threshold_u, threshold_u


def test_evaluator_score_equals_production_path(tmp_path):
    """The evaluation score must be the same number the production analyze() serves."""
    import soundfile as sf

    from app.services.model_loader import load_detector_from_path

    make_fixture_dataset(tmp_path / "ds", n_per_class=12)
    manifest = _train(tmp_path / "ds", tmp_path / "models_cal", calibrate=True)
    _evaluate(
        tmp_path / "models_cal" / "detector.joblib", manifest,
        tmp_path / "ds", tmp_path / "reports_cal",
    )

    detector = load_detector_from_path(tmp_path / "models_cal" / "detector.joblib")
    clips = sorted((tmp_path / "ds" / "sourceA" / "real").glob("*.wav"))
    data, sample_rate = sf.read(clips[0], dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data.reshape(-1)
    result = detector.analyze(mono.astype("float32"), sample_rate, 1)
    assert result.calibrated_probability is not None  # calibrated artifact

    # score_features is the shared production scoring path used by analyze();
    # the evaluator threshold applies to the identical classifier output.
    features = detector.feature_config_asdict()  # sanity: artifact carries config
    assert features is not None
    import numpy as np

    from app.services.features_enhanced import extract_enhanced_features
    x = extract_enhanced_features(
        np.asarray(mono, dtype="float32"), sample_rate
    )
    prod_score = detector.score_features(x)
    assert abs(prod_score - result.calibrated_probability) < 1e-9
