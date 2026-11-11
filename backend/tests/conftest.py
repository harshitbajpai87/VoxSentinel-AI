"""Shared test fixtures.

`MockVoiceDetector` is a TEST DOUBLE: it returns canned values so the API
pipeline can be exercised end to end. It is not a model, it is never installed
in the application, and its outputs are never presented as real detection
performance.
"""
from __future__ import annotations

import io
import wave
from dataclasses import replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.detector import (
    BaseVoiceDetector,
    DetectorResult,
    ModelDescription,
    ModelNotReadyError,
)


def make_wav(
    duration_s: float = 3.0,
    sample_rate: int = 8000,
    channels: int = 1,
    amplitude: float = 0.5,
    freq: float = 440.0,
) -> bytes:
    """Tiny synthetic WAV fixture generated in code (no personal audio needed)."""
    frames = int(duration_s * sample_rate)
    t = np.arange(frames, dtype=np.float64) / float(sample_rate)
    mono = np.sin(2 * np.pi * freq * t) * amplitude
    pcm = np.clip(mono * 32767.0, -32768, 32767).astype("<i2")
    if channels > 1:
        pcm = np.repeat(pcm.reshape(-1, 1), channels, axis=1).reshape(-1)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def make_silent_wav(duration_s: float = 3.0, sample_rate: int = 8000) -> bytes:
    return make_wav(duration_s=duration_s, sample_rate=sample_rate, amplitude=0.0)


def make_corrupt_wav() -> bytes:
    """Valid RIFF/WAVE header with an unusable payload: decoding must fail cleanly."""
    payload = b"\x00\x11\x22\x33" * 64
    return b"RIFF" + (36 + len(payload)).to_bytes(4, "little") + b"WAVEfmt " + b"\xff\xff\xff\xff" + payload


class MockVoiceDetector(BaseVoiceDetector):
    """TEST DOUBLE — canned scores, not a real detector.

    Used only to verify that the API pipeline (validation -> detector -> risk
    policy -> persistence) works end to end with a genuine detector installed.
    """

    def __init__(self, result: DetectorResult | None = None, broken: bool = False) -> None:
        self._result = result or DetectorResult(
            raw_score=0.42,
            raw_score_definition="TEST DOUBLE score, not a real model prediction",
            calibrated_probability=None,
            uncertainty=None,
            uncertainty_method=None,
            model_name="mock-detector",
            model_version="0.0-test",
            model_mode="baseline",
        )
        self._broken = broken
        self.calls = 0

    def is_ready(self) -> bool:
        # A 'broken' double models a detector that claims readiness but fails
        # during inference, so the API's 500 path can be exercised.
        return True

    def analyze(self, samples: np.ndarray, sample_rate: int, channels: int) -> DetectorResult:
        self.calls += 1
        if self._broken:
            raise RuntimeError("simulated detector crash (test double)")
        if samples.size == 0:
            raise ModelNotReadyError("no samples")
        return self._result

    def describe(self) -> ModelDescription:
        return ModelDescription(
            id="mock_detector",
            name="MockVoiceDetector (TEST DOUBLE)",
            version="0.0-test",
            mode="baseline",
            kind="detection",
            description="Test double used by the automated test suite. Never installed in the app.",
            loaded=not self._broken,
        )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        db_url=f"sqlite:///{tmp_path}/test.db",
        detector_model_path=str(tmp_path / "missing-detector.joblib"),
        speaker_model_path=str(tmp_path / "missing-speaker.joblib"),
        # Isolate from any REAL models installed under backend/models/ so the
        # honest "no model installed" tests are reproducible on every machine.
        speaker_model_dir=str(tmp_path / "missing-speaker-dir"),
        # The torch-free fallbacks are OFF here so this fixture keeps describing
        # a build with nothing available at all. Their behaviour is covered by
        # tests/test_fallbacks.py with the flags enabled.
        forensics_enabled=False,
        mfcc_speaker_enabled=False,
    )


@pytest.fixture
def client(settings) -> TestClient:
    """App with NO detector installed — the honest Phase 4 state."""
    app = create_app(settings=settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def mock_detector() -> MockVoiceDetector:
    return MockVoiceDetector()


@pytest.fixture
def client_with_detector(settings, mock_detector) -> TestClient:
    """App with a clearly-labelled TEST DOUBLE detector installed."""
    app = create_app(settings=settings, detector=mock_detector)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_with_broken_detector(settings) -> TestClient:
    """A detector that fails mid-inference: exercises the safe 500 path.

    raise_server_exceptions=False lets the test see the JSON error response
    the client would actually receive."""
    app = create_app(settings=settings, detector=MockVoiceDetector(broken=True))
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def post_analyze(client: TestClient, data: bytes, name: str = "clip.wav", consent: str = "true"):
    return client.post(
        "/api/v1/analyze",
        files={"file": (name, data, "audio/wav")},
        data={"consent": consent},
    )


@pytest.fixture
def small_limit_settings(settings) -> Settings:
    """Upload cap small enough to exercise 413 without big fixtures."""
    return replace(settings, max_upload_bytes=50_000)


@pytest.fixture
def short_limit_settings(settings) -> Settings:
    """Two-second duration ceiling for the excessive-duration case."""
    return replace(settings, max_duration_s=2.0)


def build_client(settings: Settings, detector: BaseVoiceDetector | None = None) -> TestClient:
    app = create_app(settings=settings, detector=detector)
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Phase 5: shared, clearly-labelled SMOKE fixtures for the ML pipeline.
# The two classes are generated tones and noise — NOT speech. They exist to
# exercise pipeline mechanics; their metrics are never real-world detection
# performance. See scripts/make_fixture_dataset.py.
# ---------------------------------------------------------------------------

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

BACKEND_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def fixture_dataset(tmp_path_factory) -> Path:
    """A tiny generated smoke dataset built through the real CLI generator."""
    out_dir = tmp_path_factory.mktemp("fixture_dataset")
    result = subprocess.run(
        [
            sys.executable,
            str(BACKEND_ROOT / "scripts" / "make_fixture_dataset.py"),
            "--out-dir",
            str(out_dir),
            "--per-class-per-partition",
            "6",
        ],
        capture_output=True,
        text=True,
        cwd=BACKEND_ROOT,
    )
    assert result.returncode == 0, result.stderr
    return out_dir


@pytest.fixture(scope="session")
def trained_model_dir(fixture_dataset, tmp_path_factory) -> Path:
    """Train the real training script once on the smoke fixture (TEST ONLY).

    The artifact classifies tones vs noise. It proves the training, loading
    and serving pipeline works; it is NOT a voice-cloning detector.
    """
    model_dir = tmp_path_factory.mktemp("models")
    result = subprocess.run(
        [
            sys.executable,
            str(BACKEND_ROOT / "scripts" / "train_detector.py"),
            "--manifest",
            str(fixture_dataset / "manifest.jsonl"),
            "--model-dir",
            str(model_dir),
            "--seed",
            "42",
        ],
        capture_output=True,
        text=True,
        cwd=BACKEND_ROOT,
    )
    assert result.returncode == 0, result.stderr
    return model_dir


@pytest.fixture(scope="session")
def evaluation_report(trained_model_dir, fixture_dataset, tmp_path_factory) -> dict:
    """Run the real evaluation CLI once on the held-out fixture partition."""
    report_dir = tmp_path_factory.mktemp("reports")
    result = subprocess.run(
        [
            sys.executable,
            str(BACKEND_ROOT / "scripts" / "evaluate_detector.py"),
            "--artifact",
            str(trained_model_dir / "detector.joblib"),
            "--manifest",
            str(fixture_dataset / "manifest.jsonl"),
            "--partition",
            "eval",
            "--report-dir",
            str(report_dir),
        ],
        capture_output=True,
        text=True,
        cwd=BACKEND_ROOT,
    )
    assert result.returncode == 0, result.stderr
    return json.loads((report_dir / "evaluation_eval.json").read_text())
