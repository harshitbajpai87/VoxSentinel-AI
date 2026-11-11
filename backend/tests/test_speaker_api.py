"""Module 3 tests: speaker-verification pipeline.

The API-level tests use StubEmbedder — a clearly-labelled TEST DOUBLE with
deterministic vectors, so route behaviour is verified without pretending the
stub is a model. The real ECAPA model gets one honest integration test that
is SKIPPED (and reported as skipped) whenever the model files are absent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.core.config import Settings
from app.main import create_app
from app.services.speaker import EcapaSpeakerVerifier
from app.services.speaker_model import SimilarityThresholds
from tests.conftest import make_wav

BACKEND_ROOT = Path(__file__).resolve().parents[1]


class StubEmbedder:
    """TEST DOUBLE — returns fixed vectors, NOT a real model.

    By default the stub maps content to a voice: high zero-crossing rate
    (a 400 Hz tone) -> voice 'a', low rate (a 50 Hz tone) -> voice 'b'.
    With ``preset_vectors`` it ignores content and returns the vectors in
    order — used only for direct verifier-math tests.
    """

    def __init__(self, preset_vectors: list[np.ndarray] | None = None) -> None:
        self.calls = 0
        self._preset = preset_vectors

    def is_ready(self) -> bool:
        return True

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        self.calls += 1
        if self._preset is not None:
            return self._preset[(self.calls - 1) % len(self._preset)]
        crossings = int(np.sum(np.diff(np.signbit(samples))))
        return np.array([1.0, 0.0, 0.0, 0.0]) if crossings > samples.size // 20 \
            else np.array([0.0, 1.0, 0.0, 0.0])


@pytest.fixture
def stub_client(settings):
    verifier = EcapaSpeakerVerifier(
        StubEmbedder(),
        _thresholds_for_tests(settings, threshold=0.5),
    )
    # TEST DOUBLE threshold for route behavior; production artifacts require
    # >=100 non-test calibration pairs and cannot use this fixture metadata.
    verifier.decision_threshold = lambda: 0.5
    app = create_app(settings=settings, speaker=verifier)
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        yield client


def _thresholds_for_tests(
    settings: Settings, threshold: float, placeholder: bool = False
) -> SimilarityThresholds:
    import json

    path = Path(settings.speaker_thresholds_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "format": "voxsentinel-speaker-thresholds",
        "artifact_version": 1,
        "metric": "cosine_similarity",
        "eer_threshold": threshold,
        "calibrated_on": (
            "TEST FIXTURE (not real calibration data)"
            if placeholder
            else "unit-test synthetic embedding pairs"
        ),
        "calibration_pairs": 0 if placeholder else 2,
        "created_at": "1970-01-01T00:00:00+00:00",
    }))
    return SimilarityThresholds(str(path))


def _post_verify(client, ref: bytes, test: bytes, consent="true", authorized="true"):
    return client.post(
        "/api/v1/speaker-verify",
        files={"reference": ("ref.wav", ref, "audio/wav"),
               "test": ("test.wav", test, "audio/wav")},
        data={"consent": consent, "reference_authorized": authorized},
    )


def _tone_wav(freq: float, seconds: float = 3.0, amplitude: float = 0.5) -> bytes:
    import io
    import wave

    sr = 8000
    t = np.arange(int(sr * seconds)) / sr
    pcm = (amplitude * np.sin(2 * np.pi * freq * t) * 32767).astype("<f2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _two_voices():
    """Two loud tones (VAD-safe) the stub maps to different voices."""
    return _tone_wav(400.0), _tone_wav(50.0)


def test_same_stub_voice_reports_high_similarity(stub_client):
    response = _post_verify(stub_client, _tone_wav(400.0), _tone_wav(400.0))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["similarity"] == pytest.approx(1.0, abs=1e-3)
    assert body["outcome"] == "likely_same_speaker"
    assert "cosine" in " ".join(body["limitations"]).lower()


def test_different_stub_voices_report_low_similarity(stub_client):
    ref, test = _two_voices()
    response = _post_verify(stub_client, ref, test)
    assert response.status_code == 200
    body = response.json()
    assert body["similarity"] == pytest.approx(0.0, abs=1e-3)
    assert body["outcome"] == "likely_different_speaker"


def test_borderline_similarity_is_inconclusive(settings):
    # two stub vectors at 45 degrees -> cosine ~0.707, exactly the threshold
    # fixture value, so the decision must fall inside the ±0.03 uncertainty
    # band and be reported honestly as inconclusive.
    a = np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2)
    b = np.array([1.0, 0.0, 0.0, 0.0])
    verifier = EcapaSpeakerVerifier(
        StubEmbedder(preset_vectors=[a, b]),
        _thresholds_for_tests(settings, 0.7071),
    )
    verifier.decision_threshold = lambda: 0.7071
    result = verifier.compare(np.ones(16000, dtype=np.float32) * 0.5, 16000,
                              np.ones(16000, dtype=np.float32) * 0.5, 16000)
    similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert abs(similarity - 0.7071) < 0.03
    assert result.outcome == "inconclusive"
    assert result.similarity == pytest.approx(similarity, abs=1e-6)


def test_placeholder_threshold_reports_similarity_without_decision(settings):
    verifier = EcapaSpeakerVerifier(
        StubEmbedder(), _thresholds_for_tests(settings, 0.5, placeholder=True)
    )
    result = verifier.compare(
        np.ones(16000, dtype=np.float32) * 0.5,
        16000,
        np.ones(16000, dtype=np.float32) * 0.5,
        16000,
    )
    assert result.status == "completed"
    assert result.similarity == pytest.approx(1.0)
    assert result.threshold is None
    assert result.outcome is None
    assert any("non-placeholder calibration" in warning for warning in result.warnings)


def test_missing_consent_is_rejected(stub_client):
    ref, test = _two_voices()
    response = _post_verify(stub_client, ref, test, consent="false")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_consent"


def test_missing_authorization_is_rejected(stub_client):
    ref, test = _two_voices()
    response = _post_verify(stub_client, ref, test, authorized="false")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_authorization"


def test_corrupt_reference_is_rejected(stub_client):
    response = _post_verify(stub_client, b"garbage" * 500, make_wav(3.0))
    assert response.status_code in (415, 422)


def test_silent_recording_is_rejected(stub_client):
    response = _post_verify(stub_client, make_wav(3.0, amplitude=0.0), make_wav(3.0))
    assert response.status_code == 422


def test_unavailable_model_reports_honestly(settings):
    """A server WITHOUT model files must report unavailable, not invent a score."""
    import dataclasses

    from fastapi.testclient import TestClient

    bare = dataclasses.replace(
        settings, speaker_model_dir=str(settings.speaker_model_dir) + "_absent"
    )
    app = create_app(settings=bare)
    with TestClient(app) as client:
        ref, test = _two_voices()
        response = _post_verify(client, ref, test)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "unavailable"
        assert body["similarity"] is None
        assert client.get("/health").json()["speaker_model_ready"] is False


def test_real_ecapa_model_if_present(settings, tmp_path):
    """Honest integration test with the REAL model — skipped when not installed."""
    from app.services.speaker_model import EcapaSpeakerEmbedder

    real_dir = BACKEND_ROOT / "models" / "speaker_ecapa"
    model_dir = (
        real_dir if (real_dir / "hyperparams.yaml").is_file() else settings.speaker_model_dir
    )
    embedder = EcapaSpeakerEmbedder(model_dir)
    if not embedder.is_ready():
        pytest.skip("ECAPA model files not present (see docs/DATASET_GUIDE.md)")

    np.random.default_rng(0)
    t = np.arange(16000 * 2) / 16000
    speech = (0.4 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    emb_a = embedder.embed(speech, 16000)
    emb_b = embedder.embed(speech * 0.9, 16000)
    assert emb_a.shape == (192,)
    similarity = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))
    assert np.isfinite(similarity) and -1.0 <= similarity <= 1.0
    assert similarity > 0.99  # same signal must be near-identical
