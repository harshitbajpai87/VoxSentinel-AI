"""Speaker verification and evaluation endpoints (Phase 4 status)."""
from __future__ import annotations

from tests.conftest import make_wav


def _post_verify(client, reference, test, consent="true", authorized="true"):
    return client.post(
        "/api/v1/speaker-verify",
        files=[
            ("reference", ("ref.wav", reference, "audio/wav")),
            ("test", ("test.wav", test, "audio/wav")),
        ],
        data={"consent": consent, "reference_authorized": authorized},
    )


def test_missing_consent_is_rejected(client):
    response = _post_verify(client, make_wav(2.0), make_wav(2.0), consent="false")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_consent"


def test_missing_authorization_is_rejected(client):
    response = _post_verify(client, make_wav(2.0), make_wav(2.0), authorized="false")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_authorization"


def test_valid_files_without_speaker_model_return_explicit_unavailable(client):
    response = _post_verify(client, make_wav(3.0), make_wav(3.0))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unavailable"
    assert body["similarity"] is None
    assert body["threshold"] is None
    assert body["embedding_model"] is None
    assert any("no similarity score" in lim.lower() for lim in body["limitations"])


def test_no_fabricated_similarity_without_model(client):
    """The unavailable response must not smuggle a number in anywhere."""
    body = _post_verify(client, make_wav(3.0), make_wav(3.0)).json()
    assert body["similarity"] is None
    assert body["outcome"] is None


def test_invalid_reference_file_is_rejected_before_model_check(client):
    response = _post_verify(client, b"not audio data" * 20, make_wav(3.0))
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_evaluation_metrics_report_not_evaluated(settings, mock_detector, tmp_path):
    """With an isolated (empty) report dir the metrics endpoint is honest."""
    from dataclasses import replace as dc_replace

    from fastapi.testclient import TestClient

    from app.main import create_app

    settings = dc_replace(settings, report_dir=str(tmp_path / "reports"))
    with TestClient(create_app(settings=settings, detector=mock_detector)) as c:
        body = c.get("/api/v1/evaluation/metrics").json()
        assert body == {"evaluated": False, "metrics": None, "evaluated_at": None}


def test_evaluation_summary_explains_next_step(settings, mock_detector):
    """With an isolated (empty) report dir the summary is honestly unevaluated."""
    from dataclasses import replace as dc_replace

    from fastapi.testclient import TestClient

    from app.main import create_app

    settings = dc_replace(settings, report_dir=str(settings.db_url.replace("sqlite:///", "") + "_reports"))
    with TestClient(create_app(settings=settings, detector=mock_detector)) as c:
        body = c.get("/api/v1/evaluation").json()
        assert body["evaluated"] is False
        assert "evaluation/run" in body["message"]
