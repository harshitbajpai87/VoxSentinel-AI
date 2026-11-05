"""Health, readiness and model-metadata honesty."""
from __future__ import annotations

from tests.conftest import MockVoiceDetector, build_client  # noqa: F401


def test_health_reports_api_and_database_ready(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ok"
    assert body["backend_ready"] is True
    assert isinstance(body["version"], str) and body["version"]
    assert body["mode"] in {"baseline", "demo", "production"}


def test_health_reports_missing_detector_as_not_ready(client):
    """A running API must never imply the model is ready."""
    body = client.get("/health").json()
    assert body["model_ready"] is False
    assert body["speaker_model_ready"] is False


def test_health_does_not_crash_when_detector_missing(client):
    for _ in range(3):
        assert client.get("/health").status_code == 200


def test_health_reports_ready_when_detector_present(client_with_detector):
    body = client_with_detector.get("/health").json()
    assert body["model_ready"] is True
    assert body["backend_ready"] is True


def test_health_reports_degraded_when_database_unavailable(settings):
    """A database that stops answering is reported honestly, without crashing."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app(settings=settings)
    app.state.db.ping = lambda: False  # simulate the database going down
    with TestClient(app) as client:
        body = client.get("/health").json()
        assert body["backend_ready"] is False
        assert body["status"] == "degraded"
        assert body["model_ready"] is False


def test_models_endpoint_is_truthful_without_model(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    models = response.json()["models"]

    kinds = {m["kind"] for m in models}
    assert kinds == {"detection", "speaker"}
    for model in models:
        assert model["loaded"] is False
        assert "not installed" in model["name"].lower()
        # No invented versions or metrics anywhere in the payload.
        assert model["version"] == "n/a"
        assert "accuracy" not in model and "precision" not in model


def test_models_endpoint_reflects_test_double_when_installed(client_with_detector):
    models = client_with_detector.get("/api/v1/models").json()["models"]
    detection = next(m for m in models if m["kind"] == "detection")
    assert detection["loaded"] is True
    assert "TEST DOUBLE" in detection["name"]


def test_openapi_docs_available(client):
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    for path in ["/health", "/api/v1/models", "/api/v1/analyze", "/api/v1/analyses"]:
        assert path in schema["paths"]
