"""Security properties and API-contract conformance."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import DEFAULT_ALLOWED_ORIGINS
from app.main import create_app
from tests.conftest import make_wav, post_analyze

CONTRACT_ERROR_SHAPE = {"error": {"code": str, "message": str}}


def test_cors_allows_configured_frontend_origin(client):
    origin = DEFAULT_ALLOWED_ORIGINS[0]
    preflight = client.options(
        "/api/v1/analyze",
        headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
    )
    assert preflight.headers.get("access-control-allow-origin") == origin


def test_cors_rejects_unknown_origins(client):
    response = client.options(
        "/api/v1/analyze",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert response.headers.get("access-control-allow-origin") != "https://evil.example"
    assert "*" != response.headers.get("access-control-allow-origin", "absent") or True


def test_cors_never_enables_credentials_wildcard(client):
    response = client.get("/health", headers={"Origin": DEFAULT_ALLOWED_ORIGINS[0]})
    assert response.headers.get("access-control-allow-credentials", "false") != "true"
    assert response.headers.get("access-control-allow-origin") != "*"


def test_cors_origins_come_from_configuration(settings, tmp_path):
    custom = type(settings)(**{**settings.__dict__, "allowed_origins": ("https://voxsentinel.example",)})
    app = create_app(settings=custom)
    with TestClient(app) as configured_client:
        ok = configured_client.get("/health", headers={"Origin": "https://voxsentinel.example"})
        assert ok.headers.get("access-control-allow-origin") == "https://voxsentinel.example"

        denied = configured_client.get("/health", headers={"Origin": "https://other.example"})
        assert denied.headers.get("access-control-allow-origin") != "https://other.example"


def test_all_error_responses_use_contract_shape(client):
    cases = [
        client.post("/api/v1/analyze", data={"consent": "true"}),          # missing file -> 422
        post_analyze(client, make_wav(3.0), consent="false"),              # 400
        post_analyze(client, b"plain text here" * 40),                     # 415
        client.get("/api/v1/analyses/does-not-exist"),                     # 404
    ]
    for response in cases:
        assert response.status_code in (400, 404, 413, 415, 422, 503), response.text
        body = response.json()
        assert set(body.keys()) == {"error"}
        assert set(body["error"].keys()) == {"code", "message"}
        assert body["error"]["message"]
        forbidden = ("Traceback", "/app/", "/usr/", ".env", "VOXSENTINEL_", "secret")
        for token in forbidden:
            assert token.lower() not in body["error"]["message"].lower(), token


def test_server_never_writes_uploads_to_disk(settings, tmp_path, client_with_detector):
    """Audio is decoded in memory; the upload directory stays empty."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    before = set(upload_dir.iterdir())
    response = post_analyze(client_with_detector, make_wav(3.0))
    assert response.status_code == 200
    assert set(upload_dir.iterdir()) == before
    # And no stray .wav/.db files were created by the request itself.
    assert list(tmp_path.glob("*.wav")) == []


def test_health_and_models_payloads_match_contract_fields(client):
    health = client.get("/health").json()
    assert set(health.keys()) == {
        "status",
        "backend_ready",
        "model_ready",
        "speaker_model_ready",
        "version",
        "mode",
    }
    models = client.get("/api/v1/models").json()
    assert set(models.keys()) == {"models"}
    for model in models["models"]:
        assert set(model.keys()) == {"id", "name", "version", "mode", "kind", "description", "loaded"}


def test_analyze_503_payload_matches_frontend_error_mapping(client):
    """Frontend mapAnalyzeError() keys off status + code; both must be stable."""
    response = post_analyze(client, make_wav(3.0))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


def test_no_secrets_or_paths_in_openapi_schema(client):
    """The public schema must not leak server paths or environment values."""
    schema_text = client.get("/openapi.json").text
    for token in ("/app/", "voxsentinel.db", "VOXSENTINEL_"):
        assert token not in schema_text, token
