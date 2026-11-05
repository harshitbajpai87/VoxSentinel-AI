"""CORS configuration — the real cause of "Could not reach the backend".

The browser reports any CORS rejection as a network error, so a missing
Access-Control-Allow-Origin header surfaces to users as
"Could not reach the backend (network error or CORS). The file was not analyzed."
These tests pin the dev/LAN behaviour and the production strictness.
"""
from __future__ import annotations

import re

from app.core.config import Settings, get_settings
from tests.conftest import MockVoiceDetector, build_client  # noqa: F401


def _settings(**overrides) -> Settings:
    """Copy the live settings and override a field (no env mutation needed)."""
    base = get_settings()
    data = {k: v for k, v in base.__dict__.items() if k != "allow_dev_origins"}
    data.update(overrides)
    return Settings(**data)


def test_regex_allows_loopback_and_lan_on_any_port():
    regex = _settings(app_env="development").cors_origin_regex()
    assert regex is not None
    for origin in (
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://192.168.1.5:5173",
        "http://10.0.0.9:4173",
        "http://172.20.0.4:8080",
    ):
        assert re.match(regex, origin), origin


def test_regex_rejects_public_and_malformed_origins():
    regex = _settings(app_env="development").cors_origin_regex()
    for origin in (
        "https://evil.example.com",
        "http://8.8.8.8:5173",
        "http://localhost.evil.com",
    ):
        assert not re.match(regex, origin), origin


def test_production_has_no_wildcard_dev_regex():
    assert _settings(app_env="production").cors_origin_regex() is None


def test_dev_regex_can_be_disabled():
    assert _settings(app_env="development", allow_dev_origins=False).cors_origin_regex() is None


def test_allowlisted_origin_gets_cors_header(client):
    """localhost:5173 must always receive an explicit allow header."""
    response = client.options(
        "/api/v1/analyze",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_lan_origin_gets_cors_header_in_dev(client):
    """A phone on the LAN opening the dev server must not be blocked."""
    response = client.options(
        "/api/v1/analyze",
        headers={
            "Origin": "http://192.168.1.5:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.headers.get("access-control-allow-origin") == "http://192.168.1.5:5173"


def test_external_origin_gets_no_cors_header(client):
    response = client.options(
        "/api/v1/analyze",
        headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in response.headers


def test_error_responses_still_carry_cors_headers(client):
    """A 4xx must not look like a network failure to the browser."""
    response = client.post(
        "/api/v1/analyze",
        headers={"Origin": "http://localhost:5173"},
        files={"file": ("bad.wav", b"not-audio", "audio/wav")},
        data={"consent": "true"},
    )
    assert response.status_code >= 400
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
