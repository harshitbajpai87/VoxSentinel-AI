"""History endpoints and persistence."""
from __future__ import annotations

import uuid

from tests.conftest import make_wav, post_analyze


def test_list_is_empty_by_default_and_never_fabricated(client):
    body = client.get("/api/v1/analyses").json()
    assert body == {"analyses": [], "total": 0}


def test_get_unknown_analysis_returns_404_contract_error(client):
    response = client.get(f"/api/v1/analyses/{uuid.uuid4().hex}")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "not_found"
    assert "traceback" not in error["message"].lower()


def test_persist_and_retrieve_round_trip(client_with_detector):
    created = post_analyze(client_with_detector, make_wav(3.0), name="kept.wav").json()

    fetched = client_with_detector.get(f"/api/v1/analyses/{created['id']}").json()
    assert fetched["id"] == created["id"]
    assert fetched["input"]["display_name"] == "kept.wav"
    assert fetched["scores"]["raw_score"] == created["scores"]["raw_score"]
    assert fetched["decision"] == created["decision"]
    assert fetched["evidence"]["measured_features"] == created["evidence"]["measured_features"]


def test_pagination_params_are_validated(client):
    for bad in ({"limit": 0}, {"limit": -1}, {"limit": 101}, {"offset": -5}):
        response = client.get("/api/v1/analyses", params=bad)
        assert response.status_code == 422, bad
        assert response.json()["error"]["code"] == "validation_error"


def test_pagination_slices_results(client_with_detector):
    for i in range(3):
        post_analyze(client_with_detector, make_wav(3.0), name=f"clip{i}.wav")
    page1 = client_with_detector.get("/api/v1/analyses", params={"limit": 2, "offset": 0}).json()
    page2 = client_with_detector.get("/api/v1/analyses", params={"limit": 2, "offset": 2}).json()
    assert page1["total"] == 3 and len(page1["analyses"]) == 2
    assert len(page2["analyses"]) == 1
    ids1 = {a["id"] for a in page1["analyses"]}
    ids2 = {a["id"] for a in page2["analyses"]}
    assert not (ids1 & ids2)


def test_newest_records_sort_first(client_with_detector):
    first = post_analyze(client_with_detector, make_wav(3.0), name="first.wav").json()
    second = post_analyze(client_with_detector, make_wav(3.0), name="second.wav").json()
    listing = client_with_detector.get("/api/v1/analyses").json()
    assert listing["analyses"][0]["id"] == second["id"]
    assert listing["analyses"][1]["id"] == first["id"]


def test_503_responses_do_not_pollute_history(client):
    post_analyze(client, make_wav(3.0))  # 503: model unavailable
    listing = client.get("/api/v1/analyses").json()
    assert listing["total"] == 0
