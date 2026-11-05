"""Tests for the Module 4-7 endpoints: persist flag, window evidence, CSV
export, audit trail, evaluation jobs and retention.

Everything here runs against REAL behavior: a mock detector (clearly labelled
test double) supplies the analysis scores, but persistence, export, audit,
evaluation jobs and retention are the genuine implementations under test.
"""
from __future__ import annotations

import csv
import io
import time
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import make_wav


@pytest.fixture
def client(settings, mock_detector) -> TestClient:
    app = create_app(settings=settings, detector=mock_detector)
    with TestClient(app) as test_client:
        yield test_client


def _analyze(c: TestClient, name: str = "clip.wav", duration: float = 3.0, **form) -> dict:
    response = c.post(
        "/api/v1/analyze",
        files={"file": (name, make_wav(duration), "audio/wav")},
        data={"consent": "true", **form},
    )
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Module 2/4 — persist=false provisional analyses
# ---------------------------------------------------------------------------
def test_persist_false_computes_but_never_stores(client):
    _analyze(client, persist="false")
    _analyze(client, persist="true")
    listing = client.get("/api/v1/analyses").json()
    assert listing["total"] == 1  # only the persisted one

    events = client.get("/api/v1/audit?event_type=analysis&limit=10").json()
    actions = [e["action"] for e in events["events"]]
    assert "provisional_window" in actions  # honestly audited
    assert "create" in actions


def test_persist_default_stores(client):
    """No persist field at all: the historical default is to store."""
    _analyze(client)
    assert client.get("/api/v1/analyses").json()["total"] == 1


# ---------------------------------------------------------------------------
# Module 6 — time-window evidence
# ---------------------------------------------------------------------------
def test_window_scores_present_for_long_clip(settings, trained_model_dir):
    settings = replace(settings, detector_model_path=str(trained_model_dir / "detector.joblib"))
    app = create_app(settings=settings)
    with TestClient(app) as c:
        body = _analyze(c, duration=9.0)
        windows = body["evidence"]["time_window_scores"]
        assert len(windows) == 2
        for w in windows:
            assert 0.0 <= w["score"] <= 1.0
            assert w["end_s"] > w["start_s"]
        assert windows[0]["start_s"] == 0.0


def test_short_clip_has_no_window_scores(settings, trained_model_dir):
    settings = replace(settings, detector_model_path=str(trained_model_dir / "detector.joblib"))
    app = create_app(settings=settings)
    with TestClient(app) as c:
        body = _analyze(c, duration=3.0)
        assert body["evidence"]["time_window_scores"] == []


# ---------------------------------------------------------------------------
# Module 4 — CSV export
# ---------------------------------------------------------------------------
def _rows(text: str):
    return list(csv.reader(io.StringIO(text.lstrip("\ufeff"))))


def test_csv_export_matches_persisted_records(client):
    r1 = _analyze(client, name="one.wav")
    r2 = _analyze(client, name="two.wav")
    response = client.get("/api/v1/analyses/export.csv")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    rows = _rows(response.text)
    header, data = rows[0], rows[1:]
    assert len(data) == 2
    assert "analysis_id" in header and "raw_score" in header
    ids = {row[header.index("analysis_id")] for row in data}
    assert ids == {r1["id"], r2["id"]}


def test_csv_export_escapes_hostile_filenames(client):
    """A filename with commas, quotes and a formula-injection prefix must be
    safely quoted, never spreadsheet-executable."""
    # Hostile but VALID audio filename: commas, quotes, a newline and a
    # spreadsheet formula prefix. The extension gate rightly rejects .csv.
    evil = 'clip,"=SUM(A1:A2)","\n.wav'
    _analyze(client, name=evil)
    response = client.get("/api/v1/analyses/export.csv")
    assert response.status_code == 200
    rows = _rows(response.text)
    data = rows[1:]
    assert len(data) == 1
    # The display name is percent-encoded by the upload sanitizer (existing
    # behaviour); the point here is that whatever arrives survives as ONE
    # intact, quoted cell and nothing is spreadsheet-executable.
    name_cell = data[0][2]
    assert "SUM(A1:A2)" in name_cell and "%22" in name_cell
    assert "\n" not in name_cell
    # Nothing is spreadsheet-executable: no raw cell starts with '='.
    assert not any(cell.startswith("=") for cell in data[0])
    # Every row has the full column count despite embedded delimiters.
    assert len(data[0]) == len(rows[0])


def test_csv_export_empty_history(client):
    response = client.get("/api/v1/analyses/export.csv")
    assert response.status_code == 200
    rows = _rows(response.text)
    assert len(rows) == 1  # header only


# ---------------------------------------------------------------------------
# Module 7 — audit trail
# ---------------------------------------------------------------------------
def test_audit_events_are_written_and_filterable(client):
    _analyze(client)
    _analyze(client, persist="false")
    all_events = client.get("/api/v1/audit?limit=50").json()
    assert all_events["total"] >= 2

    only_analyses = client.get("/api/v1/audit?event_type=analysis&limit=50").json()
    assert only_analyses["total"] >= 2
    assert all(e["event_type"] == "analysis" for e in only_analyses["events"])

    page = client.get("/api/v1/audit?limit=1&offset=1").json()
    assert len(page["events"]) == 1


# ---------------------------------------------------------------------------
# Module 7 — retention (dry-run + enforcement)
# ---------------------------------------------------------------------------
def test_retention_dry_run_and_enforcement(settings, mock_detector):
    settings = replace(settings, retention_analyses_days=30.0, retention_audit_days=365.0)
    app = create_app(settings=settings, detector=mock_detector)
    with TestClient(app) as c:
        _analyze(c)
        # Age the record beyond the analysis cutoff, directly in the DB.
        from sqlalchemy import text as sql_text

        with app.state.db.session() as session:
            session.execute(
                sql_text("UPDATE analysis_records SET created_at = '2020-01-01T00:00:00.000Z'")
            )
            session.commit()

        preview = c.post("/api/v1/admin/retention/run?dry_run=true").json()
        assert preview["analyses_deleted"] == 1
        assert c.get("/api/v1/analyses").json()["total"] == 1  # nothing deleted yet

        executed = c.post("/api/v1/admin/retention/run?dry_run=false").json()
        assert executed["analyses_deleted"] == 1
        assert c.get("/api/v1/analyses").json()["total"] == 0

        events = c.get("/api/v1/audit?event_type=data_retention&limit=10").json()
        assert events["total"] >= 2  # both runs audited

        status = c.get("/api/v1/admin/retention").json()
        assert status["last_run"] is not None


def test_retention_disabled_deletes_nothing(settings, mock_detector):
    """An explicit 0-day policy means KEEP FOREVER: nothing is ever deleted."""
    from dataclasses import replace as dc_replace

    settings = dc_replace(settings, retention_analyses_days=0.0, retention_audit_days=0.0)
    app = create_app(settings=settings, detector=mock_detector)
    with TestClient(app) as c:
        _analyze(c)
        from sqlalchemy import text as sql_text

        with app.state.db.session() as session:
            session.execute(
                sql_text("UPDATE analysis_records SET created_at = '2020-01-01T00:00:00.000Z'")
            )
            session.commit()
        executed = c.post("/api/v1/admin/retention/run?dry_run=false").json()
        assert executed["analyses_deleted"] == 0
        assert c.get("/api/v1/analyses").json()["total"] == 1


# ---------------------------------------------------------------------------
# Module 5 — evaluation job (real background inference)
# ---------------------------------------------------------------------------
def test_evaluation_job_end_to_end(settings, trained_model_dir, fixture_dataset, tmp_path):
    settings = replace(
        settings,
        detector_model_path=str(trained_model_dir / "detector.joblib"),
        eval_manifest_path=str(fixture_dataset / "manifest.jsonl"),
        eval_dataset_dir=str(fixture_dataset),
        report_dir=str(tmp_path / "reports"),
    )
    app = create_app(settings=settings)
    with TestClient(app) as c:
        assert c.get("/health").json()["model_ready"] is True

        started = c.post("/api/v1/evaluation/run")
        assert started.status_code in (200, 202)
        job_id = started.json()["job_id"]

        for _ in range(120):  # poll until the real job finishes
            state = c.get(f"/api/v1/evaluation/run/{job_id}").json()
            if state["status"] in ("completed", "failed"):
                break
            time.sleep(0.25)
        assert state["status"] == "completed", state
        metrics = state["metrics"]
        assert metrics["n"] > 0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert metrics["confusion_matrix"]["tp"] + metrics["confusion_matrix"]["tn"] > 0

        # unknown job id -> honest 404
        assert c.get("/api/v1/evaluation/run/does-not-exist").status_code == 404


def test_evaluation_summary_reflects_latest_run(settings, trained_model_dir, fixture_dataset, tmp_path):
    settings = replace(
        settings,
        detector_model_path=str(trained_model_dir / "detector.joblib"),
        eval_manifest_path=str(fixture_dataset / "manifest.jsonl"),
        eval_dataset_dir=str(fixture_dataset),
        report_dir=str(tmp_path / "reports"),
    )
    app = create_app(settings=settings)
    with TestClient(app) as c:
        summary = c.get("/api/v1/evaluation").json()
        assert summary["evaluated"] is False  # nothing ran yet — honest

        started = c.post("/api/v1/evaluation/run")
        job_id = started.json()["job_id"]
        for _ in range(120):
            state = c.get(f"/api/v1/evaluation/run/{job_id}").json()
            if state["status"] in ("completed", "failed"):
                break
            time.sleep(0.25)
        assert state["status"] == "completed"

        summary = c.get("/api/v1/evaluation").json()
        assert summary["evaluated"] is True
        assert summary["model"]["version"]  # real model version recorded
