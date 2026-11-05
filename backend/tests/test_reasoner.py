"""Second-pass reasoning review (Reasoner4All pattern) — offline tests.

Network access is mocked with httpx.MockTransport. The properties pinned here
are the safety ones: the reviewer may only ever *soften* a verdict, only with
verbatim transcript evidence, and a failure must leave stage one untouched.
"""
from __future__ import annotations

import json

import httpx
import pytest

from app.protection.reasoner import ReasonerConfig, ReasoningReviewer, _extract_json

SCAM = "Papa they have me, send 5 lakh now, do not call the police or they will hurt me."
AWARENESS = "In this training video we show how a scam call sounds, so remember to verify."


def _transport(payload, status: int = 200) -> httpx.MockTransport:
    body = payload if isinstance(payload, str) else json.dumps(payload)

    def handler(request: httpx.Request) -> httpx.Response:
        if status != 200:
            return httpx.Response(status, json={"error": "boom"})
        return httpx.Response(200, json={"choices": [{"message": {"content": body}}]})

    return httpx.MockTransport(handler)


@pytest.fixture
def reviewer_of(monkeypatch):
    """Build a reviewer whose HTTP client uses the supplied mock transport."""
    made = []

    def build(payload, status=200, api_key="gsk_test"):
        transport = _transport(payload, status)
        original = httpx.Client

        class _Patched(original):  # type: ignore[misc]
            def __init__(self, *a, **kw):
                kw["transport"] = transport
                super().__init__(*a, **kw)

        monkeypatch.setattr(httpx, "Client", _Patched)
        reviewer = ReasoningReviewer(
            ReasonerConfig(enabled=True, provider="groq", api_key=api_key, model="m", timeout_s=5.0)
        )
        made.append(reviewer)
        return reviewer

    yield build
    monkeypatch.undo()


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("PROTECTION_REASONER", raising=False)
    cfg = ReasonerConfig.from_env("groq")
    assert cfg.enabled is False
    assert ReasoningReviewer(cfg).review("hello", "LIKELY_SCAM").status == "disabled"


def test_missing_key_is_unavailable_not_a_crash(monkeypatch):
    monkeypatch.setenv("PROTECTION_REASONER", "enabled")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    got = ReasoningReviewer(ReasonerConfig.from_env("groq")).review("x", "LIKELY_SCAM")
    assert got.status == "unavailable"


def test_agreement_reported_without_override(reviewer_of):
    review = reviewer_of(
        {"agrees": True, "verdict": "LIKELY_SCAM", "critique": "Demands ransom.",
         "context_signals": ["do not call the police"]}
    ).review(SCAM, "LIKELY_SCAM", ["KIDNAPPING_OR_RANSOM"])
    assert review.status == "completed"
    assert review.agrees is True
    assert review.overridden is False
    assert review.context_signals == ["do not call the police"]
    assert review.evidence_verified is True


def test_softening_requires_verbatim_evidence(reviewer_of):
    review = reviewer_of(
        {"agrees": False, "verdict": "EDUCATIONAL_OR_DISCUSSION",
         "critique": "Sounds like a training clip.", "context_signals": ["in this training video"]}
    ).review(AWARENESS, "LIKELY_SCAM")
    assert review.overridden is True
    assert review.evidence_verified is True


def test_softening_without_evidence_is_refused(reviewer_of):
    """The core guarantee: no quoting, no downgrade."""
    review = reviewer_of(
        {"agrees": False, "verdict": "LIKELY_LEGITIMATE",
         "critique": "Probably fine.", "context_signals": ["he calmly asks for money"]}
    ).review(SCAM, "LIKELY_SCAM")
    assert review.verdict == "LIKELY_LEGITIMATE"
    assert review.overridden is False
    assert review.evidence_verified is False
    assert review.dropped_signals == 1
    assert "stage-one result kept" in (review.detail or "")


def test_invented_quotes_are_dropped(reviewer_of):
    review = reviewer_of(
        {"agrees": False, "verdict": "EDUCATIONAL_OR_DISCUSSION",
         "critique": "x",
         "context_signals": ["this is a public service announcement", "do not call the police"]}
    ).review(SCAM, "LIKELY_SCAM")
    assert review.context_signals == ["do not call the police"]
    assert review.dropped_signals == 1
    assert review.overridden is False  # one real quote is evidence, but verdict must be verbatim-backed


def test_never_escalates(reviewer_of):
    review = reviewer_of({"agrees": False, "verdict": "LIKELY_SCAM", "critique": "worse"})
    got = review.review("hello", "SUSPICIOUS")
    assert got.verdict == "LIKELY_SCAM"
    assert got.overridden is False


def test_invalid_json_degrades(reviewer_of):
    assert reviewer_of("no json here").review(SCAM, "LIKELY_SCAM").status == "failed"


def test_http_error_degrades(reviewer_of):
    got = reviewer_of({}, status=429).review(SCAM, "LIKELY_SCAM")
    assert got.status == "failed" and "429" in (got.detail or "")


def test_unknown_verdict_ignored(reviewer_of):
    got = reviewer_of({"agrees": True, "verdict": "TOTALLY_FINE"}).review(SCAM, "LIKELY_SCAM")
    assert got.verdict is None and got.overridden is False


def test_empty_transcript_not_sent(reviewer_of):
    assert reviewer_of({"verdict": "LIKELY_SCAM"}).review("   ", "LIKELY_SCAM").status == "unavailable"


def test_extract_json_variants():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('Answer: {"a": 2} done') == {"a": 2}
    assert _extract_json("no json") is None


def test_describe_credits_source_space():
    d = ReasoningReviewer(ReasonerConfig(True, "groq", "k", "m")).describe()
    assert d["inspired_by"] == "huggingface.co/spaces/McLoviniTtt/Reasoner4All"
    assert d["enabled"] is True
