"""Regression tests for the multilingual scam content engine.

These lock in the failure that started this round: a textbook family-emergency
call was rated merely "medium" because detection depended on one phrase per
category. The parametrised block runs the authored evaluation set so a future
pattern edit cannot quietly reintroduce a miss.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.protection import rules
from app.protection.rules import analyze_transcript

EVAL_PATH = Path(__file__).parent / "data" / "scam_eval_v2.jsonl"
EVAL_ROWS = [json.loads(line) for line in EVAL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("item", EVAL_ROWS, ids=[row["id"] for row in EVAL_ROWS])
def test_eval_set_verdicts(item):
    result = analyze_transcript(item["text"])
    suspicious = result.signal_strength >= 3 and not result.discussion_context
    detail = (
        f"{item['id']}: expected suspicious={item['expect_suspicious']}, "
        f"strength={result.signal_strength}, "
        f"categories={[c.category for c in result.categories]}"
    )
    assert suspicious is item["expect_suspicious"], detail


@pytest.mark.parametrize("item", EVAL_ROWS, ids=[row["id"] for row in EVAL_ROWS])
def test_eval_set_category_coverage(item):
    """Every expected category must be detected somewhere in the analysis."""

    if not item["expected_categories"]:
        return
    result = analyze_transcript(item["text"])
    matched = {c.category for c in result.categories}
    missing = set(item["expected_categories"]) - matched
    assert not missing, f"{item['id']}: missing {missing}"


def test_family_emergency_script_reaches_high_strength():
    """The original miss: relative + crisis + money + secrecy in Hinglish."""

    result = analyze_transcript(
        "Papa main musibat mein hoon, turant 20000 bhejo warna mere saath kuch ho jayega. "
        "Kisi ko mat batana, police ko bhi nahi."
    )
    assert result.signal_strength >= 7
    assert result.primary_category == "family_emergency_impersonation"
    assert any(c.category == "family_emergency_impersonation" for c in result.categories)
    assert result.discussion_context is False


def test_benign_family_accident_stays_below_threshold():
    result = analyze_transcript(
        "Papa, I had a small accident on the way. I'm fine, the doctor checked me. "
        "I'll call you from the hospital landline later."
    )
    assert result.signal_strength < 3


def test_awareness_advice_is_not_flagged():
    result = analyze_transcript(
        "Remember, never share your OTP with anyone. The bank never asks for it, "
        "and report any fraud call to 1930."
    )
    assert result.signal_strength == 0
    assert result.discussion_context is True


def test_ransom_word_in_benign_job_story_is_withdrawn():
    result = analyze_transcript(
        "The police rescued the child within four hours and arrested the gang that had "
        "demanded the ransom."
    )
    assert result.signal_strength < 3


def test_evidence_quotes_are_verbatim_spans():
    text = "Papa mujhe kidnap kar liya hai, 50000 abhi bhejo warna hum nahi chhodenge."
    result = analyze_transcript(text)
    quotes = getattr(result, "evidence_quotes", [])
    assert quotes, "a ransom transcript should quote its evidence"
    for quote in quotes:
        assert quote in text


def test_primary_category_is_never_from_an_empty_finding_list():
    """Guard for the crash: filtered-out findings must yield zero strength."""

    result = analyze_transcript("We discussed the ransom scene in the film for an hour.")
    assert result.signal_strength == 0
    assert result.primary_category == "legitimate_or_no_clear_scam_evidence"
    assert rules  # module import kept for clarity
