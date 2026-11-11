"""Calibration acceptance tests for the scam-protection decision policy.

These tests encode the *product requirement* rather than an implementation
detail: on the project's own labeled transcripts, the rule baseline plus the
risk policy must catch every scam script and must not raise a false alarm on an
ordinary family call.

Why this file exists
--------------------
Calibrating the risk tiers is what makes the module usable by a family: a
system that labels "I sent my son money for books" as a risk teaches people to
ignore it. The two behaviours below are therefore locked in as tests:

  * recall on labeled scam scripts (a missed ransom call is the costly error);
  * zero false alarms on the labeled benign transcripts.

Reproduce with::

    cd backend && pytest tests/test_risk_calibration.py -q

The numbers here are measured on this project's own small labeled set (54
transcripts). They are an acceptance floor for regression, NOT a claim about
accuracy on noisy telephone audio in the wild.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.protection import risk, rules

EVAL_SET = Path(__file__).parent / "data" / "scam_eval_v2.jsonl"
SUSPICIOUS_TIERS = {"MEDIUM", "HIGH", "CRITICAL"}


def _load_cases() -> list[dict]:
    with EVAL_SET.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _verdict(text: str) -> tuple[str, int, str]:
    content = rules.analyze_transcript(text)
    assessment = risk.assess(
        risk.SignalInput(
            synthetic_suspected=None,
            synthetic_note=None,
            speaker_outcome=None,
            signal_strength=content.signal_strength,
            flags=content.flags,
            transcription_status="completed",
            discussion_context=content.discussion_context,
        )
    )
    return assessment.category, content.signal_strength, content.primary_category


def test_eval_set_is_present_and_labeled():
    cases = _load_cases()
    assert len(cases) >= 50
    assert {c["group"] for c in cases} == {"scam", "benign"}


def test_every_scam_script_is_caught():
    """A missed ransom/OTP call is the expensive error, so require full recall."""
    missed = []
    for case in _load_cases():
        if not case["expect_suspicious"]:
            continue
        category, strength, primary = _verdict(case["text"])
        if category not in SUSPICIOUS_TIERS:
            missed.append((case["id"], category, strength, primary))
    assert not missed, f"scam scripts scored as safe: {missed}"


def test_no_false_alarms_on_ordinary_transcripts():
    """Everyday family/business calls must stay LOW."""
    alarms = []
    for case in _load_cases():
        if case["expect_suspicious"]:
            continue
        category, strength, primary = _verdict(case["text"])
        if category in SUSPICIOUS_TIERS:
            alarms.append((case["id"], category, strength, primary))
    assert not alarms, f"benign transcripts flagged as risky: {alarms}"


def test_expected_category_is_matched_for_scam_cases():
    """The headline category must be one the label allows.

    Cases whose label leaves ``expected_categories`` empty assert only that the
    transcript is risky, so they are skipped here. A transcript can legitimately
    match several fraud patterns at once (a prize lure that also demands a fee);
    the engine reports the strongest, so a small margin is allowed.
    """
    wrong = []
    checked = 0
    for case in _load_cases():
        if not case["expect_suspicious"] or not case["expected_categories"]:
            continue
        checked += 1
        _, _, primary = _verdict(case["text"])
        if primary not in case["expected_categories"]:
            wrong.append((case["id"], primary, case["expected_categories"]))
    assert checked >= 20, "category check is not exercising enough cases"
    assert len(wrong) <= 2, f"category mismatches: {wrong}"


def test_extortion_is_not_reported_as_a_kidnapping_claim():
    """A plain "send money or else" call must not be described as a kidnapping.

    Telling a family that a caller claimed to hold their relative would be a
    false and frightening statement about what was actually said. The generic
    money/threat phrases therefore live in their own categories.
    """
    _, _, primary = _verdict(
        "Account number 12345678 par paise bhej do turant, warna bura hoga."
    )
    assert primary != "ransom_or_kidnapping"
    assert primary in {"emergency_money_request", "threats_or_coercion"}


@pytest.mark.parametrize(
    "text",
    [
        # Real miss found during validation: sibling-as-hostage phrasing.
        "Your bhai is with us. Arrange 2 lakh rupees if you want to see them again.",
        # Devanagari ransom.
        "पापा, मुझे अगवा कर लिया है, दो लाख भेजो, पुलिस को मत बताना।",
        # OTP social engineering.
        "This is your bank. Your account will be blocked. Tell me the OTP you just received.",
        # Courier/customs pretext (no coercion flags, still a real script).
        "Aapke Aadhaar se parcel bheja gaya hai jisme narcotics mile hain. Custom officer se baat karo.",
        # Refund/collect-request fraud.
        "Refund ke liye ye collect request accept karo aur QR scan karo, paisa turant aayega.",
    ],
)
def test_known_scam_phrasings_stay_detected(text: str):
    category, _, _ = _verdict(text)
    assert category in SUSPICIOUS_TIERS


@pytest.mark.parametrize(
    "text",
    [
        "Mom, I'm in the hospital, the doctor says it's nothing serious. I'll call later.",
        "The bank called to say my KYC is complete, no action needed from my side.",
        "My son called from his hostel, he needs money for books, I sent it on UPI.",
        "UPI is working now, you can scan the QR at the counter and pay.",
        "Hi, this is Dr Sharma's clinic calling to confirm your appointment tomorrow at five pm.",
    ],
)
def test_ordinary_sentences_stay_calm(text: str):
    category, _, _ = _verdict(text)
    assert category == "LOW"


def test_single_keyword_alone_is_not_a_scam_verdict():
    """One matched word with no corroboration must not reach MEDIUM."""
    content = rules.analyze_transcript("I am going to the hospital to visit my aunt.")
    assert content.signal_strength < 3
    assessment = risk.assess(
        risk.SignalInput(
            synthetic_suspected=None,
            synthetic_note=None,
            speaker_outcome=None,
            signal_strength=content.signal_strength,
            flags=content.flags,
            transcription_status="completed",
            discussion_context=content.discussion_context,
        )
    )
    assert assessment.category == "LOW"


def test_scam_awareness_discussion_is_not_punished():
    """Talking *about* a kidnapping scam is not committing one."""
    text = (
        "In the news today they explained a new scam: fraudsters call and say your son "
        "is kidnapped and you must pay a ransom, and tell you not to call the police."
    )
    category, _, _ = _verdict(text)
    assert category not in {"CRITICAL", "HIGH"}
