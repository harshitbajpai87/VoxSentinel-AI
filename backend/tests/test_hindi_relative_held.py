"""Hindi/Hinglish relative-held patterns: contextual evidence, not keywords.

Requirement: saying a relative is being held must never be enough for a CRITICAL
warning on its own. Ransom/scam categories need a demand, a threat, secrecy, or
an explicit custody claim that is not framed as discussion.
"""
from __future__ import annotations

import pytest

from app.protection import risk
from app.protection.rules import analyze_transcript


def assess(text: str):
    content = analyze_transcript(text)
    return content, risk.assess(
        risk.SignalInput(
            synthetic_suspected=None,
            synthetic_note=None,
            speaker_outcome=None,
            signal_strength=content.signal_strength,
            flags=content.flags,
            transcription_status="completed",
            discussion_context=content.discussion_context,
            gemini_status="unavailable",
        )
    )


# --- relative held, no payment demand -------------------------------------
def test_relative_held_without_demand_is_not_critical():
    content, assessment = assess("आपकी बेटी हमारे पास है।")
    assert content.primary_category != "ransom_or_kidnapping" or content.signal_strength < 7
    assert assessment.category != "CRITICAL"


def test_relative_custody_claim_with_demand_is_ransom():
    content, assessment = assess(
        "आपकी बेटी हमारे पास है, पाँच लाख रुपये तुरंत भेजो।"
    )
    assert content.flags["ransom_or_kidnapping"] is True
    assert content.flags["payment_demand"] is True
    assert assessment.category in {"HIGH", "CRITICAL"}


# --- relative held with full ransom framing ------------------------------
def test_hindi_full_ransom_is_critical():
    content, assessment = assess(
        "आपकी बेटी हमारे पास है, पाँच लाख रुपये तुरंत भेजो, "
        "पुलिस को मत बताना वरना उसे चोट पहुँचाएँगे।"
    )
    assert content.primary_category == "ransom_or_kidnapping"
    assert content.signal_strength >= 7
    assert assessment.category == "CRITICAL"


def test_hinglish_full_ransom_is_critical():
    content, assessment = assess(
        "Beti hamare paas hai, paanch lakh rupaye abhi bhejo, "
        "police ko mat batana warna hum usko chhodenge nahi."
    )
    assert content.flags["ransom_or_kidnapping"] is True
    assert assessment.category in {"HIGH", "CRITICAL"}


# --- threats against a family member -------------------------------------
def test_threat_against_family_member_is_flagged():
    content, _ = assess("अगर पैसे नहीं दिए तो हम तुम्हारे बेटे को मार देंगे।")
    assert content.flags["threat"] is True
    assert content.flags["family_claim"] is True


def test_hinglish_threat_is_flagged():
    content, _ = assess("Paise nahi diye toh hum tumhare bete ko maar denge.")
    assert content.flags["threat"] is True


# --- secrecy / police-contact prevention ---------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "पुलिस को मत बताना, वरना सब खत्म।",
        "Kisi ko mat batana, police ko bhi nahi.",
    ],
)
def test_secrecy_and_police_prevention(text):
    content, _ = assess(text)
    assert content.flags["secrecy_or_urgency"] is True


# --- ordinary family talk mentioning relatives ---------------------------
@pytest.mark.parametrize(
    "text",
    [
        "मेरी बेटी हमारे पास है और वह ठीक है, हम सब घर पर हैं।",
        "बेटा आज घर आ रहा है, खाना बना लो।",
        "Papa, aaj college mein exam tha, main ghar aa raha hoon.",
        "मम्मी, मैं हॉस्टल पहुँच गया हूँ, चिंता मत करो।",
    ],
)
def test_ordinary_family_conversation_is_not_a_scam(text):
    content, assessment = assess(text)
    assert content.primary_category == "legitimate_or_no_clear_scam_evidence"
    assert assessment.category in {"LOW", "MEDIUM"}
    assert content.flags["ransom_or_kidnapping"] is False


# --- news, fiction, awareness discussion ---------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "समाचार में बताया गया कि एक बच्चे को अगवा कर लिया गया था।",
        "In this awareness video we explain how callers claim your daughter is with us "
        "and then demand money, so never pay.",
        "हमारी ट्रेनिंग में बताया गया है कि ऐसे कॉल आते हैं जहाँ कहते हैं बेटी हमारे पास है।",
    ],
)
def test_discussion_and_news_are_downgraded(text):
    content, assessment = assess(text)
    assert assessment.category in {"LOW", "MEDIUM"}
    if content.discussion_context:
        assert assessment.category != "CRITICAL"


def test_evidence_quotes_are_verbatim_for_hindi():
    text = "आपकी बेटी हमारे पास है, पाँच लाख रुपये तुरंत भेजो, पुलिस को मत बताना।"
    content = analyze_transcript(text)
    assert content.evidence_quotes
    for quote in content.evidence_quotes:
        assert quote in text.lower()
