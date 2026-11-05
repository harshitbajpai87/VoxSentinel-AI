"""Hindi/Hinglish kidnapping detection, false-positive guard, evidence quotes."""
from __future__ import annotations

import pytest

from app.protection.rules import analyze_transcript

RANSOM = [
    "Tumhare bete ko humne kidnap kar liya hai. Paise bhejo warna use chhodenge nahi.",
    "Papa, mujhe kisi ne pakad liya hai, please paise bhejo.",
    "Agar police ko bataya toh bura hoga, paise bhejo abhi.",
    "Your daughter is with us. Transfer the money immediately.",
    "Beti hamare paas hai, dus lakh transfer karo warna police ko mat batana.",
]
BENIGN = [
    "Hi Papa, I reached college safely and I will call you after class.",
    "Mummy, main hostel pahunch gaya.",
    "Papa, aaj college mein exam tha.",
    "Please call me when you reach home.",
    "Beta, khaana kha liya? Main ghar par hoon.",
]


@pytest.mark.parametrize("text", RANSOM)
def test_ransom_detected_in_all_languages(text):
    r = analyze_transcript(text)
    assert r.primary_category == "ransom_or_kidnapping"
    assert r.signal_strength >= 7


@pytest.mark.parametrize("text", BENIGN)
def test_ordinary_family_talk_is_not_a_scam(text):
    r = analyze_transcript(text)
    assert r.primary_category == "legitimate_or_no_clear_scam_evidence"
    assert r.signal_strength <= 2
    assert not r.flags["ransom_or_kidnapping"]


def test_keyword_alone_cannot_create_ransom_category():
    """'kidnap' in passing, with no demand/threat/secrecy, is not a ransom call."""
    r = analyze_transcript("We watched a movie about a kidnap yesterday, it was boring.")
    assert r.primary_category != "ransom_or_kidnapping"


def test_evidence_quotes_are_verbatim_slices():
    text = "Tumhare bete ko humne kidnap kar liya hai. Paise bhejo warna use chhodenge nahi."
    r = analyze_transcript(text)
    assert r.evidence_quotes, "every scam classification must cite transcript evidence"
    for quote in r.evidence_quotes:
        assert quote in text.lower(), "quote must appear verbatim in the transcript"


def test_benign_conversation_has_no_evidence_quotes():
    r = analyze_transcript("Please call me when you reach home.")
    assert r.evidence_quotes == []


def test_discussion_context_is_downweighted():
    r = analyze_transcript(
        "In this training video we show how scammers say your daughter is with us "
        "and demand money, so never share details."
    )
    assert r.discussion_context is True
