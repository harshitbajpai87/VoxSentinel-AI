"""A family address term alone must not be labelled impersonation."""
from __future__ import annotations

from app.protection.rules import analyze_transcript


def _analyse(text: str):
    return analyze_transcript(text)


def test_ordinary_call_home_is_not_impersonation():
    result = _analyse("Hi Papa, I reached college safely and I will call you after class.")
    assert result.flags["family_claim"] is False
    assert result.primary_category != "family_emergency_impersonation"


def test_family_claim_with_payment_demand_still_escalates():
    result = _analyse(
        "Papa they have me, send five lakh rupees now, do not call the police "
        "or they will hurt me."
    )
    assert result.flags["family_claim"] is True
    assert result.primary_category in {"family_emergency_impersonation", "ransom_or_kidnapping"}


def test_family_claim_with_urgency_without_money_kept():
    result = _analyse("Mummy I am in trouble at the hospital, come quickly.")
    assert result.flags["family_claim"] is True
