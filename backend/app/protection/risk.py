"""Risk policy for the protection module.

Keeps components independent (voice authenticity, speaker comparison, content
analysis) and combines them with transparent, configurable rules. Never
averages unrelated scores, never treats an uncalibrated detector output as a
probability, and never claims certainty.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SignalInput:
    """Everything the risk policy is allowed to consider."""

    synthetic_suspected: bool | None  # None = voice analysis unavailable
    synthetic_note: str | None
    speaker_outcome: str | None  # consistent/not_consistent/inconclusive/reference_unavailable/None
    signal_strength: int  # 0..10 from the rule engine
    flags: dict[str, bool]
    transcription_status: str  # completed | unavailable | failed
    discussion_context: bool
    # Gemini contextual classification (primary signal when configured).
    gemini_classification: str | None = None  # LIKELY_SCAM | SUSPICIOUS | ... | None
    gemini_risk: str | None = None  # LOW | MEDIUM | HIGH | CRITICAL | INCONCLUSIVE
    gemini_categories: tuple[str, ...] = ()
    gemini_status: str = "unavailable"  # completed | unavailable | failed
    # Human-readable provider label ("Groq (GPT-OSS)", "Gemini", ...) used in
    # user-facing reasoning so the report never credits the wrong model.
    llm_label: str = "Contextual AI"


@dataclass
class RiskAssessment:
    category: str  # LOW | MEDIUM | HIGH | CRITICAL | INCONCLUSIVE
    reasoning: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)


_ACTION_VERIFY = (
    "Pause before transferring any money. Verify the family member through a previously saved, "
    "trusted number or a separate trusted channel."
)
_ACTION_SECRETS = "Never share OTPs, PINs, passwords, CVV numbers or banking credentials on a call."
_ACTION_EMERGENCY = (
    "If there is credible evidence of kidnapping or immediate danger, contact local emergency "
    "services or the police directly using a number you looked up yourself."
)
_ACTION_RELATIVE = "Contact another trusted relative to confirm the situation independently."
_ACTION_PRESERVE = "Preserve call details and the recording when it is safe and lawful to do so."


def assess(inp: SignalInput) -> RiskAssessment:
    """Finalize independent authenticity and conversation-safety evidence.

    A usable synthetic signal imposes a MEDIUM verification floor even for
    benign/educational content. It is not proof of scam intent and cannot by
    itself cause HIGH or CRITICAL. Every contextual early return passes here.
    Missing voice analysis is never interpreted as evidence of authenticity.
    """
    result = _assess_conversation(inp)
    result.reasoning.insert(
        0, f"Conversation safety assessment before the independent voice warning: {result.category}."
    )
    if inp.synthetic_suspected is True:
        result.reasoning.append("Voice authenticity independently indicates possible synthetic speech.")
        result.limitations.append(
            "Synthetic speech is not proof of scam intent; legitimate uses exist "
            "and detector errors are possible."
        )
        if result.category == "LOW":
            result.category = "MEDIUM"
        if result.category == "MEDIUM":
            result.reasoning.append(
                "The voice signal warrants independent verification, not a HIGH or CRITICAL scam verdict."
            )
            if _ACTION_VERIFY not in result.recommended_actions:
                result.recommended_actions.append(_ACTION_VERIFY)
        if inp.gemini_status == "completed" and (inp.gemini_classification or "").upper() in {
            "LIKELY_LEGITIMATE",
            "EDUCATIONAL_OR_DISCUSSION",
        }:
            result.reasoning.append(
                "Evidence conflict: contextual analysis describes benign content while voice analysis "
                "flags possible synthesis. These address different questions; neither cancels the other."
            )
    elif inp.synthetic_suspected is False:
        result.reasoning.append(
            "Voice analysis found no strong evidence of synthetic speech; "
            "this does not establish caller identity or safety."
        )
    else:
        result.reasoning.append("Voice authenticity could not be assessed for this recording.")
        result.limitations.append("Unavailable voice analysis is not evidence that the voice is genuine.")
    if inp.synthetic_note:
        result.limitations.append(inp.synthetic_note)
    if (
        inp.gemini_status == "completed"
        and (inp.gemini_classification or "").upper() in {"LIKELY_LEGITIMATE", "EDUCATIONAL_OR_DISCUSSION"}
        and inp.signal_strength >= 7
    ):
        result.reasoning.append(
            "Evidence conflict: strong rule-based content indicators disagree with the benign contextual "
            "classification; review the transcript and matched evidence rather than assuming certainty."
        )
    if inp.speaker_outcome == "not_consistent_with_reference":
        result.reasoning.append("The recording is not consistent with the enrolled family voice reference.")
    elif inp.speaker_outcome == "consistent_with_reference":
        result.reasoning.append(
            "The recording is consistent with the enrolled family voice reference (not proof of identity)."
        )
    elif inp.speaker_outcome == "inconclusive":
        result.reasoning.append("Family voice comparison was inconclusive.")
    elif inp.speaker_outcome == "reference_unavailable":
        result.reasoning.append("No authorized family voice reference was available for comparison.")
    result.limitations.append(
        "Voice similarity cannot prove identity, and a suspicious transcript "
        "does not by itself prove a crime."
    )
    return result


def _assess_conversation(inp: SignalInput) -> RiskAssessment:
    reasoning: list[str] = []
    actions: list[str] = []
    limitations: list[str] = []

    content_high = inp.signal_strength >= 7
    content_mid = 4 <= inp.signal_strength < 7
    content_low = 1 <= inp.signal_strength < 4

    # ---- Gemini contextual classification is the PRIMARY signal when present.
    # The rule engine remains an independent, separately reported support layer.
    g_cls = (inp.gemini_classification or "").upper()
    g_risk = (inp.gemini_risk or "").upper()
    g_cats = {c.upper() for c in inp.gemini_categories}
    label = (inp.llm_label or "Contextual AI").strip() or "Contextual AI"
    gemini_ok = inp.gemini_status == "completed" and bool(g_cls)
    # A broad category such as FAMILY_EMERGENCY does not prove ransom, a
    # payment demand, or immediate danger. CRITICAL requires the provider's
    # explicit critical severity; category-only scam findings remain HIGH.
    credential_combo = bool(
        g_cats
        & {
            "OTP_OR_CREDENTIAL_THEFT",
            "BANKING_FRAUD",
            "UPI_OR_PAYMENT_FRAUD",
            "PHISHING_OR_SOCIAL_ENGINEERING",
        }
    )

    if gemini_ok:
        reasoning.append(f"{label} contextual analysis classified the transcript as {g_cls}.")
        if g_cats:
            reasoning.append(f"{label} categories: " + ", ".join(sorted(g_cats)) + ".")
        if inp.signal_strength > 0:
            reasoning.append(
                "Rule-based baseline independently flagged matching indicators "
                f"(strength {inp.signal_strength}/10)."
            )
        elif g_cls in {"LIKELY_SCAM", "SUSPICIOUS"}:
            reasoning.append(
                "The rule baseline found no keyword matches; the contextual model "
                "identified risk from meaning and context."
            )

        if g_cls == "LIKELY_SCAM" and g_risk == "CRITICAL":
            reasoning.append(
                f"{label} rated this likely-scam transcript as CRITICAL; verify the claims "
                "through a known channel because model output is not proof of a crime."
            )
            return RiskAssessment(
                "CRITICAL",
                reasoning,
                [_ACTION_VERIFY, _ACTION_SECRETS, _ACTION_EMERGENCY, _ACTION_RELATIVE, _ACTION_PRESERVE],
                limitations
                + ["A transcript can show coercion without proving a crime occurred; verify independently."],
            )
        if g_cls == "LIKELY_SCAM":
            return RiskAssessment(
                "HIGH",
                reasoning,
                [_ACTION_VERIFY, _ACTION_SECRETS, _ACTION_RELATIVE],
                limitations,
            )
        if g_cls == "SUSPICIOUS":
            cat = "HIGH" if (credential_combo or g_risk in {"HIGH", "CRITICAL"}) else "MEDIUM"
            return RiskAssessment(cat, reasoning, [_ACTION_VERIFY, _ACTION_SECRETS], limitations)
        if g_cls == "EDUCATIONAL_OR_DISCUSSION":
            reasoning.append(
                "Context indicates a discussion, warning or educational content rather than an active demand."
            )
            return RiskAssessment("LOW", reasoning, [], limitations)
        # Benign/inconclusive contextual results do not veto strong rule evidence.
        if g_cls == "INCONCLUSIVE":
            reasoning.append(
                f"{label} could not classify the content reliably; the rule baseline still applies."
            )

    if inp.gemini_status != "completed":
        reasoning.append(
            f"{label} contextual analysis was not available for this request; "
            "this assessment relies on the rule-based content baseline."
        )

    serious_content = (
        inp.flags.get("ransom_or_kidnapping")
        or inp.flags.get("otp_or_credentials")
        or inp.flags.get("bank_impersonation")
    )
    threat = inp.flags.get("threat") or inp.flags.get("secrecy_or_urgency")

    if inp.transcription_status != "completed":
        reasoning.append("Speech transcription was unavailable, so content could not be verified.")

    # --- Policy ------------------------------------------------------------
    if inp.transcription_status != "completed" and inp.synthetic_suspected is not True:
        category = "INCONCLUSIVE"
        reasoning.append(
            "Content analysis produced no usable evidence, and voice analysis supplied no positive warning."
        )
        actions.append("Retry with a clearer recording, or verify the caller by calling back a known number.")
        limitations.append("No conclusion can be drawn from this recording; this is not a clean result.")
        return RiskAssessment(category, reasoning, actions, limitations)

    synthetic = inp.synthetic_suspected is True

    if serious_content and content_high and (inp.flags.get("payment_demand") or threat):
        # CRITICAL needs contextual evidence, never a keyword: either a
        # explicit threat, or a ransom
        # demand paired with instructions to stay silent / avoid the police.
        ransom_with_coercion = bool(
            inp.flags.get("ransom_or_kidnapping")
            and inp.flags.get("payment_demand")
            and inp.flags.get("secrecy_or_urgency")
        )
        category = "CRITICAL" if inp.flags.get("threat") or ransom_with_coercion else "HIGH"
    elif content_high:
        category = "HIGH"
    elif content_mid and (serious_content or threat):
        category = "HIGH" if serious_content else "MEDIUM"
    elif content_low and inp.discussion_context:
        category = "LOW"
        reasoning.append("Scam-related words appeared only in a discussion/awareness context.")
    elif content_low:
        # Calibration: 1-2/10 is usually one ordinary word in an everyday
        # sentence ("I sent my son money for books"). Treating that as MEDIUM
        # produced false alarms on ordinary family calls, which teaches people
        # to ignore the warning. From 3/10 there is at least one corroborated
        # suspicious pattern, which deserves a verify-first verdict.
        if inp.signal_strength >= 3:
            category = "MEDIUM"
            reasoning.append(
                "A corroborated suspicious pattern was found, but the surrounding "
                "context is not conclusive."
            )
        else:
            category = "LOW"
            reasoning.append(
                "A single ordinary phrase matched a scam pattern, with no supporting "
                "indicator; this is not treated as a scam signal."
            )
    elif content_mid and inp.flags.get("scam_category_present"):
        # A concrete fraud category was matched repeatedly (fee demand, courier
        # or customs pretext, remote-access request, job/prize lure) without the
        # ransom/OTP/bank-coercion flags. That is a real script, so it must not
        # fall through to LOW just because the "serious" flags are absent.
        category = "MEDIUM"
        reasoning.append(
            "The transcript matched a concrete fraud script, but the strongest "
            "coercion indicators were not present."
        )
    else:
        category = "LOW"

    if inp.discussion_context and category in ("CRITICAL", "HIGH") and inp.signal_strength < 7:
        category = "MEDIUM"
        reasoning.append(
            "Some wording resembled scam-awareness discussion, so the content signal was down-weighted."
        )

    if category in ("CRITICAL", "HIGH"):
        reasoning.append(
            "The transcript contains serious scam indicators (ransom, threats, OTP or bank impersonation)."
        )
        actions += [_ACTION_VERIFY, _ACTION_SECRETS, _ACTION_EMERGENCY]
        if inp.flags.get("secrecy_or_urgency"):
            actions.append("Do not follow instructions to keep the call secret or avoid the police.")
        actions.append(_ACTION_RELATIVE)
        actions.append(_ACTION_PRESERVE)
    elif category == "MEDIUM":
        reasoning.append("Some suspicious indicators were found, but the evidence is not conclusive.")
        actions += [
            _ACTION_SECRETS,
            "Confirm the caller's identity through an independent, trusted channel before acting.",
        ]
    elif category == "LOW":
        reasoning.append("No strong conversation-level scam indicators were found.")
        actions.append("Stay cautious: a clean result does not guarantee the call is genuine.")

    if synthetic:
        limitations.append("A synthetic-speech detector can miss advanced voice clones.")
    if inp.speaker_outcome in ("inconclusive", "reference_unavailable", None):
        limitations.append(
            "Voice similarity cannot prove identity; a real family member's voice can also be used in a scam."
        )
    limitations.append(
        "A suspicious transcript does not by itself prove a crime occurred. Treat "
        "findings as evidence to verify, not proof."
    )

    return RiskAssessment(category, reasoning, actions, limitations)
