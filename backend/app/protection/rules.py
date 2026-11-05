"""Multilingual scam / ransom / family-emergency rule engine.

Deterministic, explainable layer (English, Hindi, Hinglish) that finds the
spans supporting each scam category and derives contextual signals. It is
deliberately conservative: keywords alone never classify a call as fraud —
combination logic and discussion/awareness detection guard against false
positives (e.g. a scam-awareness conversation mentioning "kidnap" and "OTP").

An optional LLM classifier hook can refine results when configured, but this
module never fabricates findings and never follows instructions found inside
a transcript.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Patterns. Each entry: (regex, human label). Regexes run on the normalized
# transcript (lowercased). Devanagari and romanized Hindi/Hinglish included.
# ---------------------------------------------------------------------------

_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "ransom_or_kidnapping": [
        (r"(बेटा|बेटी|लड़का|लड़की|बच्चा).{0,25}हमारे पास", "relative held (Hindi)"),
        (r"(पकड़|उठा) लिया", "abduction claim (Hindi)"),
        (r"नहीं छोड़ेंगे|छोड़ देंगे नहीं", "will not release (Hindi)"),
        (r"\bkidnap(ped)? (kar|kar liya|kar diya)\b", "kidnapping claim (Hinglish)"),
        (r"\bhumne\b.{0,30}\bkidnap\b", "kidnapper speaks in first person (Hinglish)"),
        (r"\bpakad liya\b|\butha liya\b|\bagawa kar\b", "abduction claim (Hinglish)"),
        (r"\bchhodenge nahi\b|\bchhod denge nahi\b|\bnahi chhodenge\b", "will not release (Hinglish)"),
        (r"\bbete ko\b|\bbeti ko\b|\bbete ko hum\b", "relative held (Hinglish)"),
        (r"\byour (son|daughter|child) is with us\b", "relative held (English)"),
        (r"\b(paise|paisa|rupaye|rupees?) (bhejo|bhej do|transfer karo)\b", "demands money (Hinglish)"),
        (r"\bwarna\b.{0,40}\b(maar|hurt|chhod|police)\b", "conditional threat (Hinglish)"),
        (r"\bpolice ko (bataya|bataoge|batana)\b", "forbids contacting police (Hinglish)"),
        (r"\bbura hoga\b|\banjaam\b", "threat of consequences (Hinglish)"),

        (r"\bkidnap(?:ped|ping)?\b", "kidnap"),
        (r"\bhostage\b", "hostage"),
        (r"\bransom\b", "ransom"),
        (r"\bfirou?ti\b|\bfiroti\b", "extortion/firoti"),
        (r"अगवा", "kidnapping (Hindi)"),
        (r"अपहरण", "abduction (Hindi)"),
        (r"फिरौती", "ransom (Hindi)"),
        (r"बंधक", "hostage (Hindi)"),
        (r"\bbandhak\b|\bbandhak bana\b", "hostage (Hinglish)"),
    ],
    "family_emergency_impersonation": [
        (r"\bi am your (son|daughter|father|mother|brother|sister)\b", "claims to be a relative"),
        (r"\byour (son|daughter|father|mother|brother|sister) (is|has)\b", "speaks as/about a relative"),
        (r"\bpapa\b|\bmummy\b|\bpapa,?\s", "family address term (papa/mummy)"),
        (r"\bbeta\b|\bbeti\b", "family address term (beta/beti)"),
        (r"\bmain (?:tumhara|aapka|tera) (?:beta|ladka|bhai)\b", "claims to be your son (Hinglish)"),
        (r"(?:मैं|मैं तो) (?:आपका|तेरा|तुम्हारा) (?:बेटा|लड़का|भाई)", "claims to be your son (Hindi)"),
        (r"मैं आपका बेटा|बेटा", "son (Hindi)"),
        (r"बेटी", "daughter (Hindi)"),
        (r"बेटे|बेटों|लड़के|पत्नी|पति", "relative (Hindi, oblique)"),
        (r"\baccident\b|\bhospital\b|\bicu\b", "accident/hospital"),
        (r"दुर्घटना|अस्पताल", "accident/hospital (Hindi)"),
        (r"\bmusibat\b|\bmushkil\b|\bproblem mein\b", "in trouble (Hinglish)"),
        (r"मुसीबत|परेशानी|खतरे में", "in trouble (Hindi)"),
    ],
    "emergency_money_request": [
        (r"(पैसे|पैसा|रुपये|रुपए)\s*(भेजो|भेज दो|भेजिए|ट्रांसफर)", "demands money (Hindi)"),
        (r"(turant|abhi)\s*(paise|rupaye|rupees)", "urgent money (Hinglish)"),
        (
            r"\b(?:send|transfer|pay|wire|deposit) (?:me )?(?:the |₹|rs\.? )?\d[\d,]*",
            "urgent money demand with amount",
        ),
        (r"\bpaise (?:bhejo|bhej do|transfer karo|de do)\b", "send money (Hinglish)"),
        (r"पैसे (?:भेजो|भेज दो|ट्रांसफर करो|दो)\b", "send money (Hindi)"),
        (r"\bimmediately\b|\bright now\b|\babhi\b|\bturant\b|जल्दी|तुरंत|अभी", "urgency wording"),
        (r"\b(?:50|50000|50,000|one lakh|1 lakh)\b", "amount mention"),
    ],
    "threats_or_coercion": [
        (r"\bbura hoga\b|\banjaam bura\b", "threat of consequences (Hinglish)"),
        (r"\bmaar denge\b|\bmaar dalenge\b|\bjan se maar\b", "threat of violence (Hinglish)"),
        (r"\bthey will hurt (?:me|you)\b|\bwill kill\b|\bwon't see (?:me|him|her) again\b", "threat of harm"),
        (
            r"\byou will be arrested\b|\bwarrant (?:has been|is) issued\b|\byou will be sorry\b|"
            r"\blegal action will be taken\b|गिरफ्तार हो जाएगा",
            "threat of arrest or consequences",
        ),
        (r"मार डालेंगे|जान से मार|नुकसान", "threat of harm (Hindi)"),
        (r"मार देंगे|चोट पहुँचा|अंजाम बुरा|सब खत्म", "threat of harm (Hindi, common forms)"),
        (r"\bwarna\b|वरना", "or else (conditional threat)"),
        (r"मेरे साथ कुछ हो जाएगा|कुछ हो जाएगा", "something will happen to me (Hindi)"),
    ],
    "police_or_government_impersonation": [
        (
            r"\bi am (?:a )?police officer\b|\bthis is the police\b|\bcyber crime (?:branch|cell)\b|"
            r"\bcbi officer\b|\bincome tax department\b|\bcustoms officer\b|"
            r"पुलिस अधिकारी",
            "claims to be police or government",
        ),
    ],
    "suspicious_secrecy_or_urgency": [
        (r"पुलिस को (मत )?बता", "forbids contacting police (Hindi)"),
        (r"(kisi ko )?mat batana|kisi ko na batao", "demands secrecy (Hinglish)"),
        (r"(जल्दी|तुरंत|अभी)\s*(करो|कर|आना|भेजो)", "urgency (Hindi)"),
        (
            r"\bdo(?:n'?t| not) tell (?:anyone|anybody|any person|your)\b|\bkeep "
            "(?:it|this) (?:secret|quiet)\b",
            "secrecy instruction",
        ),
        (r"\bkisi ko mat batana\b|\bkisi ko na batao\b", "secrecy instruction (Hinglish)"),
        (r"\bkisi ko mat bata\b|\bkisi ko mat batao\b", "secrecy instruction (Hinglish)"),
        (r"किसी को मत बताना|किसी को नहीं बताना", "secrecy instruction (Hindi)"),
        (r"\bdo(?:n'?t| not) (?:call|inform|contact) the police\b|\bno police\b", "avoid police"),
        (r"\bpolice ko (?:phone |call )?mat kar", "avoid police (Hinglish)"),
        (r"पुलिस को (?:फोन |कॉल )?मत कर", "avoid police (Hindi)"),
        (r"\bdon'?t hang up\b|\bstay on the line\b", "keep the victim on the line"),
    ],
    "otp_or_credential_theft": [
        (r"\b(?:otp|o\.t\.p)\b", "OTP mention"),
        (r"\bone ?time password\b", "OTP mention"),
        (r"ओटीपी", "OTP (Hindi)"),
        (
            r"\b(?:tell|share|read|give) (?:me )?(?:the |your )?(?:otp|code|pin|password)\b",
            "asks for OTP/PIN/password",
        ),
        (r"\b(?:otp|pin|password) (?:batao|bata do|share karo|bolo)\b", "asks for OTP (Hinglish)"),
        (r"(?:ओटीपी|पिन|पासवर्ड) (?:बताओ|बता दो|बता दीजिए)", "asks for OTP (Hindi)"),
        (r"\bshare (?:your )?screen\b|\banydesk\b|\bteamviewer\b", "remote-access request"),
    ],
    "bank_account_or_card_fraud": [
        (r"\byour (?:account|card) (?:will be|is) (?:blocked|suspended|frozen)\b", "account-blocking threat"),
        (r"\baccount (?:block|band) ho jayega\b", "account-blocking threat (Hinglish)"),
        (r"आपका (?:खाता|अकाउंट|कार्ड) (?:बंद|ब्लॉक) हो जाएगा", "account-blocking threat (Hindi)"),
        (r"\bbank (?:se|se baat|officer|manager)\b", "bank impersonation"),
        (r"\bcvv\b|\bcard number\b|\bdebit card\b|\bcredit card\b", "card data request"),
    ],
    "upi_or_payment_fraud": [
        (r"\bupi\b|\bgpay\b|\bphonepe\b|\bpaytm\b", "UPI app mention"),
        (r"\bupi (?:kar do|karo|kardo)\b", "UPI payment demand (Hinglish)"),
        (r"यूपीआई|यूपीआई कर दो", "UPI (Hindi)"),
        (r"\bscan (?:this )?qr\b|\bqr code\b", "QR code payment"),
        (r"\b(?:send|transfer) (?:money|amount|₹|rs)\b", "money transfer demand"),
    ],
    "investment_or_crypto_scam": [
        (r"\b(?:crypto|bitcoin|usdt|ethereum)\b", "cryptocurrency"),
        (r"\bdouble your (?:money|investment)\b|\bguaranteed returns?\b", "guaranteed-returns claim"),
        (r"\binvest(?:ment)? (?:now|today)\b", "investment urgency"),
    ],
    "blackmail_or_extortion": [
        (r"\bblackmail\b|\bextort(?:ion)?\b", "blackmail/extortion"),
        (r"\b(?:have|got) your (?:photos|videos|recording)\b", "compromising-material threat"),
        (r"\b(?:video|photo) (?:viral|leak) kar dunga\b", "leak threat (Hinglish)"),
        (r"(?:वीडियो|फोटो) (?:वायरल|लीक) कर दूंगा", "leak threat (Hindi)"),
    ],
    "phishing_or_identity_theft": [
        (r"\bclick (?:this )?link\b|\bverify your (?:account|identity)\b", "phishing link/verify"),
        (r"\baadhaar\b|\bpan card\b|\bkyc\b", "KYC/identity documents"),
        (r"\bkyc (?:update|complete) kar\b", "KYC demand (Hinglish)"),
    ],
}

# Words that signal the speaker is *discussing* scams (news, awareness,
# warning others) rather than performing one. They reduce, never inflate risk.
_DISCUSSION_MARKERS = re.compile(
    r"\b(?:scam awareness|fraud awareness|beware of|stay safe|cyber crime|"
    r"news|report(?:ed|ing)?|police (?:said|says|have|are advising)|"
    r"advisory|training|webinar|example of|according to|"
    r"in this video|we explain|explains how|fraudsters|scammers|con artists|"
    r"in the film|movie|only a scene|fictional|"
    r"dhokhadhadi|dhokha|savdhan|jagruk|jaagruk|"
    r"धोखाधड़ी|सावधान|जागरूक|समाचार|रिपोर्ट)\b"
)

# Second-person imperative demands: the strongest first-person scam signal.
_IMPERATIVE_DEMAND = re.compile(
    r"\b(?:send|transfer|pay|share|tell|read|give|bhejo|bhej do|karo|kardo|"
    r"batao|bata do|kar do|भेजो|भेज दो|करो|कर दो|बताओ|बता दो)\b"
)

# First-person framing: the hallmark of an actual scam call vs. a discussion.
_FIRST_PERSON = re.compile(r"\b(?:i|me|my|i'm|i am|mujhe|mera|meri|main|मैं|मुझे|मेरा|मेरी)\b")

_SEVERITY = {
    "ransom_or_kidnapping": 3,
    "blackmail_or_extortion": 3,
    "otp_or_credential_theft": 3,
    "bank_account_or_card_fraud": 2,
    "family_emergency_impersonation": 2,
    "emergency_money_request": 2,
    "threats_or_coercion": 2,
    "upi_or_payment_fraud": 2,
    "suspicious_secrecy_or_urgency": 2,
    "investment_or_crypto_scam": 1,
    "phishing_or_identity_theft": 1,
    "police_or_government_impersonation": 3,
}


@dataclass
class Finding:
    category: str
    label: str
    severity: int
    matches: list[dict] = field(default_factory=list)  # {phrase, start, end}


@dataclass
class ScamAnalysis:
    """Result of the deterministic content analysis."""

    categories: list[Finding]
    flags: dict[str, bool]
    discussion_context: bool
    imperative_demand: bool
    signal_strength: int  # 0..10, transparent sum of severities (not a probability)
    summary: str
    primary_category: str
    limitations: list[str]
    evidence_quotes: list[str] = field(default_factory=list)


_RANSOM_HOLD = re.compile(
    r"(with us|hamare paas|humne .{0,25}kidnap|pakad liya|bandhak|abduct|agawaa|agwa)",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace. Keeps Devanagari intact."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _evidence(norm: str, findings: list[Finding], pad: int = 60) -> list[str]:
    """Exact transcript slices (verbatim, from stored offsets) per finding."""
    quotes: list[str] = []
    for f in findings:
        for match in f.matches:
            start, end = match.get("start"), match.get("end")
            if not isinstance(start, int) or not isinstance(end, int):
                phrase = str(match.get("phrase") or "").lower()
                start = norm.find(phrase)
                if start < 0:
                    continue
                end = start + len(phrase)
            snippet = norm[max(0, start - pad) : min(len(norm), end + pad)].strip()
            if snippet and snippet not in quotes:
                quotes.append(snippet)
    return quotes[:5]


def analyze_transcript(text: str) -> ScamAnalysis:
    """Analyze a transcript. Returns findings with exact matched spans."""
    norm = normalize(text)
    findings: list[Finding] = []
    limitations: list[str] = []

    if not norm:
        return ScamAnalysis(
            categories=[],
            flags={},
            discussion_context=False,
            imperative_demand=False,
            signal_strength=0,
            summary="No transcript text was available, so content-based scam signals could not be assessed.",
            primary_category="insufficient_evidence",
            limitations=["Transcript is empty; content analysis requires recognizable speech."],
        )

    for category, patterns in _PATTERNS.items():
        matches: list[dict] = []
        for pattern, label in patterns:
            for m in re.finditer(pattern, norm):
                matches.append(
                    {"phrase": m.group(0).strip(), "label": label, "start": m.start(), "end": m.end()}
                )
        if matches:
            # Deduplicate identical spans.
            seen, unique = set(), []
            for m in matches:
                key = (m["start"], m["end"])
                if key not in seen:
                    seen.add(key)
                    unique.append(m)
            findings.append(
                Finding(
                    category=category,
                    label=category.replace("_", " "),
                    severity=_SEVERITY.get(category, 1),
                    matches=unique[:12],
                )
            )

    present = {f.category for f in findings}
    discussion = bool(_DISCUSSION_MARKERS.search(norm))
    imperative = bool(_IMPERATIVE_DEMAND.search(norm))

    flags = {
        "family_claim": "family_emergency_impersonation" in present,
        "ransom_or_kidnapping": "ransom_or_kidnapping" in present,
        "payment_demand": "emergency_money_request" in present or "upi_or_payment_fraud" in present,
        "threat": "threats_or_coercion" in present,
        "secrecy_or_urgency": "suspicious_secrecy_or_urgency" in present,
        "otp_or_credentials": "otp_or_credential_theft" in present,
        "bank_impersonation": "bank_account_or_card_fraud" in present,
        "discussion_context": discussion,
        "imperative_demand": imperative,
    }

    # Contextual combination logic: raise for genuine first-person scam
    # structure; dampen when the speech looks like awareness/news discussion.
    strength = sum(f.severity for f in findings)
    if flags["family_claim"] and flags["ransom_or_kidnapping"] and flags["payment_demand"]:
        strength += 3
    if flags["ransom_or_kidnapping"] and flags["threat"]:
        strength += 2
    if flags["family_claim"] and flags["payment_demand"]:
        strength += 2
    if flags["otp_or_credentials"] and (flags["bank_impersonation"] or flags["payment_demand"]):
        strength += 2
    if flags["secrecy_or_urgency"] and flags["payment_demand"]:
        strength += 2
    first_person = bool(_FIRST_PERSON.search(norm))
    if discussion and not (imperative and first_person):
        strength = max(0, strength - 6)
        limitations.append(
            "The transcript reads like scam-awareness/news discussion rather than a live call; "
            "keyword hits were strongly down-weighted."
        )
    # A family address term alone ("Hi Papa, I reached college") is normal speech.
    # Impersonation needs at least one coercion/payment signal alongside it,
    # otherwise every ordinary call to a parent would be labelled a scam.
    family_supported = bool(
        flags["payment_demand"]
        or flags["threat"]
        # Uses the custody regex rather than the ransom flag: by this point a
        # keyword-only ransom flag may already have been withdrawn.
        or bool(_RANSOM_HOLD.search(norm))
        or flags["secrecy_or_urgency"]
        or flags["otp_or_credentials"]
        # A stated accident/hospital/trouble is the classic opening of a
        # relative-impersonation call and keeps the signal even before money is
        # mentioned.
        or any(
            marker in norm
            for marker in (
                "hospital", "accident", "icu", "musibat", "mushkil", "problem mein",
                "अस्पताल", "दुर्घटना", "मुसीबत", "परेशानी",
            )
        )
    )
    if flags["family_claim"] and not family_supported:
        findings = [f for f in findings if f.category != "family_emergency_impersonation"]
        flags["family_claim"] = False
        strength = sum(f.severity for f in findings)
        limitations.append(
            "A family address term was used without any payment, threat or urgency "
            "signal, which is normal conversation; it was not treated as impersonation."
        )
    strength = min(10, strength)

    # A kidnapping/ransom label needs contextual support (a demand, a threat,
    # secrecy, or an explicit "we have them" claim). A stray keyword alone must
    # never produce that category or a critical warning.
    ransom_supported = bool(
        flags["payment_demand"] or flags["threat"] or flags["secrecy_or_urgency"]
    ) or bool(_RANSOM_HOLD.search(norm))
    if "ransom_or_kidnapping" in present and not ransom_supported:
        findings = [f for f in findings if f.category != "ransom_or_kidnapping"]
        flags["ransom_or_kidnapping"] = False
        strength = sum(f.severity for f in findings)
        limitations.append(
            "A kidnapping-related word appeared without any demand, threat or secrecy "
            "context, so it was not treated as a ransom attempt."
        )

    if strength == 0:
        primary = "legitimate_or_no_clear_scam_evidence"
    else:
        primary = max(findings, key=lambda f: (f.severity, len(f.matches))).category

    if strength >= 7:
        summary = "Multiple serious scam indicators were found in the transcript."
    elif strength >= 4:
        summary = "Several scam-related indicators were found in the transcript."
    elif strength >= 1:
        summary = "Weak scam-related signals were found; context does not confirm fraud."
    else:
        summary = "No clear scam indicators were found in the transcript."

    limitations.append(
        "Content analysis is rule-based and explainable; it can miss unusual "
        "phrasing and does not prove criminal intent."
    )
    return ScamAnalysis(
        categories=findings,
        flags=flags,
        discussion_context=discussion,
        imperative_demand=imperative,
        signal_strength=strength,
        summary=summary,
        primary_category=primary,
        limitations=limitations,
        evidence_quotes=_evidence(norm, findings),
    )
