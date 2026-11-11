"""Generate a scam/hard-negative corpus for training (not for evaluation).

The hand-authored evaluation set stays small and untouched by training: it is
the yardstick. This generator produces the *volume* a model needs by expanding
labelled scenario templates with slot fills (names, relations, amounts,
urgency, language) in English, Hindi and Hinglish, plus benign hard negatives
that deliberately contain tempting keywords.

Honesty rules:
  * every row records the template and slots that produced it, so duplicate or
    leaking rows can be traced;
  * the evaluation sets in `tests/` are never emitted here;
  * labels are authored, not model output.

Run: python -m scripts.generate_scam_corpus --per-template 8
"""

# ruff: noqa: E501  (scenario templates are data: one line each, easier to review)

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

NAMES = ["beta", "Rohit", "Anjali", "Papa", "Mummy", "bhai", "didi", "uncle"]
RELATIONS = ["son", "daughter", "beta", "bhai"]
AMOUNTS = ["5,000", "10,000", "20,000", "25,000", "50,000", "80,000", "1 lakh", "2 lakh"]
URGENCY = ["abhi", "turant", "right now", "immediately", "jaldi", "within 10 minutes"]
ACCOUNTS = ["SBI account", "account number", "UPI ID", "Google Pay number", "Paytm wallet"]

# (template, language, suspicious, categories)
SCAM_TEMPLATES: list[tuple[str, str, list[str]]] = [
    (
        "{name}, it's me. I'm in trouble, send {amount} rupees {urgency}. Don't tell anyone, don't call the police.",
        "en",
        ["family_emergency_impersonation"],
    ),
    (
        "{name} main hoon, accident ho gaya, hospital mein hoon. {amount} {urgency} bhejo, papa ko mat batana.",
        "hi-en",
        ["family_emergency_impersonation"],
    ),
    (
        "Papa mujhe kidnap kar liya hai, {amount} {urgency} bhejo warna hum nahi chhodenge. Police ko mat batana.",
        "hi-en",
        ["ransom_or_kidnapping"],
    ),
    (
        "मैं सीबीआई से बोल रहा हूँ, आपके नाम पर {amount} का हवाला मिला है, {urgency} भेजो वरना गिरफ्तारी होगी।",
        "hi",
        ["police_or_government_impersonation"],
    ),
    (
        "Main cyber cell se bol raha hoon. Aapke Aadhaar par parcel mein drugs mile hain. Digital arrest ho gaya, {amount} settlement bhejo {urgency}.",
        "hi-en",
        ["police_or_government_impersonation"],
    ),
    (
        "Sir bank se bol raha hoon, aapka account block ho jayega. {urgency} OTP batao warna card suspend ho jayega.",
        "hi-en",
        ["otp_or_credential_theft"],
    ),
    (
        "Aapka KYC expire ho gaya hai, is link par update karo warna account freeze ho jayega, {urgency}.",
        "hi-en",
        ["bank_account_or_card_fraud"],
    ),
    (
        "Refund ke liye collect request accept karo aur QR scan karo, {amount} {urgency} aa jayega, {{account}} details bhejo.",
        "hi-en",
        ["upi_or_payment_fraud"],
    ),
    (
        "Telegram group join karo, task complete karo, per task commission milega. {amount} invest karo, profit pakka.",
        "hi-en",
        ["investment_or_crypto_scam"],
    ),
    (
        "Guaranteed returns! Double your money in 30 days. Join our trading plan and invest {amount} {urgency}.",
        "en",
        ["investment_or_crypto_scam"],
    ),
    (
        "Aapki lottery lagi hai {amount}, processing fee {amount} {urgency} bhejo is account par.",
        "hi-en",
        ["job_or_prize_scam"],
    ),
    (
        "I have your video. Send {amount} {urgency} or I will leak it and make it viral. Don't tell anyone.",
        "en",
        ["blackmail_or_extortion"],
    ),
    (
        "तुम्हारी वीडियो मेरे पास है, {amount} {urgency} भेजो वरना वायरल कर दूंगा।",
        "hi",
        ["blackmail_or_extortion"],
    ),
    (
        "This is Microsoft support. Your computer has a virus, install remote access and pay {amount} {urgency} with a gift card.",
        "en",
        ["tech_support_scam"],
    ),
    (
        "Aapki bijli {urgency} disconnect ho jayegi, is number par {amount} bill pay karo warna connection band ho jayega.",
        "hi-en",
        ["emergency_money_request"],
    ),
    (
        "Aapka loan approve ho gaya hai, processing fee {amount} {urgency} bhejo warna CIBIL kharab ho jayega.",
        "hi-en",
        ["job_or_prize_scam"],
    ),
    (
        "Indian Army bharti ke liye {amount} bhejo, seat confirm karni hai, {urgency} karo.",
        "hi-en",
        ["job_or_prize_scam"],
    ),
    (
        "Humne tumhara accident video record kiya hai, {amount} bhejo warna police complaint kar dunga.",
        "hi-en",
        ["blackmail_or_extortion"],
    ),
    (
        "Credit card ke reward points expire ho rahe hain, link par CVV aur card number daalo {urgency}.",
        "hi-en",
        ["otp_or_credential_theft"],
    ),
    (
        "Supreme Court notice hai, final warning, {amount} {urgency} bhejo warna warrant jari ho jayega.",
        "hi-en",
        ["threats_or_coercion"],
    ),
    (
        "Main police station se bol raha hoon, tumhare {relation} ko giraftar kar liya hai, {amount} bhejo warna case file kar denge.",
        "hi-en",
        ["police_or_government_impersonation"],
    ),
    (
        "Your {relation} is with us. Arrange {amount} rupees if you want to see them again. No police.",
        "en",
        ["ransom_or_kidnapping"],
    ),
    (
        "Account number par {amount} {urgency} bhej do, warna bura hoga, kisi ko mat batana.",
        "hi-en",
        ["emergency_money_request"],
    ),
    (
        "Work from home job, sirf mobile se 2 ghante, daily {amount} earning, registration fee {amount} bhejo {urgency}.",
        "hi-en",
        ["job_or_prize_scam"],
    ),
]

# Benign rows that carry scam vocabulary in a harmless context.
BENIGN_TEMPLATES: list[tuple[str, str]] = [
    ("{name}, I reached college safely, will call you at night.", "en"),
    ("{name}, I had a small accident on the way. I'm fine, the doctor checked me. I'll call from the hospital later.", "en"),
    ("The bank called to say my KYC is complete, no action needed.", "en"),
    ("I received a scam call yesterday about a parcel with drugs, I reported it on the cyber crime portal.", "en"),
    ("News report: cyber police arrested six people for digital arrest scams in Noida.", "en"),
    ("In this video we explain how scammers say OTP batao to steal money, so never share it with anyone.", "hi-en"),
    ("Your reports are normal, take the tablets for five days and come back next week.", "en"),
    ("I'm at your gate with the parcel, please come down.", "en"),
    ("The client wants the invoice before Friday, please send it to their accounts team.", "en"),
    ("I'll pay you back the {amount} rupees I borrowed yesterday.", "en"),
    ("{name} main traffic mein phas gaya hoon, dinner ke liye late ho jaunga.", "hi-en"),
    ("{name}, I lost my wallet near the station, I'll file a complaint at the police station tomorrow.", "en"),
    ("Please transfer the rent by the 5th as usual, and send me the receipt.", "hi-en"),
    ("My son called from his hostel, he needs money for books, I sent it on UPI within a minute.", "en"),
    ("आजकल बहुत घोटाले हो रहे हैं, किसी को ओटीपी मत बताओ।", "hi"),
    ("Remember, never share your OTP with anyone, the bank never asks for it.", "en"),
    ("Parents, please note the exam schedule is on the notice board.", "en"),
    ("HR said the salary will be credited tonight, check your account tomorrow.", "en"),
    ("UPI is working now, you can scan the QR at the counter and pay.", "hi-en"),
    ("मम्मी, मैं होस्टल पहुँच गया हूँ, खाना खा लिया, आप घबराओ मत।", "hi"),
    ("पापा, मेरी फीस जमा करनी है, कॉलेज ने लिंक भेजा है, कल तक भर देना।", "hi"),
    ("Doctor said the insurance will cover the {amount} bill, so no payment is needed from us.", "en"),
]


def _fill(template: str, rng: random.Random) -> tuple[str, dict]:
    slots = {
        "name": rng.choice(NAMES),
        "relation": rng.choice(RELATIONS),
        "amount": rng.choice(AMOUNTS),
        "urgency": rng.choice(URGENCY),
    }
    text = template.format(**slots).replace("{account}", rng.choice(ACCOUNTS))
    return text, slots


def _row_id(text: str) -> str:
    return "g" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-template", type=int, default=8)
    parser.add_argument("--out", default="data/corpus/scam_corpus.jsonl")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict] = []
    seen: set[str] = set()

    for template, lang, categories in SCAM_TEMPLATES:
        for _ in range(args.per_template):
            text, slots = _fill(template, rng)
            rid = _row_id(text)
            if rid in seen:
                continue
            seen.add(rid)
            rows.append(
                {
                    "id": rid,
                    "text": text,
                    "language": lang,
                    "expect_suspicious": True,
                    "expected_categories": categories,
                    "group": "scam",
                    "template": template,
                    "slots": slots,
                    "source": "template-generator",
                }
            )

    for template, lang in BENIGN_TEMPLATES:
        for _ in range(args.per_template):
            text, slots = _fill(template, rng)
            rid = _row_id(text)
            if rid in seen:
                continue
            seen.add(rid)
            rows.append(
                {
                    "id": rid,
                    "text": text,
                    "language": lang,
                    "expect_suspicious": False,
                    "expected_categories": [],
                    "group": "benign",
                    "template": template,
                    "slots": slots,
                    "source": "template-generator",
                }
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    scams = sum(1 for r in rows if r["expect_suspicious"])
    print(json.dumps({"out": str(out), "rows": len(rows), "scam": scams, "benign": len(rows) - scams,
                      "templates": len(SCAM_TEMPLATES) + len(BENIGN_TEMPLATES)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
