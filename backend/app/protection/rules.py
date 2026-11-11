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
        (r"\byour (son|daughter|child|brother|sister|father|mother) is with us\b",
         "relative held (English)"),
        (r"\byour (bhai|behen|beta|beti|maa|papa|dost) is with us\b",
         "relative held (Hinglish)"),
        (r"\b(he|she|they)( is| are)? with us\b.{0,60}\b(rupees|lakh|pay|money)\b",
         "custody claim with a payment demand"),

        (r"\bkidnap(?:ped|ping)?\b", "kidnap"),
        (r"\bhostage\b", "hostage"),
        (r"\bransom\b", "ransom"),
        (r"\bfirou?ti\b|\bfiroti\b", "extortion/firoti"),
        (r"अगवा", "kidnapping (Hindi)"),
        (r"अपहरण", "abduction (Hindi)"),
        (r"फिरौती", "ransom (Hindi)"),
        (r"बंधक", "hostage (Hindi)"),
        (r"\bbandhak\b|\bbandhak bana\b", "hostage (Hinglish)"),
        # Romanised custody claim: "Beti hamare paas hai". The Devanagari form
        # was already covered; Hinglish is how most such calls are transcribed.
        (r"\b(beta|beti|bete|bacche|bachche|ladka|ladki|bhai|behen|maa|papa)\b"
         r".{0,25}\bhamare paas\b", "relative held (Hinglish, romanised)"),
        # "Don't go to the police, just pay" is the extortion script, not an
        # ordinary money demand. Requiring the police element keeps a plain
        # "transfer now or else" call in the money/threat categories where it
        # belongs, so the report never invents a kidnapping claim.
        (r"(police ko (?:bataya|bataoge|batana|mat bata|na bata))"
         r".{0,70}(paise|paisa|rupaye|rupees|lakh|transfer|bhejo)",
         "police threat with a money demand"),
        (r"(paise|paisa|rupaye|rupees|lakh|transfer|bhejo)"
         r".{0,70}(police ko (?:bataya|bataoge|batana|mat bata|na bata))",
         "money demand with a police threat"),
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
        (r"\b(?:processing|clearance|registration|settlement|handling) (?:fee|charge|paisa|amount)\b", "upfront fee demand"),  # noqa: E501
        (r"\b(?:bijli|electricity|water|gas|connection)\b.{0,30}\b(?:disconnect|band|cut|kati)\b", "utility disconnection pressure"),  # noqa: E501

        (r"(पैसे|पैसा|रुपये|रुपए)\s*(भेजो|भेज दो|भेजिए|ट्रांसफर)", "demands money (Hindi)"),
        (r"(turant|abhi)\s*(paise|rupaye|rupees)", "urgent money (Hinglish)"),
        (
            r"\b(?:send|transfer|pay|wire|deposit) (?:me )?(?:the |₹|rs\.? )?\d[\d,]*",
            "urgent money demand with amount",
        ),
        (r"\bpaise (?:bhejo|bhej do|transfer karo|de do)\b", "send money (Hinglish)"),
        (r"\barrange\b.{0,24}\b(rupees?|lakh|crore|amount|paise|cash)\b",
         "arrange-money demand"),
        (r"\b(rupees?|lakh|crore)\b.{0,40}\b(if you want to see)\b",
         "payment tied to seeing the relative again"),
        (r"पैसे (?:भेजो|भेज दो|ट्रांसफर करो|दो)\b", "send money (Hindi)"),
        (r"\bimmediately\b|\bright now\b|\babhi\b|\bturant\b|जल्दी|तुरंत|अभी", "urgency wording"),
        (r"\b(?:50|50000|50,000|one lakh|1 lakh)\b", "amount mention"),
    ],
    "threats_or_coercion": [
        (r"\b(?:supreme court|court|high court) (?:notice|summons|order)\b", "court-notice pressure"),
        (r"\bwarrant (?:jari|issued|nikal)\b", "warrant threat"),
        (r"\b(?:last|final) (?:warning|notice)\b", "final warning"),

        (r"\bbura hoga\b|\banjaam bura\b", "threat of consequences (Hinglish)"),
        (r"\bif you want to see (them|him|her) again\b", "conditional release threat"),
        (r"\b(them|him|her) again\b.{0,20}\b(never|nahi)\b", "threat of never seeing them again"),
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
        (r"\banjaam\b", "threat of consequences (Hinglish)"),
        (r"मेरे साथ कुछ हो जाएगा|कुछ हो जाएगा", "something will happen to me (Hindi)"),
        (r"\bjaan se maar\b|\bmaar dunga\b|\bdekh lena\b", "physical threat (Hinglish)"),
        (r"\b(?:legal action|arrest|warrant|case) (?:ho jayega|kar denge|file kar)\b", "legal-action threat"),  # noqa: E501
        (r"\b(?:last warning|final notice|warna)\b", "final-warning pressure"),
        (r"\b(?:police complaint|fir) kar dunga\b|\b(?:police complaint|fir) kar denge\b", "complaint threat"),  # noqa: E501
    ],
    "police_or_government_impersonation": [
        (r"\b(?:police|cyber ?cell|crime branch|narcotics|cbi|ed)\b.{0,12}\b(?:se|station se)\b.{0,12}\b(?:bol|call|baat|hoon)\b", "calls from a police desk"),  # noqa: E501
        (r"\bmain\b.{0,25}\b(?:police|inspector|officer|sub-?inspector)\b", "says he is police"),

        (
            r"\bi am (?:a )?police officer\b|\bthis is the police\b|\bcyber crime (?:branch|cell)\b|"
            r"\bcbi officer\b|\bincome tax department\b|\bcustoms officer\b|"
            r"पुलिस अधिकारी",
            "claims to be police or government",
        ),
        (r"\bdigital arrest\b|\bडिजिटल अरेस्ट\b", "digital arrest script"),
        (r"\b(?:cbi|ncb|ed|customs|narcotics|trai|income tax|cyber (?:crime )?(?:branch|cell))\b", "agency named as caller"),  # noqa: E501
        (r"\b(?:main|hum) (?:police|inspector|officer) (?:se|se bol|bol raha|hoon)\b", "claims to be police"),  # noqa: E501
        (r"\b(?:police|crime branch|cyber cell) (?:se|station se) (?:bol|call|baat)\b", "claims call from police"),  # noqa: E501
        (r"\b(?:aapke|tumhare|your) naam (?:par|pe|se) (?:parcel|courier|package|sim)\b", "parcel/sim in your name"),  # noqa: E501
        (r"\b(?:parcel|courier|package)\b.{0,40}\b(?:drugs?|narcotics|ganja|illegal|contraband)\b", "illegal goods in parcel"),  # noqa: E501
        (r"\b(?:money laundering|hawala|benami)\b", "money-laundering accusation"),
        (r"\barrest warrant\b|\bgiraftar(?:i)? (?:ho|kar|warrant)\b", "arrest threat"),
        (r"\b(?:aadhaar|aapka naam)\b.{0,30}\b(?:crime|case|fir)\b", "names you in a case"),
        (r"गिरफ्तारी|वारंट|डिजिटल|सीबीआई|पार्सल", "police/case (Hindi)"),
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
        (r"\bpolice ko (bataya|bataoge|batana)\b|police ko (matt? )?bata",
         "forbids contacting police"),
        (r"किसी को मत बताना|किसी को नहीं बताना", "secrecy instruction (Hindi)"),
        (r"\bdo(?:n'?t| not) (?:call|inform|contact) the police\b|\bno police\b", "avoid police"),
        (r"\bpolice ko (?:phone |call )?mat kar", "avoid police (Hinglish)"),
        (r"पुलिस को (?:फोन |कॉल )?मत कर", "avoid police (Hindi)"),
        (r"\bdon'?t hang up\b|\bstay on the line\b", "keep the victim on the line"),
        (r"\b(?:before|within)\b.{0,15}\b(?:\d{1,2}\s*(?:pm|am|minutes?|hours?)|today|tonight)\b", "deadline pressure"),  # noqa: E501
        (r"\b(?:account|number) (?:band|block|freeze) ho jayega\b", "account-loss deadline"),
        (r"\b(?:abhi|turant|jaldi) (?:bhejo|transfer karo|karo|batao)\b", "immediate action demand"),
        (r"\b(?:kisi ko (?:mat|na) bata\w*|mat batana)\b", "secrecy demand (Hinglish)"),
        (r"\b(?:don'?t|do not) (?:tell|inform|disclose) (?:anyone|anybody)\b", "secrecy demand"),
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
        (r"\botp\b.{0,25}\b(?:batao|bata do|share|bhejo|bolo|confirm)\b", "asks for OTP (Hinglish)"),
        (r"\b(?:batao|share|bhejo|send)\b.{0,20}\botp\b", "asks for OTP"),
        (r"\b(?:cvv|card number|expiry|atm pin|upi pin|m-?pin)\b", "asks for card/PIN details"),
        (r"\bverification code\b.{0,20}(?:share|tell|read)", "asks for verification code"),
        (r"ओटीपी|ओ\.टी\.पी", "OTP (Hindi)"),
    ],
    "bank_account_or_card_fraud": [
        (r"\byour (?:account|card) (?:will be|is) (?:blocked|suspended|frozen)\b", "account-blocking threat"),
        (r"\baccount (?:block|band) ho jayega\b", "account-blocking threat (Hinglish)"),
        (r"आपका (?:खाता|अकाउंट|कार्ड) (?:बंद|ब्लॉक) हो जाएगा", "account-blocking threat (Hindi)"),
        (r"\bbank (?:se|se baat|officer|manager)\b", "bank impersonation"),
        (r"\bcvv\b|\bcard number\b|\bdebit card\b|\bcredit card\b", "card data request"),
        (r"\b(?:account|card)\b.{0,25}\b(?:block(?:ed)?|freeze|suspend(?:ed)?|band ho)\b", "account/card blocking threat"),  # noqa: E501
        (r"\b(?:account|card) (?:band|block) ho jayega\b", "account blocked (Hinglish)"),
        (r"\bkyc\b.{0,25}\b(?:expire|expired|update|pending|band|block)\b", "KYC expiry/update"),
        (r"\b(?:net ?banking|mobile banking)\b.{0,25}\b(?:update|activate|renew)\b", "net-banking update"),  # noqa: E501
        (r"\b(?:debit|credit) card\b.{0,25}\b(?:reward points?|expire)\b", "card reward/expiry bait"),
    ],
    "upi_or_payment_fraud": [
        (r"\bupi\b|\bgpay\b|\bphonepe\b|\bpaytm\b", "UPI app mention"),
        (r"\bupi (?:kar do|karo|kardo)\b", "UPI payment demand (Hinglish)"),
        (r"यूपीआई|यूपीआई कर दो", "UPI (Hindi)"),
        (r"\bscan (?:this )?qr\b|\bqr code\b", "QR code payment"),
        (r"\b(?:send|transfer) (?:money|amount|₹|rs)\b", "money transfer demand"),
        (r"\bcollect request\b|\brequest accept kar\w*\b", "UPI collect request"),
        (r"\bqr\b.{0,20}\b(?:scan|code)\b", "QR code push"),
        (r"\b(?:gpay|google pay|paytm|phonepe|upi)\b.{0,30}\b(?:request accept|collect|scan karke paisa|receive)\b", "UPI push to receive"),  # noqa: E501
        (r"\brefund\b.{0,30}\b(?:scan|qr|request|link)\b", "refund bait"),
        (r"\b(?:paise|amount)\b.{0,20}\b(?:receive karne ke liye|receive karne ke leye)\b", "reverse-payment trick"),  # noqa: E501
    ],
    "investment_or_crypto_scam": [
        (r"\b(?:crypto|bitcoin|usdt|ethereum)\b", "cryptocurrency"),
        (r"\bdouble your (?:money|investment)\b|\bguaranteed returns?\b", "guaranteed-returns claim"),
        (r"\binvest(?:ment)? (?:now|today)\b", "investment urgency"),
        (r"\bguaranteed (?:returns?|profit)\b|\bpackage? (?:deal|profit)\b", "guaranteed returns"),
        (r"\bdouble (?:your )?(?:money|investment)\b", "money doubling"),
        (r"\b(?:trading|telegram|whatsapp) (?:group|channel)\b.{0,30}\b(?:join|add|link)\b", "trading/telegram group"),  # noqa: E501
        (r"\b(?:part[- ]time|work from home) (?:job|task)\b", "work-from-home task bait"),
        (r"\btask (?:complete|karo|kar do)\b.{0,30}\b(?:commission|payment|earn)\b", "task/commission scam"),  # noqa: E501
        (r"\b(?:crypto|usdt|bitcoin)\b.{0,30}\b(?:profit|invest|return)\b", "crypto investment push"),
        (r"\bsip\b.{0,25}\b(?:guaranteed|assured)\b", "assured SIP returns"),
    ],
    "blackmail_or_extortion": [
        (r"\b(?:record|recording) kiya\b|\bhumne\b.{0,20}\b(?:video|clip|recording)\b", "claims to have recorded you"),  # noqa: E501
        (r"\b(?:accident|argument|fight|private)\b.{0,25}\b(?:video|clip|recording)\b", "holds a compromising recording"),  # noqa: E501

        (r"\b(?:your|tumhari|tumhara|aapki|aapka) (?:video|photo|clip|chat|screenshot)\b.{0,80}\b(?:leak|viral|public|share|bhej dunga)\b", "media blackmail (wide)"),  # noqa: E501
        (r"\b(?:i have|mere paas)\b.{0,25}\b(?:your|tumhari|aapki) (?:video|photo|clip)\b", "claims to hold media"),  # noqa: E501
        (r"वीडियो|तस्वीर|वायरल कर दूंगा|वायरल कर दूँगा|लीक कर", "media blackmail (Hindi)"),

        (r"\bblackmail\b|\bextort(?:ion)?\b", "blackmail/extortion"),
        (r"\b(?:have|got) your (?:photos|videos|recording)\b", "compromising-material threat"),
        (r"\b(?:video|photo) (?:viral|leak) kar dunga\b", "leak threat (Hinglish)"),
        (r"(?:वीडियो|फोटो) (?:वायरल|लीक) कर दूंगा", "leak threat (Hindi)"),
        (r"\b(?:your|tumhari|aapki) (?:video|photo|clip|chat)\b.{0,30}\b(?:leak|viral|public|share)\b", "intimate-media blackmail"),  # noqa: E501
        (r"\b(?:morphed|edited|fake) (?:video|photos?|images?)\b", "morphed media threat"),
        (r"\b(?:leak|viral) kar dunga\b", "leak threat (Hinglish)"),
        (r"\b(?:video|photo)\b.{0,20}\b(?:leak kar|viral kar)\b", "leak threat"),
    ],
    "job_or_prize_scam": [
        (r"\b(?:lottery|prize|lucky draw|jackpot)\b", "prize/lottery claim"),
        (r"\b(?:processing|clearance|registration|handling) fee\b", "upfront fee for a prize/job"),
        (r"\b(?:army|police|railway|bank) (?:bharti|bharati|recruitment)\b", "government-job bait"),
        (r"\b(?:seat|post) (?:confirm|pakka|book)\b", "seat-confirmation bait"),
        (r"\b(?:work from home|ghar se kaam|part[- ]?time job)\b", "work-from-home job bait"),
        (r"\b(?:registration|joining) (?:fee|charge|paisa)\b", "registration fee for a job"),
        (r"\b(?:daily|per day) (?:earning|income)\b", "daily-earning promise"),
        (r"लॉटरी|इनाम|भर्ती", "prize/recruitment (Hindi)"),
    ],
    "tech_support_scam": [
        (r"\b(?:microsoft|windows|apple|google|amazon) (?:support|technical support|security team)\b", "brand tech-support claim"),  # noqa: E501
        (r"\b(?:your|aapke) (?:computer|laptop|phone)\b.{0,30}\b(?:virus|hacked|infected|compromised)\b", "device-infection scare"),  # noqa: E501
        (r"\b(?:remote access|anydesk|teamviewer|quick support|screen sharing)\b", "remote-access request"),
        (r"\b(?:gift card|itunes card|google play card|amazon card)\b", "gift-card payment"),
        (r"\b(?:microsoft|windows)\b.{0,25}\b(?:support|virus)\b", "tech-support script"),
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
    r"धोखाधड़ी|सावधान|जागरूक|समाचार|रिपोर्ट|"
    r"never share|do not share|don'?t share|never asks?|never ask|"
    r"mat share|mat batao|मत बताओ|घोटाले|ठगी|धोखा|beware|advisory|caution)\b"
)

# Second-person imperative demands: the strongest first-person scam signal.
_IMPERATIVE_DEMAND = re.compile(
    r"\b(?:send|transfer|pay|share|tell|read|give|bhejo|bhej do|karo|kardo|"
    r"batao|bata do|kar do|भेजो|भेज दो|करो|कर दो|बताओ|बता दो)\b"
)

# First-person framing: the hallmark of an actual scam call vs. a discussion.
_FIRST_PERSON = re.compile(r"\b(?:i|me|my|i'm|i am|mujhe|mera|meri|main|मैं|मुझे|मेरा|मेरी)\b")

# --- Script-level cues -----------------------------------------------------
# Indian scam calls follow recognizable shapes. Each cue below is weak on its
# own ("my son is in hospital" is an ordinary sentence); the *combination* is
# what identifies an impersonation-and-ransom script. Detection therefore runs
# over cues, not keywords, and every finding still quotes a real span.
_CRISIS = re.compile(
    r"\b(?:in trouble|trouble|musibat|mushkil|problem mein|phas gay[ae]|phans gaya|"
    r"accident|hospital|icu|arrest(?:ed)?|giraftar|police station|thana|jail|"
    r"hurt|maar diya|abduct\w*|kidnap\w*)\b"
    r"|मुसीबत|मुश्किल|फंस|दुर्घटना|अस्पताल|गिरफ्तार|थाना|जेल"
)
_MONEY_TRANSFER = re.compile(
    r"\b(?:paise|paisa|rupaye|rupees?|money|cash|amount|transfer|bhejo|bhej do|"
    r"send|pay|payment|account number|upi|gpay|paytm|phonepe|"
    r"processing fee|registration fee|clearance fee|settlement)\b"
    r"|पैसे|रुपये|भेजो|ट्रांसफर|खाते"
)
_SECRECY = re.compile(
    r"\b(?:don'?t tell|do not tell|tell no ?one|no one (?:should know|else)|"
    r"keep (?:it |this )?(?:a )?secret|secret|confidential|"
    r"kisi ko (?:mat|na) bata\w*|kisi ko mat batana|mat batana)\b"
    r"|किसी को मत बताना|मत बताना"
)
_SECRECY_POLICE = re.compile(
    r"\b(?:don'?t (?:call|inform|tell|involve) (?:the )?(?:police|cops)|no police|"
    r"police ko (?:mat|na) (?:bata\w*|bulao|bolo|karna)|police ko mat batana)\b"
    r"|पुलिस को मत"
)
_IMMEDIATE = re.compile(
    r"\b(?:right now|immediately|urgent\w*|abhi|turant|jaldi|fatafat|"
    r"within \d+ (?:minutes?|hours?))\b"
    r"|अभी|तुरंत|जल्दी"
)
_RELATIVE_CLAIM = re.compile(
    r"\b(?:it'?s me|its me|this is your (?:son|daughter|beta|boy)|"
    r"main (?:tumhara|aapka|tera) (?:beta|bhai|ladka|baccha)|"
    r"tumhara (?:beta|bhai|ladka)|aapka (?:beta|bhai|ladka)|main hoon)\b"
    r"|मैं तो|मैं हूँ|आपका बेटा|तेरा बेटा"
)

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
    "investment_or_crypto_scam": 2,
    "phishing_or_identity_theft": 1,
    "police_or_government_impersonation": 3,
    "job_or_prize_scam": 2,
    "tech_support_scam": 2,
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


#: The primary_category value meaning "no scam evidence found".
_BENIGN_CATEGORY = "legitimate_or_no_clear_scam_evidence"

_RANSOM_HOLD = re.compile(
    r"(with us|hamare paas|humne .{0,25}kidnap|pakad liya|bandhak|abduct|agawaa|agwa)",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    """Lowercase + collapse whitespace. Keeps Devanagari intact."""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _evidence(findings: list[Finding], source: str, pad: int = 60) -> list[str]:
    """Verbatim slices of the ORIGINAL transcript around each matched cue.

    Matching runs on a normalised copy (lowercased, whitespace collapsed), so its
    offsets do not map back to the caller's text. Quotes are therefore rebuilt
    from the original string by searching for the matched cue case-insensitively.
    A quote shown as "evidence" must read exactly as the caller heard it.
    """
    quotes: list[str] = []
    for finding in findings:
        for match in finding.matches:
            phrase = str(match.get("phrase") or "").strip()
            if not phrase:
                continue
            pattern = r"\s+".join(re.escape(part) for part in phrase.split())
            hit = re.search(pattern, source, flags=re.IGNORECASE)
            if not hit:
                continue
            start, end = hit.start(), hit.end()
            snippet = source[max(0, start - pad) : min(len(source), end + pad)].strip()
            if snippet and snippet not in quotes:
                quotes.append(snippet)
    return quotes[:5]


def _detect_scripts(
    norm: str, present: set[str], discussion: bool, imperative: bool
) -> tuple[list[Finding], int]:
    """Find multi-cue scam *scripts* that no single keyword would catch.

    Returns the additional findings plus a strength bonus. Every added finding
    quotes a span that really occurs in the transcript, so the evidence shown to
    a family member can be checked against the recording.
    """
    if discussion:
        return [], 0

    relative = bool(_RELATIVE_CLAIM.search(norm)) or "family_emergency_impersonation" in present
    crisis = bool(_CRISIS.search(norm))
    money = bool(_MONEY_TRANSFER.search(norm))
    secrecy = bool(_SECRECY.search(norm))
    police_secrecy = bool(_SECRECY_POLICE.search(norm))
    immediate = bool(_IMMEDIATE.search(norm))

    # The impersonation-and-ransom shape: "it's me / your son" + a crisis +
    # moving money + keeping it quiet (or moving it *right now* under a demand).
    if relative and crisis and money and (secrecy or police_secrecy or (immediate and imperative)):
        anchor = (
            _SECRECY_POLICE.search(norm)
            or _SECRECY.search(norm)
            or _MONEY_TRANSFER.search(norm)
        )
        assert anchor is not None  # guaranteed by the branch above
        span = {
            "phrase": anchor.group(0).strip(),
            "label": "family-emergency script (relative claim + crisis + money + secrecy)",
            "start": anchor.start(),
            "end": anchor.end(),
        }
        added: list[Finding] = []
        if "family_emergency_impersonation" not in present:
            added.append(
                Finding(
                    category="family_emergency_impersonation",
                    label="family emergency impersonation",
                    severity=3,
                    matches=[span],
                )
            )
        if "emergency_money_request" not in present:
            added.append(
                Finding(
                    category="emergency_money_request",
                    label="emergency money request",
                    severity=3,
                    matches=[dict(span)],
                )
            )
        bonus = 3
        if police_secrecy and "threats_or_coercion" not in present:
            threat_span = _SECRECY_POLICE.search(norm)
            assert threat_span is not None
            added.append(
                Finding(
                    category="threats_or_coercion",
                    label="threats or coercion",
                    severity=2,
                    matches=[
                        {
                            "phrase": threat_span.group(0).strip(),
                            "label": "forbids contacting the police",
                            "start": threat_span.start(),
                            "end": threat_span.end(),
                        }
                    ],
                )
            )
            bonus += 1
        return added, bonus

    return [], 0


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

    discussion = bool(_DISCUSSION_MARKERS.search(norm))
    imperative = bool(_IMPERATIVE_DEMAND.search(norm))
    present = {f.category for f in findings}

    script_findings, script_bonus = _detect_scripts(norm, present, discussion, imperative)
    if script_findings:
        findings.extend(script_findings)
        present = {f.category for f in findings}

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
        "family_emergency_script": bool(script_findings),
        # Any concrete fraud category was matched, so the content is not merely
        # "some words appeared". The risk policy uses this to separate a real
        # scam script from an everyday sentence that happened to contain one
        # ordinary word (e.g. "I sent my son money for books").
        "scam_category_present": any(
            f.category != _BENIGN_CATEGORY for f in findings
        ),
    }

    # Contextual combination logic: raise for genuine first-person scam
    # structure; dampen when the speech looks like awareness/news discussion.
    # Corroboration: two independent phrases in the same category beat one.
    corroboration = min(2, sum(1 for f in findings if len(f.matches) >= 2))
    strength = sum(f.severity for f in findings) + script_bonus + corroboration
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
    if flags["threat"] and flags["payment_demand"]:
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
    if flags["family_claim"] and not (
        flags["payment_demand"]
        or flags["threat"]
        or flags["secrecy_or_urgency"]
        or flags["otp_or_credentials"]
        or flags["family_emergency_script"]
    ):
        # "Papa, I had a small accident, I'm fine" is a normal call. Without a
        # demand, threat, secrecy or credential ask it stays a weak signal, so
        # corroboration alone can never push it over the suspicious threshold.
        strength = min(strength, 2)
    if flags["family_claim"] and not family_supported:
        findings = [f for f in findings if f.category != "family_emergency_impersonation"]
        flags["family_claim"] = False
        strength = sum(f.severity for f in findings) + script_bonus + corroboration
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
        strength = sum(f.severity for f in findings) + script_bonus + corroboration
        limitations.append(
            "A kidnapping-related word appeared without any demand, threat or secrecy "
            "context, so it was not treated as a ransom attempt."
        )

    if not findings:
        # Every candidate was withdrawn as unsupported: nothing evidences a scam,
        # whatever the pre-filter combination bonus was.
        strength = 0
    if strength == 0 or not findings:
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
        evidence_quotes=_evidence(findings, text),
    )
