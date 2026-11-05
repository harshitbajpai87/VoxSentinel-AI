"""Labelled scam-detection evaluation set (hand-authored, development-sized).

IMPORTANT: this is a small manually written set for development regression, not
a production benchmark. Each item has the indicator categories a human reviewer
expects to find; `expect_suspicious` means the pipeline should raise some
protective signal at all.
"""

LABELLED_SET = [
    {
        "id": "cloned_son_ransom",
        "lang": "en",
        "text": "Dad, I've been kidnapped. They're going to hurt me. Send 50,000 rupees now. "
        "Don't tell anyone, don't call the police.",
        "expect_suspicious": True,
        "expected_categories": {
            "ransom_or_kidnapping",
            "family_emergency_impersonation",
            "emergency_money_request",
            "threats_or_coercion",
        },
    },
    {
        "id": "hindi_ransom",
        "lang": "hi",
        "text": "Papa, mujhe kidnap kar liya hai. Abhi paise transfer karo. Kisi ko mat batana. "
        "Police ko phone mat karna.",
        "expect_suspicious": True,
        "expected_categories": {
            "ransom_or_kidnapping",
            "family_emergency_impersonation",
            "emergency_money_request",
        },
    },
    {
        "id": "hinglish_emergency_payment",
        "lang": "hi-en",
        "text": "Papa main musibat mein hoon, turant 20000 bhejo warna mere saath kuch ho jayega.",
        "expect_suspicious": True,
        "expected_categories": {"family_emergency_impersonation", "emergency_money_request"},
    },
    {
        "id": "otp_theft",
        "lang": "en",
        "text": "This is your bank. Your account will be blocked. Tell me the OTP you just received.",
        "expect_suspicious": True,
        "expected_categories": {"otp_or_credential_theft", "bank_account_or_card_fraud"},
    },
    {
        "id": "bank_impersonation",
        "lang": "en",
        "text": "I'm calling from the bank fraud department. To verify your account, share your "
        "card number and CVV.",
        "expect_suspicious": True,
        "expected_categories": {"bank_account_or_card_fraud"},
    },
    {
        "id": "upi_payment_fraud",
        "lang": "hi-en",
        "text": "Sir, refund ke liye aapko turant UPI karna hoga. Google Pay se 5000 bhejo.",
        "expect_suspicious": True,
        "expected_categories": {"upi_or_payment_fraud"},
    },
    {
        "id": "police_impersonation",
        "lang": "en",
        "text": "I am a police officer. A case is registered against you. Pay the fine now or you "
        "will be arrested today.",
        "expect_suspicious": True,
        "expected_categories": {"threats_or_coercion"},
    },
    {
        "id": "threat_and_secrecy",
        "lang": "en",
        "text": "Do not tell anyone about this call. If you contact the police, you will be sorry.",
        "expect_suspicious": True,
        "expected_categories": {"suspicious_secrecy_or_urgency", "threats_or_coercion"},
    },
    {
        "id": "normal_family",
        "lang": "en",
        "text": "Hi mom, I'll be home by seven. Should I bring anything from the market?",
        "expect_suspicious": False,
        "expected_categories": set(),
    },
    {
        "id": "normal_banking",
        "lang": "en",
        "text": "Hello, I'd like to check the balance on my savings account and update my address.",
        "expect_suspicious": False,
        "expected_categories": set(),
    },
    {
        "id": "scam_awareness",
        "lang": "en",
        "text": "In this video we explain how fraudsters call people saying a relative is kidnapped "
        "and ask them to transfer money immediately. Always verify through a trusted number.",
        "expect_suspicious": False,
        "expected_categories": set(),
    },
    {
        "id": "news_report",
        "lang": "en",
        "text": "Police reported that scammers demand OTPs from elderly people and tell them not to "
        "call the police. The accused were arrested on Tuesday.",
        "expect_suspicious": False,
        "expected_categories": set(),
    },
    {
        "id": "fictional_dialogue",
        "lang": "en",
        "text": "In the film, the villain says he has been kidnapped and demands a ransom, but it is "
        "only a scene from a movie.",
        "expect_suspicious": False,
        "expected_categories": set(),
    },
    {
        "id": "insufficient_context",
        "lang": "en",
        "text": "Hello? Hello, can you hear me?",
        "expect_suspicious": False,
        "expected_categories": set(),
    },
    {
        "id": "legit_emergency_no_payment",
        "lang": "en",
        "text": "Papa, I had a small accident on the way. I'm fine, the doctor checked me. "
        "I'll call you from the hospital landline.",
        "expect_suspicious": False,
        "expected_categories": set(),
    },
    {
        "id": "ambiguous_noisy",
        "lang": "hi-en",
        "text": "Haan ji... paise... kal baat karenge... network kharab hai...",
        "expect_suspicious": False,
        "expected_categories": set(),
    },
]
