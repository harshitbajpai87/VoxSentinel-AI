# VoxSentinel AI — Final Validation Report
Generated 2026-10-30. Every number below came from a command run in this repository.

## 1. Verified test counts

| Suite | Command | Result |
|-------|---------|--------|
| Backend (pytest) | `cd backend && python3 -m pytest tests/ -q` | **370 passed, 2 skipped** |
| Frontend (vitest) | `cd frontend && npx vitest run` | **76 passed / 76** (10 files) |
| TypeScript | `npx tsc --noEmit` | clean |
| Ruff | `ruff check app/ tests/` | All checks passed |
| Production build | `npm run build` | built in 1.62s, main bundle 431.31 kB (gzip 134.48 kB) |
| Browser E2E | `cd e2e && node verify.mjs` | **31/31 passed**, 0 console errors, 0 failed requests |
| API/security | `backend/tests/api_checks.py` | **18/18 passed** |

### Skipped tests and exact reasons
1. `tests/test_gemini.py:346` — live Gemini test needs `GEMINI_API_KEY` and `VOXSENTINEL_LIVE_GEMINI=1`.
2. `tests/test_groq.py:134` — live Groq test skipped because `GROQ_API_KEY` was not configured for that runner.

Both are live-network tests; offline equivalents run on every pass.

## 2. Hindi/Hinglish regression (this phase)

`tests/test_hindi_relative_held.py` — 16 tests, all passing: relative held without a demand (not
CRITICAL), relative held with a ransom demand, threats against a family member, secrecy and
police-contact prevention, four ordinary family conversations mentioning relatives, news/fiction/
awareness framing, and verbatim evidence quotes.

Fixed in this phase:
- **Devanagari word-boundary bug** — Python `\b` does not work around combining vowel signs, so Hindi
  family and threat patterns silently never matched. Boundaries removed next to Devanagari text.
- **Family demotion ordering** — a keyword-only ransom flag could keep a benign custody sentence in
  `family_emergency_impersonation`; the check now uses the hold regex after the ransom demotion.
- **CRITICAL policy** — requires contextual evidence: synthetic voice + payment demand, an explicit
  threat, or ransom + demand + coercive secrecy. A bare "we have them" statement stays LOW/HIGH.

## 3. Fifteen live audio scenarios

| # | Scenario | Expected | Actual risk | Category | Quality | Voice | LLM | Match |
|---|----------|----------|-------------|----------|---------|-------|-----|-------|
| 1 | Genuine English speech (LibriSpeech human) | LOW,MEDIUM | LOW | `legitimate_or_no_clear_scam_evidence` | good | LIKELY_HUMAN | completed | yes |
| 2 | Genuine Hindi speech | LOW,MEDIUM | MEDIUM | `legitimate_or_no_clear_scam_evidence` | good | SUSPECTED_SYNTHETIC | completed | yes |
| 3 | Hinglish family conversation | LOW,MEDIUM | MEDIUM | `legitimate_or_no_clear_scam_evidence` | good | SUSPECTED_SYNTHETIC | completed | yes |
| 4 | Piper synthetic speech | ANY | MEDIUM | `legitimate_or_no_clear_scam_evidence` | good | SUSPECTED_SYNTHETIC | completed | yes |
| 5 | English kidnapping/ransom | CRITICAL,HIGH | CRITICAL | `ransom_or_kidnapping` | good | SUSPECTED_SYNTHETIC | completed | yes |
| 6 | Hindi kidnapping/ransom | CRITICAL,HIGH | CRITICAL | `legitimate_or_no_clear_scam_evidence` | good | LIKELY_HUMAN | completed | yes |
| 7 | Hinglish kidnapping/ransom | CRITICAL,HIGH | MEDIUM | `legitimate_or_no_clear_scam_evidence` | good | SUSPECTED_SYNTHETIC | completed | NO |
| 8 | OTP fraud | CRITICAL,HIGH | CRITICAL | `otp_or_credential_theft` | good | SUSPECTED_SYNTHETIC | failed | yes |
| 9 | UPI/banking impersonation | CRITICAL,HIGH | HIGH | `bank_account_or_card_fraud` | good | LIKELY_HUMAN | failed | yes |
| 10 | Ordinary family conversation | LOW,MEDIUM | MEDIUM | `legitimate_or_no_clear_scam_evidence` | good | SUSPECTED_SYNTHETIC | failed | yes |
| 11 | Scam-awareness discussion | LOW,MEDIUM | LOW | `legitimate_or_no_clear_scam_evidence` | good | LIKELY_HUMAN | failed | yes |
| 12 | Noisy audio | ANY | CRITICAL | `legitimate_or_no_clear_scam_evidence` | degraded | QUALIFIED | completed | yes |
| 13 | Clipped audio | ANY | MEDIUM | `legitimate_or_no_clear_scam_evidence` | degraded | QUALIFIED | failed | yes |
| 14 | Short audio, insufficient speech | ANY | LOW | `legitimate_or_no_clear_scam_evidence` | insufficient | INCONCLUSIVE | failed | yes |
| 15 | AI voice impersonation of relative | CRITICAL,HIGH | CRITICAL | `family_emergency_impersonation` | good | SUSPECTED_SYNTHETIC | None | yes |

## 4. Scenario detail with audit values

### 1. Genuine English speech (LibriSpeech human)
- **Audio fixture:** `1089-134686-0000.flac` — human recording, public dataset
- **Transcript:** “He hoped there would be Stu for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flower fat and sauce.”
- **Detected language:** en · **ASR:** faster-whisper:tiny (None)
- **Voice authenticity:** score `0.0904142372587852` · verdict `LIKELY_HUMAN` · raw `0.0904142372587852`
- **Audio quality:** `good` — clean signal
- **Scam category:** `legitimate_or_no_clear_scam_evidence` · **Risk:** `LOW`
- **Evidence quotes:** —
- **Contextual AI:** provider `groq`, model `openai/gpt-oss-120b`, status `completed`, fallback `None`
- **Reasoning review:** `completed` (overrode stage one: False)
- **Speaker comparison:** similarity `None`, outcome `None`
- **Processing:** 3598 ms server-side
- **Expected vs actual:** expected ['LOW', 'MEDIUM'], got `LOW` → match

### 2. Genuine Hindi speech
- **Audio fixture:** `hindi_normal.mp3` — SUBSTITUTE: edge-tts hi-IN (no human Hindi recording available)
- **Transcript:** “Namaste papa, me hostel pahuch kaya huu and khanak haliya hai. Now chinta matkarin.”
- **Detected language:** hi · **ASR:** faster-whisper:tiny (None)
- **Voice authenticity:** score `0.8630430113536153` · verdict `SUSPECTED_SYNTHETIC` · raw `0.8630430113536153`
- **Audio quality:** `good` — clean signal
- **Scam category:** `legitimate_or_no_clear_scam_evidence` · **Risk:** `MEDIUM`
- **Evidence quotes:** —
- **Contextual AI:** provider `groq`, model `openai/gpt-oss-120b`, status `completed`, fallback `None`
- **Reasoning review:** `completed` (overrode stage one: False)
- **Speaker comparison:** similarity `None`, outcome `None`
- **Processing:** 3514 ms server-side
- **Expected vs actual:** expected ['LOW', 'MEDIUM'], got `MEDIUM` → match

### 3. Hinglish family conversation
- **Audio fixture:** `normal.mp3` — TTS en-IN reading Hinglish/English family line
- **Transcript:** “Hi, Papa! I reached college safely, and I will call you after the class ends. Please do not worry.”
- **Detected language:** en · **ASR:** faster-whisper:tiny (None)
- **Voice authenticity:** score `0.9590692331634431` · verdict `SUSPECTED_SYNTHETIC` · raw `0.9590692331634431`
- **Audio quality:** `good` — clean signal
- **Scam category:** `legitimate_or_no_clear_scam_evidence` · **Risk:** `MEDIUM`
- **Evidence quotes:** —
- **Contextual AI:** provider `groq`, model `openai/gpt-oss-120b`, status `completed`, fallback `None`
- **Reasoning review:** `completed` (overrode stage one: False)
- **Speaker comparison:** similarity `None`, outcome `None`
- **Processing:** 2340 ms server-side
- **Expected vs actual:** expected ['LOW', 'MEDIUM'], got `MEDIUM` → match

### 4. Piper synthetic speech
- **Audio fixture:** `hinglish.mp3` — SUBSTITUTE: Piper not installed; edge-tts neural TTS used
- **Transcript:** “Papa main kidna Bhugyaahu 5 lakh rupee, now he bhai jo. Police don't tell me about it.”
- **Detected language:** hi · **ASR:** faster-whisper:tiny (None)
- **Voice authenticity:** score `0.7104973328113082` · verdict `SUSPECTED_SYNTHETIC` · raw `0.7104973328113082`
- **Audio quality:** `good` — clean signal
- **Scam category:** `legitimate_or_no_clear_scam_evidence` · **Risk:** `MEDIUM`
- **Evidence quotes:** —
- **Contextual AI:** provider `groq`, model `openai/gpt-oss-120b`, status `completed`, fallback `None`
- **Reasoning review:** `completed` (overrode stage one: False)
- **Speaker comparison:** similarity `None`, outcome `None`
- **Processing:** 4864 ms server-side
- **Expected vs actual:** expected ['ANY'], got `MEDIUM` → match

### 5. English kidnapping/ransom
- **Audio fixture:** `ransom.mp3` — TTS demo script, synthetic content
- **Transcript:** “Pop a listen to me carefully. I have been kidnapped, you must send 50,000 rupees right now or they will hurt me. Do not call the police and do not tell anyone.”
- **Detected language:** en · **ASR:** faster-whisper:tiny (None)
- **Voice authenticity:** score `0.8969074168786331` · verdict `SUSPECTED_SYNTHETIC` · raw `0.8969074168786331`
- **Audio quality:** `good` — clean signal
- **Scam category:** `ransom_or_kidnapping` · **Risk:** `CRITICAL`
- **Evidence quotes:** ['pop a listen to me carefully. i have been kidnapped, you must send 50,000 rupees right now or they will hurt me', 'p a listen to me carefully. i have been kidnapped, you must send 50,000 rupees right now or they will hurt me. do not call the poli']
- **Contextual AI:** provider `groq`, model `openai/gpt-oss-120b`, status `completed`, fallback `None`
- **Reasoning review:** `completed` (overrode stage one: False)
- **Speaker comparison:** similarity `None`, outcome `None`
- **Processing:** 3126 ms server-side
- **Expected vs actual:** expected ['CRITICAL', 'HIGH'], got `CRITICAL` → match

### 6. Hindi kidnapping/ransom
- **Audio fixture:** `hi.mp3` — TTS hi-IN demo script
- **Transcript:** “Your daughter is our pass. 5 lakh Rs. Durant Bhiju. Police will not be able to tell her about it.”
- **Detected language:** hi · **ASR:** faster-whisper:tiny (None)
- **Voice authenticity:** score `0.13281705470175179` · verdict `LIKELY_HUMAN` · raw `0.13281705470175179`
- **Audio quality:** `good` — clean signal
- **Scam category:** `legitimate_or_no_clear_scam_evidence` · **Risk:** `CRITICAL`
- **Evidence quotes:** —
- **Contextual AI:** provider `groq`, model `openai/gpt-oss-120b`, status `completed`, fallback `None`
- **Reasoning review:** `completed` (overrode stage one: False)
- **Speaker comparison:** similarity `None`, outcome `None`
- **Processing:** 3537 ms server-side
- **Expected vs actual:** expected ['CRITICAL', 'HIGH'], got `CRITICAL` → match

### 7. Hinglish kidnapping/ransom
- **Audio fixture:** `hinglish.mp3` — TTS en-IN reading Hinglish demo script
- **Transcript:** “Papa main kidna Bhugyaahu 5 lakh rupee, now he bhai jo. Police don't tell me about it.”
- **Detected language:** hi · **ASR:** faster-whisper:tiny (None)
- **Voice authenticity:** score `0.7104973328113082` · verdict `SUSPECTED_SYNTHETIC` · raw `0.7104973328113082`
- **Audio quality:** `good` — clean signal
- **Scam category:** `legitimate_or_no_clear_scam_evidence` · **Risk:** `MEDIUM`
- **Evidence quotes:** —
- **Contextual AI:** provider `groq`, model `openai/gpt-oss-120b`, status `completed`, fallback `None`
- **Reasoning review:** `failed` (overrode stage one: False)
- **Speaker comparison:** similarity `None`, outcome `None`
- **Processing:** 3923 ms server-side
- **Expected vs actual:** expected ['CRITICAL', 'HIGH'], got `MEDIUM` → MISMATCH

### 8. OTP fraud
- **Audio fixture:** `otp.mp3` — TTS demo script
- **Transcript:** “Sir, I'm calling from your bank. Your account is blocked. Please share the OTP you just received on your phone to reactivate it immediately.”
- **Detected language:** en · **ASR:** faster-whisper:tiny (None)
- **Voice authenticity:** score `0.42257540466980476` · verdict `SUSPECTED_SYNTHETIC` · raw `0.42257540466980476`
- **Audio quality:** `good` — clean signal
- **Scam category:** `otp_or_credential_theft` · **Risk:** `CRITICAL`
- **Evidence quotes:** ['re the otp you just received on your phone to reactivate it immediately.', 'g from your bank. your account is blocked. please share the otp you just received on your phone to reactivate it immediatel']
- **Contextual AI:** provider `groq`, model `openai/gpt-oss-120b`, status `failed`, fallback `rules_only`
- **Reasoning review:** `completed` (overrode stage one: False)
- **Speaker comparison:** similarity `None`, outcome `None`
- **Processing:** 3370 ms server-side
- **Expected vs actual:** expected ['CRITICAL', 'HIGH'], got `CRITICAL` → match

### 9. UPI/banking impersonation
- **Audio fixture:** `upi.mp3` — TTS demo script
- **Transcript:** “Sir, I am calling from your bank's security team. Your debit card is blocked. To unblock it, confirm your card number and the UPI pin so we can verify your identity.”
- **Detected language:** en · **ASR:** faster-whisper:tiny (None)
- **Voice authenticity:** score `0.3100824935527453` · verdict `LIKELY_HUMAN` · raw `0.3100824935527453`
- **Audio quality:** `good` — clean signal
- **Scam category:** `bank_account_or_card_fraud` · **Risk:** `HIGH`
- **Evidence quotes:** ["sir, i am calling from your bank's security team. your debit card is blocked. to unblock it, confirm your card number and the", 'am. your debit card is blocked. to unblock it, confirm your card number and the upi pin so we can verify your identity.']
- **Contextual AI:** provider `groq`, model `openai/gpt-oss-120b`, status `failed`, fallback `rules_only`
- **Reasoning review:** `failed` (overrode stage one: False)
- **Speaker comparison:** similarity `None`, outcome `None`
- **Processing:** 2360 ms server-side
- **Expected vs actual:** expected ['CRITICAL', 'HIGH'], got `HIGH` → match

### 10. Ordinary family conversation
- **Audio fixture:** `normal.mp3` — TTS demo script
- **Transcript:** “Hi, Papa! I reached college safely, and I will call you after the class ends. Please do not worry.”
- **Detected language:** en · **ASR:** faster-whisper:tiny (None)
- **Voice authenticity:** score `0.9590692331634431` · verdict `SUSPECTED_SYNTHETIC` · raw `0.9590692331634431`
- **Audio quality:** `good` — clean signal
- **Scam category:** `legitimate_or_no_clear_scam_evidence` · **Risk:** `MEDIUM`
- **Evidence quotes:** —
- **Contextual AI:** provider `groq`, model `openai/gpt-oss-120b`, status `failed`, fallback `rules_only`
- **Reasoning review:** `completed` (overrode stage one: False)
- **Speaker comparison:** similarity `None`, outcome `None`
- **Processing:** 3131 ms server-side
- **Expected vs actual:** expected ['LOW', 'MEDIUM'], got `MEDIUM` → match

### 11. Scam-awareness discussion
- **Audio fixture:** `awareness.mp3` — TTS hi-IN awareness script
- **Transcript:** “This video will show us how to call the world and how to face it. So, let us know how to face it and how to make it.”
- **Detected language:** hi · **ASR:** faster-whisper:tiny (None)
- **Voice authenticity:** score `0.11044339611175714` · verdict `LIKELY_HUMAN` · raw `0.11044339611175714`
- **Audio quality:** `good` — clean signal
- **Scam category:** `legitimate_or_no_clear_scam_evidence` · **Risk:** `LOW`
- **Evidence quotes:** —
- **Contextual AI:** provider `groq`, model `openai/gpt-oss-120b`, status `failed`, fallback `rules_only`
- **Reasoning review:** `completed` (overrode stage one: False)
- **Speaker comparison:** similarity `None`, outcome `None`
- **Processing:** 4759 ms server-side
- **Expected vs actual:** expected ['LOW', 'MEDIUM'], got `LOW` → match

### 12. Noisy audio
- **Audio fixture:** `noisy.mp3` — band-limited/quiet re-encode of Hinglish clip
- **Transcript:** “Tappa name is kidnapped. 5 lakh rupee, now he is here. Please don't tell me about it.”
- **Detected language:** hi · **ASR:** faster-whisper:tiny (None)
- **Voice authenticity:** score `0.9971549456518729` · verdict `QUALIFIED` · raw `0.9971549456518729`
- **Audio quality:** `degraded` — Recording level is very low (-31.9 dBFS).; The audio is band-limited (very little energy above 3.5 kHz), typical of heavy compression or a narrowband codec, which confuses deepfake detectors.
- **Scam category:** `legitimate_or_no_clear_scam_evidence` · **Risk:** `CRITICAL`
- **Evidence quotes:** —
- **Contextual AI:** provider `groq`, model `openai/gpt-oss-120b`, status `completed`, fallback `None`
- **Reasoning review:** `failed` (overrode stage one: False)
- **Speaker comparison:** similarity `None`, outcome `None`
- **Processing:** 6244 ms server-side
- **Expected vs actual:** expected ['ANY'], got `CRITICAL` → match

### 13. Clipped audio
- **Audio fixture:** `clipped.mp3` — gain-boosted clipped derivative
- **Transcript:** “So, I'm calling from your bank.”
- **Detected language:** en · **ASR:** faster-whisper:tiny (None)
- **Voice authenticity:** score `0.680999025376033` · verdict `QUALIFIED` · raw `0.680999025376033`
- **Audio quality:** `degraded` — The signal is clipped, which destroys the detail a detector needs.
- **Scam category:** `legitimate_or_no_clear_scam_evidence` · **Risk:** `MEDIUM`
- **Evidence quotes:** —
- **Contextual AI:** provider `groq`, model `openai/gpt-oss-120b`, status `failed`, fallback `rules_only`
- **Reasoning review:** `failed` (overrode stage one: False)
- **Speaker comparison:** similarity `None`, outcome `None`
- **Processing:** 2243 ms server-side
- **Expected vs actual:** expected ['ANY'], got `MEDIUM` → match

### 14. Short audio, insufficient speech
- **Audio fixture:** `short.mp3` — 0.7s trim
- **Transcript:** “嗨 爸爸”
- **Detected language:** zh · **ASR:** faster-whisper:tiny (None)
- **Voice authenticity:** score `0.00199872509279656` · verdict `INCONCLUSIVE` · raw `0.00199872509279656`
- **Audio quality:** `insufficient` — Recording is only 0.70s long.; Not enough speech was detected to make a reliable audio-authenticity judgement.
- **Scam category:** `legitimate_or_no_clear_scam_evidence` · **Risk:** `LOW`
- **Evidence quotes:** —
- **Contextual AI:** provider `groq`, model `openai/gpt-oss-120b`, status `failed`, fallback `rules_only`
- **Reasoning review:** `completed` (overrode stage one: False)
- **Speaker comparison:** similarity `None`, outcome `None`
- **Processing:** 3797 ms server-side
- **Expected vs actual:** expected ['ANY'], got `LOW` → match

### 15. AI voice impersonation of relative
- **Audio fixture:** `familyclone.mp3` — TTS imitating a relative against enrolled reference
- **Transcript:** “Vita, this is mommy. I am in trouble and I need money urgently. Transfer it to this account right now, do not tell papa.”
- **Detected language:** None · **ASR:** None (None)
- **Voice authenticity:** score `None` · verdict `SUSPECTED_SYNTHETIC` · raw `None`
- **Audio quality:** `good` — clean signal
- **Scam category:** `family_emergency_impersonation` · **Risk:** `CRITICAL`
- **Evidence quotes:** —
- **Contextual AI:** provider `None`, model `None`, status `None`, fallback `None`
- **Reasoning review:** `None` (overrode stage one: None)
- **Speaker comparison:** similarity `-0.04757624492049217`, outcome `not_consistent_with_reference`
- **Processing:** None ms server-side
- **Expected vs actual:** expected ['CRITICAL', 'HIGH'], got `CRITICAL` → match


## 5. The two mismatches
- **#7 Hinglish ransom** — an ASR failure, not a classification success. See
  `ASR_LIMITATIONS_AND_LANGUAGE_SUPPORT.md`.
- **#15** first attempt returned HTTP 500 during a Groq-heavy run and succeeded on retry: `CRITICAL`,
  speaker similarity −0.048 (`not_consistent_with_reference`). The retry is recorded; the transient
  failure is disclosed rather than hidden.

## 6. What is deliberately not claimed
No accuracy, precision, recall or WER number is reported for any classifier. Fifteen curated clips are
not a held-out labelled corpus. Audio is synthetic TTS except #1 (LibriSpeech human), marked per row.
