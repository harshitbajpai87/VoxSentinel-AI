# ASR Limitations and Language Support

## In use today
- ASR: `faster-whisper` **tiny** (CTranslate2), multilingual, auto language detection with probability.
- Working well: clean English. Working acceptably: Hindi (Devanagari output). **Unreliable: Hinglish.**
- Scam rules: English, Hinglish transliteration, Devanagari Hindi.

## The Hinglish failure (scenario #7)
Input script: "Papa main kidnap ho gaya hoon, paanch lakh rupaye abhi bhejo, police ko mat batana."
Transcript produced by `faster-whisper:tiny`:

> "Papa main kidna Bhugyaahu 5 lakh rupee, now he bhai jo. Police don't tell me about it."

Consequences, traced end to end:
1. "kidnap", the demand verb and the police instruction were all destroyed.
2. The rule engine found **no** indicators (signal strength 0).
3. The contextual classifier read nonsense and returned `LIKELY_LEGITIMATE`.
4. Final risk **MEDIUM**; expected CRITICAL/HIGH.

**This is an ASR limitation, not a classifier capability.** English and Hindi ransom scripts reached
CRITICAL through the same pipeline. It is not evidence that Hinglish ransom calls are handled reliably.

## Why the tiny model fails
- 39M parameters, trained largely on clean read speech.
- Intra-sentence Hindi/English switching is rare in its training data.
- Indian phonology (retroflexes, aspirated stops) is underrepresented.
- No vocabulary bias for scam terms (kidnap, ransom, OTP, UPI).

## Recommended fix, in order
1. `faster-whisper small` or `medium` (3–8× slower on CPU, materially better on code-mixing).
2. Hindi-first: `ai4bharat/indicwhisper`, `vasista22/whisper-hindi-large-v2`.
3. Domain hotword list via `initial_prompt` (kidnap, ransom, firौती, OTP, UPI, account blocked).
4. Confidence gating: low `language_probability`, or a transcript with no recognisable keywords, should
   show "transcription uncertain" rather than silently classifying.
5. Build a held-out speaker-disjoint Hindi/Hinglish eval set and publish WER before claiming accuracy.

## Guardrails already shipped
The transcript is displayed and stored with every record, the ASR model name is recorded, and any
provider/ASR failure sets `fallback: rules_only` with the reason, so the product never implies
contextual analysis ran when it did not.
