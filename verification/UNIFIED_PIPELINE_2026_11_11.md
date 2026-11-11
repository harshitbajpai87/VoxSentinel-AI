# Unified pipeline verification - 2026-11-11

Every number below comes from a command run on this machine (linux, CPU only,
Python 3.11). Nothing here is projected or estimated.

## Suites

| Check | Result |
|---|---|
| `python -m pytest -q` (backend) | **795 passed, 5 skipped** in 125.46 s |
| `npx vitest run` (frontend) | **112 passed**, 17 files, 15.89 s |
| `npm run build` | green, 423.18 kB JS / 134.96 kB gzip |
| `ruff check app tests` | clean |
| `npx tsc --noEmit` | clean |

The 5 skips are the gated-model and network tests that refuse to fake a result
(they assert the *reason* instead of running the weights).

## Live endpoint checks (uvicorn on :8099, real models)

### `GET /api/capabilities`

```
asr: [('whisper_small', True), ('indic_hindi', False), ('faster_whisper', True), ('gnani_asr', False)]
indic reason: ai4bharat/indicwav2vec-hindi is a gated Hugging Face repository: accept the model
              terms and set HF_TOKEN, or place a copy under models/hf/indicwav2vec-hindi.
llm: provider=router | configured=True | model=voxisentinel-mistral | fallback=mistral-small-latest
chain: [('ollama', configured=True, 'voxisentinel-mistral')]
```

### `POST /api/analyze` - text input

```
badges: risk=critical, scam_intent=critical, transcription=ok, speaker_match=info
scam score: 100.0 | transformer ensemble: completed 0.9607
speaker: skipped | voice: skipped
verdict: HIGH | evidence: 6 | why-not-conclusive: 7
```

### `POST /api/analyze` - audio input (`tests/fixtures/audio/clean_speech.mp3`)

```
input audio | status completed | 18559 ms
transcript: whisper_small -> "Hi Papa, I reached college safely and I will call you after ..."
voice: completed, calibrated_probability 0.6014
forensics fusion: ['detector', 'handcrafted'] score 0.4438
speaker: skipped | scam: 0.0 | verdict: MEDIUM
```

Honest note: a benign clip scored **MEDIUM** because the detector's calibrated
probability (0.60) sits in the warn band for this voice. That is the shipped
model's behaviour, not a UI bug; the explanation lists it as an unresolved
forensic signal rather than a scam finding.

### Reasoning router - real chain, stand-in daemon

```
provider: ollama | model: voxisentinel-mistral | status: completed
attempts: [('ollama', 'completed')]
severity: HIGH | confidence: 0.75 | fallback_used: False
badge llm: {'id': 'llm_reasoning', 'label': 'Local voxisentinel-mistral', 'tone': 'ok', 'value': 'HIGH'}
verdict: HIGH
evidence[0]: urgency: artificial deadline ("now or your account will be blocked")
```

### Reasoning router - daemon down, no fallback key

```
provider: router | label: Local voxisentinel-mistral | status: failed
attempts: [('ollama', 'failed', 'ollama_unreachable')]
fallback_used: False
verdict: HIGH  (from rules; llm severity None)
```

### Gated checkpoint download

```
$ python scripts/download_hf_models.py --only indicwav2vec-hindi
FAILED  indicwav2vec-hindi   ai4bharat/indicwav2vec-hindi
        401 Client Error. (gated repo: accept the terms on the model page and set HF_TOKEN)
0/1 downloaded into backend/models/hf
```

## Bugs this verification pass found and fixed

1. **`/api/capabilities` reported the legacy Grok provider** (`provider: xai`,
   `model: grok-4.7`) as the reasoning engine, which is not what `/api/analyze`
   uses. It now reports the resolved router chain.
2. **`indic_hindi` reported `available: true`** while the directory held only a
   README from a failed download; the engine would have failed at load time with
   HTTP 401. Gated repos now resolve to unavailable with the reason.
3. **`/health` reported `contextual_ai: true`** whenever a router existed, even
   with a stopped Ollama daemon and no hosted key: it claimed a working reasoner
   that could not answer once. Readiness now requires a chain member that is
   actually reachable (cached 30 s probe, 0.6 s timeout).
4. **The default provider was the legacy hosted one.** `_build_llm` preferred
   `build_router(force_local=False)`, which needs a hosted fallback key, so a
   plain local install silently fell through to xAI. The default is now
   local-first: `voxisentinel-mistral` via Ollama, Mistral cloud on failure.

Regression cover: `tests/test_honest_capabilities.py` (11 tests) pins all four.
