# NVIDIA NemotronLabs VoiceChat 11B — Verified Integration Record (Phase 6C)

Status: **API contract verified against the official model card and the
official deployment API reference on 2026-10-25.** The model itself is NOT
bundled or run here: it requires an NVIDIA GPU deployment, which this backend
reaches over the network. No fabricated calls were made; until the deployment
exists, the endpoint honestly refuses with `503 provider_disabled`.

## 1. Model identification (all from the official model card)

- Model: **NVIDIA NemotronLabs VoiceChat 11B** (nvidia /
  NVIDIA-NemotronLabs-VoiceChat-11B, Hugging Face, release 2026-08-03)
- What it is: an 11B end-to-end **full-duplex speech-to-speech** model that
  jointly performs streaming speech understanding and speech generation
  (Fast Conformer speech encoder → Nemotron Nano v2 9B LLM → NVIDIA TTS
  decoder/codec; separate output channel for tool-calling scripts).
- Capabilities (per the card): natural turn-taking (~450 ms), barge-in, live
  tool calling; first open full-duplex model supporting tool calls.
- License: OpenMDW 1.1. Use case per card: ASR, TTS, voice-assistant research
  and development. (Training-use confirmation is handled by the same explicit
  `--confirm-usage-rights` gate as other generated data; see section 6.)
- It is NOT a synthetic-speech detection model. Nothing in the card or the
  deployment docs claims detection. VoxSentinel's local classifier remains
  the sole detection source; `GET /api/v1/providers` reports
  `detection_supported: false` for this provider.
- Input: text (prompt/system) + audio (16 kHz internally). Output: agent text
  + agent audio (22.05 kHz internally) + user transcripts.

## 2. Hardware/deployment reality (stated honestly)

- Verified supported hardware: NVIDIA A100 / H100 / H200 / B100 / B200 /
  RTX-6000; Linux; vLLM + Triton via the optimized NVIDIA inference container.
- Two documented serving paths:
  1. **Offline inference** (batch speech-to-speech checks): the NeMo `Speech`
     repo on branch `nemotron-labs-voicechat`, conda env, torch 2.10.0, run
     `examples/speechlm2/offline_voicechat_infer.py --checkpoint ... --wav ...`.
     Useful for ad-hoc experiments on a GPU box; NOT wired into the API.
  2. **Interactive streaming** (the path we integrated): the official
     NVIDIA inference container exposes a bidirectional WebSocket API.
     Docs: `voicechat_realtime_instructions/` (prerequisites.md, deploy.md,
     generate-model-repo.md, api-reference.md) in the same repo.
- Nothing in this repo can substitute for that GPU deployment. Typical route:
  an AWS/other GPU instance (e.g. g5.48xlarge / p4de class) running the
  container, fronted by TLS, exposed at `wss://.../v1/realtime`.

## 3. Verified wire protocol (api-reference.md, fetched 2026-10-25)

- HTTP discovery `GET /` and health `GET /v1/realtime/health`
  (200 ok / 503 not ready; JSON status, triton_status, inference stats).
- WebSocket `/v1/realtime` (alias `/realtime`), OpenAI-Realtime-compatible
  JSON messages, base64 audio payloads.
- Audio format: PCM 16-bit signed little-endian, mono; client rate 24 kHz
  in and out (server resamples 16 kHz in / 22.05 kHz out internally);
  recommended chunk ≈ 80 ms = 3,840 bytes.
- Session lifecycle implemented by the adapter (strictly, in order):
  `session.created` → client `session.update` (audio formats, instructions,
  tools) → `session.updated` → `input_audio_buffer.append`* →
  `response.output_audio.delta`* (base64 PCM chunks) →
  `response.output_audio_transcript.done` →
  `conversation.item.input_audio_transcription.completed` →
  `response.done` → client `session.close` → `session.end`.
- Known server error codes: `inference_timeout`, `inference_error`,
  `session_timeout`, `tools_not_set` — all mapped to typed exceptions.
- The adapter (`app/services/voicechat_client.py`) validates every event
  against this documented order and raises `VoiceChatProtocolError` on any
  deviation instead of guessing.

## 4. How it is exposed

`POST /api/v1/voicechat` (multipart: user speech file + optional
`instructions` system prompt):

- Disabled by default; requires BOTH `VOXSENTINEL_VOICECHAT_ENABLED=true`
  and a `VOXSENTINEL_VOICECHAT_WS_URL` (wss://, or ws:// for localhost only —
  TLS is enforced in code for non-localhost hosts).
- Input validation reuses the existing honest audio pipeline (format, size,
  duration, silence checks → 415/413/422). Local cap on input length
  (`VOXSENTINEL_VOICECHAT_MAX_INPUT_SECONDS`, default 60 s).
- Per-client sliding-window rate limit + daily quota.
- Bounded timeouts: connect 5 s, idle 60 s (configurable); every error class
  maps to a specific status (422 validation, 429 rate limited, 503
  unavailable, 502 provider error, 504 provider timeout).
- Response JSON (`VoiceChatTurnOut`): `synthetic: true`, provider, model,
  `generator_id`, base64 WAV (24 kHz mono) of the agent's speech, both
  transcripts, sample rate.
- The agent's audio is SYNTHETIC by construction and labelled as such in the
  response, in the audit log, and in every stored artifact.
- Audit events record durations, byte counts, transcript LENGTHS, and
  attribution only — never transcripts, raw audio, or the system prompt.
- Optional storage (`VOXSENTINEL_VOICECHAT_STORE_GENERATED=false` default):
  when on, each WAV is written with a `provenance.jsonl` row labelling it
  synthetic with `source: voicechat_s2s` and its generator/model.

## 5. Detection stays local

VoiceChat performs NO synthetic-speech detection (verified: the model card
and the complete deployment docs contain no detection API). If it is enabled,
`GET /api/v1/providers` lists it under `generations` with
`detection_supported: false`. A provider result (generated agent speech) is
never presented as, or mixed with, a VoxSentinel detection score, and never
enters the evaluation corpus.

## 6. Training workflow (explicit, never automatic)

Agent speech may become training data only through the existing explicit
pipeline (`scripts/build_generated_manifest.py --confirm-usage-rights`),
which appends to `train`/`validation` manifests only, rejects unprovenanced
rows, and can never touch the `eval` partition. No API request can trigger
training or modify the production model.

## 7. Verification checklist (2026-10-25)

- [x] Official model card read end-to-end (architecture, I/O, license,
      hardware, both serving paths)
- [x] Official api-reference.md read end-to-end; wire protocol implemented
      exactly as documented (events, audio format, chunk size, error codes)
- [x] TLS enforcement for non-localhost URLs
- [x] Adapter unit tests: 15 (scripted fake session — no network, no GPU)
- [x] API endpoint tests: 13 (injected test-double client)
- [x] Full backend suite 234 passed / 1 skipped; Ruff clean
- [x] Provider capability reporting honest: generation yes, detection no
- [ ] Live turn against a real GPU deployment — blocked on infrastructure
      (needs an NVIDIA GPU server running the official container; none is
      configured in this environment)
