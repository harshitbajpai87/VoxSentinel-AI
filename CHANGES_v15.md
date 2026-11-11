# VoxSentinel AI — v15 patch: Mistral AI + Ollama providers and training tooling

Drop these files over the existing project tree (same relative paths).

## New
- `backend/app/protection/mistral.py` — hosted Mistral AI adapter (OpenAI-compatible)
- `backend/app/protection/ollama.py` — local Ollama adapter (no key, loopback-only http, `health()` probe)
- `backend/scripts/build_scam_finetune_dataset.py` — labelled transcripts -> schema-validated chat JSONL
- `backend/scripts/train_mistral_finetune.py` — validated Mistral fine-tuning job (dry run by default)
- `backend/scripts/create_ollama_model.py` — builds a local Ollama model (prompt-level, or LoRA via `--adapter`)
- `backend/tests/test_mistral.py`, `test_ollama.py`, `test_finetune_dataset.py`, `test_llm_training_tools.py`
- `backend/data/finetune/{train,valid}.jsonl`, `dataset_card.json` — 16-example seed set (format proof only)
- `backend/ollama/Modelfile.generated` — generated Modelfile example
- `docs/MISTRAL_OLLAMA.md`

## Modified
- `backend/app/protection/grok.py` — provider-generic status codes (`code_prefix`) and attempt budget;
  xAI behaviour unchanged (`grok_*` codes preserved)
- `backend/app/protection/service.py` — factory accepts `mistral` and `ollama`; unsafe Ollama base URL
  fails closed to a disabled provider instead of crashing startup
- `backend/app/protection/reasoner.py` — second-pass review supports both new providers; Ollama needs no key
- `.env.example` — `MISTRAL_*`, `OLLAMA_*`, `VOXSENTINEL_OLLAMA_ENABLED`
- `README.md` — v15 section

## Verified
- 447 backend tests pass, 3 skipped (live network only); `ruff check app scripts` clean
- `/api/v1/protection/capabilities` reports `provider: mistral`, key never serialised
- real-socket test against a loopback fake Ollama daemon: correct path, no auth header, no `response_format`,
  LIKELY_SCAM/CRITICAL with a verified evidence quote

## Not done here (needs your credentials/infra)
- No live Mistral call (no `MISTRAL_API_KEY` on this machine) and no real Ollama daemon (not installed)
- No fine-tuning job was submitted and no weights were trained: the bundled 16-item label set is a
  development regression set. Add a labelled corpus with `--extra` before training.
