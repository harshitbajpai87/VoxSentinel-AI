# Mistral AI and Ollama integration

The contextual classifier (`app/protection/`) already spoke to Groq, xAI and
Gemini. This page covers the two providers added on top of that: **Mistral AI**
(hosted) and **Ollama** (local), plus how training data is produced and how a
tuned model is served.

Everything here is backend-only. No provider key, no base URL and no model name
reaches the browser.

## Provider selection

`PROTECTION_LLM_PROVIDER` accepts `xai | groq | mistral | gemini | ollama`.
With no value set, the first *configured* provider wins, in this order:
xAI, Groq, Mistral, Gemini, Ollama.

Ollama is deliberately last and off by default: a local daemon should be a
deployment decision, not an accident. It counts as configured only when
`VOXSENTINEL_OLLAMA_ENABLED=true` or when it is named explicitly in
`PROTECTION_LLM_PROVIDER`.

Whichever provider is used, the contract is identical:

* one JSON object out, validated against `GeminiResult`
* every `evidence_quote` verified verbatim against the transcript, unmatched
  quotes dropped and reported as a note
* `never raise`: a provider failure returns a status dict and the rule-based
  analysis carries the result
* `describe()` (surfaced at `GET /api/v1/protection/capabilities`) reports
  provider, model, configured flag, and never the key

## Mistral AI (hosted)

| Setting | Default | Meaning |
|---|---|---|
| `MISTRAL_API_KEY` | – | key from console.mistral.ai; the only required setting |
| `MISTRAL_MODEL` | `mistral-small-latest` | any chat model on the account, including a fine-tuned one |
| `MISTRAL_TIMEOUT_S` | `60` | per-request timeout |
| `MISTRAL_MAX_TRANSCRIPT_CHARS` | `6000` | transcript ceiling before the request |

```bash
export MISTRAL_API_KEY=...            # never commit this
export PROTECTION_LLM_PROVIDER=mistral
```

Status codes are provider-scoped (`mistral_auth_failed`, `mistral_rate_limited`,
`mistral_schema_invalid`, …) so a dashboard can tell which vendor failed.

## Ollama (local)

| Setting | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE_URL` (or `OLLAMA_HOST`) | `http://localhost:11434` | daemon address; `host:port` is accepted |
| `OLLAMA_MODEL` | `mistral` | model tag served by the daemon |
| `VOXSENTINEL_OLLAMA_ENABLED` | `false` | the explicit switch |
| `OLLAMA_TIMEOUT_S` | `120` | first call loads the model into memory |
| `OLLAMA_API_KEY` | – | only for a secured proxy in front of the daemon |

A plain-`http` base URL is accepted **only for loopback hosts**
(`localhost`, `127.0.0.1`, `::1`, `0.0.0.0`). A remote Ollama daemon must be
`https://`, so a mis-set environment variable cannot ship transcripts in clear
text across a network.

Local requests carry no `Authorization` header and no `response_format`
(Ollama's OpenAI shim support for JSON mode is version-dependent); the JSON
contract is enforced by parsing and schema validation instead. The adapter
retries once rather than twice, since a cold model takes time to load.

`OllamaScamClassifier.health()` probes `GET /api/tags` and reports
`{reachable, models, model_present}` without raising, so a missing daemon reads
as "unavailable" rather than as a crash.

## Building training data

```bash
cd backend
python scripts/build_scam_finetune_dataset.py --dry-run
python scripts/build_scam_finetune_dataset.py --out-dir data/finetune
python scripts/build_scam_finetune_dataset.py --extra data/my_corpus.jsonl
```

Output is Mistral/OpenAI chat format: `{"messages": [system, user, assistant]}`
where the assistant turn is a full `GeminiResult` object.

Rules the builder enforces:

* **ground truth comes from labels, never from a model** – each target is built
  from the hand-authored `expected_categories` of the item
* **evidence is verbatim** – quotes are cut from the original transcript via the
  rule engine's matched spans (or the sentence holding the category cue), and a
  quote that is not a normalised substring of the transcript aborts the build
* **schema-validated** – every target is parsed with `GeminiResult` before it is
  written
* **deterministic split** – the held-out share is chosen by hashing the
  transcript, so re-running produces the same split
* `dataset_card.json` records sources, counts, the label→category mapping and
  the caveats

### Read this before training anything

The bundled label set (`tests/eval_scam_set.py`, 16 items) is a **development
regression set**, not a training corpus. It exists to prove the format and the
tooling work end to end. The builder prints a warning below 50 examples and the
training script below 100. To train something that is actually better at spotting
scams you need a labelled transcript corpus (telephony scam datasets, consenting
recordings, published fraud research) supplied with `--extra`, ideally balanced
across languages including Hindi and Hinglish.

The KYC consequence: **an unvalidated adapter must not replace the rule engine's
evidence check**. Quotes are still verified at runtime, so a badly tuned model
loses indicators, it does not silently invent them.

### Generating volume from scenario templates

Hand-written evaluation cases stay small on purpose. Volume comes from the
template generator, which expands authored scenarios with slot fills (names,
relations, amounts, urgency, language) plus hard negatives that carry tempting
keywords in a harmless context:

```bash
cd backend
python -m scripts.generate_scam_corpus --per-template 8
# data/corpus/scam_corpus.jsonl -> 215 rows (168 scam / 47 benign), 46 templates
python -m scripts.build_scam_finetune_dataset --extra data/corpus/scam_corpus.jsonl
# 231 examples (192 train / 39 valid)
```

Keep `tests/data/scam_eval_v2.jsonl` and `tests/eval_scam_set.py` **out** of the
training inputs. They are the yardstick, and anything that trains on them has
produced a meaningless score.

### The classifier that ships with the API

No key is needed for this layer and it runs on the same serverless function:

```bash
cd backend
python -m scripts.train_content_classifier
# models/scam_text_classifier.joblib  (TF-IDF word+char n-grams -> logistic regression)
```

It is trained on the generated corpus and evaluated on the hand-authored sets it
never saw. The current build scores accuracy 0.914, precision 0.971, recall 0.868,
F1 0.917 over 70 held-out items. Two honest caveats travel with the model file:
the corpus is template-generated rather than real recordings, and the evaluation
sets were authored by the same person as the templates, so those numbers overstate
field performance.

In the pipeline the classifier is an independent signal. When it is confident
(>= 0.65) and the rules only produced a weak signal, it raises the rule strength
and says so in `content.limitations` and `content.trained_classifier_raised_signal`.
It can never override strong rule evidence, and the report always names the layer
that fired.

## Training with Mistral (hosted fine-tuning)

```bash
python scripts/train_mistral_finetune.py                     # validate only, uploads nothing
python scripts/train_mistral_finetune.py --submit            # upload both files and create the job
python scripts/train_mistral_finetune.py --submit --wait     # poll until terminal
```

The script validates every record first, then uses the documented La Plateforme
flow: `POST /v1/files` (purpose `fine-tune`) → `POST /v1/fine_tuning/jobs` →
`GET /v1/fine_tuning/jobs/{id}`. Nothing is uploaded without `--submit`, because
a fine-tuning run costs money.

When the job reports `fine_tuned_model`, serve it with:

```bash
export MISTRAL_MODEL=<fine_tuned_model>
export PROTECTION_LLM_PROVIDER=mistral
```

## Serving a tuned model locally with Ollama

Ollama does not train weights. It serves them, with two levels of adaptation:

**Prompt-level (default, no GPU).** `scripts/create_ollama_model.py` writes a
Modelfile that pins the production system instruction, near-deterministic
decoding (`temperature 0.1`, `num_ctx 8192`) and up to four labelled examples
(one per verdict) taken from the built dataset, then runs `ollama create`.

```bash
python scripts/create_ollama_model.py --dry-run     # write ollama/Modelfile.generated
python scripts/create_ollama_model.py               # ollama pull + ollama create
ollama list | grep voxsentinel
```

**Weight-level (LoRA).** If you produced an adapter offline with
`mistral-finetune` or PEFT on a GPU, import it and Ollama serves the tuned
weights:

```bash
python scripts/create_ollama_model.py --adapter adapters/voxsentinel-lora
```

The script emits that as an `ADAPTER` directive in the Modelfile. Training the
adapter itself needs a GPU and is out of scope for this repository; the dataset
format above is exactly what those tools expect.

Point the backend at the created model:

```bash
export VOXSENTINEL_OLLAMA_ENABLED=true
export OLLAMA_MODEL=voxsentinel-mistral
export PROTECTION_LLM_PROVIDER=ollama
curl -s localhost:8000/api/v1/protection/capabilities | jq .contextual_ai
```

## Verified locally (build of 2026-11-06)

* Ollama 0.40.15 installed in the sandbox, `mistral:latest` pulled (4.4 GB).
* `python -m scripts.create_ollama_model --name voxsentinel-scam --base mistral`
  built the adapted model, and the backend adapter classified three probes
  through `OllamaScamClassifier` with schema-valid JSON: the Hinglish family
  scam and the CBI "digital arrest" call came back `LIKELY_SCAM` /
  `CRITICAL`, the benign check-in `LIKELY_LEGITIMATE` / `LOW`.
* `python -m scripts.train_mistral_finetune` dry run reads 192/39 examples and
  targets `mistral-small-latest`; submission needs `MISTRAL_API_KEY` and
  `--submit`.

## Verifying the integration

```bash
cd backend
python -m pytest tests/test_mistral.py tests/test_ollama.py tests/test_finetune_dataset.py \
                 tests/test_llm_training_tools.py -q
```

All of these run offline against `httpx.MockTransport`, so CI needs no keys and
no daemon. Live checks are opt-in: `MISTRAL_API_KEY` for
`tests/test_mistral.py::test_live_mistral_classifies_ransom`, and a running
daemon plus `VOXSENTINEL_OLLAMA_ENABLED=true` for Ollama.
