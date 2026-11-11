# Model reliability repairs — 2026-11-11

## Scope

This change set improves model routing, score semantics, fallback behavior, result provenance, and failure reporting. It does **not** retrain the project’s models or prove production accuracy. Accuracy still requires the deployed model weights, representative labeled audio/text, and held-out evaluation across languages, recording conditions, and attack types.

## Corrections made

- **ASR:** the API records the engine that actually produced the selected transcript. An unavailable or failing preferred engine now falls back to the next available engine when enabled; disabling fallback is respected. Adapter exceptions are retained as failed attempts rather than escaping. An empty or unavailable transcript is no longer treated as usable scam evidence.
- **Scam text ensemble:** branch selection is language-aware, cache keys include language, code-mixed Hindi is routed more cautiously, and a transformer head without an explicit `scam` class mapping fails instead of choosing an arbitrary argmax. Failed model loads are reported as failed rather than available.
- **Speaker comparison:** raw cosine similarity is preserved, but a same/different-speaker decision now requires a finite cosine threshold and at least 100 non-test calibration pairs with non-placeholder provenance. With the shipped placeholder threshold, the API reports similarity (when embedding inference is available) without an identity verdict. Capabilities report the active backend and whether it has decision calibration.
- **Contextual LLM/reasoner:** provider labels follow the provider that actually answered; unsupported second-pass providers do not silently become Groq; and trivial substring matches cannot authorize a verdict downgrade. LLM severity is no longer converted into a made-up confidence probability.
- **Unified results:** speaker outcomes are normalized to canonical values, model attribution is present only when a comparison ran, and client-supplied text is distinguished from an audio transcript. Missing or empty ASR is explicitly “scam intent not assessed,” not evidence of low risk.
- **Acoustic forensics and fusion:** AASIST softmax is explicitly marked uncalibrated. Joint confidence is `null` until a joint calibration is evaluated. Content evidence strength cannot independently create a `CRITICAL` fusion level. Acoustic bandwidth and silence cues are described as non-exclusive processing signals rather than proof of synthesis.
- **Frontend:** detector percentages are shown only for a calibrated probability. Other outputs display their raw score and score definition; text-only analysis types allow a null audio duration.

## Artifact and validation limits in this checkout

The project includes detector/scam-text/scam-head joblib artifacts and an ECAPA speaker checkpoint. The speaker threshold artifact reports zero calibration pairs, so it is deliberately rejected for identity decisions. The local Hugging Face encoder directories (ASR, MuRIL/DeBERTa/XLM-R, and XLSR), AASIST ONNX graph, and a trained XLSR forensics head are not present in this extracted project. Therefore, optional integration tests that require those weights are skipped or report the branch unavailable; passing unit/API tests is not evidence that those absent models can run here.

Before deployment, install the intended model weights and runtime dependencies using the project’s documented setup scripts, regenerate the speaker threshold from a sufficiently large held-out labeled set, then run the real-model tests and the evaluation scripts. Do not use the unit-test threshold stub as a production calibration artifact.

## Verification

Backend and frontend test/build results are recorded in the delivery summary. Real-model accuracy and calibration remain unverified until the missing artifacts and labeled evaluation data are supplied.
