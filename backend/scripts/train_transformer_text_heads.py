"""Train the MuRIL and DeBERTa scam-intent heads used by the text ensemble.

The encoders stay frozen: this trains one small logistic head per encoder over
mean-pooled hidden states. That keeps the artifacts tiny (~10 KB), CPU inference
fast, and the training honest about its data - the labelled corpus in
``data/corpus/scam_corpus.jsonl`` plus a held-out evaluation on the hand-authored
sets that the corpus never contains.

Run:
    python -m scripts.train_transformer_text_heads                # both branches
    python -m scripts.train_transformer_text_heads --branch muril
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

CORPUS = BACKEND_DIR / "data" / "corpus" / "scam_corpus.jsonl"
OUT_DIR = BACKEND_DIR / "models" / "text_heads"

BRANCHES = {
    "muril": {
        "model_id": "google/muril-base-cased",
        "slug": "muril-base-cased",
        "artifact": "muril_scam_head.joblib",
    },
    "xlmr": {
        "model_id": "FacebookAI/xlm-roberta-base",
        "slug": "xlmr-base",
        "artifact": "xlmr_scam_head.joblib",
    },
    "deberta": {
        "model_id": "microsoft/deberta-v3-small",
        "slug": "deberta-v3-small",
        "artifact": "deberta_scam_head.joblib",
    },
}
MAX_TOKENS = 256


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _eval_rows() -> list[dict]:
    rows = _load_jsonl(BACKEND_DIR / "tests" / "data" / "scam_eval_v2.jsonl")
    sys.path.insert(0, str(BACKEND_DIR))
    from tests.eval_scam_set import LABELLED_SET

    for item in LABELLED_SET:
        rows.append({"id": item["id"], "text": item["text"], "expect_suspicious": item["expect_suspicious"]})
    return rows


def _embed(model, tokenizer, texts: list[str], batch_size: int = 8) -> np.ndarray:
    import torch

    vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", truncation=True, max_length=MAX_TOKENS, padding=True
        )
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        vectors.append(pooled.numpy().astype(np.float32))
    return np.vstack(vectors)


def _metrics(truth: list[str], probs: np.ndarray, threshold: float) -> dict:
    predicted = ["scam" if p >= threshold else "benign" for p in probs]
    tp = sum(1 for t, p in zip(truth, predicted, strict=True) if t == "scam" and p == "scam")
    fn = sum(1 for t, p in zip(truth, predicted, strict=True) if t == "scam" and p != "scam")
    fp = sum(1 for t, p in zip(truth, predicted, strict=True) if t != "scam" and p == "scam")
    tn = sum(1 for t, p in zip(truth, predicted, strict=True) if t != "scam" and p != "scam")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n": len(truth),
        "accuracy": round((tp + tn) / max(len(truth), 1), 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positives": fp,
        "false_negatives": fn,
    }


def train_branch(branch: str, threshold: float, seed: int) -> dict:
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModel, AutoTokenizer

    spec = BRANCHES[branch]
    local = BACKEND_DIR / "models" / "hf" / spec["slug"]
    source = str(local) if (local / "config.json").exists() else spec["model_id"]

    corpus = _load_jsonl(CORPUS)
    if not corpus:
        raise SystemExit("corpus is empty; run scripts.generate_scam_corpus first")
    texts = [row["text"] for row in corpus]
    labels = ["scam" if row["expect_suspicious"] else "benign" for row in corpus]

    print(f"[{branch}] loading {source}")
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModel.from_pretrained(source).eval()

    print(f"[{branch}] embedding {len(texts)} training rows")
    train_x = _embed(model, tokenizer, texts)

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0, random_state=seed)),
        ]
    )
    pipeline.fit(train_x, labels)

    eval_rows = _eval_rows()
    eval_x = _embed(model, tokenizer, [row["text"] for row in eval_rows])
    scam_index = list(pipeline.classes_).index("scam")
    probs = pipeline.predict_proba(eval_x)[:, scam_index]
    truth = ["scam" if row["expect_suspicious"] else "benign" for row in eval_rows]
    metrics = _metrics(truth, probs, threshold)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = OUT_DIR / spec["artifact"]
    payload = {
        "format": "voxsentinel-transformer-text-head",
        "version": f"{branch}-{datetime.now(UTC).strftime('%Y%m%d')}",
        "branch": branch,
        "model_id": spec["model_id"],
        "embedding_dim": int(train_x.shape[1]),
        "pooling": "mean of last hidden state",
        "threshold": threshold,
        "trained_on": "data/corpus/scam_corpus.jsonl",
        "training_rows": len(texts),
        "metrics": metrics,
        "trained_at": datetime.now(UTC).isoformat(),
        "pipeline": pipeline,
    }
    joblib.dump(payload, artifact)
    print(f"[{branch}] wrote {artifact.relative_to(BACKEND_DIR)} | metrics={metrics}")
    return {"branch": branch, "artifact": str(artifact), "metrics": metrics}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch", choices=[*BRANCHES, "all"], default="all")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args(argv)

    branches = list(BRANCHES) if args.branch == "all" else [args.branch]
    results = [train_branch(branch, args.threshold, args.seed) for branch in branches]
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
