"""Train the small content classifier that ships with the API.

Cheap, deterministic and dependency-light (scikit-learn only): this is the layer
that keeps the scam module working when no LLM key is configured or the provider
is rate-limited. It never replaces the rule engine or the contextual model; it is
reported as its own signal.

Split policy (no leakage):
  * train on the generated corpus (`data/corpus/scam_corpus.jsonl`)
  * evaluate on the hand-authored sets in `tests/` which the corpus never contains

Run: python -m scripts.train_content_classifier
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

CORPUS = BACKEND_DIR / "data" / "corpus" / "scam_corpus.jsonl"
MODEL_OUT = BACKEND_DIR / "models" / "scam_text_classifier.joblib"


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _eval_set() -> list[dict]:
    rows = _load_jsonl(BACKEND_DIR / "tests" / "data" / "scam_eval_v2.jsonl")
    from tests.eval_scam_set import LABELLED_SET

    for item in LABELLED_SET:
        rows.append(
            {
                "id": item["id"],
                "text": item["text"],
                "expect_suspicious": item["expect_suspicious"],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(CORPUS))
    parser.add_argument("--out", default=str(MODEL_OUT))
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.pipeline import FeatureUnion, Pipeline

    corpus = _load_jsonl(Path(args.corpus))
    if not corpus:
        raise SystemExit("corpus is empty; run scripts.generate_scam_corpus first")

    texts = [row["text"] for row in corpus]
    labels = ["scam" if row["expect_suspicious"] else "benign" for row in corpus]

    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0, random_state=args.seed),
            ),
        ]
    )
    pipeline.fit(texts, labels)

    # --- held-out evaluation on the hand-authored sets ------------------
    eval_rows = _eval_set()
    scores = pipeline.predict_proba([r["text"] for r in eval_rows])
    scam_index = list(pipeline.classes_).index("scam")
    probs = scores[:, scam_index]
    predicted = ["scam" if p >= args.threshold else "benign" for p in probs]
    truth = ["scam" if r["expect_suspicious"] else "benign" for r in eval_rows]

    tp = sum(1 for t, p in zip(truth, predicted, strict=True) if t == "scam" and p == "scam")
    fn = sum(1 for t, p in zip(truth, predicted, strict=True) if t == "scam" and p != "scam")
    fp = sum(1 for t, p in zip(truth, predicted, strict=True) if t != "scam" and p == "scam")
    tn = sum(1 for t, p in zip(truth, predicted, strict=True) if t != "scam" and p != "scam")
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    report = classification_report(truth, predicted, output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=["scam", "benign"]).tolist()

    metrics = {
        "n": len(eval_rows),
        "accuracy": round((tp + tn) / len(eval_rows), 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positives": fp,
        "false_negatives": fn,
        "confusion_matrix_labels": ["scam", "benign"],
        "confusion_matrix": matrix,
        "per_class": {
            key: {
                k: round(v, 3)
                for k, v in report[key].items()
                if k in ("precision", "recall", "f1-score")
            }
            for key in ("scam", "benign")
            if key in report
        },
        "eval_sets": ["tests/data/scam_eval_v2.jsonl", "tests/eval_scam_set.py"],
    }

    payload = {
        "pipeline": pipeline,
        "version": f"sklearn-tfidf-lr-{datetime.now(UTC).strftime('%Y%m%d')}",
        "trained_on": str(Path(args.corpus).name),
        "train_rows": len(corpus),
        "threshold": args.threshold,
        "metrics": metrics,
        "created_at": datetime.now(UTC).isoformat(),
        "caveats": [
            "Trained on template-generated text plus authored labels, not on real call recordings.",
            "Probabilities are model scores on that distribution; they are not real-fraud probabilities.",
            "The hand-authored evaluation sets were written by the same author as the "
            "templates, so these numbers overstate real-world performance.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out, compress=3)
    print(json.dumps({"model": str(out), "train_rows": len(corpus), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
