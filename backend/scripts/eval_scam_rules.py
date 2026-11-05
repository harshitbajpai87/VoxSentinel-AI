"""Offline evaluation of the RULE-BASED baseline on the labelled set.

Run: python -m scripts.eval_scam_rules
Reports real, computed metrics for the rule engine only. Gemini metrics require
a live API key and are NOT reported here (nothing is fabricated).
"""

from __future__ import annotations

import json
import sys

from app.protection.rules import analyze_transcript
from tests.eval_scam_set import LABELLED_SET


def main() -> int:
    tp = fp = tn = fn = 0
    cat_hits = cat_total = 0
    rows = []
    for item in LABELLED_SET:
        r = analyze_transcript(item["text"])
        predicted = r.signal_strength >= 3 and not r.discussion_context
        found = {c.category for c in r.categories}
        hit = bool(found & item["expected_categories"]) if item["expected_categories"] else True
        if item["expected_categories"]:
            cat_total += 1
            cat_hits += 1 if hit else 0
        if item["expect_suspicious"] and predicted:
            tp += 1
        elif item["expect_suspicious"] and not predicted:
            fn += 1
        elif not item["expect_suspicious"] and predicted:
            fp += 1
        else:
            tn += 1
        rows.append(
            {
                "id": item["id"],
                "expected": item["expect_suspicious"],
                "predicted": predicted,
                "strength": r.signal_strength,
                "categories": sorted(found),
            }
        )

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "n": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round((tp + tn) / total, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positive_rate": round(fp / (fp + tn), 3) if fp + tn else 0.0,
        "false_negative_rate": round(fn / (fn + tp), 3) if fn + tp else 0.0,
        "category_recall": round(cat_hits / cat_total, 3) if cat_total else None,
        "dataset_note": "Hand-authored development set; not a production benchmark.",
        "engine": "rule-based baseline (Gemini not evaluated offline)",
    }
    print(json.dumps({"metrics": metrics, "rows": rows}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
