"""Evaluate the content layers on the larger labelled set (v2).

Reports real, computed numbers only. Two engine configurations are measured:

  rules   — the deterministic rule engine alone (offline, no keys)
  hybrid  — rules + the trained text classifier shipped in models/, combined
            the way `app/protection/service.py` combines them

Gemini/Mistral/Ollama are NOT evaluated here: they need a live key or a local
Ollama daemon, and this script never invents numbers for a model it could not
call. Use `scripts/eval_scam_llm.py` for a live provider run.

Run: python -m scripts.eval_scam_set_v2 [--show-failures]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "tests" / "data" / "scam_eval_v2.jsonl"


def load_items() -> list[dict]:
    return [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]


def _score(rows: list[dict]) -> dict:
    tp = sum(1 for r in rows if r["expected"] and r["predicted"])
    fn = sum(1 for r in rows if r["expected"] and not r["predicted"])
    fp = sum(1 for r in rows if not r["expected"] and r["predicted"])
    tn = sum(1 for r in rows if not r["expected"] and not r["predicted"])
    n = len(rows)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    cat_total = sum(1 for r in rows if r["expected_categories"])
    cat_hits = sum(1 for r in rows if r["expected_categories"] and r["category_hit"])
    return {
        "n": n,
        "accuracy": round((tp + tn) / n, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positives": fp,
        "false_negatives": fn,
        "false_positive_rate": round(fp / (fp + tn), 3) if (fp + tn) else 0.0,
        "category_recall": round(cat_hits / cat_total, 3) if cat_total else None,
    }


def evaluate(engine: str, items: list[dict]) -> tuple[dict, list[dict]]:
    from app.protection.content_classifier import classify_text
    from app.protection.rules import analyze_transcript

    rows = []
    for item in items:
        rule_result = analyze_transcript(item["text"])
        rule_suspicious = rule_result.signal_strength >= 3 and not rule_result.discussion_context
        predicted = rule_suspicious
        categories = {c.category for c in rule_result.categories}
        detail = {"rule_strength": rule_result.signal_strength}

        if engine == "hybrid":
            model = classify_text(item["text"])
            detail["model_label"] = model.label
            detail["model_confidence"] = model.confidence
            detail["model_available"] = model.status == "completed"
            # Same combination rule as the service: the trained classifier can
            # raise a rules-only "weak" verdict, but never vetoes strong rules.
            if model.status == "completed" and model.label == "scam" and model.confidence >= 0.65:
                predicted = predicted or rule_result.signal_strength >= 1
                categories = categories | {model.top_category} if model.top_category else categories

        rows.append(
            {
                "id": item["id"],
                "group": item["group"],
                "expected": item["expect_suspicious"],
                "predicted": predicted,
                "categories": sorted(c for c in categories if c),
                "expected_categories": item["expected_categories"],
                "category_hit": (
                    bool(set(item["expected_categories"]) & categories)
                    if item["expected_categories"]
                    else True
                ),
                **detail,
            }
        )
    return _score(rows), rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-failures", action="store_true")
    parser.add_argument("--engine", choices=["rules", "hybrid", "both"], default="both")
    args = parser.parse_args()

    items = load_items()
    engines = ["rules", "hybrid"] if args.engine == "both" else [args.engine]
    report: dict = {"dataset": str(DATA.name), "n": len(items), "engines": {}}
    all_rows: dict[str, list[dict]] = {}
    for engine in engines:
        metrics, rows = evaluate(engine, items)
        all_rows[engine] = rows
        report["engines"][engine] = metrics

    print(json.dumps(report, indent=2))

    if args.show_failures:
        for engine, rows in all_rows.items():
            missed = [r for r in rows if r["expected"] != r["predicted"]]
            cat_missed = [r for r in rows if r["expected_categories"] and not r["category_hit"]]
            print(f"\n[{engine}] verdict errors: {len(missed)}")
            for r in missed:
                print(f"  {r['id']}: expected={r['expected']} predicted={r['predicted']} "
                      f"strength={r.get('rule_strength')} cats={r['categories']}")
            print(f"[{engine}] category misses: {len(cat_missed)}")
            for r in cat_missed:
                print(f"  {r['id']}: expected={r['expected_categories']} got={r['categories']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
