"""Unseen-generator holdout evaluation: per-generator recall on held-out synthetics.

The holdout partition contains ONLY synthetic clips from generators never seen
in training (xTTS, FlashSpeech, VoiceBox, NaturalSpeech3), so precision/FPR are
undefined there; the meaningful generalization metric is the recall per
generator at the standard evaluation threshold (0.5), computed through the
SAME production inference path (DetectorService.analyze) the API uses.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.manifest import load_manifest  # noqa: E402
from app.services.model_loader import load_detector_from_path  # noqa: E402
from scripts.train_detector import _read_audio  # noqa: E402


def _score_of(result) -> float:
    """Same score the risk engine uses: calibrated probability if available."""
    if result.calibrated_probability is not None:
        return float(result.calibrated_probability)
    return float(result.raw_score)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--name", default="model")
    args = parser.parse_args()

    detector = load_detector_from_path(args.artifact)
    data_root = Path(args.dataset_dir)

    entries = [e for e in load_manifest(Path(args.manifest)) if e.partition == "eval"]
    per_gen: dict[str, list[float]] = defaultdict(list)
    for entry in entries:
        mono, sample_rate = _read_audio(entry, data_root)
        result = detector.analyze(mono, sample_rate, 1)
        per_gen[entry.generator_id or "unknown"].append(_score_of(result))

    report = {"model": args.name, "artifact": args.artifact, "n": len(entries)}
    print(f"== {args.name} on unseen generators (n={len(entries)}, threshold 0.5) ==")
    flagged = 0
    for gen in sorted(per_gen):
        scores = per_gen[gen]
        preds = [s >= 0.5 for s in scores]
        recall = sum(preds) / len(scores)
        mean_score = sum(scores) / len(scores)
        flagged += sum(preds)
        report[gen] = {
            "n": len(scores),
            "recall": round(recall, 4),
            "mean_score": round(mean_score, 4),
        }
        print(f"  {gen:16s} n={len(scores):4d} recall={recall:.4f} mean_score={mean_score:.3f}")
    overall_recall = flagged / len(entries)
    report["overall_recall"] = round(overall_recall, 4)
    print(f"  {'OVERALL':16s} n={len(entries):4d} recall={overall_recall:.4f}")
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "holdout_generators.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
