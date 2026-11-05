"""Build a training manifest from Gnani-generated synthetic samples (Phase 6).

Explicit, reproducible, and safe by default:

  - Input is the provenance manifest written by the /api/v1/synthesize
    storage path (data/generated/gnani/provenance.jsonl), where every row is
    labelled synthetic with full generator attribution.
  - Every row must carry provenance (generator_id, model, voice, language);
    rows without it are REJECTED, never guessed.
  - --confirm-usage-rights must be passed explicitly to state that the
    program's terms permit training use. The script refuses to run without it.
  - Samples are appended only to `train` or `validation` partitions. The
    `eval` partition is reserved for the existing test corpus and can never
    receive generated samples; the script rejects it loudly.
  - This script only builds a manifest. It NEVER retrains, NEVER overwrites
    the production model, and is never triggered by an API request. Model
    replacement happens only through scripts/train_candidate.py plus an
    explicit, separately evaluated approval step.

Usage:
    python scripts/build_generated_manifest.py \
        --generated-manifest data/generated/gnani/provenance.jsonl \
        --out-manifest data/gnani_synth_train.jsonl \
        --partition train \
        --confirm-usage-rights
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_PROVENANCE_FIELDS = ("generator_id", "model", "voice", "language")
ALLOWED_PARTITIONS = ("train", "validation")
SOURCE_ID = "gnani_timbre"

FAIL = 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-manifest", required=True,
                        help="provenance.jsonl produced by /api/v1/synthesize storage")
    parser.add_argument("--out-manifest", required=True,
                        help="output training manifest (JSONL, appended if it exists)")
    parser.add_argument("--partition", default="train", choices=ALLOWED_PARTITIONS,
                        help="partition label for the appended samples")
    parser.add_argument("--confirm-usage-rights", action="store_true",
                        help="required explicit statement that training use is permitted")
    parser.add_argument("--dataset-root", default=None,
                        help="optional root prefix recorded in each manifest path")
    args = parser.parse_args()

    if not args.confirm_usage_rights:
        print(
            "Refusing to build a training manifest: pass --confirm-usage-rights to "
            "explicitly confirm that the provider's terms of service and the "
            "program rules permit using generated audio as training data.",
            file=sys.stderr,
        )
        return FAIL

    src_path = Path(args.generated_manifest)
    if not src_path.is_file():
        print(f"Generated provenance manifest not found: {src_path}", file=sys.stderr)
        return FAIL

    accepted = 0
    rejected = 0
    out_path = Path(args.out_manifest)
    rows: list[str] = []
    with src_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"line {line_no}: not valid JSON — skipped", file=sys.stderr)
                rejected += 1
                continue
            missing = [f for f in REQUIRED_PROVENANCE_FIELDS if not row.get(f)]
            if row.get("label") != "synthetic" or missing:
                print(
                    f"line {line_no}: missing provenance "
                    f"({', '.join(missing) or 'label!=synthetic'}) — skipped",
                    file=sys.stderr,
                )
                rejected += 1
                continue
            path = row.get("path", "")
            if args.dataset_root:
                path = f"{args.dataset_root.rstrip('/')}/{path}"
            entry = {
                "path": path,
                "label": "synthetic",
                "partition": args.partition,
                "speaker_id": None,
                "source": SOURCE_ID,
                "generator_id": row["generator_id"],
                "model": row["model"],
                "voice": row["voice"],
                "language": row["language"],
            }
            rows.append(json.dumps(entry))
            accepted += 1

    if accepted == 0:
        print("No usable rows; nothing written.", file=sys.stderr)
        return FAIL

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row + "\n")

    print(
        f"Appended {accepted} synthetic training rows to {out_path} "
        f"(partition={args.partition}); rejected {rejected} rows. "
        "Evaluation partitions were not touched. No training was run and no "
        "model was modified."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
