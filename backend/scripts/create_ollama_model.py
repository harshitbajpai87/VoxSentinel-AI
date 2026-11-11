"""Create a domain-adapted Ollama model for the scam classifier.

Ollama cannot train weights by itself. What it can do is *serve* an adapted
model, in two supported ways:

1. Prompt-level adaptation (this script's default). It builds a Modelfile that
   pins the production system instruction, decoding parameters and a few
   labelled examples from the fine-tuning dataset, then runs ``ollama create``.
   No GPU, no training run: the behaviour is steered by the prompt the model
   always sees, which is exactly the contract the backend expects.
2. Weight-level adaptation (LoRA). If you have produced an adapter with
   ``mistral-finetune`` / PEFT on a GPU, pass ``--adapter <dir>`` and the
   Modelfile will import it via the ADAPTER directive so Ollama serves the tuned
   weights.

Usage:
    python scripts/create_ollama_model.py --dry-run          # write the Modelfile only
    python scripts/create_ollama_model.py                    # create the model
    python scripts/create_ollama_model.py --adapter adapters/voxsentinel-lora

Then point the backend at it:
    VOXSENTINEL_OLLAMA_ENABLED=true
    OLLAMA_MODEL=voxsentinel-mistral
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.protection.gemini import SYSTEM_INSTRUCTION  # noqa: E402

DEFAULT_MODEL_NAME = "voxsentinel-mistral"
DEFAULT_BASE = "mistral"
DEFAULT_FEWSHOT = 4


def _triple(text: str) -> str:
    if '"""' in text:
        raise SystemExit("text contains a triple quote and cannot be embedded in a Modelfile")
    return f'"""{text}"""'


def _read_examples(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_fewshot(rows: list[dict], count: int) -> list[dict]:
    """One example per distinct verdict, so the prompt shows the whole spectrum."""
    picked: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        try:
            label = json.loads(row["messages"][2]["content"]).get("classification", "")
        except Exception:  # noqa: BLE001 - a malformed row is just skipped here
            continue
        if label in seen:
            continue
        seen.add(label)
        picked.append(row)
        if len(picked) >= count:
            break
    return picked


def build_modelfile(base: str, examples: list[dict], adapter: str | None) -> str:
    lines = [
        f"FROM {base}",
        "",
        "# Decoding: the classifier must be near-deterministic and must not ramble.",
        "PARAMETER temperature 0.1",
        "PARAMETER top_p 0.9",
        "PARAMETER num_ctx 8192",
        "",
        "SYSTEM " + _triple(SYSTEM_INSTRUCTION.strip()),
        "",
    ]
    for example in examples:
        user = example["messages"][1]["content"]
        assistant = example["messages"][2]["content"]
        lines.append("MESSAGE user " + _triple(user))
        lines.append("MESSAGE assistant " + _triple(assistant))
        lines.append("")
    if adapter:
        lines.append("# LoRA adapter produced by an offline fine-tuning run")
        lines.append(f"ADAPTER {adapter}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/finetune/train.jsonl"))
    parser.add_argument("--name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--base", default=DEFAULT_BASE, help="base Ollama model to adapt (mistral, mistral-nemo, ...)"
    )
    parser.add_argument(
        "--fewshot", type=int, default=DEFAULT_FEWSHOT, help="labelled examples baked into the prompt"
    )
    parser.add_argument("--adapter", default=None, help="path to a LoRA adapter to import (optional)")
    parser.add_argument("--out", type=Path, default=Path("ollama/Modelfile.generated"))
    parser.add_argument("--dry-run", action="store_true", help="write the Modelfile and stop")
    args = parser.parse_args()

    examples = select_fewshot(_read_examples(args.dataset), args.fewshot)
    content = build_modelfile(args.base, examples, args.adapter)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content, encoding="utf-8")
    print(f"Modelfile: {args.out} ({len(examples)} example(s), base {args.base})")
    if not examples:
        print("note: no dataset examples found, the model gets the system prompt only", file=sys.stderr)

    if args.dry_run:
        print("dry run: model not created")
        return 0

    if not shutil.which("ollama"):
        raise SystemExit(
            "the 'ollama' CLI is not on PATH. Install it from https://ollama.com/download, "
            f"then run: ollama create {args.name} -f {args.out}"
        )

    pull = subprocess.run(["ollama", "pull", args.base], check=False)
    if pull.returncode != 0:
        raise SystemExit(f"could not pull base model {args.base}")

    created = subprocess.run(["ollama", "create", args.name, "-f", str(args.out)], check=False)
    if created.returncode != 0:
        raise SystemExit("ollama create failed; see the output above")

    print(f"\ncreated local model: {args.name}")
    print("point the backend at it:")
    print(
        "  VOXSENTINEL_OLLAMA_ENABLED=true\n"
        f"  OLLAMA_MODEL={args.name}\n"
        "  PROTECTION_LLM_PROVIDER=ollama"
    )
    print(f"smoke test: ollama run {args.name} \"Papa mujhe kidnap kar liya hai, 50000 bhejo\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
