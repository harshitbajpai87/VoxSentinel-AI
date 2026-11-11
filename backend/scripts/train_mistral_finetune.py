"""Submit a Mistral AI fine-tuning job for the scam classifier.

Uses the documented La Plateforme flow:
  POST /v1/files                 (purpose=fine-tune)  -> file id
  POST /v1/fine_tuning/jobs      (model, training_files, validation_files)
  GET  /v1/fine_tuning/jobs/{id}                      -> status / fine_tuned_model

The dataset must already exist (build it with
``scripts/build_scam_finetune_dataset.py``). Nothing is uploaded without an
explicit ``--submit``: the default is a validated dry run, because fine-tuning
costs money and cannot be undone.

    python scripts/train_mistral_finetune.py                    # validate only
    python scripts/train_mistral_finetune.py --submit            # real job
    python scripts/train_mistral_finetune.py --submit --wait     # poll to the end

The API key is read from MISTRAL_API_KEY. It is never printed, logged, or written
to disk. When the job finishes, serve the resulting model either through Mistral's
hosted endpoint (MISTRAL_MODEL=<fine_tuned_model>) or convert it for Ollama.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.protection.gemini import GeminiResult  # noqa: E402

API_BASE = "https://api.mistral.ai/v1"
DEFAULT_MODEL = "mistral-small-latest"
TERMINAL_STATES = {"SUCCESS", "SUCCESSFUL", "FAILED", "FAILURE", "CANCELLED", "CANCELLATION_REQUESTED"}
RECOMMENDED_MIN_EXAMPLES = 100


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no} is not valid JSON: {exc}") from exc
    return rows


def validate_dataset(path: Path) -> list[dict]:
    """Structural + schema validation before anything is uploaded."""
    if not path.exists():
        raise SystemExit(f"dataset not found: {path}\nRun scripts/build_scam_finetune_dataset.py first.")
    rows = _read_jsonl(path)
    if not rows:
        raise SystemExit(f"{path} is empty")
    for index, row in enumerate(rows, 1):
        messages = row.get("messages")
        if not isinstance(messages, list) or [m.get("role") for m in messages] != [
            "system",
            "user",
            "assistant",
        ]:
            raise SystemExit(f"{path}:{index} must have system/user/assistant messages")
        try:
            payload = json.loads(messages[2]["content"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise SystemExit(f"{path}:{index} assistant turn is not a JSON object: {exc}") from exc
        try:
            GeminiResult.model_validate(payload)
        except Exception as exc:  # noqa: BLE001 - any schema error blocks the run
            raise SystemExit(f"{path}:{index} assistant turn fails schema validation: {exc}") from exc
    return rows


def _headers() -> dict[str, str]:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise SystemExit(
            "MISTRAL_API_KEY is not set. Export it in the deployment environment (never commit it)."
        )
    return {"Authorization": f"Bearer {key}"}


def upload_file(path: Path, base_url: str) -> str:
    headers = _headers()
    with path.open("rb") as handle:
        files = {"file": (path.name, handle, "application/jsonl")}
        data = {"purpose": "fine-tune"}
        response = httpx.post(f"{base_url}/files", headers=headers, files=files, data=data, timeout=300.0)
    if response.status_code >= 400:
        raise SystemExit(f"upload failed: HTTP {response.status_code} {response.text[:300]}")
    file_id = response.json().get("id")
    if not file_id:
        raise SystemExit("upload succeeded but no file id was returned")
    print(f"uploaded {path.name} -> file id {file_id}")
    return file_id


def create_job(
    base_url: str,
    model: str,
    train_file: str,
    valid_file: str | None,
    suffix: str | None,
    steps: int | None,
) -> dict:
    body: dict = {"model": model, "training_files": [train_file]}
    if valid_file:
        body["validation_files"] = [valid_file]
    if suffix:
        body["suffix"] = suffix[:18]
    if steps:
        body["hyperparameters"] = {"training_steps": steps}
    response = httpx.post(f"{base_url}/fine_tuning/jobs", headers=_headers(), json=body, timeout=120.0)
    if response.status_code >= 400:
        raise SystemExit(f"job creation failed: HTTP {response.status_code} {response.text[:500]}")
    return response.json()


def poll_job(base_url: str, job_id: str, interval_s: int = 30) -> dict:
    print(f"polling job {job_id} every {interval_s}s (Ctrl-C stops watching; the job keeps running)")
    while True:
        response = httpx.get(f"{base_url}/fine_tuning/jobs/{job_id}", headers=_headers(), timeout=60.0)
        if response.status_code >= 400:
            raise SystemExit(f"status check failed: HTTP {response.status_code} {response.text[:300]}")
        job = response.json()
        status = str(job.get("status") or "").upper()
        print(f"  status: {status}")
        if status in TERMINAL_STATES:
            return job
        time.sleep(interval_s)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train", type=Path, default=Path("data/finetune/train.jsonl"))
    parser.add_argument("--valid", type=Path, default=Path("data/finetune/valid.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL, help="base model to fine-tune")
    parser.add_argument("--suffix", default="voxsentinel", help="name suffix for the tuned model")
    parser.add_argument("--steps", type=int, default=None, help="training_steps hyperparameter (optional)")
    parser.add_argument("--api-base", default=os.environ.get("MISTRAL_API_BASE", API_BASE))
    parser.add_argument("--submit", action="store_true", help="actually upload and start the job")
    parser.add_argument("--wait", action="store_true", help="poll until the job reaches a terminal state")
    parser.add_argument("--poll-interval", type=int, default=30)
    args = parser.parse_args()

    rows = validate_dataset(args.train)
    valid_rows: list[dict] = []
    if args.valid.exists():
        valid_rows = validate_dataset(args.valid)

    print(f"train: {len(rows)} examples ({args.train})")
    print(f"valid: {len(valid_rows)} examples ({args.valid})" if valid_rows else "valid: none")
    print(f"base model: {args.model}")
    if len(rows) < RECOMMENDED_MIN_EXAMPLES:
        print(
            f"WARNING: {len(rows)} examples is below the {RECOMMENDED_MIN_EXAMPLES} recommended for a "
            "usable adapter. A job started now will fit the format, not the problem.",
            file=sys.stderr,
        )

    if not args.submit:
        print("dry run only. Re-run with --submit to upload and start a fine-tuning job.")
        return 0

    train_id = upload_file(args.train, args.api_base)
    valid_id = upload_file(args.valid, args.api_base) if valid_rows else None
    job = create_job(args.api_base, args.model, train_id, valid_id, args.suffix, args.steps)
    job_id = job.get("id")
    print(f"job created: {job_id} (status {job.get('status')})")
    if args.wait and job_id:
        final = poll_job(args.api_base, job_id, args.poll_interval)
        print(json.dumps(final, indent=2)[:2000])
        model = final.get("fine_tuned_model")
        if model:
            print(f"\nserve it with: MISTRAL_MODEL={model} PROTECTION_LLM_PROVIDER=mistral")
    else:
        print(
            "check it with: curl -H \"Authorization: Bearer $MISTRAL_API_KEY\" "
            f"{args.api_base}/fine_tuning/jobs/{job_id}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
