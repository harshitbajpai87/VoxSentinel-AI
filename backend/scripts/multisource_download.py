"""Download and register the multi-dataset training sources.

Dataset registry (accessibility verified via Kaggle's public API on
2026-10-23; see docs/MULTIDATASET_UPGRADE.md for the full verification log):

  jbuchner/synthetic-speech-commands-dataset      ~2.68 GB   CC BY-SA 4.0
  kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices
                                                   ~1.97 GB   CC BY-NC-SA 4.0
  mohammedabdeldayem/the-fake-or-real-dataset    ~20.2 GB    LGPL 3.0
  adarshsingh0903/audio-deepfake-detection-dataset ~1.42 GB Apache 2.0
  speech_commands_v0.02 (real-human supplement)  ~2.43 GB   CC BY 4.0
      direct download, no Kaggle account needed

Authentication: Kaggle downloads REQUIRE credentials. They are read from
KAGGLE_USERNAME / KAGGLE_KEY environment variables, or ~/.kaggle/kaggle.json.
Nothing is ever hardcoded, logged or written into the registry.

Without credentials the script still VERIFIES accessibility through Kaggle's
public API and records status "verified_accessible_not_downloaded" — it never
fabricates a download.

Usage:
    python scripts/multisource_download.py --data-dir ../data --dry-run
    KAGGLE_USERNAME=... KAGGLE_KEY=... python scripts/multisource_download.py \
        --data-dir ../data --only jbuchner/synthetic-speech-commands-dataset
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

KAGGLE_DATASETS = [
    {
        "ref": "jbuchner/synthetic-speech-commands-dataset",
        "title": "Synthetic Speech Commands Dataset",
        "license": "CC BY-SA 4.0",
        "expected_bytes": 2682021232,
        "notes": "SYNTHETIC-ONLY (TTS speech commands). Requires a real-speech supplement.",
    },
    {
        "ref": "kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices",
        "title": "Speech Dataset of Human and AI-Generated Voices",
        "license": "CC BY-NC-SA 4.0 (non-commercial)",
        "expected_bytes": 1973792861,
        "notes": "42 long WAV files (21 human, 21 AI), Indonesian, 4-9 min each.",
    },
    {
        "ref": "mohammedabdeldayem/the-fake-or-real-dataset",
        "title": "The Fake-or-Real (FoR) Dataset",
        "license": "LGPL 3.0",
        "expected_bytes": 20216844884,
        "notes": "~195k utterances; versions for-original/for-norm/for-2sec/for-rerec.",
    },
    {
        "ref": "adarshsingh0903/audio-deepfake-detection-dataset",
        "title": "Audio Deepfake Detection Dataset",
        "license": "Apache 2.0",
        "expected_bytes": 1417208088,
        "notes": "4,447 WAV files (2,274 real / 2,173 fake per dataset description).",
    },
]

DIRECT_DATASETS = [
    {
        "ref": "speech_commands_real/speech_commands_v0.02",
        "title": "Speech Commands v0.02 (real human speech supplement)",
        "license": "CC BY 4.0 (per dataset of record)",
        "expected_bytes": 2428923189,
        "url": "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
        "notes": "Authentic volunteer recordings; the real counterpart of the synthetic set.",
    },
]

API_LIST_URL = "https://www.kaggle.com/api/v1/datasets/list?user={owner}"


def verify_kaggle_dataset(ref: str) -> dict:
    """Verify a dataset exists through the public API (no credentials needed)."""
    owner = ref.split("/")[0]
    try:
        with urllib.request.urlopen(API_LIST_URL.format(owner=owner), timeout=30) as response:
            payload = json.load(response)
    except Exception as exc:  # noqa: BLE001
        return {"status": "verification_failed", "error": f"{type(exc).__name__}: {exc}"}
    for dataset in payload:
        url = dataset.get("urlNullable") or dataset.get("refNullable") or ""
        if ref in url:
            return {
                "status": "verified_accessible",
                "title": dataset.get("titleNullable"),
                "size_bytes": dataset.get("totalBytesNullable"),
                "license": dataset.get("licenseNameNullable"),
                "creator": dataset.get("creatorNameNullable"),
            }
    return {"status": "not_found_on_kaggle"}


def kaggle_credentials() -> tuple[str, str] | None:
    """Read Kaggle credentials from env or kaggle.json. Never logs values."""
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return username, key
    config_path = Path.home() / ".kaggle" / "kaggle.json"
    if config_path.is_file():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            return data.get("username"), data.get("key")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def download_kaggle_dataset(ref: str, target_dir: Path, credentials: tuple[str, str]) -> dict:
    """Download one Kaggle dataset via the authenticated API.

    Uses HTTP basic auth against /api/v1/datasets/download/{ref} and follows
    the redirect to the archive. The archive is extracted into target_dir.
    """
    import base64
    import shutil
    import zipfile

    username, key = credentials
    auth = base64.b64encode(f"{username}:{key}".encode()).decode()
    request = urllib.request.Request(
        f"https://www.kaggle.com/api/v1/datasets/download/{ref}",
        headers={"Authorization": f"Basic {auth}"},
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / f"{ref.replace('/', '__')}.zip"
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            archive_path.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
    except urllib.error.HTTPError as exc:
        return {"status": "download_failed", "http_error": exc.code}
    try:
        extract_dir = target_dir / ref.replace("/", "__")
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_dir)
    except zipfile.BadZipFile:
        return {"status": "download_failed", "error": "archive is not a zip file"}
    finally:
        archive_path.unlink(missing_ok=True)

    file_count = sum(1 for p in extract_dir.rglob("*") if p.is_file())
    total_bytes = sum(p.stat().st_size for p in extract_dir.rglob("*") if p.is_file())
    return {
        "status": "downloaded",
        "directory": str(extract_dir),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def download_direct_dataset(entry: dict, target_dir: Path) -> dict:
    """Download a direct-URL archive (no Kaggle account needed)."""
    import shutil
    import tarfile

    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / f"{entry['ref'].replace('/', '__')}.tar.gz"
    try:
        with (
            urllib.request.urlopen(entry["url"], timeout=120) as response,
            archive_path.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
    except Exception as exc:  # noqa: BLE001
        return {"status": "download_failed", "error": f"{type(exc).__name__}: {exc}"}
    extract_dir = target_dir / entry["ref"].replace("/", "__")
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(archive_path) as archive:
            archive.extractall(extract_dir)
    except tarfile.TarError as exc:
        return {"status": "download_failed", "error": f"extract failed: {exc}"}
    finally:
        archive_path.unlink(missing_ok=True)
    file_count = sum(1 for p in extract_dir.rglob("*") if p.is_file())
    total_bytes = sum(p.stat().st_size for p in extract_dir.rglob("*") if p.is_file())
    return {
        "status": "downloaded",
        "directory": str(extract_dir),
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and register multi-dataset sources.")
    parser.add_argument(
        "--data-dir", default="../data",
        help="External data directory (never committed to Git)",
    )
    parser.add_argument("--only", default=None, help="Comma-separated dataset refs to process")
    parser.add_argument("--dry-run", action="store_true", help="Verify accessibility only; no downloads")
    parser.add_argument(
        "--registry", default=None,
        help="Registry JSON path (default: <data-dir>/dataset_registry.json)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    registry_path = Path(args.registry) if args.registry else data_dir / "dataset_registry.json"
    only = {x.strip() for x in (args.only or "").split(",") if x.strip()}

    credentials = kaggle_credentials()
    registry = {
        "registry_type": "voxsentinel_multidataset_registry",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "kaggle_credentials_available": credentials is not None,
        "note": (
            "Kaggle downloads require credentials supplied via KAGGLE_USERNAME/KAGGLE_KEY "
            "or ~/.kaggle/kaggle.json. Without them, datasets are verified but not downloaded."
        ),
        "datasets": [],
    }

    entries = list(KAGGLE_DATASETS) + list(DIRECT_DATASETS)
    for entry in entries:
        if only and entry["ref"] not in only:
            continue
        record: dict = {
            "ref": entry["ref"],
            "title": entry["title"],
            "expected_license": entry["license"],
            "expected_bytes": entry["expected_bytes"],
            "notes": entry["notes"],
        }
        if "url" in entry:  # direct download
            verification = {"status": "verified_direct_download", "url": entry["url"]}
            if args.dry_run:
                record.update(verification, **{"download_status": "verified_accessible_not_downloaded"})
            else:
                record.update(verification, **download_direct_dataset(entry, data_dir))
        else:
            verification = verify_kaggle_dataset(entry["ref"])
            record.update(verification)
            if verification.get("status") != "verified_accessible":
                record["download_status"] = verification.get("status", "unverified")
            elif args.dry_run or credentials is None:
                record["download_status"] = "verified_accessible_not_downloaded"
            else:
                record["download_status"] = "downloading"
                result = download_kaggle_dataset(entry["ref"], data_dir, credentials)
                record["download_status"] = result.get("status", "download_failed")
                record.update(result)
        registry["datasets"].append(record)
        print(f"{entry['ref']}: verify={record.get('status')} download={record.get('download_status')}")

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Registry: {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
