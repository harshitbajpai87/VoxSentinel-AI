"""Re-save a detector artifact in the CURRENT environment.

Why: a joblib artifact is a pickle, and a pickle records the module paths of
the library versions that saved it. An artifact trained with scikit-learn 1.9
references ``sklearn.frozen._frozen``; a runtime with scikit-learn < 1.6 has no
such module and rejects the artifact, which used to surface as the opaque
``could not load artifact (ImportError)`` and silently degraded the API to
"model unavailable".

Run this inside the deployment environment once, and the artifact is written
with that environment's own module paths, so it loads natively from then on:

    python backend/scripts/repack_detector_artifact.py models/detector.joblib

    # verify first, write nothing:
    python backend/scripts/repack_detector_artifact.py models/detector.joblib --dry-run

Everything except the estimators is preserved byte-for-byte in meaning; the
estimators are re-pickled. A ``dependencies`` fingerprint is recorded in the
artifact so a later mismatch is reported with the exact versions.

Note: repacking requires the artifact to be loadable in THIS environment, which
is what the compatibility shims in app.services.artifact_compat provide.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib  # noqa: E402

from app.services.artifact_compat import (  # noqa: E402
    ArtifactLoadError,
    dependency_fingerprint,
    load_artifact_or_explain,
)

PRESERVED_KEYS = (
    "format",
    "artifact_version",
    "kind",
    "model_name",
    "model_version",
    "model_mode",
    "classes",
    "feature_config",
    "calibrated",
)


def repack(path: Path, dry_run: bool) -> int:
    if not path.is_file():
        print(f"error: no artifact at {path}", file=sys.stderr)
        return 2

    try:
        artifact = load_artifact_or_explain(path)
    except ArtifactLoadError as exc:
        print(f"error: {exc.detail}", file=sys.stderr)
        return 1

    if not isinstance(artifact, dict) or "classifier" not in artifact:
        print("error: this is not a VoxSentinel detector artifact", file=sys.stderr)
        return 1

    fingerprint = dependency_fingerprint()
    print("saving with: " + ", ".join(f"{k} {v}" for k, v in fingerprint.items()))

    rebuilt = {key: artifact[key] for key in PRESERVED_KEYS if key in artifact}
    rebuilt["scaler"] = artifact["scaler"]
    rebuilt["classifier"] = artifact["classifier"]
    rebuilt["dependencies"] = fingerprint
    rebuilt["repacked_at"] = datetime.now(UTC).isoformat(timespec="seconds")

    if dry_run:
        print(f"dry run: {path} is loadable here; nothing written")
        return 0

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    joblib.dump(rebuilt, path)
    print(f"repacked {path} (previous copy kept at {backup})")

    # Prove the rewrite loads on its own path.
    check = joblib.load(path)
    print(f"verified: reloaded keys {sorted(check)[:6]}...")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact",
        nargs="?",
        default="models/detector.joblib",
        help="path to detector.joblib (default: models/detector.joblib)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="only check that the artifact loads here"
    )
    args = parser.parse_args()
    return repack(Path(args.artifact), args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
