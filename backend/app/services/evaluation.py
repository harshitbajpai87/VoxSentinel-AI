"""Evaluation jobs (Module 5): REAL inference runs with progress tracking.

POST /api/v1/evaluation/run starts a background thread that loads the real
manifest, extracts features with the SAME pipeline as serving, runs the real
detector, computes metrics with app.services.metrics (shared with the CLI
script — single source of truth), persists a report, and appends an audit
event. Job state is real: queued -> running (with n/total progress) ->
completed/failed. No mock progress, no fabricated numbers.
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings
from app.services.audit import log_event
from app.services.features import FeatureConfig, extract_features
from app.services.manifest import ManifestError, load_manifest, require_partition_classes
from app.services.metrics import compute_binary_metrics

logger = logging.getLogger("voxsentinel.evaluation")


class EvaluationError(RuntimeError):
    """Honest failure: missing dataset/model/labels — never fabricated."""


class _JobState:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self.status = "queued"
        self.progress_done = 0
        self.progress_total = 0
        self.error: str | None = None
        self.started_at = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        self.finished_at: str | None = None
        self.result: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": {"done": self.progress_done, "total": self.progress_total},
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metrics": self.result.get("metrics") if self.result else None,
            "report": self.result,
        }


class EvaluationJobManager:
    """In-process job registry (single-worker deployment, documented)."""

    def __init__(self, settings: Settings, detector_service) -> None:
        self._settings = settings
        self._detector_service = detector_service
        self._jobs: dict[str, _JobState] = {}
        self._lock = threading.Lock()

    def latest_job(self) -> _JobState | None:
        with self._lock:
            if not self._jobs:
                return None
            return self._jobs[max(self._jobs)]

    def get(self, job_id: str) -> _JobState | None:
        return self._jobs.get(job_id)

    def start(self, partition: str | None = None) -> _JobState:
        with self._lock:
            running = [j for j in self._jobs.values() if j.status in ("queued", "running")]
            if running:
                raise EvaluationError(
                    "An evaluation job is already running "
                    f"(id {running[0].job_id}); wait for it to finish."
                )
            job_id = uuid.uuid4().hex[:16]
            state = _JobState(job_id)
            self._jobs[job_id] = state
        thread = threading.Thread(
            target=self._execute, args=(state, partition or self._settings.eval_partition),
            daemon=True, name=f"eval-{job_id}",
        )
        thread.start()
        return state

    # ------------------------------------------------------------------
    def _execute(self, state: _JobState, partition: str) -> None:
        try:
            state.status = "running"
            result = self._run_partition(state, partition)
            state.result = result
            state.status = "completed"
            state.finished_at = _now_iso()
            self._audit(state.job_id, "evaluation_run", "success",
                        {"partition": partition, "n": result["split"]["n"],
                         "model_version": result["model"]["version"]})
        except Exception as exc:  # noqa: BLE001 - surfaced safely to the client
            state.status = "failed"
            state.error = str(exc) or type(exc).__name__
            state.finished_at = _now_iso()
            logger.warning("evaluation job %s failed: %s", state.job_id, state.error)
            self._audit(state.job_id, "evaluation_run", "failed",
                        {"partition": partition, "error": state.error})

    def _audit(self, job_id: str, action: str, outcome: str, meta: dict) -> None:
        try:
            from app.db.session import Database  # local import avoids cycles

            db = Database(self._settings.db_url)
            session = db.session()
            try:
                log_event(session, "model_evaluation", action, outcome,
                          resource_id=job_id, metadata=meta)
            finally:
                session.close()
                db.dispose()
        except Exception:  # noqa: BLE001 - audit must not break the job
            logger.warning("evaluation audit write failed", exc_info=True)

    def _run_partition(self, state: _JobState, partition: str) -> dict[str, Any]:
        settings = self._settings
        manifest_path = Path(settings.eval_manifest_path)
        if not manifest_path.is_file():
            raise EvaluationError(
                f"No evaluation dataset manifest at {manifest_path}. Prepare one with "
                "scripts/prepare_dataset.py (see docs/DATASET_GUIDE.md)."
            )
        entries = load_manifest(manifest_path)
        try:
            require_partition_classes(entries, partition)
        except ManifestError as exc:
            raise EvaluationError(str(exc)) from exc

        dataset_dir = Path(settings.eval_dataset_dir)
        detector = self._detector_service
        if not detector.is_ready():
            raise EvaluationError("No trained detector model is installed on this server.")

        state.progress_total = len([e for e in entries if e.partition == partition])
        rows: list[tuple[int, float]] = []
        failed: list[str] = []
        for entry in entries:
            if entry.partition != partition:
                continue
            path = dataset_dir / entry.path
            try:
                samples, sr = _read_audio(path)
                label = 1 if entry.label == "synthetic" else 0
                features = extract_features(samples, sr, FeatureConfig(**detector.feature_config_asdict()))
                score = detector.score_features(features)
                rows.append((label, score))
            except Exception:  # noqa: BLE001 - recorded, never guessed
                failed.append(entry.path)
            state.progress_done += 1

        if not rows:
            raise EvaluationError(
                f"No evaluation samples could be processed from partition '{partition}'."
            )
        y = np.array([r[0] for r in rows], dtype=int)
        scores = np.array([r[1] for r in rows], dtype=float)
        if len(np.unique(y)) < 2:
            raise EvaluationError(
                f"Partition '{partition}' ended up with a single class; "
                "class-dependent metrics are undefined."
            )

        metrics = compute_binary_metrics(y, scores, threshold=0.5)
        report = {
            "report_type": "evaluation_run",
            "job_id": state.job_id,
            "model": {
                "name": detector.model_name(),
                "version": detector.model_version(),
            },
            "dataset": {
                "manifest": str(manifest_path),
                "partition": partition,
                "sources": sorted({e.source for e in entries if e.partition == partition}),
                "class_distribution": metrics["class_distribution"],
                "processed": len(rows),
                "failed": len(failed),
                "failed_examples": failed[:10],
            },
            "split": {
                "partition": partition,
                "n": len(rows),
                "note": (
                    "Held-out partition: the training script only reads train/validation; "
                    "this partition was never used for model selection."
                ),
            },
            "metrics": metrics,
            "evaluated_at": _now_iso(),
        }

        # persist report (latest + per-run) — used by GET /evaluation/metrics
        report_dir = Path(settings.report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "evaluation_latest.json").write_text(json.dumps(report, indent=2))
        with (report_dir / "evaluation_runs.jsonl").open("a") as handle:
            handle.write(json.dumps(report) + "\n")
        return report


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.ndim == 2 else data
    if mono.size == 0:
        raise ValueError("empty audio")
    return mono, int(sr)
