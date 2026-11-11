"""Speaker diarization: who spoke when, as cluster labels.

The segmentation step finds speech regions, the ECAPA embedder turns each region
into a vector, and agglomerative clustering on cosine distance groups the vectors
into speakers. Labels are cluster ids (SPEAKER_01, ...). Nothing here identifies a
person: naming a speaker is out of scope and is never inferred from a cluster.

Segmentation sources, in order of preference:
  1. ``pyannote/segmentation-3.0`` when it can be loaded (it is a gated repo, so a
     token or a local copy is required);
  2. energy-based voice activity detection over 20 ms frames, which needs no
     download and is reported as ``energy_vad`` so the weaker boundary quality is
     visible in the response.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("voxsentinel.diarization")

BACKEND_DIR = Path(__file__).resolve().parents[2]
ECAPA_DIR = BACKEND_DIR / "models" / "speaker_ecapa"
CALIBRATION_PATH = BACKEND_DIR / "models" / "diarization_calibration.json"

SAMPLE_RATE = 16000
HOP_S = 0.5
CHUNK_S = 2.0
MIN_TURN_S = 0.35
MAX_SPEAKERS = 6
DEFAULT_THRESHOLD = 0.55
PYANNOTE_ID = "pyannote/segmentation-3.0"


def _load_calibration() -> dict | None:
    if not CALIBRATION_PATH.is_file():
        return None
    try:
        return json.loads(CALIBRATION_PATH.read_text())
    except Exception:  # noqa: BLE001 - a broken calibration file must not break analysis
        return None


_CALIBRATION = _load_calibration()


@dataclass
class Turn:
    start: float
    end: float
    speaker: str
    confidence: float | None = None

    def as_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "speaker": self.speaker,
            "confidence": self.confidence,
        }


def _regions_from_grid(grid: np.ndarray, grid_step: float = 0.02) -> list[tuple[float, float]]:
    """Contiguous speech runs from a per-frame activity grid."""
    regions: list[tuple[float, float]] = []
    start: float | None = None
    for index, active in enumerate(grid):
        if active and start is None:
            start = index * grid_step
        elif not active and start is not None:
            if index * grid_step - start >= 0.12:
                regions.append((start, index * grid_step))
            start = None
    if start is not None:
        regions.append((start, len(grid) * grid_step))
    return regions


def _energy_regions(samples: np.ndarray, sample_rate: int) -> list[tuple[float, float]]:
    """Energy VAD on 20 ms frames, thresholded against the clip's own floor."""
    step = int(0.02 * sample_rate)
    if samples.size < step * 2:
        return [(0.0, samples.size / sample_rate)] if samples.size else []
    frames = samples[: samples.size - samples.size % step].reshape(-1, step)
    rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    noise = float(np.percentile(db, 15))
    speech = float(np.percentile(db, 90))
    threshold = noise + max(6.0, 0.35 * (speech - noise))
    return [
        (start, end)
        for start, end in _regions_from_grid(db >= threshold)
        if end - start >= 0.2
    ]


class Diarizer:
    """pyannote/ECAPA diarization with an energy-VAD fallback."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._embedder = None
        self._seg_model = None
        self._seg_error: str | None = None
        self._seg_checked = False
        self._enabled = True

    # -- capability probe --------------------------------------------------
    def _enabled_flag(self) -> bool:
        import os

        return (os.environ.get("VOXSENTINEL_DIARIZATION") or "true").lower() not in {
            "0",
            "false",
            "no",
        }

    def _pyannote_ready(self) -> tuple[bool, str | None]:
        """Whether the gated segmentation model can be used, and why not."""
        if self._seg_checked:
            return (self._seg_model is not None), self._seg_error
        self._seg_checked = True
        import os

        local = BACKEND_DIR / "models" / "hf" / "pyannote-segmentation-3.0"
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        try:
            from pyannote.audio import Model  # noqa: PLC0415 - optional dependency

            if local.is_dir() and any(local.glob("*.bin")):
                self._seg_model = Model.from_pretrained(str(local))
            elif token:
                self._seg_model = Model.from_pretrained(PYANNOTE_ID, use_auth_token=token)
            else:
                self._seg_error = (
                    f"{PYANNOTE_ID} is a gated repository: set HF_TOKEN or place a copy at "
                    f"{local}. Energy-based boundaries were used instead."
                )
        except Exception as exc:  # noqa: BLE001 - never fatal
            self._seg_error = f"{type(exc).__name__}: {str(exc)[:160]}"
        return (self._seg_model is not None), self._seg_error

    def _ecapa(self):
        if self._embedder is not None:
            return self._embedder
        with self._lock:
            if self._embedder is not None:
                return self._embedder
            from speechbrain.inference.speaker import EncoderClassifier  # noqa: PLC0415

            source = str(ECAPA_DIR) if ECAPA_DIR.is_dir() else "speechbrain/spkrec-ecapa-voxceleb"
            self._embedder = EncoderClassifier.from_hparams(
                source=source, savedir=str(ECAPA_DIR), run_opts={"device": "cpu"}
            )
            return self._embedder

    def status(self) -> dict:
        self._enabled = self._enabled_flag()
        if not self._enabled:
            state, detail = "unavailable", "Disabled by configuration (VOXSENTINEL_DIARIZATION=false)."
            embedder = None
        else:
            try:
                embedder = self._ecapa()
                state, detail = "completed", None
            except Exception as exc:  # noqa: BLE001
                embedder = None
                state = "failed"
                detail = f"ECAPA embeddings unavailable at {ECAPA_DIR}: {type(exc).__name__}"
        pyannote_ok, pyannote_reason = self._pyannote_ready() if self._enabled else (False, None)
        return {
            "status": state,
            "detail": detail,
            "backend": "pyannote_segmentation+ecapa_clustering" if pyannote_ok else "ecapa_clustering",
            "segmentation": "pyannote" if pyannote_ok else "energy_vad",
            "segmentation_model": PYANNOTE_ID if pyannote_ok else None,
            "embedding_model": "speechbrain/spkrec-ecapa-voxceleb" if embedder else None,
            "pyannote_note": pyannote_reason,
            "cosine_threshold": self._threshold(),
            "calibrated": _CALIBRATION is not None,
            "calibration": _CALIBRATION,
            "min_turn_s": MIN_TURN_S,
            "max_speakers": MAX_SPEAKERS,
        }

    def _threshold(self) -> float:
        if _CALIBRATION and isinstance(_CALIBRATION.get("threshold"), (int, float)):
            return float(_CALIBRATION["threshold"])
        return DEFAULT_THRESHOLD

    # -- inference ---------------------------------------------------------
    def _activity(self, samples: np.ndarray, sample_rate: int) -> tuple[list[tuple[float, float]], str]:
        ok, _ = self._pyannote_ready()
        if ok:
            try:
                import torch  # noqa: PLC0415

                waveform = torch.from_numpy(samples.astype("float32")).unsqueeze(0)
                with torch.no_grad():
                    output = self._seg_model(waveform.unsqueeze(0), torch.tensor([sample_rate]))
                powerset = output if torch.is_tensor(output) else output[0]
                active = powerset[0, 0].detach().cpu().numpy().reshape(-1)
                grid = active > 0.5
                frames_per_step = max(1, int(round(0.02 * sample_rate * (samples.size / sample_rate) / max(1, grid.size))))
                if samples.size and grid.size:
                    seconds_per_frame = (samples.size / sample_rate) / grid.size
                    del frames_per_step
                    regions = [
                        (start, end)
                        for start, end in _regions_from_grid(grid, seconds_per_frame)
                        if end - start >= 0.12
                    ]
                    if regions:
                        return regions, "pyannote"
            except Exception as exc:  # noqa: BLE001
                logger.info("pyannote segmentation failed, using energy VAD: %s", exc)
        return _energy_regions(samples, sample_rate), "energy_vad"

    def _embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch  # noqa: PLC0415

        embedder = self._ecapa()
        wav = torch.from_numpy(samples.astype("float32")).unsqueeze(0)
        with torch.no_grad():
            vectors = embedder.encode_batch(wav)
        return vectors.detach().cpu().numpy().reshape(-1)

    def diarize(self, samples: np.ndarray, sample_rate: int) -> dict:
        started = time.perf_counter()
        self._enabled = self._enabled_flag()
        status = self.status()
        if not self._enabled or status["status"] != "completed":
            return {
                "status": status["status"],
                "backend": status.get("backend"),
                "segmentation": status.get("segmentation"),
                "segmentation_model": status.get("segmentation_model"),
                "embedding_model": status.get("embedding_model"),
                "speakers": [],
                "speaker_count": 0,
                "segments": [],
                "turns": 0,
                "speech_s": 0.0,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "detail": status.get("detail"),
                "limitations": [
                    "Diarization did not run, so no speaker boundaries are reported.",
                    "Speaker labels are cluster ids, not identities: nothing here names a person.",
                ],
            }

        regions, segmentation = self._activity(samples, sample_rate)
        if len(regions) < 2:
            return {
                "status": "completed",
                "backend": status["backend"],
                "segmentation": segmentation,
                "segmentation_model": status["segmentation_model"],
                "embedding_model": status["embedding_model"],
                "speakers": ["SPEAKER_01"] if regions else [],
                "speaker_count": 1 if regions else 0,
                "segments": (
                    [Turn(regions[0][0], regions[0][1], "SPEAKER_01").as_dict()] if regions else []
                ),
                "turns": 1 if regions else 0,
                "speech_s": round(sum(end - start for start, end in regions), 2),
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "detail": "Fewer than two speech regions: a single speaker cannot be ruled out.",
                "limitations": [
                    "Speaker labels are cluster ids, not identities: nothing here names a person.",
                ],
            }

        def windows(regions: list[tuple[float, float]]) -> list[tuple[float, float]]:
            out: list[tuple[float, float]] = []
            for start, end in regions:
                cursor = start
                while cursor < end - 1e-6:
                    stop = min(end, cursor + CHUNK_S)
                    if stop - cursor >= 1.2:
                        out.append((cursor, stop))
                    cursor = stop
            return out

        chunks = windows(regions)
        if not chunks:
            chunks = [(start, end) for start, end in regions if end - start >= 0.75] or regions
        embeddings = np.stack(
            [
                self._embed(
                    samples[int(start * sample_rate) : int(end * sample_rate)], sample_rate
                )
                for start, end in chunks
            ]
        )
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        unit = embeddings / norms
        similarity = unit @ unit.T
        distance = np.clip(1.0 - similarity, 0.0, 2.0)

        labels, margin = self._cluster(distance)
        uncalibrated = _CALIBRATION is None
        if uncalibrated and len(set(labels.tolist())) > 1 and margin < 0.08:
            # The clusters are not cleanly separated. Reporting "N speakers" here
            # would be a guess dressed as a measurement, so the count is withheld.
            labels = labels[:] = 0
            margin = 0.0

        turns: list[Turn] = []
        for (start, end), label in zip(chunks, labels, strict=False):
            if turns and turns[-1].speaker == f"SPEAKER_{label + 1:02d}" and start - turns[-1].end <= HOP_S:
                turns[-1] = Turn(turns[-1].start, end, turns[-1].speaker, turns[-1].confidence)
            else:
                turns.append(Turn(start, end, f"SPEAKER_{label + 1:02d}"))

        durations: dict[str, float] = {}
        for turn in turns:
            durations[turn.speaker] = durations.get(turn.speaker, 0.0) + (turn.end - turn.start)
        kept = {name for name, total in durations.items() if total >= MIN_TURN_S}
        turns = [turn for turn in turns if turn.speaker in kept] or turns
        speakers = sorted({turn.speaker for turn in turns})
        speech_s = float(sum(end - start for start, end in regions))
        limitations = [
            "Speaker labels are cluster ids, not identities: nothing here names a person.",
            "Near-simultaneous speech and very short turns can merge into one speaker.",
        ]
        if segmentation != "pyannote":
            limitations.append(
                "Boundaries come from energy VAD, not the pyannote segmentation model, so turn edges are approximate."
            )
        if _CALIBRATION is None:
            limitations.append(
                f"The clustering cutoff ({DEFAULT_THRESHOLD:.2f}) is a documented default, not a fitted value on this host."
            )
        return {
            "status": "completed",
            "backend": status["backend"],
            "segmentation": segmentation,
            "segmentation_model": status["segmentation_model"],
            "embedding_model": status["embedding_model"],
            "segments": [turn.as_dict() for turn in turns[:400]],
            "speakers": speakers,
            "speaker_count": len(speakers),
            "speech_s": round(speech_s, 2),
            "turns": len(turns),
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "detail": None if turns else "No speaker turns survived the minimum-duration filter.",
            "limitations": limitations,
        }

    def _cluster(self, distance: np.ndarray) -> tuple[np.ndarray, float]:
        """Average-linkage clustering; also returns the merge margin.

        The margin is how much lower the last accepted merge distance is than the
        next rejected one. A small margin means the cut is arbitrary: the caller
        uses that to decide whether a speaker count is defensible.
        """
        count = distance.shape[0]
        if count == 0:
            return np.zeros(0, dtype=int), 0.0
        clusters = [[index] for index in range(count)]
        threshold = self._threshold()
        last_accepted = 0.0
        next_rejected: float | None = None
        while len(clusters) > 1 and len(clusters) > 1:
            best: tuple[float, int, int] | None = None
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    block = distance[np.ix_(clusters[i], clusters[j])]
                    mean = float(block.mean())
                    if best is None or mean < best[0]:
                        best = (mean, i, j)
            if best is None:
                break
            if best[0] > threshold or len(clusters) <= 1:
                next_rejected = best[0]
                break
            _, i, j = best
            last_accepted = best[0]
            if len(clusters) <= 1:
                break
            clusters[i] = sorted(clusters[i] + clusters[j])
            clusters.pop(j)
        labels = np.zeros(count, dtype=int)
        for label, members in enumerate(sorted(clusters, key=lambda c: min(c))):
            for member in members:
                labels[member] = label
        if next_rejected is None:
            margin = threshold - last_accepted
        else:
            margin = next_rejected - last_accepted
        return labels, float(margin)


_DIARIZER: Diarizer | None = None
_DIARIZER_LOCK = threading.Lock()


def get_diarizer() -> Diarizer:
    global _DIARIZER
    if _DIARIZER is None:
        with _DIARIZER_LOCK:
            if _DIARIZER is None:
                _DIARIZER = Diarizer()
    return _DIARIZER
