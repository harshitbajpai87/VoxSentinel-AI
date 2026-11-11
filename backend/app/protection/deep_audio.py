"""Deep-audio evidence: AASIST, the SSL spoof branches and diarization.

This module adds the model families that need the decoded waveform to an
analysis result that is already built. It never fabricates a value: a detector
that cannot run appears with its status and the reason, and the fusion block says
which families actually contributed.

Nothing here changes the verdict of the scam-text layer. Fusion keeps the content
level as the overall level (see :mod:`app.protection.fusion`).
"""

from __future__ import annotations

import logging
import time

from app.protection.fusion import FusionInput, Vote, fuse_evidence, fuse_synthetic_audio
from app.services import ssl_spoof
from app.services.aasist import get_detector
from app.services.diarization import get_diarizer

logger = logging.getLogger("voxsentinel.deep_audio")


SAMPLE_RATE = 16000


def _decode(data: bytes, filename: str):
    """Decode to 16 kHz mono: the rate AASIST, ECAPA and the SSL branches expect."""
    from app.core.config import Settings  # noqa: PLC0415
    from app.services.audio_validation import validate_and_decode  # noqa: PLC0415

    decoded = validate_and_decode(data, filename, Settings())
    samples = decoded.samples
    rate = int(decoded.sample_rate)
    if rate == SAMPLE_RATE:
        return samples, SAMPLE_RATE
    import numpy as np  # noqa: PLC0415

    duration = samples.size / rate
    target = int(round(duration * SAMPLE_RATE))
    if target <= 1:
        return samples, rate
    positions = np.linspace(0, samples.size - 1, target)
    resampled = np.interp(positions, np.arange(samples.size), samples).astype("float32")
    return resampled, SAMPLE_RATE


def synthetic_audio_block(samples, sample_rate) -> dict:
    """AASIST plus the SSL branches, each voting separately."""
    votes: list[Vote] = []

    detector = get_detector()
    detector_status = detector.status()
    if detector_status["status"] == "completed":
        started = time.perf_counter()
        try:
            scored = detector.score(samples, sample_rate)
        except Exception as exc:  # noqa: BLE001 - a detector fault must not fail the request
            scored = {
                "status": "failed",
                "probability": None,
                "detail": f"{type(exc).__name__}: {str(exc)[:160]}",
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
        votes.append(
            Vote(
                name=detector_status["model"],
                family="aasist",
                probability=scored.get("probability"),
                threshold=detector_status.get("threshold") or 0.5,
                calibrated=bool(detector_status.get("calibrated")),
                status=scored.get("status", "completed"),
                detail=scored.get("detail"),
                latency_ms=scored.get("latency_ms"),
            )
        )
    else:
        votes.append(
            Vote(
                name=detector_status["model"],
                family="aasist",
                probability=None,
                status="unavailable",
                detail=detector_status.get("detail"),
            )
        )

    for branch in ssl_spoof.build_branches():
        status = branch.status()
        if status["status"] != "completed":
            votes.append(
                Vote(
                    name=status["model"],
                    family=branch.key,
                    probability=None,
                    status="unavailable",
                    detail=status.get("detail"),
                )
            )
            continue
        try:
            scored = branch.score(samples, sample_rate)
        except Exception as exc:  # noqa: BLE001
            scored = {"status": "failed", "probability": None, "detail": type(exc).__name__}
        votes.append(
            Vote(
                name=status["model"],
                family=branch.key,
                probability=scored.get("probability"),
                threshold=scored.get("threshold"),
                calibrated=bool(status.get("calibrated")),
                status=scored.get("status", "completed"),
                detail=scored.get("detail"),
                latency_ms=scored.get("latency_ms"),
                metrics=scored.get("metrics") or status.get("metrics"),
            )
        )

    return fuse_synthetic_audio(votes)


def diarization_block(samples, sample_rate) -> dict:
    diarizer = get_diarizer()
    try:
        return diarizer.diarize(samples, sample_rate)
    except Exception as exc:  # noqa: BLE001
        logger.info("diarization failed: %s", exc)
        return {
            "status": "failed",
            "detail": f"{type(exc).__name__}: {str(exc)[:160]}",
            "speakers": [],
            "speaker_count": 0,
            "segments": [],
            "turns": 0,
            "speech_s": 0.0,
            "limitations": ["Diarization failed, so no speaker boundaries are reported."],
        }


def augment(payload: dict, data: bytes | None, filename: str = "recording") -> dict:
    """Attach the deep-audio sections and the fusion verdict to a result."""
    samples = sample_rate = None
    decode_detail = None
    if data:
        try:
            samples, sample_rate = _decode(data, filename)
        except Exception as exc:  # noqa: BLE001 - the analysis already ran; keep it
            decode_detail = f"{type(exc).__name__}: {str(exc)[:160]}"

    if samples is not None and sample_rate:
        synthetic = synthetic_audio_block(samples, sample_rate)
        diarization = diarization_block(samples, sample_rate)
    else:
        reason = decode_detail or "No audio was supplied (text-only analysis)."
        synthetic = {
            "status": "unavailable",
            "level": None,
            "probability": None,
            "agreement": None,
            "confidence": 0.0,
            "corroborated": None,
            "detectors_used": 0,
            "detectors_available": 0,
            "disagreement": [],
            "summary": reason,
            "evidence": [],
            "models": [],
            "limitations": [
                "Voice-authenticity models need audio, so they did not run for this analysis."
            ],
        }
        diarization = {
            "status": "skipped",
            "detail": reason,
            "speakers": [],
            "speaker_count": 0,
            "segments": [],
            "turns": 0,
            "speech_s": 0.0,
            "limitations": ["Diarization needs audio, so it did not run for this analysis."],
        }

    scam = payload.get("scam_risk") or {}
    speaker = payload.get("speaker_verification") or payload.get("speaker_comparison") or {}
    transcription = payload.get("transcription") or {}
    fusion = fuse_evidence(
        FusionInput(
            synthetic=synthetic,
            scam=scam if scam else None,
            speaker=speaker,
            diarization=diarization,
            transcription=transcription,
            warnings=list(payload.get("warnings") or []),
        )
    )

    payload["synthetic_detection"] = synthetic
    payload["speaker_diarization"] = diarization
    payload["evidence_fusion"] = fusion
    payload["model_families_used"] = [
        name
        for name, section in fusion.get("sections", {}).items()
        if isinstance(section, dict) and (section.get("status") == "completed")
    ]
    return payload
