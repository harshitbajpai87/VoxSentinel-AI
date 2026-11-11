"""Evidence fusion: one documented level over several model families.

Two rules decide every number in this module.

1. Scores from different families are never averaged into a "confidence". Each
   model keeps its own score, and a voice score is reported as a voice score.
   Confidence is null until a joint calibration set has been evaluated.
2. Content-level scam evidence sets the overall level. Voice, speaker and
   diarization evidence can corroborate concern, but they can
   never on their own produce a HIGH or CRITICAL finding, and no model output is
   treated as an identification of a person.
"""

from __future__ import annotations

from dataclasses import dataclass, field

POLICY = (
    "Content-level scam evidence sets the overall level. Voice, speaker and diarization "
    "evidence can corroborate concern but never alone produce a HIGH or "
    "CRITICAL finding, and no model output is treated as an identification of a person."
)

LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL", "INCONCLUSIVE")
LEVEL_ORDER = {"INCONCLUSIVE": -1, "LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

FAMILIES = (
    "synthetic_audio",
    "scam_content",
    "speaker_similarity",
    "diarization",
    "transcription",
)

#: Prior trust in each detector family, used only to weight votes inside the
#: synthetic-audio family. A measured EER halves the distance to the prior.
BASE_WEIGHTS = {
    "aasist": 0.65,
    "wavlm": 0.55,
    "xlsr": 0.50,
    "acoustic": 0.45,
    "whisper": 0.40,
}

_LIMITS = [
    "A synthetic-voice score is evidence about the audio, not proof of fraud or of who spoke.",
    "Detectors trained on ASVspoof-style corpora underperform on unseen generators; "
    "a low score is not a clean bill of health.",
    "No model here identifies a person; speaker labels are clusters and similarity "
    "is to an enrolled reference only.",
]


@dataclass
class Vote:
    """One model's opinion, kept separate from every other model's opinion."""

    name: str
    family: str
    probability: float | None
    threshold: float | None = None
    calibrated: bool = False
    status: str = "completed"
    weight: float | None = None
    detail: str | None = None
    latency_ms: int | None = None
    metrics: dict | None = None

    def reliability(self) -> float:
        base = BASE_WEIGHTS.get(self.family, 0.5) if self.weight is None else float(self.weight)
        if self.metrics and isinstance(self.metrics.get("eer"), (int, float)):
            base = 0.5 * base + 0.5 * max(0.05, 1.0 - float(self.metrics["eer"]))
        if not self.calibrated:
            base *= 0.7  # an uncalibrated score is a vote, not a probability
        return round(base, 4)

    def suspect(self) -> bool | None:
        if self.probability is None:
            return None
        cut = self.threshold if self.threshold is not None else 0.5
        return self.probability >= float(cut)

    def as_dict(self) -> dict:
        return {
            "model": self.name,
            "family": self.family,
            "status": self.status,
            "probability": None if self.probability is None else round(self.probability, 4),
            "threshold": self.threshold,
            "calibrated": self.calibrated,
            "vote": (
                "synthetic"
                if self.suspect()
                else ("natural" if self.probability is not None else None)
            ),
            "reliability": self.reliability(),
            "latency_ms": self.latency_ms,
            "detail": self.detail,
            "metrics": self.metrics,
        }


def _level_from(probability: float, agreement: float, scored: int) -> str:
    if scored == 0:
        return "INCONCLUSIVE"
    if probability >= 0.75 and agreement >= 0.6:
        return "HIGH"
    if probability >= 0.5:
        return "MEDIUM"
    if probability <= 0.35:
        return "LOW"
    return "INCONCLUSIVE"


def fuse_synthetic_audio(votes: list[Vote]) -> dict:
    """Weighted vote over the synthetic-audio detectors.

    A detector that did not score is listed with the reason it could not run. It
    is never replaced by an invented score, and it does not dilute the weighted
    probability of the detectors that did run.
    """
    models = [vote.as_dict() for vote in votes]
    scored = [v for v in votes if v.probability is not None and v.status == "completed"]
    available = len(votes)

    if not scored:
        reasons = "; ".join(
            f"{v.name}: {v.detail or v.status}" for v in votes if v.status != "completed"
        )
        return {
            "status": "unavailable",
            "level": None,
            "probability": None,
            "agreement": None,
        "confidence": None,
        "confidence_definition": "No calibrated joint confidence model is installed.",
            "corroborated": None,
            "detectors_used": 0,
            "detectors_available": available,
            "disagreement": [],
            "summary": "No synthetic-audio detector could score this recording.",
            "evidence": [r for r in reasons.split("; ") if r],
            "models": models,
            "limitations": _LIMITS,
        }

    weight_total = sum(v.reliability() for v in scored) or 1.0
    probability = sum(v.reliability() * float(v.probability) for v in scored) / weight_total
    synthetic_votes = sum(1 for v in scored if v.suspect())
    majority = synthetic_votes * 2 >= len(scored)
    agreement = (synthetic_votes if majority else len(scored) - synthetic_votes) / len(scored)
    level = _level_from(probability, agreement, len(scored))
    disagreement = [
        v.name for v in scored if v.suspect() is not None and bool(v.suspect()) != majority
    ]
    evidence = [
        (
            f"{v.name}: {float(v.probability):.2f} "
            f"(cutoff {'n/a' if v.threshold is None else f'{float(v.threshold):.2f}'}, "
            f"{'calibrated' if v.calibrated else 'uncalibrated'}) -> "
            f"{'synthetic' if v.suspect() else 'natural'}"
        )
        for v in scored
    ]
    missing = [v for v in votes if v not in scored]
    summary = (
        f"{len(scored)} of {available} synthetic-audio models scored this recording"
        f"{f'; {len(missing)} did not run' if missing else ''}. "
        f"Weighted score {probability:.2f}, detector agreement {agreement:.0%}, level {level}."
    )
    limitations = list(_LIMITS)
    if missing:
        limitations.append(
            "Some detectors did not run; their families are absent from this verdict rather than assumed clean."
        )
    if any(not v.calibrated for v in scored):
        limitations.append(
            "At least one detector is uncalibrated, so its score is a vote, not a probability."
        )
    return {
        "status": "completed",
        "level": level,
        "probability": round(probability, 4),
        "agreement": round(agreement, 4),
        "confidence": None,
        "confidence_definition": "No calibrated joint confidence model is installed; detector agreement is reported separately.",
        "corroborated": majority and len(scored) > 1,
        "detectors_used": len(scored),
        "detectors_available": available,
        "disagreement": disagreement,
        "summary": summary,
        "evidence": evidence + [f"{v.name}: {v.detail or v.status} (not scored)" for v in missing],
        "models": models,
        "limitations": limitations,
    }


@dataclass
class FusionInput:
    synthetic: dict = field(default_factory=dict)
    scam: dict | None = None
    speaker: dict | None = None
    diarization: dict | None = None
    transcription: dict | None = None
    warnings: list[str] = field(default_factory=list)


def _scam_level(scam: dict | None) -> str:
    """The scam layer's own level, from its evidence strength (0-100)."""
    if not scam:
        return "INCONCLUSIVE"
    status = str(scam.get("status") or "completed").lower()
    if status in {"unavailable", "failed", "skipped"}:
        return "INCONCLUSIVE"
    score = scam.get("score")
    if score is None:
        level = str(scam.get("level") or "").upper()
        return level if level in LEVELS else "INCONCLUSIVE"
    try:
        score = float(score)
    except (TypeError, ValueError):
        return "INCONCLUSIVE"
    if not 0.0 <= score <= 100.0:
        return "INCONCLUSIVE"
    # This is the existing rule/model evidence-strength scale, not a probability.
    # Its bands mirror the 0-10 policy strength ranges. It never creates CRITICAL;
    # that requires explicit critical severity from the risk policy.
    if score >= 70:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


def fuse_evidence(payload: FusionInput) -> dict:
    """Per-family report plus one documented overall level."""
    synthetic = payload.synthetic or {}
    scam_level = _scam_level(payload.scam)
    audio_level = synthetic.get("level") or "INCONCLUSIVE"
    reasons: list[str] = []
    corroboration = False

    if scam_level in ("HIGH", "CRITICAL"):
        level = scam_level
        reasons.append(f"Content analysis is {scam_level}; that sets the overall level.")
        if LEVEL_ORDER.get(audio_level, -1) >= LEVEL_ORDER["MEDIUM"]:
            corroboration = True
            reasons.append(
                f"Voice analysis independently reports {audio_level}, which corroborates the content finding."
            )
    elif scam_level == "MEDIUM":
        level = "MEDIUM"
        reasons.append("Content analysis is MEDIUM: verify through a known channel before acting.")
        if audio_level == "HIGH":
            corroboration = True
            reasons.append("Voice analysis is HIGH, which corroborates concern but does not create a calibrated joint confidence.")
    elif scam_level == "LOW":
        strong_voice = audio_level == "HIGH" and float(synthetic.get("agreement") or 0) >= 0.75
        level = "MEDIUM" if strong_voice else "LOW"
        reasons.append(
            "Content analysis found no fraud pattern."
            + (
                " Synthetic-voice evidence is strong, so the audio warrants verification "
                "before you trust the speaker."
                if level == "MEDIUM"
                else ""
            )
        )
    else:
        level = "INCONCLUSIVE"
        reasons.append("Content analysis could not run, so the content level is inconclusive.")

    speaker = payload.speaker or {}
    diarization = payload.diarization or {}
    transcription = payload.transcription or {}

    sections = {
        "synthetic_audio": synthetic,
        "scam_content": {
            "level": scam_level,
            "score": (payload.scam or {}).get("score"),
            "flags": (payload.scam or {}).get("flags"),
            "categories": (payload.scam or {}).get("categories"),
            "status": (payload.scam or {}).get("status", "unavailable" if not payload.scam else "completed"),
        },
        "speaker_similarity": {
            "status": speaker.get("status", "skipped"),
            "outcome": speaker.get("outcome"),
            "score": speaker.get("score"),
            "threshold": speaker.get("threshold"),
            "calibrated": speaker.get("calibrated"),
            "detail": speaker.get("detail"),
        },
        "diarization": {
            "status": diarization.get("status", "skipped"),
            "backend": diarization.get("backend"),
            "segmentation": diarization.get("segmentation"),
            "speaker_count": diarization.get("speaker_count"),
            "turns": diarization.get("turns"),
            "speech_s": diarization.get("speech_s"),
            "detail": diarization.get("detail"),
        },
        "transcription": {
            "status": transcription.get("status"),
            "model": transcription.get("model"),
            "language": transcription.get("language"),
            "language_probability": transcription.get("language_probability"),
            "reliability": transcription.get("reliability"),
        },
    }

    return {
        "status": "completed",
        "level": level,
        "confidence": None,
        "confidence_definition": "No calibrated joint confidence model is installed; component scores and statuses remain separate.",
        "corroborated": corroboration,
        "sections": sections,
        "reasons": reasons,
        "warnings": payload.warnings,
        "policy": POLICY,
        "limitations": _LIMITS,
    }
