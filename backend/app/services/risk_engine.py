"""Risk policy: turns detector output into a documented decision category.

Vocabulary is kept explicit (see docs/ARCHITECTURE.md section 4):
  - raw model score        : unscaled output, not a probability
  - calibrated probability : only when calibration ran on validation data
  - uncertainty estimate   : only when the detector reports one, with method
  - risk category          : this policy decision, from the thresholds below

NOTE: the thresholds here are provisional defaults. They are NOT validated
against a labelled dataset yet; Phase 7 will calibrate and document them. The
API reports this limitation with every result so nobody mistakes a threshold
for a measured performance claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.services.audio_validation import DecodedAudio
from app.services.detector import DetectorResult

STANDARD_LIMITATIONS = (
    "Research prototype for decision support only; a result is never proof "
    "that audio is genuine or synthetic.",
    "Risk thresholds are provisional defaults and have not yet been validated on a labelled dataset.",
    "Detector coverage of unseen voice-cloning methods has not been evaluated yet.",
)


@dataclass(frozen=True)
class RiskThresholds:
    """Provisional policy thresholds applied to the primary detector score."""

    high: float = 0.70
    elevated: float = 0.35
    inconclusive_uncertainty: float = 0.50


@dataclass(frozen=True)
class Decision:
    category: str
    label: str
    reasons: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


RECOMMENDATIONS: dict[str, list[str]] = {
    "high_risk": [
        "Treat the request as unverified. Do not act on it from the audio alone.",
        "Verify the person through an independent trusted channel (for "
        "example, call back a known number).",
        "Escalate to a human reviewer and follow your organization's approval procedure.",
    ],
    "elevated_risk": [
        "Seek secondary verification through an independent channel before acting.",
        "Have a human reviewer listen to the recording and check the context.",
    ],
    "low_observed_risk": [
        "No strong synthetic-speech evidence was detected. This does NOT prove the recording is genuine.",
        "For high-impact decisions, still verify through your standard procedure.",
    ],
    "inconclusive": [
        "Do not base a decision on this analysis alone.",
        "Re-capture the audio in better conditions, or use an independent verification method.",
    ],
}

LABELS: dict[str, str] = {
    "high_risk": "Potentially synthetic speech detected",
    "elevated_risk": "Suspicious evidence warrants additional checks",
    "low_observed_risk": "No strong synthetic-speech evidence detected",
    "inconclusive": "Inconclusive",
}


class RiskEngine:
    def __init__(self, thresholds: RiskThresholds | None = None) -> None:
        self._thresholds = thresholds or RiskThresholds()

    @property
    def thresholds(self) -> RiskThresholds:
        return self._thresholds

    def decide(self, result: DetectorResult, audio: DecodedAudio) -> Decision:
        """Apply the documented policy to a real detector result."""
        if result.calibrated_probability is not None:
            score = float(result.calibrated_probability)
            basis = "calibrated probability"
        else:
            score = float(result.raw_score)
            basis = "raw model score (uncalibrated, not a probability)"

        reasons = [
            f"Decision derived from the {basis} of {result.model_name} v{result.model_version}.",
            f"Provisional thresholds applied: elevated>={self._thresholds.elevated}, "
            f"high>={self._thresholds.high}.",
        ]
        reasons.extend(audio.warnings)

        if (
            result.uncertainty is not None
            and result.uncertainty >= self._thresholds.inconclusive_uncertainty
        ):
            category = "inconclusive"
            reasons.append(
                "Detector uncertainty is high, so no risk category is asserted."
            )
        elif score >= self._thresholds.high:
            category = "high_risk"
        elif score >= self._thresholds.elevated:
            category = "elevated_risk"
        else:
            category = "low_observed_risk"

        return Decision(
            category=category,
            label=LABELS[category],
            reasons=reasons,
            recommendations=RECOMMENDATIONS[category],
        )
