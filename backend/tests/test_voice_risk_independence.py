"""Policy regressions use explicit test doubles, not real model artifacts."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.protection import risk
from app.protection.quality import QualityReport
from app.protection.service import ProtectionService
from app.protection.transcription import TranscriptionResult
from app.services.detector import DetectorResult, NullVoiceDetector, VoiceDetectorService
from tests.conftest import MockVoiceDetector, make_wav


def _input(**kwargs):
    base = risk.SignalInput(
        synthetic_suspected=True,
        synthetic_note=None,
        speaker_outcome=None,
        signal_strength=0,
        flags={},
        transcription_status="completed",
        discussion_context=False,
    )
    return replace(base, **kwargs)


@pytest.mark.parametrize(
    "classification",
    [
        None,
        "LIKELY_LEGITIMATE",
        "EDUCATIONAL_OR_DISCUSSION",
        "INCONCLUSIVE",
    ],
)
@pytest.mark.parametrize("strength,discussion", [(0, False), (2, True)])
def test_synthetic_voice_never_silently_becomes_low(classification, strength, discussion):
    result = risk.assess(
        _input(
            signal_strength=strength,
            discussion_context=discussion,
            gemini_classification=classification,
            gemini_risk="LOW",
            gemini_status="completed" if classification else "unavailable",
        )
    )
    assert result.category == "MEDIUM"
    assert any("Conversation safety" in x for x in result.reasoning)
    assert any("Voice authenticity" in x for x in result.reasoning)
    assert any("not proof of scam intent" in x for x in result.limitations)
    if classification in {"LIKELY_LEGITIMATE", "EDUCATIONAL_OR_DISCUSSION"}:
        assert any("Evidence conflict" in x for x in result.reasoning)


@pytest.mark.parametrize(
    "classification,categories,model_risk,expected",
    [
        ("LIKELY_SCAM", (), None, "HIGH"),
        ("LIKELY_SCAM", ("KIDNAPPING_OR_RANSOM",), None, "HIGH"),
        ("LIKELY_SCAM", ("KIDNAPPING_OR_RANSOM",), "CRITICAL", "CRITICAL"),
        ("SUSPICIOUS", (), None, "MEDIUM"),
        ("EDUCATIONAL_OR_DISCUSSION", (), None, "LOW"),
    ],
)
@pytest.mark.parametrize("synthetic", [True, False, None])
def test_early_classifier_returns_include_independent_voice_evidence(
    classification, categories, model_risk, expected, synthetic
):
    result = risk.assess(
        _input(
            synthetic_suspected=synthetic,
            gemini_classification=classification,
            gemini_risk=model_risk,
            gemini_categories=categories,
            gemini_status="completed",
            speaker_outcome="not_consistent_with_reference",
        )
    )
    assert result.category == ("MEDIUM" if synthetic and expected == "LOW" else expected)
    assert any("Voice authenticity" in x or "Voice analysis" in x for x in result.reasoning)
    assert any("enrolled family voice reference" in x for x in result.reasoning)


def test_strong_rule_disagreement_is_recorded_even_for_educational_early_return():
    result = risk.assess(
        _input(
            synthetic_suspected=False,
            signal_strength=9,
            gemini_status="completed",
            gemini_classification="EDUCATIONAL_OR_DISCUSSION",
        )
    )
    assert any("strong rule-based" in x for x in result.reasoning)


def test_voice_does_not_upgrade_content_to_critical():
    flags = {"otp_or_credentials": True, "payment_demand": True}
    for synthetic in (True, False, None):
        assert (
            risk.assess(_input(signal_strength=9, flags=flags, synthetic_suspected=synthetic)).category
            == "HIGH"
        )


def test_no_transcript_and_no_positive_voice_warning_is_inconclusive():
    assert (
        risk.assess(
            _input(
                synthetic_suspected=False,
                transcription_status="unavailable",
            )
        ).category
        == "INCONCLUSIVE"
    )


class _Transcriber:
    def transcribe(self, *args):
        return TranscriptionResult(
            status="completed", text="Let us meet for lunch tomorrow.", model="TEST DOUBLE"
        )

    def describe(self):
        return {"service": "TEST DOUBLE"}


class _LLM:
    provider_label = "TEST DOUBLE"
    provider_key = "test"

    def classify(self, *args):
        return None, {"status": "unavailable", "detail": "TEST DOUBLE disabled"}

    def describe(self):
        return {"configured": False}


class _Reviewer:
    def review(self, **kwargs):
        return SimpleNamespace(status="disabled", as_dict=lambda: {"status": "disabled"})

    def describe(self):
        return {"configured": False}


def _service(settings, detector, monkeypatch, quality="good", forensics=None):
    service = ProtectionService(
        settings,
        VoiceDetectorService(detector),
        SimpleNamespace(is_ready=lambda: False),
        transcriber=_Transcriber(),
        gemini=_LLM(),
        forensics_detector=forensics,
    )
    service._reviewer = _Reviewer()
    monkeypatch.setattr(
        "app.protection.service.assess_quality",
        lambda *args: QualityReport(
            3.0,
            2.0,
            0.66,
            -15,
            -3,
            0,
            20,
            verdict=quality,
        ),
    )
    # Never load the archived classifier while running mapping regressions.
    monkeypatch.setattr(
        "app.protection.service.content_classifier.classify_text",
        lambda *args: SimpleNamespace(
            status="unavailable",
            label=None,
            confidence=None,
            detail="TEST DOUBLE",
            to_dict=lambda: {},
        ),
    )
    return service


@pytest.mark.parametrize(
    "quality,band,expected", [("good", None, "MEDIUM"), ("insufficient", None, "LOW"), ("good", 0.1, "LOW")]
)
def test_qualified_voice_result_is_what_reaches_signal_input(settings, monkeypatch, quality, band, expected):
    detector = MockVoiceDetector(DetectorResult(raw_score=0.95, raw_score_definition="TEST DOUBLE"))
    if band is not None:
        detector.elevated_threshold = 0.95
        detector.ambiguity_band = band
    service = _service(settings, detector, monkeypatch, quality)
    result = service.analyze(make_wav(), "test.wav")
    assert result["risk"]["category"] == expected
    assert result["voice"]["score"] == 0.95  # raw evidence stays intact


def test_forensics_remains_separate_and_cannot_drive_risk(settings, monkeypatch):
    class Forensics:
        def analyze(self, *args):
            return SimpleNamespace(raw_score=0.99, explanation="TEST DOUBLE acoustic evidence")

    service = _service(settings, NullVoiceDetector(), monkeypatch, forensics=Forensics())
    result = service.analyze(make_wav(), "test.wav")
    assert result["voice"]["status"] == "unavailable"
    assert result["voice"]["score"] is None
    assert result["voice"]["forensics"]["used_for_risk"] is False
    assert result["risk"]["category"] == "LOW"
    assert any("could not be assessed" in x for x in result["risk"]["reasoning"])
