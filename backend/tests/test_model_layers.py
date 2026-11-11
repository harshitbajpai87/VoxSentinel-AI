"""The unified pipeline's model layers, exercised without loading big weights.

Everything here is a TEST DOUBLE: fake engines, fake heads, fake providers. The
point is the routing, fusion and honesty rules, not any metric.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.protection import llm_router
from app.protection.asr_engines import (
    EngineTranscript,
    MultiEngineTranscriber,
    is_hindi_text,
    looks_hinglish,
)
from app.protection.text_ensemble import BranchResult, TextEnsemble
from app.protection.fusion import FusionInput, fuse_evidence
from app.services.forensics_embeddings import (
    FUSION_WEIGHTS,
    UnifiedForensics,
    _fuse,
    _verdict,
)


# ------------------------------------------------------------------- ASR
class FakeEngine:
    def __init__(self, name: str, text: str, language: str | None = None, available: bool = True):
        self.name = name
        self._text = text
        self._language = language
        self._available = available
        self.calls = 0

    def is_available(self) -> bool:
        return self._available

    def transcribe(self, samples, sample_rate, language_hint=None) -> EngineTranscript:
        self.calls += 1
        if not self._available:
            return EngineTranscript(engine=self.name, model="fake", status="unavailable",
                                    detail="not installed on this host")
        return EngineTranscript(
            engine=self.name,
            model="fake",
            status="completed",
            text=self._text,
            language=self._language,
            language_probability=0.9 if self._language else None,
        )


def test_language_helpers():
    assert is_hindi_text("मुझे पैसे भेजो अभी")
    assert not is_hindi_text("send me the money now")
    assert looks_hinglish("bhai mujhe paisa chahiye abhi jaldi bhej de")
    assert not looks_hinglish("the quarterly report is attached")


def test_primary_engine_wins_and_engines_are_reported():
    primary = FakeEngine("whisper_small", "Please send the payment today.")
    transcriber = MultiEngineTranscriber(engines=[primary])
    result = transcriber.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    assert result.status == "completed"
    assert result.text == "Please send the payment today."
    assert result.model == "fake"
    assert result.engines[0]["engine"] == "whisper_small"
    assert primary.calls == 1


def test_hindi_model_takes_over_when_the_text_is_devanagari():
    primary = FakeEngine("whisper_small", "bhai mujhe paisa chahiye abhi jaldi bhejo")
    hindi = FakeEngine("indic_hindi", "भाई पैसे भेजो अभी", language="hi")
    transcriber = MultiEngineTranscriber(engines=[primary, hindi])
    result = transcriber.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    assert result.text == "भाई पैसे भेजो अभी"
    assert result.language == "hi"
    assert {entry["engine"] for entry in result.engines} == {"whisper_small", "indic_hindi"}


def test_no_engine_available_reports_every_reason():
    transcriber = MultiEngineTranscriber(
        engines=[FakeEngine("whisper_small", "", available=False),
                 FakeEngine("indic_hindi", "", available=False)]
    )
    result = transcriber.transcribe(np.zeros(1600, dtype=np.float32), 16000)
    assert result.status == "unavailable"
    assert "not installed on this host" in (result.error or "")
    assert len(result.engines) == 2


def test_explicit_language_hint_is_recorded():
    transcriber = MultiEngineTranscriber(
        engines=[FakeEngine("whisper_small", "namaste")], language_hint="hi"
    )
    result = transcriber.transcribe(np.zeros(1600, dtype=np.float32), 16000)
    assert result.language == "hi"
    assert result.language_source == "request_hint"


def test_unavailable_preferred_engine_falls_back_to_an_available_engine():
    preferred = FakeEngine("whisper_small", "", available=False)
    fallback = FakeEngine("faster_whisper", "fallback transcript")
    result = MultiEngineTranscriber(
        engines=[preferred, fallback], preferred="whisper_small"
    ).transcribe(np.zeros(1600, dtype=np.float32), 16000)
    assert result.status == "completed"
    assert result.text == "fallback transcript"
    assert result.selected_engine == "faster_whisper"


def test_runtime_asr_failure_falls_back_and_is_reported():
    class ExplodingEngine(FakeEngine):
        def transcribe(self, samples, sample_rate, language_hint=None):
            self.calls += 1
            raise RuntimeError("load failure")

    primary = ExplodingEngine("whisper_small", "")
    fallback = FakeEngine("faster_whisper", "usable transcript")
    result = MultiEngineTranscriber(
        engines=[primary, fallback], preferred="whisper_small"
    ).transcribe(np.zeros(1600, dtype=np.float32), 16000)
    assert result.status == "completed"
    assert result.text == "usable transcript"
    assert result.selected_engine == "faster_whisper"
    assert [attempt["status"] for attempt in result.engines] == ["failed", "completed"]


def test_disabled_asr_fallback_does_not_call_secondary(monkeypatch):
    class ExplodingEngine(FakeEngine):
        def transcribe(self, samples, sample_rate, language_hint=None):
            self.calls += 1
            raise RuntimeError("load failure")

    monkeypatch.setenv("VOXSENTINEL_ASR_FALLBACKS", "false")
    primary = ExplodingEngine("whisper_small", "")
    secondary = FakeEngine("faster_whisper", "must not run")
    result = MultiEngineTranscriber(
        engines=[primary, secondary], preferred="whisper_small"
    ).transcribe(np.zeros(1600, dtype=np.float32), 16000)
    assert result.status == "failed"
    assert secondary.calls == 0


# --------------------------------------------------------- text ensemble
class FakeHead:
    def __init__(self, name: str, score: float | None, status: str = "completed"):
        self.name = name
        self._score = score
        self._status = status
        self.calls = 0

    def is_available(self) -> bool:
        return self._status == "completed"

    def status(self) -> dict:
        return {"branch": self.name, "status": self._status, "model": "fake",
                "detail": None if self._status == "completed" else "no artifact"}

    def score(self, text: str) -> BranchResult:
        self.calls += 1
        return BranchResult(self.name, "fake", self._status, score=self._score)


def test_ensemble_weights_the_branches():
    ensemble = TextEnsemble(
        heads={"muril": FakeHead("muril", 0.9), "deberta": FakeHead("deberta", 0.5)},
        threshold=0.55,
    )
    result = ensemble.classify("send 1 lakh rupees now, don't tell anyone")
    assert result["status"] == "completed"
    assert result["score"] == pytest.approx(0.74, abs=0.01)  # 0.6*0.9 + 0.4*0.5
    assert result["label"] == "scam"


def test_language_routing_excludes_english_only_head_for_hindi():
    muril = FakeHead("muril", 0.8)
    deberta = FakeHead("deberta", 0.9)
    result = TextEnsemble(heads={"muril": muril, "deberta": deberta}).classify(
        "मुझे पैसे भेजो अभी", language="hi"
    )
    assert result["status"] == "completed"
    assert muril.calls == 1
    assert deberta.calls == 0
    branches = {item["branch"]: item for item in result["branches"]}
    assert branches["deberta"]["status"] == "skipped"


def test_text_ensemble_cache_is_separated_by_language():
    muril = FakeHead("muril", 0.8)
    deberta = FakeHead("deberta", 0.2)
    ensemble = TextEnsemble(heads={"muril": muril, "deberta": deberta})
    text = "I reached college safely and will call after class."
    hindi = ensemble.classify(text, language="hi")
    english = ensemble.classify(text, language="en")
    assert hindi["score"] != english["score"]
    assert muril.calls == 2
    assert deberta.calls == 1


def test_ensemble_says_unavailable_when_no_branch_is_ready():
    ensemble = TextEnsemble(
        heads={"muril": FakeHead("muril", None, status="unavailable"),
               "deberta": FakeHead("deberta", None, status="unavailable")},
    )
    result = ensemble.classify("anything at all")
    assert result["status"] == "unavailable"
    assert result["score"] is None
    assert "TF-IDF classifier" in result["detail"]


def test_ensemble_ignores_empty_text():
    ensemble = TextEnsemble(heads={"muril": FakeHead("muril", 0.9)})
    assert ensemble.classify("   ")["status"] == "unavailable"


def test_ensemble_can_be_disabled(monkeypatch):
    monkeypatch.setenv("VOXSENTINEL_TEXT_ENSEMBLE", "false")
    ensemble = TextEnsemble(heads={"muril": FakeHead("muril", 0.9)})
    assert ensemble.classify("send money")["status"] == "unavailable"


# -------------------------------------------------------------- forensics
class StubHandcrafted:
    """TEST DOUBLE handcrafted screen."""

    def __init__(self, score: float):
        self._score = score

    def analyze(self, samples, sample_rate, channels):
        class Result:
            raw_score = self._score
            explanation = "stub explanation"
            evidence = None

        return Result()


class StubEmbedder:
    def __init__(self, available: bool = False):
        self._available = available

    def status(self) -> dict:
        return {
            "model": "facebook/wav2vec2-large-xlsr-53",
            "status": "completed" if self._available else "unavailable",
            "detail": None if self._available else "model not installed locally",
        }

    def embed(self, samples, sample_rate):
        return np.zeros(1024, dtype=np.float32)


def test_fusion_renormalises_over_the_components_that_ran():
    score, used, weights = _fuse({"detector": 0.8})
    assert used == ["detector"]
    assert weights == {"detector": 1.0}
    assert score == pytest.approx(0.8)


def test_fusion_uses_declared_weights():
    score, used, weights = _fuse({"detector": 0.6, "handcrafted": 0.2})
    assert used == ["detector", "handcrafted"]
    assert weights["detector"] == pytest.approx(0.5 / 0.7, abs=1e-3)
    assert score == pytest.approx((0.6 * 0.5 + 0.2 * 0.2) / 0.7, abs=1e-3)
    assert set(FUSION_WEIGHTS) >= set(used)


def test_verdict_bands_keep_borderline_audio_undetermined():
    assert _verdict(0.9, 0.5) == "elevated"
    assert _verdict(0.1, 0.5) == "not_elevated"
    assert _verdict(0.5, 0.5) == "inconclusive"
    assert _verdict(None, 0.5) == "unavailable"


def test_uncalibrated_content_strength_cannot_become_critical_or_confidence():
    result = fuse_evidence(FusionInput(
        scam={"status": "completed", "score": 75, "score_scale": "evidence strength"},
        synthetic={"status": "unavailable", "level": "INCONCLUSIVE"},
    ))
    assert result["level"] == "HIGH"
    assert result["sections"]["scam_content"]["level"] == "HIGH"
    assert result["confidence"] is None


def test_unified_forensics_reports_the_missing_learned_branch():
    engine = UnifiedForensics(
        handcrafted=StubHandcrafted(0.3),
        embedder=StubEmbedder(False),
    )
    block = engine.analyze(
        np.zeros(16000, dtype=np.float32), 16000, 1,
        {"status": "completed", "score": 0.7, "calibrated_probability": 0.7,
         "model": "detector.joblib", "threshold": 0.5},
    )
    assert block["status"] == "completed"
    assert block["fusion"]["components_used"] == ["detector", "handcrafted"]
    assert "wav2vec2_head" in block["fusion"]["components_missing"]
    assert block["wav2vec2_head"]["status"] == "unavailable"
    assert "model not installed locally" in block["explanation"]


def test_unified_forensics_is_honest_when_nothing_ran():
    engine = UnifiedForensics(handcrafted=None, embedder=StubEmbedder(False))
    block = engine.analyze(np.zeros(1600, dtype=np.float32), 16000, 1, {"status": "unavailable"})
    assert block["status"] == "unavailable"
    assert block["fusion"]["score"] is None
    assert block["fusion"]["verdict"] == "unavailable"


# -------------------------------------------------------------- LLM router
class FakeProvider:
    def __init__(self, configured: bool = True, result=None, status=None):
        self._config = type("C", (), {"configured": configured, "model": "fake-model"})()
        self._result = result
        self._status = status or {"provider": "fake", "status": "completed"}
        self.calls = 0

    @property
    def provider_label(self) -> str:
        return "Fake provider"

    def describe(self) -> dict:
        return {"provider": "fake"}

    def classify(self, transcript, language=None):
        self.calls += 1
        return self._result, dict(self._status)


def test_router_uses_the_local_model_first():
    primary = FakeProvider(result={"classification": "LIKELY_SCAM", "risk_level": "HIGH",
                                   "scam_categories": ["FAMILY_EMERGENCY_IMPERSONATION"],
                                   "summary": "money demand"})
    fallback = FakeProvider(result={"summary": "from fallback"})
    router = llm_router.ThreatReasoningRouter(primary=primary, fallback=fallback)
    result, status = router.classify("send money now")
    assert result["classification"] == "LIKELY_SCAM"
    assert status["provider"] == "ollama"
    assert status["fallback_used"] is False
    assert primary.calls == 1
    assert fallback.calls == 0


def test_router_falls_back_and_labels_the_fallback():
    primary = FakeProvider(result=None, status={"provider": "ollama", "status": "failed",
                                                "code": "ollama_connection_error"})
    fallback = FakeProvider(result={"classification": "LIKELY_SCAM", "risk_level": "MEDIUM",
                                    "scam_categories": ["KYC_PHISHING"], "summary": "kyc"})
    router = llm_router.ThreatReasoningRouter(primary=primary, fallback=fallback)
    result, status = router.classify("your kyc is expiring")
    assert result["classification"] == "LIKELY_SCAM"
    assert status["provider"] == "mistral"
    assert status["fallback_used"] is True
    assert status["provider_label"] == "Fake provider"
    assert [attempt["provider"] for attempt in status["attempts"]] == ["ollama", "mistral"]


def test_router_reports_unavailable_when_nothing_is_configured():
    router = llm_router.ThreatReasoningRouter(
        primary=FakeProvider(configured=False), fallback=FakeProvider(configured=False)
    )
    result, status = router.classify("hello")
    assert result is None
    assert status["status"] == "unavailable"
    assert status["code"] == "no_provider"


def test_router_reports_all_providers_failed():
    primary = FakeProvider(result=None, status={"provider": "ollama", "status": "failed",
                                                "code": "ollama_schema_invalid"})
    fallback = FakeProvider(result=None, status={"provider": "mistral", "status": "failed",
                                                 "code": "mistral_http_error"})
    router = llm_router.ThreatReasoningRouter(primary=primary, fallback=fallback)
    result, status = router.classify("hello")
    assert result is None
    assert status["status"] == "failed"
    assert status["code"] == "all_providers_failed"


def test_verdict_normalises_a_provider_result():
    verdict = llm_router.to_verdict(
        {
            "classification": "LIKELY_SCAM",
            "risk_level": "CRITICAL",
            "scam_categories": ["RANSOM", "FAMILY_EMERGENCY_IMPERSONATION"],
            "summary": "Caller claims a kidnapping and demands payment.",
            "indicators": [{"category": "urgency", "evidence_quote": "within 10 minutes",
                            "explanation": "artificial deadline", "severity": "HIGH"}],
            "manipulation_tactics": ["isolation"],
            "requested_actions": ["transfer money"],
            "protective_recommendations": ["call the family member directly"],
            "uncertainty_reasons": ["first-time caller"],
        },
        {"provider": "ollama", "status": "completed", "model": "voxisentinel-mistral"},
    ).as_dict()
    assert verdict["provider"] == "ollama"
    assert verdict["model"] == "voxisentinel-mistral"
    assert verdict["severity"] == "CRITICAL"
    assert verdict["confidence"] is None
    assert verdict["threat_type"].startswith("RANSOM")
    assert verdict["indicators"][0]["quote"] == "within 10 minutes"


def test_verdict_without_a_result_claims_nothing():
    verdict = llm_router.to_verdict(None, {"provider": "router", "status": "unavailable"}).as_dict()
    assert verdict["status"] == "unavailable"
    assert verdict["threat_type"] is None
    assert verdict["indicators"] == []
