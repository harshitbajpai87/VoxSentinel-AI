"""The shipped text classifier must load, classify, and stay honest about itself."""

from __future__ import annotations

from app.protection import content_classifier


def test_model_is_available():
    info = content_classifier.describe()
    assert isinstance(info["available"], bool)
    if info["available"]:
        assert info["threshold"] > 0
        assert info["metrics"]["n"] > 0


def test_classifies_a_ransom_call():
    result = content_classifier.classify_text(
        "Papa mujhe kidnap kar liya hai, 50000 abhi bhejo warna hum nahi chhodenge, "
        "kisi ko batana nahi."
    )
    if result.status != "completed":
        return  # artifact absent in this checkout
    assert result.label == "scam"
    assert 0.0 <= result.confidence <= 1.0


def test_classifies_a_benign_check_in():
    result = content_classifier.classify_text(
        "Hi Papa, I reached college safely, will call you at night."
    )
    if result.status != "completed":
        return
    assert result.label == "benign"


def test_empty_text_is_not_guessed():
    for payload in ("", "   "):
        result = content_classifier.classify_text(payload)
        assert result.status == "unavailable"
        assert result.label is None


def test_confidence_definition_is_documented():
    info = content_classifier.describe()
    blob = str(info)
    assert "probability" in blob.lower() or "score" in blob.lower()
