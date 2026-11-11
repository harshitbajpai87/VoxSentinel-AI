"""Tests for the fine-tuning dataset builder (offline, no model downloads)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
SCRIPT = BACKEND / "scripts" / "build_scam_finetune_dataset.py"

_spec = importlib.util.spec_from_file_location("build_scam_finetune_dataset", SCRIPT)
assert _spec and _spec.loader
builder = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = builder
_spec.loader.exec_module(builder)

from app.protection.gemini import SYSTEM_INSTRUCTION, GeminiResult, _normalise  # noqa: E402
from tests.eval_scam_set import LABELLED_SET  # noqa: E402


@pytest.mark.parametrize("item", LABELLED_SET, ids=[i["id"] for i in LABELLED_SET])
def test_every_labelled_item_builds_a_schema_valid_target(item):
    target = builder.build_target(item)
    GeminiResult.model_validate(target)
    assert target["summary"] and len(target["summary"]) <= 600


@pytest.mark.parametrize("item", LABELLED_SET, ids=[i["id"] for i in LABELLED_SET])
def test_evidence_quotes_are_verbatim(item):
    target = builder.build_target(item)
    haystack = _normalise(item["text"])
    for indicator in target["indicators"]:
        assert _normalise(indicator["evidence_quote"]) in haystack


def test_example_is_chat_format_with_system_prompt():
    example = builder._example(LABELLED_SET[0])
    roles = [m["role"] for m in example["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert example["messages"][0]["content"] == SYSTEM_INSTRUCTION
    assert "<transcript>" in example["messages"][1]["content"]
    json.loads(example["messages"][2]["content"])


def test_benign_family_call_is_not_flagged_as_scam():
    item = next(i for i in LABELLED_SET if i["id"] == "normal_family")
    target = builder.build_target(item)
    assert target["classification"] == "LIKELY_LEGITIMATE"
    assert target["scam_categories"] == []
    assert target["indicators"] == []


def test_ransom_call_is_critical_with_money_request():
    item = next(i for i in LABELLED_SET if i["id"] == "cloned_son_ransom")
    target = builder.build_target(item)
    assert target["risk_level"] == "CRITICAL"
    assert "KIDNAPPING_OR_RANSOM" in target["scam_categories"]
    assert target["requested_actions"]


def test_reported_speech_is_educational_not_scam():
    item = next(i for i in LABELLED_SET if i["id"] == "news_report")
    target = builder.build_target(item)
    assert target["classification"] == "EDUCATIONAL_OR_DISCUSSION"
    assert target["risk_level"] == "LOW"


def test_money_demand_label_does_not_create_a_category():
    item = next(i for i in LABELLED_SET if i["id"] == "hinglish_emergency_payment")
    target = builder.build_target(item)
    assert "OTHER_SUSPICIOUS_ACTIVITY" not in target["scam_categories"]
    assert "Send money or transfer funds" in target["requested_actions"]


def test_split_is_deterministic():
    text = LABELLED_SET[0]["text"]
    first = builder._is_valid_split(text, 0.2)
    assert builder._is_valid_split(text, 0.2) is first


def test_validation_rejects_fabricated_quote():
    with pytest.raises(ValueError, match="not present in transcript"):
        builder._validate(
            "Papa mujhe kidnap kar liya hai.",
            {
                "classification": "LIKELY_SCAM",
                "risk_level": "HIGH",
                "scam_categories": ["KIDNAPPING_OR_RANSOM"],
                "summary": "x",
                "indicators": [
                    {
                        "category": "KIDNAPPING_OR_RANSOM",
                        "evidence_quote": "send bitcoin to this wallet",
                        "explanation": "x",
                        "severity": "HIGH",
                    }
                ],
                "transcript_quality": "ADEQUATE",
            },
        )
