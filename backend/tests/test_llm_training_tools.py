"""Tests for the local-model training/serving tooling (offline)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]


def _load(script_name: str):
    spec = importlib.util.spec_from_file_location(script_name, BACKEND / "scripts" / f"{script_name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ollama_script = _load("create_ollama_model")
train_script = _load("train_mistral_finetune")

from tests.eval_scam_set import LABELLED_SET  # noqa: E402


def _row(text: str, language: str = "en") -> dict:
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": text},
            {"role": "assistant", "content": "{}"},
        ]
    }


def test_modelfile_pins_system_prompt_and_parameters():
    content = ollama_script.build_modelfile("mistral", [], None)
    assert content.startswith("FROM mistral")
    assert "SYSTEM \"\"\"" in content
    assert "PARAMETER temperature 0.1" in content
    assert "ADAPTER" not in content


def test_modelfile_embeds_examples_as_message_turns():
    example = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Papa mujhe kidnap kar liya hai"},
            {"role": "assistant", "content": '{"classification": "LIKELY_SCAM"}'},
        ]
    }
    content = ollama_script.build_modelfile("mistral", [example], None)
    assert "MESSAGE user" in content
    assert "MESSAGE assistant" in content
    assert content.count("MESSAGE ") == 2


def test_adapter_directive_emitted_when_requested():
    content = ollama_script.build_modelfile("mistral", [], "adapters/voxsentinel-lora")
    assert "ADAPTER adapters/voxsentinel-lora" in content


def test_fewshot_selection_is_one_per_verdict():
    rows = [
        _row("a"),
        _row("b"),
        _row("c"),
    ]
    rows[0]["messages"][2]["content"] = json.dumps({"classification": "LIKELY_SCAM"})
    rows[1]["messages"][2]["content"] = json.dumps({"classification": "LIKELY_SCAM"})
    rows[2]["messages"][2]["content"] = json.dumps({"classification": "LIKELY_LEGITIMATE"})
    picked = ollama_script.select_fewshot(rows, 4)
    labels = [json.loads(p["messages"][2]["content"])["classification"] for p in picked]
    assert labels == ["LIKELY_SCAM", "LIKELY_LEGITIMATE"]


def test_training_dataset_validation_rejects_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="dataset not found"):
        train_script.validate_dataset(tmp_path / "nope.jsonl")


def test_training_dataset_validation_rejects_wrong_roles(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="system/user/assistant"):
        train_script.validate_dataset(path)


def test_training_dataset_validation_rejects_schema_violation(tmp_path):
    path = tmp_path / "bad.jsonl"
    row = _row("hello")
    row["messages"][2]["content"] = json.dumps({"classification": "NOT_A_LABEL"})
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="schema validation"):
        train_script.validate_dataset(path)


def test_training_dataset_validation_accepts_built_examples(tmp_path):
    builder = _load("build_scam_finetune_dataset")
    path = tmp_path / "train.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for item in LABELLED_SET:
            handle.write(json.dumps(builder._example(item), ensure_ascii=False) + "\n")
    rows = train_script.validate_dataset(path)
    assert len(rows) == len(LABELLED_SET)


def test_no_credentials_in_tooling_sources():
    for script in ("train_mistral_finetune.py", "create_ollama_model.py", "build_scam_finetune_dataset.py"):
        source = (BACKEND / "scripts" / script).read_text(encoding="utf-8")
        assert "sk-" not in source
        assert "Authorization: Bearer " not in source.split("purpose")[0] or "MISTRAL_API_KEY" in source
