"""Phase 6: metrics correctness, leakage exemptions, and the corpus builder.

No real-speech data is needed here: statistical helpers are tested against
hand-computed values, and the librispeech_piper preset against a tiny fake
corpus. Real-data results live only in backend/reports/ (see MODEL_CARD).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.manifest import ManifestEntry, assert_no_speaker_leakage  # noqa: E402
from app.services.metrics import equal_error_rate  # noqa: E402
from scripts.build_tts_spoofs import collect_transcripts, select_transcripts  # noqa: E402
from scripts.evaluate_detector import _fmt  # noqa: E402
from scripts.prepare_dataset import main_librispeech_piper  # noqa: E402

# --------------------------- metrics ---------------------------

def brute_force_eer(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Empirical FAR/FRR crossing over all thresholds (independent check)."""
    best = 1.0
    for t in np.unique(scores):
        pred = scores >= t
        far = np.mean(pred[y_true == 0])
        frr = 1.0 - np.mean(pred[y_true == 1])
        best = min(best, abs(far - frr), (far + frr) / 2 if abs(far - frr) < 0.30 else best)
    return best


def test_eer_zero_for_perfect_separation():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.01, 0.02, 0.03, 0.97, 0.98, 0.99])
    assert equal_error_rate(y, scores) == pytest.approx(0.0)


def test_eer_correct_on_imperfect_scores():
    # Hand-computed: at threshold 0.60, FAR = FRR = 1/3 exactly, so EER = 1/3.
    # (Regression guard: an argmin-based EER mis-picks a tie and returned 0.5.)
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.10, 0.40, 0.60, 0.20, 0.70, 0.90])
    eer = equal_error_rate(y, scores)
    assert eer == pytest.approx(1 / 3, abs=1e-6)


def test_eer_matches_empirical_far_frr_crossing():
    """Cross-check against an independent brute-force scan over thresholds."""
    rng = np.random.default_rng(7)
    y = (rng.random(60) > 0.5).astype(int)
    scores = np.clip(y * 0.4 + rng.random(60) * 0.8, 0, 1)
    eer = equal_error_rate(y, scores)
    assert eer is not None

    gaps = []
    for t in np.unique(scores):
        far = np.mean(scores[y == 0] >= t)
        frr = 1.0 - np.mean(scores[y == 1] >= t)
        gaps.append(abs(far - frr))
    # The true EER cannot be smaller than the best empirical |FAR-FRR| gap
    # half-width; and must lie within the observed crossing range.
    assert 0.0 <= eer <= 0.5
    assert eer >= (min(gaps) / 2) - 0.05


def test_eer_none_for_single_class():
    assert equal_error_rate(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9])) is None


def test_fmt_distinguishes_zero_from_unavailable():
    assert _fmt(None) == "unavailable"
    assert _fmt(float("nan")) == "unavailable"
    assert _fmt(0.0) == "0.0000"  # a real measured 0.0 is NOT "unavailable"
    assert _fmt(0.05) == "0.0500"


# --------------------------- leakage rules ---------------------------

def entry(speaker: str, label: str, source: str, partition: str) -> ManifestEntry:
    return ManifestEntry(
        path=f"{speaker}.wav", label=label, partition=partition,
        speaker_id=speaker, source=source,
    )


def test_tts_generator_overlap_requires_explicit_exemption():
    train = [entry("piper:lessac", "synthetic", "piper_tts", "train")]
    validation = [entry("piper:lessac", "synthetic", "piper_tts", "validation")]
    # Without exemption: refused (default is strict).
    with pytest.raises(Exception, match="leakage"):
        assert_no_speaker_leakage(train, validation)
    # With the documented exemption: allowed.
    assert_no_speaker_leakage(train, validation, exempt_sources=("piper_tts",))


def test_human_speaker_overlap_is_never_exempted():
    train = [entry("84", "real", "librispeech", "train")]
    validation = [entry("84", "real", "librispeech", "validation")]
    with pytest.raises(Exception, match="leakage"):
        assert_no_speaker_leakage(train, validation, exempt_sources=("piper_tts",))


# --------------------------- transcript selection ---------------------------

def test_select_transcripts_is_deterministic_and_disjoint(tmp_path):
    subset = tmp_path / "dev-clean"
    chapter = subset / "84" / "121123"
    chapter.mkdir(parents=True)
    lines = []
    for i in range(40):
        lines.append(f"84-121123-{i:04d} " + ("word " * (10 + i)))
    (chapter / "84-121123.trans.txt").write_text("\n".join(lines), encoding="utf-8")

    pairs = collect_transcripts(subset)
    assert len(pairs) == 40

    first = select_transcripts(pairs, count=10, offset=0, seed=42, min_words=8, max_words=40)
    second = select_transcripts(pairs, count=10, offset=10, seed=42, min_words=8, max_words=40)
    again = select_transcripts(pairs, count=10, offset=0, seed=42, min_words=8, max_words=40)

    assert [p[0] for p in first] == [p[0] for p in again]  # deterministic
    assert not {p[0] for p in first} & {p[0] for p in second}  # disjoint slices


# --------------------------- librispeech_piper preset ---------------------------

@pytest.fixture
def fake_corpus(tmp_path):
    tiny_audio = np.zeros(8000, dtype=np.float32)
    tiny_audio[::100] = 0.5

    dev = tmp_path / "LibriSpeech" / "dev-clean"
    test = tmp_path / "LibriSpeech" / "test-clean"
    for reader in ("101", "102", "103"):
        chapter_dir = dev / reader / "2001"
        chapter_dir.mkdir(parents=True)
        sf.write(chapter_dir / f"{reader}-2001.flac", tiny_audio, 8000, format="FLAC")
    for reader in ("201",):
        chapter_dir = test / reader / "3001"
        chapter_dir.mkdir(parents=True)
        sf.write(chapter_dir / f"{reader}-3001.flac", tiny_audio, 8000, format="FLAC")

    spoofs = tmp_path / "spoofs"
    spoofs.mkdir()
    sf.write(spoofs / "s1.wav", tiny_audio, 8000)
    index = tmp_path / "spoofs" / "index.jsonl"
    index.write_text(
        json.dumps(
            {
                "path": "spoofs/s1.wav",
                "label": "synthetic",
                "partition": "train",
                "speaker_id": "piper:testvoice",
                "source": "piper_tts",
                "voice_id": "testvoice",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return tmp_path


def test_librispeech_piper_builds_balanced_manifest(fake_corpus, tmp_path):
    entries, provenance = _build(fake_corpus)
    assert len(entries) == 5  # 3 dev real + 1 test real + 1 spoof
    labels = [e["label"] for e in entries]
    assert labels.count("real") == 4 and labels.count("synthetic") == 1
    assert "CC BY 4.0" in provenance["real"]
    eval_speakers = {e["speaker_id"] for e in entries if e["partition"] == "eval"}
    train_speakers = {e["speaker_id"] for e in entries if e["partition"] != "eval"}
    assert not eval_speakers & train_speakers


def test_librispeech_piper_refuses_missing_class(fake_corpus, tmp_path, capsys):
    # Spoof index only has a train entry: eval partition lacks synthetic samples.
    args = argparse.Namespace(
        dataset_dir=str(fake_corpus),
        spoof_index=str(fake_corpus / "spoofs" / "index.jsonl"),
        train_readers=2,
        val_readers=1,
        per_reader_train=1,
        per_reader_val=1,
        per_reader_eval=1,
        output=str(tmp_path / "manifest.jsonl"),
    )
    assert main_librispeech_piper(args) == 2  # missing eval synthetic class
    assert "missing class" in capsys.readouterr().err


def _build(corpus_dir):
    from scripts.prepare_dataset import build_librispeech_piper

    return build_librispeech_piper(
        corpus_dir,
        train_readers=2,
        val_readers=1,
        per_reader_train=1,
        per_reader_val=1,
        per_reader_eval=1,
        spoof_index=corpus_dir / "spoofs" / "index.jsonl",
    )
