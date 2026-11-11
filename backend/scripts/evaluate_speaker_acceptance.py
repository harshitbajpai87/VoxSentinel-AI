"""Measure real speaker-verification FAR/FRR/EER on librispeech-dev-clean.

This script exists because the audit could only confirm the *rejection* side of
speaker verification (a different-speaker pair scored 0.108). Accepting the same
speaker is the other half of the claim, and it must be measured, not assumed.

Protocol
--------
* Speakers are split into a calibration pool and a held-out pool by speaker id,
  so the reported numbers are not tuned on the same pairs.
* Pairs are all-vs-all within the pool: same-speaker pairs (different utterances
  of one speaker) are positives, different-speaker pairs are negatives.
* The threshold is the equal-error-rate point from the calibration pool; the
  reported FAR/FRR/EER come from the held-out pool.

Run: python scripts/evaluate_speaker_acceptance.py --root /tmp/av/LibriSpeech/dev-clean
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.services.speaker_model import EcapaSpeakerEmbedder  # noqa: E402


def _load(path: Path, sr_target: int = 16000) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != sr_target:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, sr_target)
        data = resample_poly(data, sr_target // g, sr // g).astype(np.float32)
    return data


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _metrics(pos: list[float], neg: list[float], thr: float) -> dict:
    far = sum(1 for s in neg if s >= thr) / len(neg) if neg else None
    frr = sum(1 for s in pos if s < thr) / len(pos) if pos else None
    return {"threshold": round(thr, 4), "far": round(far, 4), "frr": round(frr, 4)}


def _eer(pos: list[float], neg: list[float]) -> tuple[float, float]:
    """EER and its threshold from the two score lists."""
    grid = np.unique(np.round(np.concatenate([pos, neg]), 4))
    best = (1.0, 0.5)
    for t in grid:
        far = sum(1 for s in neg if s >= t) / len(neg)
        frr = sum(1 for s in pos if s < t) / len(pos)
        gap = abs(far - frr)
        if gap < best[0]:
            best = (gap, float(t))
    t = best[1]
    far = sum(1 for s in neg if s >= t) / len(neg)
    frr = sum(1 for s in pos if s < t) / len(pos)
    return (far + frr) / 2, t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/av/LibriSpeech/dev-clean")
    ap.add_argument("--speakers", type=int, default=36)
    ap.add_argument("--utts-per-speaker", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="reports/speaker_acceptance.json")
    args = ap.parse_args()

    random.seed(args.seed)
    root = Path(args.root)
    by_speaker: dict[str, list[Path]] = {}
    for flac in sorted(root.rglob("*.flac")):
        spk = flac.parent.parent.name  # .../speaker/chapter/file.flac
        by_speaker.setdefault(spk, []).append(flac)
    usable = {s: sorted(p) for s, p in by_speaker.items() if len(p) >= args.utts_per_speaker}
    speakers = sorted(usable)[: args.speakers]
    if len(speakers) < 6:
        print(json.dumps({"status": "insufficient_data", "speakers_found": len(speakers)}))
        return 1

    half = len(speakers) // 2
    pools = {"calibration": speakers[:half], "heldout": speakers[half:]}

    settings = get_settings()
    embedder = EcapaSpeakerEmbedder(settings.speaker_model_dir)
    if not embedder.is_ready():
        print(json.dumps({"status": "speaker_model_unavailable",
                          "detail": embedder.describe()}))
        return 1

    cache: dict[str, np.ndarray] = {}
    for spk in speakers:
        for path in usable[spk][: args.utts_per_speaker]:
            cache[str(path)] = embedder.embed(_load(path), 16000)

    report: dict[str, object] = {
        "model": "ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)",
        "dataset": str(root),
        "embedding_dim": int(next(iter(cache.values())).shape[0]),
        "pools": {k: len(v) for k, v in pools.items()},
        "speakers_used": len(speakers),
        "utts_per_speaker": args.utts_per_speaker,
    }

    per_pool: dict[str, dict] = {}
    for name, spks in pools.items():
        pos, neg = [], []
        paths = {s: [str(p) for p in usable[s][: args.utts_per_speaker]] for s in spks}
        for spk in spks:
            for a, b in itertools.combinations(paths[spk], 2):
                pos.append(_cos(cache[a], cache[b]))
        for s1, s2 in itertools.combinations(spks, 2):
            # one pair per speaker pair keeps the negatives balanced with positives
            pos_a = random.choice(paths[s1])
            pos_b = random.choice(paths[s2])
            neg.append(_cos(cache[pos_a], cache[pos_b]))
        eer, eer_thr = _eer(pos, neg)
        per_pool[name] = {
            "n_same": len(pos),
            "n_different": len(neg),
            "same_mean": round(float(np.mean(pos)), 4),
            "different_mean": round(float(np.mean(neg)), 4),
            "eer": round(eer, 4),
            "eer_threshold": round(eer_thr, 4),
            "at_eer": _metrics(pos, neg, eer_thr),
            "at_app_threshold_0.5": _metrics(pos, neg, 0.5),
            "at_app_threshold_0.7": _metrics(pos, neg, 0.7),
        }

    cal_thr = per_pool["calibration"]["eer_threshold"]
    report["pools_detail"] = per_pool
    report["heldout_at_calibration_eer_threshold"] = {
        name: per_pool["heldout"][name] for name in per_pool["heldout"]
    }
    report["decision"] = {
        "calibrated_threshold": cal_thr,
        "app_default_threshold": 0.5,
        "note": (
            "FAR/FRR here are for the embedding+cosine stage only. A cloned voice "
            "can score high against its target, so a high similarity is evidence of "
            "acoustic similarity, never proof of identity."
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
