"""Build the WavLM and XLS-R spoof heads from labelled ASVspoof 2019 LA audio.

Pipeline: ASVspoof parquet -> balanced labelled clips -> frozen SSL embeddings ->
logistic head with a fitted threshold -> measured EER on a held-out split.

Every number written into the artifact comes from the split it names. If the
dataset or a checkpoint is missing, the script says so and writes nothing, so the
branch stays honestly "unavailable" instead of shipping a made-up head.

Usage:
  python -m scripts.train_spoof_heads --branch wavlm
  python -m scripts.train_spoof_heads --branch all --per-class 250
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATASET_REPO = "Bisher/ASVspoof_2019_LA"
DATASET_FILE = "data/validation-00000-of-00001.parquet"
DATASET_DIR = BACKEND_DIR / "models" / "datasets" / "asvspoof2019-la"
HEADS_DIR = BACKEND_DIR / "models" / "spoof_heads"
HF_DIR = BACKEND_DIR / "models" / "hf"

BRANCHES = {
    "wavlm": "wavlm-base-plus",
    "xlsr": "xlsr-900m",
}

SAMPLE_RATE = 16000
MAX_SECONDS = 4.0


def _log(message: str) -> None:
    print(message, flush=True)


def ensure_parquet() -> Path:
    from huggingface_hub import hf_hub_download

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    local = DATASET_DIR / Path(DATASET_FILE).name
    if local.is_file() and local.stat().st_size > 100_000_000:
        _log(f"dataset present: {local} ({local.stat().st_size/1e6:.0f} MB)")
        return local
    _log(f"downloading {DATASET_REPO}/{DATASET_FILE}")
    path = hf_hub_download(
        repo_id=DATASET_REPO, filename=DATASET_FILE, repo_type="dataset", local_dir=str(DATASET_DIR)
    )
    return Path(path)


def load_clips(parquet: Path, per_class: int, seed: int = 13) -> tuple[list[np.ndarray], list[int]]:
    """Read a balanced, deterministic subset of clips from the parquet shard."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet)
    names = pf.schema_arrow.names
    _log(f"columns: {names}")
    audio_col = next((c for c in names if c in {"audio", "wav", "file", "audio_bytes"}), None)
    label_col = next((c for c in names if c in {"label", "labels", "key", "bonafide"}), None)
    if audio_col is None or label_col is None:
        raise RuntimeError(f"cannot identify audio/label columns in {names}")

    rng = np.random.default_rng(seed)
    kept: dict[int, list[bytes]] = {0: [], 1: []}
    seen = 0
    for batch in pf.iter_batches(batch_size=64, columns=[audio_col, label_col]):
        table = batch.to_pylist()
        for row in table:
            seen += 1
            raw = row[audio_col]
            label = row[label_col]
            if isinstance(raw, dict):
                raw = raw.get("bytes") or raw.get("path")
            if isinstance(label, str):
                low = label.lower()
                label = 0 if ("bonafide" in low or low in {"0", "real", "human"}) else 1
            else:
                label = int(label)
                # ASVspoof convention: 0 = bonafide, 1 = spoof
            if not isinstance(raw, (bytes, bytearray)) or label not in (0, 1):
                continue
            bucket = kept[label]
            if len(bucket) < per_class * 3:
                bucket.append(bytes(raw))
            if len(kept[0]) >= per_class * 3 and len(kept[1]) >= per_class * 3:
                break
        if len(kept[0]) >= per_class * 3 and len(kept[1]) >= per_class * 3:
            break
    _log(f"scanned {seen} rows; bonafide={len(kept[0])} spoof={len(kept[1])}")

    samples: list[np.ndarray] = []
    labels: list[int] = []
    for label in (0, 1):
        pool = kept[label]
        if len(pool) < per_class:
            raise RuntimeError(f"only {len(pool)} clips of class {label}")
        picks = rng.choice(len(pool), size=per_class, replace=False)
        for index in picks:
            decoded = decode(pool[int(index)])
            if decoded is None:
                continue
            samples.append(decoded)
            labels.append(label)
    return samples, labels


def decode(raw: bytes) -> np.ndarray | None:
    """Decode one container via ffmpeg to 16 kHz mono float32."""
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le",
            "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1",
        ],
        input=raw,
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    audio = np.frombuffer(proc.stdout, dtype="<f4")
    limit = int(MAX_SECONDS * SAMPLE_RATE)
    if audio.size > limit:
        start = int(np.argmax(np.abs(audio[: audio.size - limit + 1])))
        audio = audio[start : start + limit]
    if audio.size < int(0.5 * SAMPLE_RATE):
        return None
    return audio.astype("float32")


def embed_all(branch: str, clips: list[np.ndarray], cache: Path) -> np.ndarray:
    if cache.is_file():
        data = np.load(cache)
        if data["embeddings"].shape[0] == len(clips):
            _log(f"embeddings cached: {cache} {data['embeddings'].shape}")
            return data["embeddings"]
    import torch
    from transformers import AutoModel

    model_dir = HF_DIR / BRANCHES[branch]
    _log(f"loading {model_dir}")
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
    model.eval()
    vectors = []
    for index, clip in enumerate(clips):
        wav = torch.from_numpy(clip).unsqueeze(0)
        with torch.no_grad():
            out = model(wav)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            raise RuntimeError("unexpected SSL output")
        vectors.append(hidden.mean(dim=1).squeeze(0).numpy().astype("float32"))
        if (index + 1) % 50 == 0:
            _log(f"  {branch}: {index + 1}/{len(clips)}")
    embeddings = np.stack(vectors)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, embeddings=embeddings)
    return embeddings


def eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the threshold where false-accept equals false-reject."""
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fpr - fnr)))
    rate = float(max(fpr[index], fnr[index]))
    threshold = float(thresholds[min(index, len(thresholds) - 1)])
    if not np.isfinite(threshold):
        threshold = 0.5
    return rate, threshold


FIXTURES = [
    BACKEND_DIR / "tests" / "fixtures" / "audio" / "clean_speech.mp3",
    BACKEND_DIR / "tests" / "fixtures" / "audio" / "narrowband_noisy.mp3",
]


def out_of_domain_probe(branch: str, mean, scale, coef, intercept) -> dict:
    """Score genuine speech that is not from the training corpus.

    A head that calls real human speech synthetic has learned the corpus, not
    spoofing, so it is not allowed to vote until this probe passes.
    """
    import joblib  # noqa: F401  (kept for symmetry with the caller)

    scores = []
    for fixture in FIXTURES:
        if not fixture.is_file():
            continue
        audio = decode(fixture.read_bytes())
        if audio is None:
            continue
        embedding = embed_all(branch, [audio], DATASET_DIR / f"probe_{branch}_{fixture.stem}.npz")
        z = (embedding[0] - mean) / scale
        scores.append(float(1.0 / (1.0 + np.exp(-(z @ coef + intercept)))))
    if not scores:
        return {"validated": False, "reason": "no out-of-domain probe audio available", "scores": []}
    worst = max(scores)
    return {
        "validated": worst <= 0.5,
        "probe": "genuine human speech outside the training corpus",
        "scores": [round(value, 4) for value in scores],
        "worst": round(worst, 4),
        "reason": None
        if worst <= 0.5
        else "flagged genuine human speech as synthetic; the head learned corpus artefacts",
    }


def train_branch(branch: str, clips: list[np.ndarray], labels: np.ndarray) -> dict:
    import joblib
    from sklearn.linear_model import LogisticRegression

    embeddings = embed_all(branch, clips, DATASET_DIR / f"embeddings_{branch}.npz")
    rng = np.random.default_rng(7)
    index = rng.permutation(len(labels))
    split = int(len(index) * 0.7)
    train_idx, valid_idx = index[:split], index[split:]

    mean = embeddings[train_idx].mean(axis=0)
    scale = embeddings[train_idx].std(axis=0) + 1e-6
    xtr = (embeddings[train_idx] - mean) / scale
    xva = (embeddings[valid_idx] - mean) / scale

    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(xtr, labels[train_idx])
    scores = clf.predict_proba(xva)[:, 1]
    rate, threshold = eer(scores, labels[valid_idx])

    probe = out_of_domain_probe(branch, mean, scale, clf.coef_[0], float(clf.intercept_[0]))
    _log(f"{branch}: out-of-domain probe {probe}")

    train_scores = clf.predict_proba(xtr)[:, 1]
    train_rate, _ = eer(train_scores, labels[train_idx])

    # Deploy a documented 0.5 decision threshold (equal prior, the honest default
    # for a detector whose in-domain scores are this separated); the EER threshold
    # is kept as a measured statistic, not as a decision rule.
    decision_threshold = 0.5
    artifact = {
        "coef": clf.coef_[0],
        "intercept": float(clf.intercept_[0]),
        "mean": mean,
        "scale": scale,
        "threshold": decision_threshold,
        "metrics": {
            "eer": round(rate, 4),
            "train_eer": round(train_rate, 4),
            "threshold_at_eer": round(threshold, 4),
            "decision_threshold": decision_threshold,
            "valid_clips": int(len(valid_idx)),
            "train_clips": int(len(train_idx)),
            "domain": "ASVspoof2019 LA dev (in-domain)",
            "out_of_domain": probe,
        },
        "metadata": {
            "branch": branch,
            "model": BRANCHES[branch],
            "dataset": f"{DATASET_REPO} ({DATASET_FILE})",
            "calibration": "platt",
            "threshold": decision_threshold,
            "pooling": "mean over last_hidden_state",
            "sample_rate": SAMPLE_RATE,
            "metrics": {
                "eer": round(rate, 4),
                "train_eer": round(train_rate, 4),
                "threshold_at_eer": round(threshold, 4),
                "domain": "ASVspoof2019 LA dev (in-domain)",
                "out_of_domain_validated": bool(probe.get("validated")),
                "valid_clips": int(len(valid_idx)),
            },
        },
    }
    HEADS_DIR.mkdir(parents=True, exist_ok=True)
    out = HEADS_DIR / f"{branch}_spoof.joblib"
    joblib.dump(artifact, out)
    _log(f"{branch}: EER {rate:.3f} (train {train_rate:.3f}), threshold {threshold:.3f} -> {out}")
    return {"branch": branch, "eer": rate, "threshold": threshold, "artifact": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default="all", choices=["all", *BRANCHES])
    parser.add_argument("--per-class", type=int, default=250)
    args = parser.parse_args()

    parquet = ensure_parquet()
    clips, labels = load_clips(parquet, args.per_class)
    if not clips:
        _log("no clips decoded; nothing written")
        return 1
    label_array = np.asarray(labels)
    _log(f"dataset: {len(clips)} clips, {int(label_array.sum())} spoof")

    branches = list(BRANCHES) if args.branch == "all" else [args.branch]
    results = []
    for branch in branches:
        try:
            results.append(train_branch(branch, clips, label_array))
        except Exception as exc:  # noqa: BLE001 - report and continue
            _log(f"{branch}: FAILED {type(exc).__name__}: {exc}")
    _log("RESULTS:" + json.dumps(results))
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
