"""Speaker comparison without torch: MFCC statistics fallback for the slim build.

The full build uses ECAPA-TDNN embeddings (speechbrain) and a threshold
calibrated on LibriSpeech pairs. Vercel cannot ship torch, so this module gives
the serverless build a *closeness* measure that still costs no model download:

  * 20 MFCCs per 25 ms frame (numpy mel filterbank + DCT-II)
  * mean and standard deviation across speech frames, plus delta statistics
  * L2-normalised vector, cosine similarity between the two recordings

What it is not: it has no discriminative training, so it cannot separate two
people as reliably as ECAPA and its similarity is NOT a probability. It returns
no same/different verdict (outcome stays None) and says so in `limitations`.
VOXSENTINEL_MFCC_SPEAKER_ENABLED=false disables it.
"""

from __future__ import annotations

import numpy as np

from app.services.speaker import (
    BaseSpeakerVerifier,
    ModelDescription,
    SpeakerVerificationResult,
)

FRAME = 400  # 25 ms at 16 kHz
HOP = 160  # 10 ms
N_MELS = 26
N_MFCC = 20


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = N_MELS) -> np.ndarray:
    def hz_to_mel(f: float) -> float:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m: float) -> float:
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    fmin, fmax = 60.0, sr / 2.0
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        left, centre, right = bins[m - 1], bins[m], bins[m + 1]
        if centre == left:
            centre = left + 1
        if right == centre:
            right = centre + 1
        for k in range(left, min(centre, fb.shape[1])):
            fb[m - 1, k] = (k - left) / max(1, centre - left)
        for k in range(centre, min(right, fb.shape[1])):
            fb[m - 1, k] = (right - k) / max(1, right - centre)
    return fb


def _dct_matrix(n_out: int, n_in: int) -> np.ndarray:
    """Orthonormal DCT-II, so MFCCs are comparable across recordings."""
    k = np.arange(n_out)[:, None]
    n = np.arange(n_in)[None, :]
    mat = np.cos(np.pi * k * (2 * n + 1) / (2 * n_in))
    mat *= np.sqrt(2.0 / n_in)
    mat[0] /= np.sqrt(2.0)
    return mat


def mfcc(y: np.ndarray, sr: int) -> np.ndarray:
    """MFCC matrix (frames x N_MFCC) for speech frames only."""
    y = np.asarray(y, dtype=np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000 and len(y):
        # Simple decimation/interpolation keeps the frame geometry fixed.
        target = int(len(y) * 16000 / sr)
        y = np.interp(np.linspace(0, len(y) - 1, target), np.arange(len(y)), y)
        sr = 16000
    if len(y) < FRAME * 3:
        return np.zeros((0, N_MFCC))
    n = 1 + (len(y) - FRAME) // HOP
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    frames = y[idx] * np.hamming(FRAME)[None, :]
    energy = np.sqrt((frames**2).mean(axis=1))
    keep = energy > max(0.01, np.percentile(energy, 40))
    frames = frames[keep]
    if len(frames) < 3:
        return np.zeros((0, N_MFCC))
    spec = np.abs(np.fft.rfft(frames, n=1024, axis=1)) ** 2
    mel = spec @ _mel_filterbank(16000, 1024).T
    logmel = np.log(mel + 1e-10)
    return logmel @ _dct_matrix(N_MFCC, N_MELS).T


def embed(y: np.ndarray, sr: int) -> np.ndarray:
    """Mean/std/delta-statistics vector, L2 normalised."""
    feats = mfcc(y, sr)
    if len(feats) == 0:
        return np.zeros(0)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    if len(feats) > 2:
        delta = np.diff(feats, axis=0)
        dstd = delta.std(axis=0)
    else:
        dstd = np.zeros(N_MFCC)
    vec = np.concatenate([mean, std, dstd])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else np.zeros_like(vec)


class MfccStatSpeakerVerifier(BaseSpeakerVerifier):
    """Torch-free speaker *closeness* comparison (no verdict, no probability)."""

    kind = "speaker"

    def is_ready(self) -> bool:
        return True

    def compare(
        self, reference: np.ndarray, reference_sr: int, test: np.ndarray, test_sr: int
    ) -> SpeakerVerificationResult:
        emb_a = embed(reference, reference_sr)
        emb_b = embed(test, test_sr)
        if emb_a.size == 0 or emb_b.size == 0:
            return SpeakerVerificationResult(
                status="unavailable",
                similarity=None,
                threshold=None,
                outcome=None,
                embedding_model=None,
                warnings=[
                    "One of the recordings has too little speech for an MFCC comparison."
                ],
                limitations=self._limitations(),
            )
        similarity = float(np.dot(emb_a, emb_b))
        if not np.isfinite(similarity):
            raise RuntimeError("similarity computation produced a non-finite value")
        return SpeakerVerificationResult(
            status="completed",
            similarity=similarity,
            threshold=None,
            outcome=None,  # no calibrated threshold in this build: no verdict
            embedding_model="MFCC statistics (numpy fallback)",
            warnings=[
                "This serverless build uses an MFCC-statistics comparison instead of the "
                "ECAPA embedding model, and no calibrated threshold is installed on it, so "
                "no same/different-speaker decision is produced.",
            ],
            limitations=self._limitations(),
        )

    @staticmethod
    def _limitations() -> list[str]:
        return [
            "Similarity is the cosine between hand-built MFCC statistics vectors, not an "
            "embedding distance from a trained speaker model.",
            "Without a calibrated threshold this is a closeness measure only: it must not "
            "be used to claim two recordings are the same person.",
            "A cloned voice can score high similarity to its target, and a genuine clip can "
            "score low because of noise or recording conditions.",
            "Identity is never claimed from voice alone; this feature requires explicit "
            "authorization for the reference recording.",
        ]

    def describe(self) -> ModelDescription:
        return ModelDescription(
            id="speaker_verifier",
            name="MFCC-statistics speaker comparison (torch-free)",
            version="0.1.0",
            mode="baseline",
            kind="speaker",
            description=(
                "Compares MFCC mean/std/delta statistics with cosine similarity. Runs "
                "without torch so the serverless build can still give a closeness reading, "
                "but it is weaker than ECAPA and reports no same/different verdict."
            ),
            loaded=True,
        )
