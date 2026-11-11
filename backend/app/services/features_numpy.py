"""Pure-numpy re-implementation of `app/services/features_enhanced.py`.

WHY THIS MODULE EXISTS
----------------------
The production (Vercel) bundle ships numpy + scikit-learn + soundfile but no
librosa, and `api/index.py` deliberately makes `librosa` fail to import so
capability probes stay honest. The trained artifact (`models/detector.joblib`,
`kind="mfcc_enhanced"`) therefore used to be unrunnable in production and the
API silently fell back to a hand-written heuristic, so live scores did not
match local ones.

This module recomputes the *same* feature vector with numpy only, mirroring the
librosa 0.11 code paths call-for-call, so the SAME artifact can be served in
both environments. It is a port, not a re-design:

    librosa                                    here
    -----------------------------------------  --------------------------------
    librosa.resample(y, 'soxr_hq')             _resample  (polyphase FIR)
    librosa.stft (hann, center=True, pad 0)    _stft_magnitude
    librosa.filters.mel(norm='slaney')         _mel_filterbank
    librosa.power_to_db (ref=1, top_db=80)     _power_to_db
    scipy.fft.dct(type=2, norm='ortho')        _dct_matrix
    librosa.feature.delta (savgol 9/1/interp)  _savgol_delta
    zero_crossing_rate / rms / spectral_*      the frame-based helpers below

Everything librosa does with default padding/centering semantics is reproduced
including the subtle bits: Hann is *periodic* (fftbins=True), ZCR pads with
`mode="edge"` while RMS/STFT pad with `mode="constant"`, `power_to_db` clamps at
`amin=1e-10` and `top_db=80`, delta uses Savitzky-Golay with `mode="interp"`
(i.e. a degree-`polyorder` least-squares fit at both edges), and `normalize()`
falls back to a divisor of 1.0 for all-zero frames.

The legacy approximation previously used for resampling is no longer selected.
Exact inference resampling now uses soxr HQ, matching librosa. Historical context: librosa 0.11 defaults to
`soxr_hq`, a mixed-phase filter bank that cannot be reproduced exactly by a
linear-phase FIR (and is unavailable in production). `_resample` therefore uses
a Kaiser-windowed polyphase FIR, and the measured gap is documented in
tests/test_features_numpy_parity.py.

Public API: `extract_enhanced_features_numpy(samples, sample_rate, config=None)`.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from app.services.features import (
    FeatureExtractionError,
    _fixed_segment,
    _to_mono_float32,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from app.services.features_enhanced import EnhancedFeatureConfig

# librosa defaults this port has to reproduce exactly.
_AMIN = 1e-10           # power_to_db / spectral_flatness
_TOP_DB = 80.0          # power_to_db
_ZCR_THRESHOLD = 1e-10  # librosa.zero_crossings
_ROLL_PERCENT = 0.85
_DELTA_WIDTH = 9
_DELTA_POLYORDER = 1

# Resampler design. `_RESAMPLE_HALF` is the half-length per *output* sample of
# the Kaiser-windowed prototype, `_RESAMPLE_ROLLOFF` the passband edge relative
# to the lower (output) Nyquist and `_RESAMPLE_BETA` the Kaiser beta. Values
# approximate the soxr_hq passband (0.913 of the lower Nyquist) with a smooth
# transition; the achieved parity is measured in the test suite, not assumed.
_RESAMPLE_HALF = 32
_RESAMPLE_ROLLOFF = 0.9565
_RESAMPLE_BETA = 8.6
_RESAMPLE_MAX_PHASES = 4096


# --------------------------------------------------------------------------- #
# small building blocks
# --------------------------------------------------------------------------- #
def _periodic_hann(length: int) -> np.ndarray:
    """Hann window as scipy.signal.get_window('hann', N, fftbins=True).

    librosa's STFT default window is the *periodic* Hann, i.e. a symmetric
    Hann of length+1 with the last sample dropped.
    """
    if length == 1:
        return np.ones(1, dtype=np.float64)
    return np.hanning(length + 1)[:-1]


def _frame(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Frames as (n_frames, frame_length), equivalent to librosa.util.frame."""
    if x.size < frame_length:
        raise FeatureExtractionError(
            f"Input is too short (n={x.size}) for frame_length={frame_length}."
        )
    if hop_length < 1:
        raise FeatureExtractionError(f"Invalid hop_length: {hop_length}.")
    view = np.lib.stride_tricks.sliding_window_view(x, frame_length)
    return view[::hop_length]


def _stft_magnitude(y: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """Magnitude spectrogram, librosa.stft(window='hann', center=True,
    pad_mode='constant') semantics, shape (1 + n_fft // 2, n_frames).

    For centered analysis librosa zero-pads n_fft // 2 on both sides and the
    frame count is 1 + len(y) // hop_length.
    """
    pad = n_fft // 2
    padded = np.pad(y.astype(np.float64, copy=False), (pad, pad), mode="constant")
    frames = _frame(padded, n_fft, hop_length)
    windowed = frames * _periodic_hann(n_fft)[None, :]
    spectrum = np.fft.rfft(windowed, n=n_fft, axis=-1)
    return np.abs(spectrum.astype(np.complex64)).T


def _hz_to_mel(frequencies: np.ndarray) -> np.ndarray:
    """librosa.hz_to_mel(htk=False) â€” Slaney piecewise scale."""
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    log_t = frequencies >= min_log_hz
    mels = np.where(log_t, min_log_mel + np.log(np.maximum(frequencies, 1e-300) / min_log_hz) / logstep, mels)
    return mels


def _mel_to_hz(mels: np.ndarray) -> np.ndarray:
    """librosa.mel_to_hz(htk=False) â€” inverse of the Slaney scale."""
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = math.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    return np.where(log_t, min_log_hz * np.exp(logstep * (mels - min_log_mel)), freqs)


def _mel_frequencies(n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    return _mel_to_hz(np.linspace(_hz_to_mel(np.array([fmin]))[0],
                                  _hz_to_mel(np.array([fmax]))[0], n_mels))


def _mel_filterbank(
    sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    """librosa.filters.mel(norm='slaney') â€” shape (n_mels, 1 + n_fft // 2)."""
    fft_freqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sample_rate)
    mel_f = _mel_frequencies(n_mels + 2, fmin, fmax)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)
    weights = np.zeros((n_mels, fft_freqs.size), dtype=np.float32)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    # Slaney normalisation: constant energy per mel band.
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]
    return weights


def _power_to_db(S: np.ndarray) -> np.ndarray:
    """librosa.power_to_db(S, ref=1.0, amin=1e-10, top_db=80)."""
    log_spec = 10.0 * np.log10(np.maximum(_AMIN, S))
    log_spec -= 10.0 * np.log10(max(_AMIN, 1.0))
    return np.maximum(log_spec, log_spec.max() - _TOP_DB)


def _dct_matrix(size: int) -> np.ndarray:
    """DCT-II orthonormal basis, rows = output coefficients.

    Multiplying on the left by this matrix reproduces
    scipy.fft.dct(x, axis=-2, type=2, norm='ortho').
    """
    n = np.arange(size, dtype=np.float64)
    k = np.arange(size, dtype=np.float64)[:, None]
    matrix = np.sqrt(2.0 / size) * np.cos(np.pi * (2.0 * n[None, :] + 1.0) * k / (2.0 * size))
    matrix[0, :] = np.sqrt(1.0 / size)
    return matrix


def _savgol_delta(data: np.ndarray, width: int = _DELTA_WIDTH,
                  polyorder: int = _DELTA_POLYORDER) -> np.ndarray:
    """librosa.feature.delta(data, width=9, order=1, mode='interp').

    Interior frames are the Savitzky-Golay derivative (a fixed FIR); the
    `width // 2` frames at each edge use the derivative of a degree-`polyorder`
    polynomial fitted to the first/last `width` frames â€” exactly what
    scipy.signal.savgol_filter(mode='interp') does via _fit_edge.
    """
    if width < 3 or width % 2 != 1:
        raise FeatureExtractionError("delta width must be an odd integer >= 3")
    n_frames = data.shape[-1]
    if width > n_frames:
        raise FeatureExtractionError(
            f"delta width={width} cannot exceed the number of frames={n_frames}"
        )
    work = data.astype(np.float64, copy=False)
    padded = np.pad(work, [(0, 0), (width // 2, width // 2)], mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width, axis=-1)

    # Savitzky-Golay coefficients for a first derivative, degree 1, delta=1:
    # least-squares slope of the window, scaled by the sample spacing.
    t = np.arange(width, dtype=np.float64) - (width - 1) / 2.0
    coeffs = t / np.sum(t * t)

    out = np.einsum("j,ikj->ik", coeffs, windows, optimize=True)

    # Edges: scipy's mode='interp' discards the constant-padded windows and
    # instead evaluates a degree-`polyorder` least-squares fit over the first
    # and last `width` frames. For polyorder=1 the derivative is constant, so
    # each edge block is filled with that fitted slope.
    half = width // 2
    tt = np.arange(width, dtype=np.float64)
    centre = tt - tt.mean()
    weights = centre / np.sum(centre * centre)
    for edge_start, interp_slice in ((0, slice(0, half)),
                                     (n_frames - width, slice(n_frames - half, n_frames))):
        slope = work[:, edge_start : edge_start + width] @ weights
        out[:, interp_slice] = slope[:, None]
    return out


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Band-limited rational resampling with a polyphase Kaiser FIR.

    librosa 0.11 defaults to res_type='soxr_hq'. soxr cannot be reproduced
    exactly (mixed-phase filter bank, and absent from the production bundle),
    so this is a linear-phase Kaiser-windowed polyphase FIR whose passband is
    aligned with the soxr_hq roll-off. Output length matches soxr's
    `ceil(len * target / orig)`.
    """
    if orig_sr == target_sr:
        return samples
    if orig_sr <= 0 or target_sr <= 0:
        raise FeatureExtractionError(f"Invalid sample rate: {orig_sr} -> {target_sr}")

    # Match librosa's default exactly when the small soxr runtime is installed.
    # Never silently substitute another resampler: it changes model features.
    try:
        import soxr
    except ImportError as exc:
        raise FeatureExtractionError(
            "Exact resampling requires soxr; install the pinned runtime requirements."
        ) from exc
    output = soxr.resample(np.asarray(samples, dtype=np.float32), orig_sr, target_sr, quality="HQ")
    expected = int(math.ceil(len(samples) * target_sr / orig_sr))
    return np.pad(output, (0, max(0, expected - len(output))))[:expected].astype(np.float32)


def _legacy_resample_approximation(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Archived FIR implementation; NOT used in model inference."""
    ratio = Fraction(int(target_sr), int(orig_sr)).limit_denominator(_RESAMPLE_MAX_PHASES)
    up, down = ratio.numerator, ratio.denominator

    x = np.asarray(samples, dtype=np.float64)
    n_out = int(math.ceil(x.size * up / down))

    half = _RESAMPLE_HALF
    cutoff = _RESAMPLE_ROLLOFF / max(up, down)
    support = half * up
    taps = np.arange(-support, support + 1, dtype=np.float64)
    prototype = cutoff * np.sinc(cutoff * taps) * np.kaiser(taps.size, _RESAMPLE_BETA)
    prototype *= up / prototype.sum()  # unit DC gain after decimation

    # y[m] = sum_s x[q - s] * h[r + s * up]  where m*down = q*up + r.
    padded = np.pad(x, (support, support + 1), mode="constant")
    offsets = np.arange(2 * half + 1)
    out = np.zeros(n_out, dtype=np.float64)
    block = 16384
    for start in range(0, n_out, block):
        m = np.arange(start, min(start + block, n_out), dtype=np.int64)
        position = m * down + support
        phases, centres = position % up, position // up
        # Index into `padded` so that padded[centres + support - s] == x[centres - s].
        rows = (centres + support)[:, None] - offsets[None, :]
        windows = padded[rows]
        # One prototype branch per distinct phase in this block.
        def _branch(phase: int) -> np.ndarray:
            idx = phase + offsets * up
            inside = (idx >= 0) & (idx < prototype.size)
            return np.where(inside, prototype[np.clip(idx, 0, prototype.size - 1)], 0.0)

        branch_table = np.stack([_branch(phase) for phase in range(up)])
        out[start : start + m.size] = np.einsum(
            "ij,ij->i", windows, branch_table[phases]
        )
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# frame-based spectral statistics
# --------------------------------------------------------------------------- #
def _stats(values: np.ndarray) -> np.ndarray:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if not (np.isfinite(mean) and np.isfinite(std)):
        raise FeatureExtractionError("Enhanced features produced non-finite statistics.")
    return np.array([mean, std], dtype=np.float32)


def _normalize_frames(S: np.ndarray) -> np.ndarray:
    """librosa.util.normalize(S, norm=1, axis=-2) for non-negative S.

    All-zero frames (from zero padding) are divided by 1.0, matching librosa's
    `length[small_idx] = 1.0` branch (`fill=None`).
    """
    length = np.sum(np.abs(S), axis=-2, keepdims=True)
    length = np.where(length < np.finfo(length.dtype).tiny, 1.0, length)
    return S / length


def _zero_crossing_rate(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """librosa.feature.zero_crossing_rate(center=True) â€” edge padding."""
    pad = frame_length // 2
    padded = np.pad(y.astype(np.float64, copy=False), (pad, pad), mode="edge")
    frames = _frame(padded, frame_length, hop_length)
    clipped = np.where(np.abs(frames) <= _ZCR_THRESHOLD, 0.0, frames)
    signs = np.signbit(clipped)
    crossings = signs[:, 1:] != signs[:, :-1]
    # librosa uses pad=False: do not invent a crossing for the first sample.
    return np.count_nonzero(crossings, axis=1) / frame_length


def _rms(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """librosa.feature.rms(center=True, pad_mode='constant')."""
    pad = frame_length // 2
    padded = np.pad(y.astype(np.float64, copy=False), (pad, pad), mode="constant")
    frames = _frame(padded, frame_length, hop_length)
    return np.sqrt(np.mean(frames * frames, axis=1))


def _spectral_shape(
    magnitude: np.ndarray, sample_rate: int, n_fft: int
) -> tuple[np.ndarray, ...]:
    """centroid, rolloff(0.85), flatness, bandwidth from a magnitude spectrogram."""
    n_bins = magnitude.shape[0]
    freqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sample_rate)
    if freqs.size != n_bins:  # pragma: no cover - defensive
        raise FeatureExtractionError("Spectrogram bin count does not match n_fft.")

    mag = magnitude.astype(np.float64, copy=False)
    norm = _normalize_frames(mag)

    centroid = np.sum(freqs[:, None] * norm, axis=0)

    total = np.cumsum(magnitude, axis=0)
    threshold = _ROLL_PERCENT * total[-1, :]
    above = total >= threshold[None, :]
    # librosa: nanmin(where(total < threshold, nan, 1) * freq) -> first bin
    # whose cumulative energy reaches the threshold.
    # librosa: nanmin(where(total < threshold, nan, 1) * freq) -> the first bin
    # whose cumulative energy reaches the threshold. An all-zero frame has
    # threshold 0, so every bin qualifies and the result is freq[0] == 0.
    rolloff = freqs[np.argmax(above, axis=0)].astype(np.float64)

    thresholded = np.maximum(_AMIN, mag ** 2.0)
    gmean = np.exp(np.mean(np.log(thresholded), axis=0))
    amean = np.mean(thresholded, axis=0)
    flatness = gmean / amean

    deviation = np.abs(centroid[None, :] - freqs[:, None])
    bandwidth = np.sum(norm * deviation ** 2.0, axis=0) ** 0.5
    return centroid, rolloff, flatness, bandwidth


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def extract_enhanced_features_numpy(
    samples: np.ndarray,
    sample_rate: int,
    config: EnhancedFeatureConfig | None = None,
) -> np.ndarray:
    """Same vector as `extract_enhanced_features`, using numpy only.

    Raises FeatureExtractionError for empty, non-finite or otherwise unusable
    input, exactly like the librosa implementation it mirrors.
    """
    if config is None:
        from app.services.features_enhanced import EnhancedFeatureConfig

        config = EnhancedFeatureConfig()

    mono = _to_mono_float32(samples)
    if mono.size == 0:
        raise FeatureExtractionError("Cannot extract features from empty audio.")
    if not np.all(np.isfinite(mono)):
        raise FeatureExtractionError("Audio contains non-finite samples.")
    mono = _resample(mono, int(sample_rate), config.target_sample_rate)
    segment = _fixed_segment(mono, config.target_sample_rate, config.segment_seconds)
    if segment.size < 2:
        raise FeatureExtractionError("Audio segment is too short for analysis.")
    if float(np.max(np.abs(segment))) == 0.0:
        raise FeatureExtractionError("Audio segment is silent (all zeros).")

    n_fft = config.n_fft
    hop = config.hop_length
    magnitude = _stft_magnitude(segment, n_fft, hop)
    if magnitude.shape[1] < 2:
        raise FeatureExtractionError("Audio segment is too short to compute MFCC statistics.")

    mel_basis = _mel_filterbank(
        config.target_sample_rate, n_fft, config.n_mels, config.fmin, config.fmax
    )
    mel = mel_basis @ (magnitude ** 2.0)
    log_mel = _power_to_db(mel)
    mfcc = (_dct_matrix(config.n_mels) @ log_mel)[: config.n_mfcc, :]

    parts: list[np.ndarray] = [
        mfcc.mean(axis=1).astype(np.float32),
        mfcc.std(axis=1).astype(np.float32),
    ]
    if config.include_delta_mfcc:
        delta = _savgol_delta(mfcc)
        parts.append(delta.mean(axis=1).astype(np.float32))
        parts.append(delta.std(axis=1).astype(np.float32))

    if config.include_spectral_shape:
        zcr = _zero_crossing_rate(segment, n_fft, hop)
        rms = _rms(segment, n_fft, hop)
        centroid, rolloff, flatness, bandwidth = _spectral_shape(
            magnitude, config.target_sample_rate, n_fft
        )
        parts.append(np.concatenate([
            _stats(zcr), _stats(rms), _stats(centroid), _stats(rolloff),
            np.array([float(np.mean(flatness))], dtype=np.float32),
            _stats(bandwidth),
        ]))

    features = np.concatenate(parts).astype(np.float32)
    if features.size != config.feature_dim:
        raise FeatureExtractionError(
            f"Enhanced feature dimension {features.size} != expected {config.feature_dim}."
        )
    if not np.all(np.isfinite(features)):
        raise FeatureExtractionError("Enhanced features contain non-finite values.")
    return features


def resample_numpy(samples, orig_sr: int, target_sr: int):
    """Public alias for the module's polyphase resampler."""
    return _resample(samples, orig_sr, target_sr)
