"""Pitch (fundamental frequency) of speech, for the advisory enrollment range check. numpy only, RAM only.

40 ms frames with a 10 ms hop; per frame the normalised autocorrelation
r(lag) = sum x[n] x[n+lag] / sqrt(sum x[n]^2 * sum x[n+lag]^2) over lags for 60 to 500 Hz (men, women
and children). A frame is voiced when its best local peak reaches VOICED_CORR. Octave guard: a periodic
frame also peaks at twice its period, so the shortest-lag peak within OCTAVE_RATIO of the best one is
taken. The peak lag is refined by parabolic interpolation. low/high are the 10th and 90th percentiles of
the voiced-frame pitch. Working copies of the audio are zeroed before returning; nothing is stored."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

FRAME_S = 0.040
HOP_S = 0.010
F0_MIN_HZ, F0_MAX_HZ = 60.0, 500.0
VOICED_CORR = 0.5            # peak normalised autocorrelation for a voiced frame
OCTAVE_RATIO = 0.9           # a shorter-lag peak at least this fraction of the best one wins (no sub-octave picks)
MIN_VOICED_FRAMES = 10       # fewer voiced frames (100 ms) => no estimate
LOW_PERCENTILE, HIGH_PERCENTILE = 10, 90
SILENT_RMS = 1e-4            # frames quieter than about -80 dBFS are skipped
BLOCK_FRAMES = 128           # frames analysed at once (bounds the working memory)
ENERGY_EPS = 1e-20


@dataclass(frozen=True)
class PitchStats:
    median_hz: float
    low_hz: float
    high_hz: float
    voiced_frames: int


def _block_f0(x: np.ndarray, starts: np.ndarray, n: int, sample_rate: int, lags: np.ndarray) -> list[float]:
    frames = x[starts[:, None] + np.arange(n)].astype(np.float64)   # a working copy, zeroed below
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = ac = cs = None
    try:
        frames -= frames.mean(axis=1, keepdims=True)
        cs = np.concatenate([np.zeros((frames.shape[0], 1)), np.cumsum(frames * frames, axis=1)], axis=1)
        spec = np.fft.rfft(frames, nfft, axis=1)
        spec *= spec.conj()
        ac = np.fft.irfft(spec, nfft, axis=1)[:, :lags[-1] + 1]
        e0 = cs[:, n - lags]              # energy of x[0 : n-lag]
        e1 = cs[:, [n]] - cs[:, lags]     # energy of x[lag : n]
        r = ac[:, lags] / np.sqrt(e0 * e1 + ENERGY_EPS)
        quiet = cs[:, n] < (SILENT_RMS ** 2) * n
        out = []
        for i in range(r.shape[0]):
            if quiet[i]:
                continue
            f0 = _frame_f0(r[i], lags, sample_rate)
            if f0 is not None:
                out.append(f0)
        return out
    finally:
        for a in (frames, spec, ac, cs):
            if a is not None:
                a.fill(0)


def _frame_f0(r: np.ndarray, lags: np.ndarray, sample_rate: int) -> float | None:
    mid = r[1:-1]
    peaks = np.flatnonzero((mid > r[:-2]) & (mid >= r[2:])) + 1   # local maxima strictly inside the lag range
    if peaks.size == 0:
        return None
    best = float(r[peaks].max())
    if best < VOICED_CORR:
        return None
    k = int(peaks[r[peaks] >= OCTAVE_RATIO * best][0])          # shortest lag close to the best: octave guard
    a, b, c = float(r[k - 1]), float(r[k]), float(r[k + 1])
    den = a - 2.0 * b + c
    shift = 0.5 * (a - c) / den if den < 0 else 0.0
    f0 = sample_rate / (float(lags[k]) + float(np.clip(shift, -0.5, 0.5)))
    return f0 if F0_MIN_HZ <= f0 <= F0_MAX_HZ else None


def estimate(speech: np.ndarray, sample_rate: int) -> PitchStats | None:
    """Pitch of mono `speech` in [-1, 1], or None when fewer than MIN_VOICED_FRAMES frames are voiced."""
    x = np.asarray(speech)
    if x.ndim != 1 or isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        return None
    n = int(round(FRAME_S * sample_rate))
    hop = max(1, int(round(HOP_S * sample_rate)))
    lag_lo = max(2, int(np.floor(sample_rate / F0_MAX_HZ)))
    lag_hi = int(np.ceil(sample_rate / F0_MIN_HZ))
    if x.size < n or lag_hi + 1 >= n:
        return None
    lags = np.arange(lag_lo - 1, lag_hi + 2)   # one extra lag each side for the peak test and interpolation
    starts = np.arange(0, x.size - n + 1, hop)
    f0s: list[float] = []
    for b in range(0, starts.size, BLOCK_FRAMES):
        f0s += _block_f0(x, starts[b:b + BLOCK_FRAMES], n, sample_rate, lags)
    if len(f0s) < MIN_VOICED_FRAMES:
        return None
    v = np.asarray(f0s)
    return PitchStats(float(np.median(v)), float(np.percentile(v, LOW_PERCENTILE)), float(np.percentile(v, HIGH_PERCENTILE)), int(v.size))


__all__ = ["PitchStats", "estimate"]
