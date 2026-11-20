"""Sound direction of arrival (DOA) from two or more microphones (360-degree coverage plan).

GCC-PHAT time differences per microphone pair, then a far-field (plane-wave) least-squares
fit over all pairs. Results are in the shared HEAD FRAME (ADR-0060): x forward, y left, z up;
azimuth 0 ahead, +90 left, -90 right, 180 behind; elevation +90 straight up. A bearing from
here can be handed to the spatial merger as kind "sound", family "voice".

Geometry decides what can be resolved:
  * one pair (or collinear mics): angle from the pair axis only, source assumed near the
    horizontal plane -> two mirror candidates, ambiguous=True (a left/right temple pair cannot
    tell front from back);
  * coplanar mics (three or more, not collinear): the source's mirror image about the array plane
    fits equally well. In a horizontal plane both mirrors share the azimuth -> full azimuth,
    elevation None. A tilted or vertical plane -> ambiguous=True with both azimuths;
  * non-coplanar mics (rank-3 baselines): azimuth and elevation.
Pairs below MIN_CONFIDENCE are dropped first, so a dead mic in a 3-D array honestly degrades to
the planar (or pair) answer.
Below MIN_CONFIDENCE the answer is None: no guess is ever returned.

Proven on synthetic signals only (tests/test_doa.py). Not modelled: head shadow and
diffraction around a real head, near-field sources, moving sources, reverberant rooms beyond
a weak echo, per-device mic gain/phase mismatch, several simultaneous talkers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

SPEED_OF_SOUND_M_S = 343.0
FULL_TURN_DEG, HALF_TURN_DEG = 360.0, 180.0   # angle wrapping to (-180, 180] (ADR-0060 azimuth range), not tolerances
PHAT_EPSILON = 1e-12            # added to |cross-spectrum| before PHAT whitening
# Regularised PHAT: the whitening denominator is |G| + floor, floor = this x mean |G|. Pure PHAT gives
# the empty bins between a voice's harmonics the same weight as the harmonics; on synthetic harmonic
# tones at 10 dB SNR that lost the peak, while broadband accuracy is unchanged (tests/test_doa.py).
PHAT_FLOOR_FRACTION = 1.0
SILENCE_RMS = 1e-6              # channel RMS (full scale 1.0) below which a channel is silent
DEFAULT_INTERP = 8              # correlation upsampling factor (zero-padded spectrum)
# Peak / no-signal correlation RMS. Measured on independent white noise (+ hum/rumble), frames
# 512..16384, spacings 0.14/0.28 m, 2400 trials: median 4.6, maximum 8.55 -> significance 0 up to 9.
NOISE_PEAK_RATIO = 9.0
CLEAR_PEAK_RATIO = 18.0         # fully significant
MIN_CONFIDENCE = 0.2            # pair or result confidence below this is "no clear peak"
RESIDUAL_LIMIT_SAMPLES = 2.0    # weighted RMS fit residual (samples) at which DOA confidence reaches 0
RANK_TOLERANCE = 1e-3           # singular value / largest one below which a geometry axis is absent
COINCIDENT_MIC_M = 1e-6         # mics closer than this are one point: the pair is skipped
VERTICAL_AXIS_TOLERANCE = 1e-6  # horizontal part of a unit pair axis below which azimuth is unobservable
UNIT_NORM_TOLERANCE = 0.3       # 3-D fit: |u| must be 1 +- this for the pair delays to describe one plane wave
PLANE_MIRROR_TOL_DEG = 2.0      # planar array: mirror images closer than this in azimuth count as one answer

# Direction words (oiris/guard/phrases.py slot values), boundaries in degrees.
AHEAD_MAX_DEG = 45.0
SIDE_MAX_DEG = 135.0
VERTICAL_DEG = 45.0
HYSTERESIS_DEG = 10.0
DIRECTION_WORDS = ("ahead", "at left", "at right", "behind", "above", "below")


@dataclass(frozen=True)
class DoaResult:
    az_deg: float                       # head-frame azimuth; for an ambiguous pair, the front candidate
    el_deg: float | None                # None unless the array is non-coplanar
    ambiguous: bool                     # True when the geometry leaves a mirror image (front/back for a temple pair)
    candidates: list[float] = field(default_factory=list)   # every azimuth consistent with the data
    confidence: float = 0.0             # 0..1
    pairs_used: tuple[tuple[int, int], ...] = ()


def normalize_az(az_deg: float) -> float:
    """Wrap to (-180, 180]."""
    a = math.fmod(az_deg, FULL_TURN_DEG)
    if a <= -HALF_TURN_DEG:
        a += FULL_TURN_DEG
    elif a > HALF_TURN_DEG:
        a -= FULL_TURN_DEG
    return a


def max_tau_s(spacing_m: float, speed_m_s: float = SPEED_OF_SOUND_M_S) -> float:
    """Largest physically possible time difference for a mic pair."""
    return abs(spacing_m) / speed_m_s


def gcc_phat(sig_a: np.ndarray, sig_b: np.ndarray, sample_rate: float, max_tau_s: float,
             interp: int = DEFAULT_INTERP) -> tuple[float, float]:
    """Time difference of arrival (s, positive when the sound reaches b after a) and confidence 0..1.

    Lags are searched only within +-max_tau_s; the returned tau is clipped to that range.
    Confidence = peak height (1.0 for a perfectly coherent arrival) x its significance against the
    no-signal RMS of the rest of the correlation. Silent input returns (0.0, 0.0)."""
    tau, peak, ratio = _gcc_phat_peak(sig_a, sig_b, sample_rate, max_tau_s, interp)
    if peak <= 0.0:
        return tau, 0.0
    significance = float(np.clip((ratio - NOISE_PEAK_RATIO) / (CLEAR_PEAK_RATIO - NOISE_PEAK_RATIO), 0.0, 1.0))
    return tau, float(np.clip(peak, 0.0, 1.0)) * significance


def _gcc_phat_peak(sig_a: np.ndarray, sig_b: np.ndarray, sample_rate: float, max_tau_s: float,
                   interp: int) -> tuple[float, float, float]:
    """(tau_s, normalised peak height, peak / no-signal correlation RMS)."""
    a = np.asarray(sig_a, dtype=float).ravel()
    b = np.asarray(sig_b, dtype=float).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0, 0.0, 0.0
    # Hann taper: frame edges of a tonal source otherwise bias the sub-sample lag
    a = (a - a.mean()) * np.hanning(a.size)
    b = (b - b.mean()) * np.hanning(b.size)
    if np.sqrt(np.mean(a * a)) < SILENCE_RMS or np.sqrt(np.mean(b * b)) < SILENCE_RMS:
        return 0.0, 0.0, 0.0
    interp = max(1, int(interp))
    n = a.size + b.size
    cross = np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n))
    mag = np.abs(cross)
    denom = mag + PHAT_EPSILON + PHAT_FLOOR_FRACTION * float(mag.mean())
    weight = mag / denom
    cross = cross / denom
    # normalise so a perfectly coherent arrival peaks at 1.0 (irfft halves the one-sided sum)
    norm = 2.0 * float(weight.sum()) / (n * interp)
    if norm <= 0.0:
        return 0.0, 0.0, 0.0
    cc = np.fft.irfft(cross, n * interp) / norm
    rate = sample_rate * interp
    max_shift = min(int(math.ceil(max_tau_s * rate)), (n * interp) // 2 - 1)
    # lags -max_shift..+max_shift, lag 0 at index max_shift
    window = np.concatenate((cc[-max_shift:], cc[:max_shift + 1])) if max_shift > 0 else cc[:1].copy()
    k = int(np.argmax(window))
    peak = float(window[k])
    shift = float(k)
    if 0 < k < window.size - 1:
        y0, y1, y2 = window[k - 1], window[k], window[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom < 0.0:
            shift += 0.5 * (y0 - y2) / denom
    tau = (shift - max_shift) / rate
    tau = float(np.clip(tau, -max_tau_s, max_tau_s))
    # Peak height against the rest of the correlation: with no common signal every lag is a sum of
    # random-phase bins with RMS sqrt(sum w^2 / 2) / sum w (the taper correlates neighbouring bins,
    # so the observed null peak sits higher; NOISE_PEAK_RATIO is calibrated on that).
    noise_rms = math.sqrt(float(np.sum(weight ** 2)) / 2.0) / float(weight.sum())
    return tau, peak, peak / noise_rms


def _rank(m: np.ndarray) -> int:
    if m.size == 0:
        return 0
    s = np.linalg.svd(m, compute_uv=False)
    if s[0] <= 0.0:
        return 0
    return int(np.sum(s / s[0] > RANK_TOLERANCE))


def _weighted_lstsq(a: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, float]:
    sw = np.sqrt(w)
    x, *_ = np.linalg.lstsq(a * sw[:, None], y * sw, rcond=None)
    resid = a @ x - y
    rms = float(np.sqrt(np.sum(w * resid ** 2) / np.sum(w)))
    return x, rms


def doa(channels: np.ndarray, mic_positions_m: np.ndarray, sample_rate: float,
        interp: int = DEFAULT_INTERP, speed_m_s: float = SPEED_OF_SOUND_M_S) -> DoaResult | None:
    """Direction of arrival from M synchronised channels (M, N) and mic positions (M, 3) in the head frame.

    Returns None for silence, no clear correlation peak, or pairs that disagree."""
    x = np.atleast_2d(np.asarray(channels, dtype=float))
    pos = np.asarray(mic_positions_m, dtype=float).reshape(-1, 3)
    if x.shape[0] != pos.shape[0]:
        raise ValueError(f"{x.shape[0]} channels but {pos.shape[0]} mic positions")
    if x.shape[0] < 2:
        raise ValueError("direction of arrival needs at least two microphones")
    live = [i for i in range(x.shape[0]) if float(np.std(x[i])) >= SILENCE_RMS]
    baselines, taus, confs, pairs = [], [], [], []
    for i, j in combinations(live, 2):
        d = pos[j] - pos[i]
        spacing = float(np.linalg.norm(d))
        if spacing <= COINCIDENT_MIC_M:
            continue
        tau, conf = gcc_phat(x[i], x[j], sample_rate, max_tau_s(spacing, speed_m_s), interp)
        if conf < MIN_CONFIDENCE:
            continue
        baselines.append(d); taus.append(tau); confs.append(conf); pairs.append((i, j))
    if not pairs:
        return None
    d_mat = np.array(baselines)
    rank = _rank(d_mat)
    # Fewer resolved axes than the live mics offer means the pairs disagree about whether a sound is
    # there at all (weak signal). Answering from the survivors would add a geometry assumption the
    # array does not need, so abstain. A silent (dead) mic is excluded above and degrades honestly.
    if rank < _rank(np.array([pos[j] - pos[i] for i, j in combinations(live, 2)])):
        return None
    w = np.array(confs)
    # far field: tau_ij = -(p_j - p_i) . u / c, u the unit vector towards the source
    y = -speed_m_s * np.array(taus)
    sample_m = speed_m_s / sample_rate          # residuals are compared in samples
    el_deg: float | None = None
    if rank >= 3:
        u, rms = _weighted_lstsq(d_mat, y, w)
        norm = float(np.linalg.norm(u))
        if abs(norm - 1.0) > UNIT_NORM_TOLERANCE:
            return None                             # time differences do not fit one plane wave
        u = u / norm
        az = normalize_az(math.degrees(math.atan2(u[1], u[0])))
        el_deg = math.degrees(math.atan2(u[2], math.hypot(u[0], u[1])))
        candidates, ambiguous = [az], False
    elif rank == 2:
        # coplanar: the data fix the in-plane part p of u; u = p +- s n is a mirror pair about the plane.
        # A horizontal plane gives both mirrors the same azimuth (elevation sign unknown -> None).
        p, rms = _weighted_lstsq(d_mat, y, w)       # minimum-norm solution lies in the baseline plane
        p_norm = float(np.linalg.norm(p))
        if p_norm > 1.0:
            p = p / p_norm
        normal = np.linalg.svd(d_mat)[2][2]
        s = math.sqrt(max(0.0, 1.0 - float(p @ p)))
        mirrors = [normalize_az(math.degrees(math.atan2(v[1], v[0]))) for v in (p + s * normal, p - s * normal)]
        if abs(normalize_az(mirrors[0] - mirrors[1])) <= PLANE_MIRROR_TOL_DEG:
            az = normalize_az(mirrors[0] + normalize_az(mirrors[1] - mirrors[0]) / 2.0)
            candidates, ambiguous = [az], False
        else:
            front, back = sorted(mirrors, key=abs)
            az, candidates, ambiguous = front, [front, back], True
    else:
        # collinear (one pair): only the angle from the pair axis; mirror image about that axis
        axis = d_mat[int(np.argmax(np.linalg.norm(d_mat, axis=1)))]
        axis = axis / np.linalg.norm(axis)
        lengths = d_mat @ axis                      # signed baseline lengths along the axis
        proj, rms = _weighted_lstsq(lengths[:, None], y, w)
        horiz = float(math.hypot(axis[0], axis[1]))
        if horiz <= VERTICAL_AXIS_TOLERANCE:
            return None                             # vertical pair: no azimuth information
        cos_theta = float(np.clip(proj[0] / horiz, -1.0, 1.0))
        theta = math.degrees(math.acos(cos_theta))
        axis_az = math.degrees(math.atan2(axis[1], axis[0]))
        c1, c2 = normalize_az(axis_az + theta), normalize_az(axis_az - theta)
        front, back = (c1, c2) if abs(c1) <= abs(c2) else (c2, c1)
        az, candidates, ambiguous = front, [front, back], True
    resid_samples = rms / sample_m
    consistency = max(0.0, 1.0 - resid_samples / RESIDUAL_LIMIT_SAMPLES)
    confidence = float(np.mean(w)) * consistency
    if confidence < MIN_CONFIDENCE:
        return None
    return DoaResult(az_deg=az, el_deg=el_deg, ambiguous=ambiguous, candidates=candidates,
                     confidence=confidence, pairs_used=tuple(pairs))


def _word_holds(word: str, az: float, el: float | None, margin: float) -> bool:
    """Is (az, el) inside the sector of `word`, widened by `margin` degrees?"""
    e = 0.0 if el is None else el
    if word == "above":
        return e > VERTICAL_DEG - margin
    if word == "below":
        return e < -VERTICAL_DEG + margin
    if abs(e) > VERTICAL_DEG + margin:
        return False
    if word == "ahead":
        return abs(az) <= AHEAD_MAX_DEG + margin
    if word == "at left":
        return AHEAD_MAX_DEG - margin < az <= SIDE_MAX_DEG + margin
    if word == "at right":
        return -SIDE_MAX_DEG - margin <= az < -AHEAD_MAX_DEG + margin
    if word == "behind":
        return abs(az) > SIDE_MAX_DEG - margin
    raise ValueError(f"unknown direction word {word!r}")


def direction_of_bearing(az_deg: float, el_deg: float | None = None, previous: str | None = None) -> str:
    """Shared direction word for a bearing (ADR-0060). Thin helper for callers without the spatial
    package, whose tracker is the main one. With `previous`, that word is kept until the bearing is
    HYSTERESIS_DEG past its boundary."""
    az = normalize_az(az_deg)
    if previous is not None and _word_holds(previous, az, el_deg, HYSTERESIS_DEG):
        return previous
    for word in ("above", "below", "ahead", "at left", "at right", "behind"):
        if _word_holds(word, az, el_deg, 0.0):
            return word
    raise AssertionError("direction sectors cover the sphere")  # unreachable: sectors are exhaustive
