"""Sound direction of arrival (oiris.features.doa) on synthetic signals only: broadband noise bursts and
harmonic tones, delayed per mic by an exact fractional delay (frequency domain), plus sensor noise at
several SNRs and a weak echo from another direction. No recordings, no microphone."""
import math

import numpy as np
import pytest

from oiris.features import doa as D

FS = 48000                      # Hz
FRAME = 4096                    # samples analysed (85 ms)
GUARD = 512                     # samples cropped from each end so circular delays look linear
TEMPLE_SPACING_M = 0.14         # smart-glasses mics, one on each temple
SQUARE_SIDE_M = 0.2             # helmet-like square in the horizontal plane
TETRA_EDGE_M = 0.2              # tetrahedral array edge
HARMONIC_F0_HZ = 180.0
HARMONIC_COUNT = 22             # harmonics up to ~4 kHz, amplitude 1/k
ECHO_GAIN = 0.3                 # weak echo
ECHO_DELAY_S = 0.004
ECHO_AZ_OFFSET_DEG = 120.0      # the echo arrives from a different direction
PAIR_TOL_DEG = 2.0              # broadband, single pair
PAIR_HARMONIC_TOL_DEG = 5.0     # harmonic tone, single pair, away from endfire
# At endfire (+-90) the angle is acos(lag / max lag): 0.2 samples of lag error is ~10 deg. Measured on
# harmonic tones at 20 dB with the echo, 40 seeds: p95 12.7, max 14.1 deg.
PAIR_ENDFIRE_TOL_DEG = 15.0
ENDFIRE_DEG = 90.0
ARRAY_TOL_DEG = 5.0             # square / tetrahedron at >= 20 dB SNR
CANDIDATE_TOL_DEG = 0.5         # the mirror candidate is exactly 180 - az
NOISE_TRIALS = 40

PAIR_AZ = [0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 150.0, -150.0]
FULL_AZ = [0.0, 45.0, 90.0, 135.0, 180.0, -150.0, -120.0, -90.0, -30.0]
TETRA_DIRS = [(az, el) for az in (0.0, 100.0, -140.0, 180.0) for el in (-40.0, 0.0, 30.0, 60.0)]
CONDITIONS = [("noise", 30.0), ("noise", 20.0), ("noise", 10.0), ("noise", 0.0), ("harmonic", 30.0), ("harmonic", 20.0)]


def _pair() -> np.ndarray:
    h = TEMPLE_SPACING_M / 2
    return np.array([[0.0, h, 0.0], [0.0, -h, 0.0]])          # left temple, right temple


def _square() -> np.ndarray:
    h = SQUARE_SIDE_M / 2
    return np.array([[h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0], [-h, h, 0.0]])


def _tetra() -> np.ndarray:
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    return v * TETRA_EDGE_M / (2 * math.sqrt(2))               # vertex-to-vertex distance = edge


def _unit(az: float, el: float = 0.0) -> np.ndarray:
    a, e = math.radians(az), math.radians(el)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def _source(kind: str, rng: np.random.Generator) -> np.ndarray:
    n = FRAME + 2 * GUARD
    if kind == "noise":
        burst = rng.standard_normal(n)
        return burst * np.hanning(n) ** 0.25                    # tapered noise burst
    t = np.arange(n) / FS
    return sum(np.sin(2 * np.pi * HARMONIC_F0_HZ * k * t + rng.uniform(0, 2 * np.pi)) / k for k in range(1, HARMONIC_COUNT + 1))


def _render(src: np.ndarray, pos: np.ndarray, u: np.ndarray, extra_delay_s: float = 0.0) -> np.ndarray:
    """Plane wave from direction u: mic p hears the source at t = -(p . u) / c, as an exact fractional delay."""
    spec = np.fft.rfft(src)
    f = np.fft.rfftfreq(src.size, 1 / FS)
    out = [np.fft.irfft(spec * np.exp(-2j * np.pi * f * (extra_delay_s - (p @ u) / D.SPEED_OF_SOUND_M_S)), src.size) for p in pos]
    return np.array(out)[:, GUARD:-GUARD]


def _scene(kind: str, pos: np.ndarray, az: float, el: float, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    src = _source(kind, rng)
    x = _render(src, pos, _unit(az, el))
    x = x + ECHO_GAIN * _render(src, pos, _unit(az + ECHO_AZ_OFFSET_DEG), ECHO_DELAY_S)
    power = float(np.mean(x ** 2))
    return x + rng.standard_normal(x.shape) * math.sqrt(power / 10 ** (snr_db / 10))


def _az_err(a: float, b: float) -> float:
    return abs(D.normalize_az(a - b))


# --- gcc_phat ---------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["noise", "harmonic"])
@pytest.mark.parametrize("delay_samples", [-7.3, 0.0, 2.5, 11.71])
def test_gcc_phat_recovers_fractional_delay_and_sign(kind, delay_samples):
    rng = np.random.default_rng(7)
    src = _source(kind, rng)
    spec, f = np.fft.rfft(src), np.fft.rfftfreq(src.size, 1 / FS)
    b = np.fft.irfft(spec * np.exp(-2j * np.pi * f * delay_samples / FS), src.size)
    a, b = src[GUARD:-GUARD], b[GUARD:-GUARD]
    tau, conf = D.gcc_phat(a, b, FS, max_tau_s=20 / FS)
    assert tau * FS == pytest.approx(delay_samples, abs=0.05)   # positive: b hears it after a
    assert conf > D.MIN_CONFIDENCE


@pytest.mark.parametrize("spacing_m", [0.05, TEMPLE_SPACING_M, SQUARE_SIDE_M])
@pytest.mark.parametrize("delay_samples", [-60.0, 35.5, 200.0])
def test_lag_search_never_exceeds_physical_limit(spacing_m, delay_samples):
    """A delay larger than the mic spacing allows (e.g. a cable/clock fault) is never reported as one."""
    rng = np.random.default_rng(3)
    src = rng.standard_normal(FRAME)
    b = np.roll(src, int(delay_samples)) if delay_samples == int(delay_samples) else np.fft.irfft(
        np.fft.rfft(src) * np.exp(-2j * np.pi * np.fft.rfftfreq(FRAME) * delay_samples), FRAME)
    limit = D.max_tau_s(spacing_m)
    assert abs(delay_samples) / FS > limit
    for interp in (1, D.DEFAULT_INTERP):
        tau, _ = D.gcc_phat(src, b, FS, limit, interp)
        assert abs(tau) <= limit


def test_silence_returns_none():
    pos = _square()
    assert D.doa(np.zeros((4, FRAME)), pos, FS) is None
    dither = np.random.default_rng(0).standard_normal((4, FRAME)) * D.SILENCE_RMS / 10
    assert D.doa(dither, pos, FS) is None
    assert D.gcc_phat(np.zeros(FRAME), np.zeros(FRAME), FS, 1e-3) == (0.0, 0.0)


@pytest.mark.parametrize("geometry", ["pair", "square", "tetra"])
def test_uncorrelated_noise_returns_none(geometry):
    pos = {"pair": _pair(), "square": _square(), "tetra": _tetra()}[geometry]
    rng = np.random.default_rng(11)
    for _ in range(NOISE_TRIALS):
        assert D.doa(rng.standard_normal((len(pos), FRAME)), pos, FS) is None


# --- two mics (temples) -----------------------------------------------------------------------

@pytest.mark.parametrize("kind,snr_db", CONDITIONS)
@pytest.mark.parametrize("az", PAIR_AZ)
def test_temple_pair_left_right_correct_front_back_ambiguous(kind, snr_db, az):
    r = D.doa(_scene(kind, _pair(), az, 0.0, snr_db, seed=int(az) + 1000), _pair(), FS)
    assert r is not None
    tol = PAIR_TOL_DEG if kind == "noise" else PAIR_ENDFIRE_TOL_DEG if abs(az) == ENDFIRE_DEG else PAIR_HARMONIC_TOL_DEG
    assert r.ambiguous and len(r.candidates) == 2 and r.el_deg is None
    front, back = r.candidates
    assert r.az_deg == front and abs(front) <= 90.0
    assert _az_err(back, 180.0 - front) < CANDIDATE_TOL_DEG    # mirror about the left/right axis
    assert min(_az_err(c, az) for c in r.candidates) <= tol
    if abs(az) > tol:                                         # left vs right is never confused
        assert math.copysign(1, r.az_deg) == math.copysign(1, az)
    word = D.direction_of_bearing(r.az_deg)
    if 60.0 <= abs(az) <= 120.0:
        assert word == ("at left" if az > 0 else "at right")
    assert r.pairs_used == ((0, 1),) and 0.0 < r.confidence <= 1.0


# --- square of four (helmet) ------------------------------------------------------------------

@pytest.mark.parametrize("kind,snr_db", [c for c in CONDITIONS if c[1] >= 20.0])
@pytest.mark.parametrize("az", FULL_AZ)
def test_square_full_azimuth_including_behind(kind, snr_db, az):
    r = D.doa(_scene(kind, _square(), az, 0.0, snr_db, seed=int(az) + 2000), _square(), FS)
    assert r is not None
    assert not r.ambiguous and r.candidates == [r.az_deg] and r.el_deg is None
    assert _az_err(r.az_deg, az) <= ARRAY_TOL_DEG
    assert len(r.pairs_used) == 6


def test_square_names_behind_and_sides():
    for az, word in [(180.0, "behind"), (-150.0, "behind"), (90.0, "at left"), (-90.0, "at right"), (10.0, "ahead")]:
        r = D.doa(_scene("noise", _square(), az, 0.0, 20.0, seed=5), _square(), FS)
        assert r is not None and D.direction_of_bearing(r.az_deg, r.el_deg) == word


# --- tetrahedron ------------------------------------------------------------------------------

@pytest.mark.parametrize("kind,snr_db", [c for c in CONDITIONS if c[1] >= 20.0])
@pytest.mark.parametrize("az,el", TETRA_DIRS)
def test_tetrahedron_azimuth_and_elevation(kind, snr_db, az, el):
    r = D.doa(_scene(kind, _tetra(), az, el, snr_db, seed=int(az + 7 * el) + 3000), _tetra(), FS)
    assert r is not None and not r.ambiguous and r.el_deg is not None
    assert abs(r.el_deg - el) <= ARRAY_TOL_DEG
    assert _az_err(r.az_deg, az) <= ARRAY_TOL_DEG


def test_tetrahedron_overhead_says_above():
    r = D.doa(_scene("noise", _tetra(), 0.0, 70.0, 20.0, seed=9), _tetra(), FS)
    assert r is not None and D.direction_of_bearing(r.az_deg, r.el_deg) == "above"


# --- geometry and inputs ----------------------------------------------------------------------

def test_vertical_pair_has_no_azimuth():
    pos = np.array([[0.0, 0.0, 0.05], [0.0, 0.0, -0.05]])
    assert D.doa(_scene("noise", pos, 30.0, 20.0, 30.0, seed=1), pos, FS) is None


def test_front_back_pair_mirrors_left_right():
    pos = np.array([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]])   # forward axis pair: left/right is the ambiguity
    r = D.doa(_scene("noise", pos, 40.0, 0.0, 30.0, seed=2), pos, FS)
    assert r is not None and r.ambiguous
    assert sorted(r.candidates) == pytest.approx([-40.0, 40.0], abs=PAIR_TOL_DEG)


@pytest.mark.parametrize("az", [0.0, 120.0, 180.0, -75.0])
def test_horizontal_triangle_full_azimuth(az):
    pos = _square()[:3]                                         # three non-collinear mics in a plane
    r = D.doa(_scene("noise", pos, az, 0.0, 20.0, seed=4), pos, FS)
    assert r is not None and not r.ambiguous and r.el_deg is None
    assert _az_err(r.az_deg, az) <= ARRAY_TOL_DEG


def test_horizontal_square_elevated_source_keeps_azimuth():
    r = D.doa(_scene("noise", _square(), -120.0, 35.0, 20.0, seed=6), _square(), FS)
    assert r is not None and not r.ambiguous and r.el_deg is None
    assert _az_err(r.az_deg, -120.0) <= ARRAY_TOL_DEG


def test_vertical_plane_array_is_front_back_ambiguous():
    """Mics only on the front face of a helmet (y-z plane): a source ahead and its mirror behind fit equally."""
    pos = np.array([[0.1, 0.1, 0.0], [0.1, -0.1, 0.0], [0.1, 0.0, 0.1]])
    r = D.doa(_scene("noise", pos, 30.0, 0.0, 30.0, seed=8), pos, FS)
    assert r is not None and r.ambiguous and r.el_deg is None
    assert sorted(r.candidates) == pytest.approx([30.0, 150.0], abs=ARRAY_TOL_DEG)


def test_dead_mic_in_tetrahedron_degrades_honestly():
    pos = _tetra()
    x = _scene("noise", pos, 100.0, 30.0, 20.0, seed=10)
    x[3] = 0.0                                                  # mic 3 dead: its pairs are dropped
    r = D.doa(x, pos, FS)
    assert r is not None and r.el_deg is None and len(r.pairs_used) == 3
    assert all(3 not in p for p in r.pairs_used)
    assert r.ambiguous and min(_az_err(c, 100.0) for c in r.candidates) <= ARRAY_TOL_DEG


def test_mic_hearing_only_its_own_noise_makes_the_array_abstain():
    """Not silent, so not a dead mic: the surviving pairs would need a planar assumption -> None, never a guess."""
    pos = _tetra()
    x = _scene("noise", pos, 100.0, 30.0, 20.0, seed=12)
    x[3] = np.random.default_rng(13).standard_normal(x.shape[1])
    assert D.doa(x, pos, FS) is None


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        D.doa(np.zeros((3, FRAME)), _square(), FS)
    with pytest.raises(ValueError):
        D.doa(np.zeros((1, FRAME)), np.zeros((1, 3)), FS)


# --- direction words --------------------------------------------------------------------------

@pytest.mark.parametrize("az,el,word", [
    (0.0, None, "ahead"), (45.0, None, "ahead"), (-45.0, None, "ahead"), (45.1, None, "at left"),
    (135.0, None, "at left"), (135.1, None, "behind"), (180.0, None, "behind"), (-135.0, None, "at right"),
    (-135.1, None, "behind"), (-45.1, None, "at right"), (360.0, None, "ahead"), (-270.0, None, "at left"),
    (0.0, 45.0, "ahead"), (0.0, 45.1, "above"), (180.0, -45.1, "below"), (90.0, 80.0, "above"),
])
def test_direction_words(az, el, word):
    assert D.direction_of_bearing(az, el) == word
    assert word in D.DIRECTION_WORDS


def test_direction_hysteresis_keeps_word_until_ten_degrees_past():
    assert D.direction_of_bearing(50.0, previous="ahead") == "ahead"
    assert D.direction_of_bearing(55.0, previous="ahead") == "ahead"
    assert D.direction_of_bearing(55.1, previous="ahead") == "at left"
    assert D.direction_of_bearing(40.0, previous="at left") == "at left"
    assert D.direction_of_bearing(35.0, previous="at left") == "ahead"
    assert D.direction_of_bearing(-170.0, previous="behind") == "behind"
    assert D.direction_of_bearing(126.0, previous="behind") == "behind"
    assert D.direction_of_bearing(125.0, previous="behind") == "at left"
    assert D.direction_of_bearing(0.0, 40.0, previous="above") == "above"
    assert D.direction_of_bearing(0.0, 35.0, previous="above") == "ahead"
    assert D.direction_of_bearing(0.0, 50.0, previous="ahead") == "ahead"
    assert D.direction_of_bearing(0.0, 56.0, previous="ahead") == "above"
    with pytest.raises(ValueError):
        D.direction_of_bearing(0.0, previous="nearby")


def test_normalize_az_range():
    assert D.normalize_az(-180.0) == 180.0 and D.normalize_az(540.0) == 180.0 and D.normalize_az(-190.0) == 170.0
