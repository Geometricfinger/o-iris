"""Motion gate input (oiris/modules/motion.py) and the simulated IMU adapter (oiris/adapters/imu_sim.py).
Synthetic traces only: no real sensor."""
import math, random, time
import xml.etree.ElementTree as ET

import pytest

from oiris.adapters import imu_sim
from oiris.bus import SensorBus
from oiris.modules import motion as M
from oiris.modules.motion import FAST, STILL, UNKNOWN, WALKING, MotionEstimator, effective, is_moving
from oiris.registry import default_registry
from oiris.ui.glasses import GlassesState, moving_from_motion, render, state_from_result

RATE = imu_sim.RATE_HZ


class Clock:
    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def feed(est, clk, accel_at, seconds, t0, speed=None, every=1):
    """Feed `seconds` of samples starting at sensor time t0, `every` samples per add(); the estimator clock
    follows the sensor clock. Returns (end time, list of reported states after each add)."""
    n = int(round(seconds * RATE))
    seen, ts, xs = [], [], []
    for k in range(n):
        t = t0 + k / RATE
        ts.append(t); xs.append(accel_at(t))
        if len(ts) == every or k == n - 1:
            clk.t = 100.0 + t
            seen.append(est.add(ts, xs, speed))
            ts, xs = [], []
    return t0 + n / RATE, seen


def profile(name, seed=0):
    rng = random.Random(seed)
    return lambda t: imu_sim.accel(name, t, rng)


def bounce(amp, hz=imu_sim.STEP_HZ):
    return lambda t: [0.0, 0.0, imu_sim.G + amp * math.sin(2 * math.pi * hz * t)]


def transitions(states):
    return sum(1 for a, b in zip(states, states[1:]) if a != b)


# ---- classification ------------------------------------------------------------------------------------
def test_no_imu_is_unknown_and_counts_as_still():
    est = MotionEstimator(clock=Clock())
    assert est.state() == UNKNOWN and effective(UNKNOWN) == STILL and not is_moving(UNKNOWN)
    assert est.status()["state"] == UNKNOWN and est.status()["samples"] == 0


def test_still_walking_and_driving_traces():
    for name, speed, want in (("still", None, STILL), ("walking", None, WALKING), ("driving", imu_sim.DRIVE_SPEED_MPS, FAST)):
        clk = Clock()
        est = MotionEstimator(clock=clk)
        _, seen = feed(est, clk, profile(name), 6.0, 0.0, speed=speed, every=10)
        assert seen[-1] == want, name
        assert transitions([s for s in seen if s != UNKNOWN]) <= 1, name      # no flicker once judged


def test_driving_without_a_speed_input_reads_as_still():
    """An IMU alone cannot tell a smooth car from sitting: fast needs the speed input (or a running-level trace)."""
    clk = Clock()
    est = MotionEstimator(clock=clk)
    feed(est, clk, profile("driving"), 6.0, 0.0)
    assert est.state() == STILL and est.std < M.WALK_EXIT_STD


def test_sustained_high_variance_is_fast_and_a_burst_is_not():
    clk = Clock()
    est = MotionEstimator(clock=clk)
    t, _ = feed(est, clk, profile("walking"), 4.0, 0.0)
    assert est.state() == WALKING
    t, seen = feed(est, clk, bounce(8.0, hz=2.6), M.HOLD_UP_S * 0.5, t)        # a short jolt
    assert est.state() == WALKING
    t, seen = feed(est, clk, bounce(8.0, hz=2.6), 4.0, t)                       # running
    assert est.state() == FAST and transitions(seen) == 1


def test_hysteresis_between_thresholds_and_hold_times():
    mid = bounce(0.65)            # magnitude deviation about 0.46: between WALK_EXIT_STD and WALK_ENTER_STD
    clk = Clock()
    est = MotionEstimator(clock=clk)
    t, _ = feed(est, clk, profile("still"), 3.0, 0.0)
    t, _ = feed(est, clk, mid, 5.0, t)
    assert M.WALK_EXIT_STD < est.std < M.WALK_ENTER_STD and est.state() == STILL     # not enough to start walking
    t, _ = feed(est, clk, profile("walking"), 4.0, t)
    assert est.state() == WALKING
    t, seen = feed(est, clk, mid, 5.0, t)
    assert est.state() == WALKING and set(seen) == {WALKING}                         # not little enough to stop
    # A short stop (a crossing) shorter than HOLD_DOWN_S keeps walking; a longer one ends it.
    t, seen = feed(est, clk, profile("still"), M.WINDOW_S + M.HOLD_DOWN_S * 0.5, t)
    assert set(seen) == {WALKING}
    t, seen = feed(est, clk, profile("still"), M.HOLD_DOWN_S + 1.0, t)
    assert est.state() == STILL and transitions(seen) == 1


def test_walking_starts_after_hold_up_not_at_the_first_stride():
    clk = Clock()
    est = MotionEstimator(clock=clk)
    t, _ = feed(est, clk, profile("still"), 3.0, 0.0)
    t, seen = feed(est, clk, profile("walking"), 0.5, t)
    assert est.state() == STILL
    t, seen = feed(est, clk, profile("walking"), 3.0, t)
    assert est.state() == WALKING and transitions(seen) == 1


def test_speed_input_with_hysteresis_and_staleness():
    clk = Clock()
    est = MotionEstimator(clock=clk)
    t, _ = feed(est, clk, profile("still"), 3.0, 0.0)
    t, _ = feed(est, clk, profile("still"), 2.0, t, speed=M.FAST_ENTER_SPEED_MPS)
    assert est.state() == FAST
    t, _ = feed(est, clk, profile("still"), 4.0, t, speed=(M.FAST_ENTER_SPEED_MPS + M.FAST_EXIT_SPEED_MPS) / 2)
    assert est.state() == FAST                                                       # between exit and enter: stays
    t, _ = feed(est, clk, profile("still"), 4.0, t, speed=M.FAST_EXIT_SPEED_MPS - 0.5)
    assert est.state() == STILL
    est.set_speed(M.FAST_ENTER_SPEED_MPS + 1)
    clk.t += M.SPEED_STALE_S + 0.1
    assert est.status()["speed_mps"] is None                                         # an old GPS speed is ignored


def test_stale_imu_reads_unknown_and_bad_samples_are_skipped():
    clk = Clock()
    est = MotionEstimator(clock=clk)
    t, _ = feed(est, clk, profile("walking"), 4.0, 0.0)
    assert est.state() == WALKING
    clk.t += M.STALE_S + 0.1
    assert est.state() == UNKNOWN
    n = est.samples
    assert est.add([t, t + 0.02, "x", float("nan")], [[0, 0, "?"], None, [0, 0, 9.8], [0, 0, 9.8]]) == UNKNOWN
    assert est.samples == n                                    # all four skipped: no fresh data, still unknown
    assert est.add_sample("garbage") == UNKNOWN and est.add_sample({"t_s": None}) == UNKNOWN
    assert est.add([t + 0.04], [[0, 0, 9.8]]) == WALKING       # one good sample: fresh again, state kept


def test_sensor_clock_going_back_restarts_the_window():
    clk = Clock()
    est = MotionEstimator(clock=clk)
    feed(est, clk, profile("walking"), 4.0, 50.0)
    assert est.state() == WALKING
    feed(est, clk, profile("still"), 0.5, 0.0)          # adapter restarted: its clock starts again
    assert len(est._win) == 25 and est.state() == WALKING   # too little window to judge yet; state kept


# ---- the simulated IMU adapter ------------------------------------------------------------------------
def test_imu_sim_is_deterministic_and_shaped_like_a_real_imu_sample():
    a, b = imu_sim.make("walking", seed=3, period_s=0), imu_sim.make("walking", seed=3, period_s=0)
    sa, sb = [a.read(1.0) for _ in range(3)], [b.read(1.0) for _ in range(3)]
    assert [s.data for s in sa] == [s.data for s in sb]
    d = sa[1].data
    assert a.id == "imu" and a.caps == {"imu"} and a.bus == "sim" and sa[0].units == "m/s^2"
    assert len(d["t_s"]) == imu_sim.BATCH and d["t_s"][0] == pytest.approx(imu_sim.BATCH / RATE)
    assert all(len(x) == 3 for x in d["accel"]) and d["speed_mps"] is None and d["profile"] == "walking"
    assert imu_sim.make("driving", period_s=0).read(1.0).data["speed_mps"] == imu_sim.DRIVE_SPEED_MPS
    with pytest.raises(ValueError):
        imu_sim.make("flying")
    with pytest.raises(ValueError):
        imu_sim.make("still", rate_hz=0)


def test_imu_sim_through_the_registry_and_the_bus_feeds_the_estimator():
    reg = default_registry({"entry_points": False})
    a = reg.create("sensor", "imu", profile="walking", period_s=0.005)
    bus = SensorBus(queue_size=256)
    bus.add(a)
    bus.start()
    try:
        t0 = time.monotonic()
        while bus.health()["imu"]["samples"] < 40 and time.monotonic() - t0 < 10:
            time.sleep(0.02)
    finally:
        bus.stop()
    samples = bus.drain("imu")
    assert len(samples) >= 40 and bus.health()["imu"]["health"] == "ok"
    est = MotionEstimator()
    for s in samples:
        est.add_sample(s.data)
    assert est.state() == WALKING


# ---- the glasses -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("motion,moving", [(STILL, False), (UNKNOWN, False), (None, False), (WALKING, True), (FAST, True)])
def test_glasses_moving_comes_from_the_motion_state(motion, moving):
    assert moving_from_motion(motion) is moving
    st = state_from_result({"level": "attention", "signatures": ["A1", "A4"], "health": {}}, motion=motion)
    assert st.moving is moving and st.level == "medium"
    svg = render(st)
    ET.fromstring(svg)
    assert ('id="G2"' in svg) is not moving and 'id="G1"' in svg          # the level strip always shows
    assert GlassesState().moving is False and state_from_result({}).moving is False
