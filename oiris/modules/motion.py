"""Motion gate input: still, walking, fast or unknown, from a head-worn accelerometer (docs/ARCHITECTURE.md,
"Outputs and motion"). The output arbiter (oiris/modules/output.py) uses it to keep pop-ups and speech from
distracting a wearer on the move (Module Map 6.1 glasses constraints: "no pop-ups during movement").

- Input: 3-axis accelerometer samples in m/s^2 with timestamps in seconds (the sensor's own clock), from any
  sensor adapter with the "imu" capability (oiris/adapters/imu_sim.py today, the glasses IMU later).
- Still versus walking: the standard deviation of the acceleration magnitude over WINDOW_S. Gravity is a
  constant offset of the magnitude, so how the glasses are worn does not matter.
- Fast: a sustained very high deviation (running), or an optional speed input (GPS, later) at or above
  FAST_ENTER_SPEED_MPS.
- Hysteresis: each boundary has an enter and a lower exit threshold, and a new state must hold for
  HOLD_UP_S (towards more motion) or HOLD_DOWN_S (towards less) of sensor time before it is reported, so it
  does not flicker at a stop or between strides.
- Unknown: no IMU, or no sample for STALE_S. The arbiter treats unknown as still.
All thresholds are PROVISIONAL: set from typical head-worn walking traces, to be measured on the glasses."""
from __future__ import annotations
import collections, math, threading, time
from typing import Any, Callable, Iterable, Sequence

STILL, WALKING, FAST, UNKNOWN = "still", "walking", "fast", "unknown"
STATES = (STILL, WALKING, FAST, UNKNOWN)
MOVING = frozenset({WALKING, FAST})
IMU_CAP = "imu"                  # adapter capability whose samples feed the estimator
_RANK = {STILL: 0, WALKING: 1, FAST: 2}

WINDOW_S = 2.0                   # magnitude deviation is measured over this much sensor time (about 4 steps)
MIN_WINDOW_S = 1.0               # no judgement until the window covers at least this much
WALK_ENTER_STD = 0.6             # m/s^2: a deviation at or above this starts walking...
WALK_EXIT_STD = 0.35             # ...and walking ends only below this
FAST_ENTER_STD = 4.0             # m/s^2: running-level deviation, sustained for HOLD_UP_S
FAST_EXIT_STD = 3.0
FAST_ENTER_SPEED_MPS = 4.0       # about 14 km/h: faster than a run, e.g. a bike or a car
FAST_EXIT_SPEED_MPS = 2.5
SPEED_STALE_S = 5.0              # a speed reading older than this (estimator clock) is ignored
HOLD_UP_S = 1.0                  # a more active state must hold this long before it is reported
HOLD_DOWN_S = 3.0                # a calmer state must hold this long (stops at a crossing do not flicker)
STALE_S = 3.0                    # no accelerometer sample for this long (estimator clock): unknown
MAX_WINDOW_SAMPLES = 2000        # bounded memory whatever the sample rate


def magnitude(ax: float, ay: float, az: float) -> float:
    return math.sqrt(ax * ax + ay * ay + az * az)


def _std(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / n)


class MotionEstimator:
    """Thread-safe: the adapter thread adds samples, the arbiter and snapshot read state()."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self._lock = threading.Lock()
        self._win: collections.deque = collections.deque(maxlen=MAX_WINDOW_SAMPLES)   # (t_s, magnitude)
        self._state = UNKNOWN
        self._candidate: str | None = None
        self._candidate_since = 0.0
        self._last_seen: float | None = None   # estimator clock of the last add()
        self._speed: tuple[float, float] | None = None   # (m/s, estimator clock)
        self.std = 0.0
        self.samples = 0

    # -- input ----------------------------------------------------------------------------------------
    def add(self, times_s: Iterable[float], accel: Iterable[Sequence[float]], speed_mps: float | None = None) -> str:
        """Add accelerometer samples (seconds, [ax, ay, az] in m/s^2). Bad values are skipped. Returns the state."""
        now = self.clock()
        with self._lock:
            if speed_mps is not None:
                self._set_speed(speed_mps, now)
            for t, a in zip(times_s, accel):
                try:
                    t, (ax, ay, az) = float(t), a
                    m = magnitude(float(ax), float(ay), float(az))
                except (TypeError, ValueError):  # python-brain: allow SILENT_EXCEPTION - a malformed sample is skipped, never fatal
                    continue
                if not (math.isfinite(t) and math.isfinite(m)):
                    continue
                if self._win and t <= self._win[-1][0]:
                    self._win.clear()            # the sensor clock went back (adapter restart): start again
                    self._candidate = None
                self._win.append((t, m))
                self.samples += 1
                self._last_seen = now
                self._evaluate(t)
            return self._current(now)

    def add_sample(self, data: Any) -> str:
        """An "imu" adapter sample's data: {"t_s": [...], "accel": [[ax, ay, az], ...], "speed_mps": x | None}."""
        if not isinstance(data, dict):
            return self.state()
        return self.add(data.get("t_s") or [], data.get("accel") or [], data.get("speed_mps"))

    def set_speed(self, speed_mps: float | None) -> None:
        """Optional speed input (GPS, later). None forgets it."""
        with self._lock:
            if speed_mps is None:
                self._speed = None
            else:
                self._set_speed(speed_mps, self.clock())

    def _set_speed(self, v: Any, now: float) -> None:
        try:
            v = float(v)
        except (TypeError, ValueError):  # python-brain: allow SILENT_EXCEPTION - a malformed speed is ignored
            return
        if math.isfinite(v) and v >= 0:
            self._speed = (v, now)

    # -- classification -------------------------------------------------------------------------------
    def _speed_now(self, now: float) -> float | None:
        if self._speed is None or now - self._speed[1] > SPEED_STALE_S:
            return None
        return self._speed[0]

    def _raw(self, std: float, speed: float | None) -> str:
        cur = self._state
        if speed is not None and speed >= (FAST_EXIT_SPEED_MPS if cur == FAST else FAST_ENTER_SPEED_MPS):
            return FAST
        if std >= (FAST_EXIT_STD if cur == FAST else FAST_ENTER_STD):
            return FAST
        if std >= (WALK_EXIT_STD if cur in MOVING else WALK_ENTER_STD):
            return WALKING
        return STILL

    def _evaluate(self, t: float) -> None:
        while self._win and t - self._win[0][0] > WINDOW_S:
            self._win.popleft()
        if not self._win or t - self._win[0][0] < MIN_WINDOW_S:
            return
        self.std = _std([m for _, m in self._win])
        raw = self._raw(self.std, self._speed_now(self.clock()))
        if self._state == UNKNOWN:              # first judgement: no previous state to hold against
            self._state, self._candidate = raw, None
            return
        if raw == self._state:
            self._candidate = None
            return
        if raw != self._candidate:
            self._candidate, self._candidate_since = raw, t
        hold = HOLD_UP_S if _RANK[raw] > _RANK[self._state] else HOLD_DOWN_S
        if t - self._candidate_since >= hold:
            self._state, self._candidate = raw, None

    def _current(self, now: float) -> str:
        if self._last_seen is None or now - self._last_seen > STALE_S:
            return UNKNOWN
        return self._state

    def state(self) -> str:
        with self._lock:
            return self._current(self.clock())

    def status(self) -> dict:
        with self._lock:
            now = self.clock()
            return {"state": self._current(now), "std": round(self.std, 3), "samples": self.samples,
                    "speed_mps": self._speed_now(now)}


def effective(state: str) -> str:
    """Unknown (no IMU) counts as still."""
    return STILL if state == UNKNOWN else state


def is_moving(state: str) -> bool:
    return state in MOVING


__all__ = ["MotionEstimator", "STILL", "WALKING", "FAST", "UNKNOWN", "STATES", "MOVING", "IMU_CAP", "effective",
           "is_moving", "magnitude"]
