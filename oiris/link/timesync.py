"""One clock for several devices: NTP-style offset and drift between the host's monotonic clock and a remote
device's (pod, glasses, phone), measured over the Noise link (pod_service op "time_sync").

Planning code for 360-degree coverage: detections from several devices are "at the same moment" only once their
timestamps (schema Sample.t_mono_ns, per-device monotonic ns) are on one timeline. The host converts each remote
stamp with remote_to_host before merging.

Exchange (integer nanoseconds throughout): the host sends at t0 (host clock); the device receives at t1 and replies
at t2 (device clock); the host receives at t3.
  offset_ns = remote - host = floor(((t1 - t0) + (t2 - t3)) / 2)   (floor division, as Python //)
  delay_ns  = (t3 - t0) - (t2 - t1)                                  (round trip minus the device's turnaround)
The estimate is exact when the two legs take equal time; an asymmetry a = (up leg - down leg) biases it by a / 2,
which is bounded by delay / 2, so the minimum-delay sample is the most trustworthy.

TimeSync keeps the last MAX_SAMPLES accepted exchanges and uses the minimum-delay one (ties: the most recent).
Drift (remote clock rate minus host rate, ppm) is a Theil-Sen fit (median of pairwise slopes of offset against the
host midpoint) over the lowest-delay half, once there are DRIFT_MIN_SAMPLES samples whose host midpoints span
DRIFT_MIN_SPAN_NS. Pure and deterministic; android/core TimeSync.kt is the same estimator, checked against
tests/vectors/time_sync.json (scripts/make_android_vectors.py)."""
from __future__ import annotations
import json, math
from collections import deque
from dataclasses import dataclass

MAX_SAMPLES = 64                       # exchanges kept (the window)
MAX_DELAY_NS = 250_000_000             # round trip above this is refused: a BLE connection event is 7.5-100 ms
STALE_NS = 30_000_000_000              # quality "stale" when no accepted exchange is this recent (host clock)
DRIFT_MIN_SAMPLES = 8                  # accepted exchanges before a drift is fitted
DRIFT_MIN_SPAN_NS = 10_000_000_000     # host-midpoint span of the fitted (lowest-delay) half
PPM = 1_000_000
# Timestamps are monotonic ns since boot, refused outside [0, MAX_TIMESTAMP_NS): 2^61 ns is about 73 years of
# uptime, and with every input below it each sum and difference stays below 2^62, inside a signed 64-bit long.
MAX_TIMESTAMP_NS = 1 << 61
LONG_LIMIT = 1 << 63                   # a reply integer outside a signed 64-bit long is malformed (as on Kotlin)

OK, UNSYNCED, STALE = "ok", "unsynced", "stale"
REJECT_RANGE = "timestamp out of range"
REJECT_ORDER = "device replied before it received"
REJECT_NEGATIVE = "negative delay"
REJECT_SLOW = "delay above maximum"
REJECT_REPLY = "bad time_sync reply"


class TimeSyncError(ValueError):
    pass


def request(t0: int) -> str:
    """The pod op's request body (canonical JSON, as the Kotlin side's TimeSync.request)."""
    return json.dumps({"op": "time_sync", "t0": t0}, sort_keys=True, separators=(",", ":"))


def offset_ns(t0: int, t1: int, t2: int, t3: int) -> int:
    return ((t1 - t0) + (t2 - t3)) // 2


def delay_ns(t0: int, t1: int, t2: int, t3: int) -> int:
    return (t3 - t0) - (t2 - t1)


def reject_reason(t0: int, t1: int, t2: int, t3: int, max_delay_ns: int = MAX_DELAY_NS) -> str | None:
    """Why an exchange is refused, or None when it is usable."""
    if not all(0 <= t < MAX_TIMESTAMP_NS for t in (t0, t1, t2, t3)):
        return REJECT_RANGE
    if t2 < t1:
        return REJECT_ORDER
    d = delay_ns(t0, t1, t2, t3)
    if d < 0:
        return REJECT_NEGATIVE
    if d > max_delay_ns:
        return REJECT_SLOW
    return None


@dataclass(frozen=True)
class Exchange:
    t0: int
    t1: int
    t2: int
    t3: int

    @property
    def offset_ns(self) -> int:
        return offset_ns(self.t0, self.t1, self.t2, self.t3)

    @property
    def delay_ns(self) -> int:
        return delay_ns(self.t0, self.t1, self.t2, self.t3)

    @property
    def host_mid_ns(self) -> int:
        """Host time the offset refers to: the midpoint of send and receive (floor)."""
        return self.t0 + (self.t3 - self.t0) // 2


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


class TimeSync:
    def __init__(self, max_samples: int = MAX_SAMPLES, max_delay_ns: int = MAX_DELAY_NS, stale_ns: int = STALE_NS):
        if max_samples < 1:
            raise ValueError("max_samples must be at least 1")
        self.max_delay_ns, self.stale_ns = max_delay_ns, stale_ns
        self.samples: deque[Exchange] = deque(maxlen=max_samples)
        self.last_t3: int | None = None

    def add(self, t0: int, t1: int, t2: int, t3: int) -> str | None:
        """Record one exchange. Returns None when accepted, else the refusal reason (the sample is dropped)."""
        why = reject_reason(t0, t1, t2, t3, self.max_delay_ns)
        if why is None:
            self.samples.append(Exchange(t0, t1, t2, t3))
            self.last_t3 = t3 if self.last_t3 is None else max(self.last_t3, t3)
        return why

    def add_reply(self, reply: str | bytes, t0_sent: int, t3: int) -> str | None:
        """Record the pod's reply to request(t0_sent), received at t3: None when accepted, else why not (REJECT_REPLY
        for a refusal, a malformed reply or one that echoes another request's t0)."""
        try:
            r = json.loads(reply)
        except ValueError:
            return REJECT_REPLY
        if not isinstance(r, dict) or r.get("ok") is not True or r.get("t0") != t0_sent \
                or not all(type(r.get(k)) is int and -LONG_LIMIT <= r[k] < LONG_LIMIT for k in ("t0", "t1", "t2")):
            return REJECT_REPLY
        return self.add(t0_sent, r["t1"], r["t2"], t3)

    def best(self) -> Exchange | None:
        """Minimum delay; ties go to the most recent (larger t3)."""
        return min(self.samples, key=lambda e: (e.delay_ns, -e.t3), default=None)

    def drift(self) -> float | None:
        """Remote rate minus host rate as a slope (ns per ns), or None while too few samples or too short a span."""
        if len(self.samples) < DRIFT_MIN_SAMPLES:
            return None
        ranked = sorted(self.samples, key=lambda e: (e.delay_ns, -e.t3))
        half = ranked[:(len(ranked) + 1) // 2]
        mids = [e.host_mid_ns for e in half]
        if max(mids) - min(mids) < DRIFT_MIN_SPAN_NS:
            return None
        slopes = [(b.offset_ns - a.offset_ns) / (b.host_mid_ns - a.host_mid_ns)
                  for i, a in enumerate(half) for b in half[i + 1:] if b.host_mid_ns != a.host_mid_ns]
        return _median(slopes) if slopes else None

    def drift_ppm(self) -> float | None:
        d = self.drift()
        return None if d is None else d * PPM

    def offset_at(self, at_host_ns: int) -> int:
        """remote - host at a host time: the best sample's offset, carried along the drift (floor of the ns)."""
        b = self.best()
        if b is None:
            raise TimeSyncError("not synced")
        d = self.drift()
        return b.offset_ns if d is None else b.offset_ns + math.floor(d * (at_host_ns - b.host_mid_ns))

    def remote_to_host(self, t_remote_ns: int, at_host_ns: int) -> int:
        """A remote monotonic stamp on the host timeline. `at_host_ns`: roughly when (host clock) the stamp was
        taken, for the drift term; the reception time is close enough (1 s off at 100 ppm is 100 us)."""
        return t_remote_ns - self.offset_at(at_host_ns)

    def quality(self, now_host_ns: int) -> str:
        if self.last_t3 is None:
            return UNSYNCED
        return STALE if now_host_ns - self.last_t3 > self.stale_ns else OK
