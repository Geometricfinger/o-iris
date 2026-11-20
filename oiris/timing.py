"""Per-stage timing counters (latency budgets come from measurements; spec R-17/E-20, docs/LIVE.md "Timing").

A StageTimer keeps the last WINDOW durations of each named stage in a bounded ring buffer and summarises them
as count, p50, p95 and max in milliseconds. Each owner (Engine, VoiceListener, OutputArbiter) gets its own
timer passed in; there is no module-level timer, so tests always start from an empty one.

Numbers only: stage names are fixed strings from the code (STAGE_RE, at most MAX_STAGES), and the values are
durations. Never a name, audio, identifier or free text."""
from __future__ import annotations
import math, re, threading, time
from collections import deque
from contextlib import contextmanager
from typing import Callable, Iterator

WINDOW = 512                  # durations kept per stage (ring buffer)
MAX_STAGES = 32               # distinct stage names per timer; more are ignored, so keys cannot grow without bound
MS_DECIMALS = 3               # summary values in ms, rounded to microseconds
MS_PER_S = 1000.0
P50, P95 = 50.0, 95.0
PERCENT = 100.0
STAGE_RE = re.compile(r"[a-z][a-z0-9_]{0,39}")   # code-chosen names only (e.g. "speech_wait")


def percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile of an ascending list (p in (0, 100]); 0.0 for an empty list."""
    if not sorted_values:
        return 0.0
    k = max(1, math.ceil(p / PERCENT * len(sorted_values)))
    return sorted_values[min(k, len(sorted_values)) - 1]


class StageTimer:
    def __init__(self, window: int = WINDOW, clock: Callable[[], float] = time.perf_counter):
        """window: durations kept per stage. clock: monotonic seconds (tests inject a fake)."""
        self.window = max(1, int(window))
        self.clock = clock
        self._lock = threading.Lock()
        self._stages: dict[str, deque] = {}
        self._counts: dict[str, int] = {}     # every record since start, not just the kept window

    def now(self) -> float:
        return self.clock()

    def record(self, stage: str, seconds: float) -> None:
        """Add one duration. Unknown-shaped names, a full stage table, and negative or non-finite durations are
        ignored: timing never raises into, or changes, the code it measures."""
        if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
            return
        with self._lock:
            ring = self._stages.get(stage)
            if ring is None:
                if len(self._stages) >= MAX_STAGES or not isinstance(stage, str) or not STAGE_RE.fullmatch(stage):
                    return
                ring = self._stages[stage] = deque(maxlen=self.window)
                self._counts[stage] = 0
            ring.append(float(seconds))
            self._counts[stage] += 1

    def since(self, stage: str, start: float) -> None:
        """Record now() - start (start taken with now())."""
        self.record(stage, self.clock() - start)

    @contextmanager
    def time(self, stage: str) -> Iterator[None]:
        """Time a block; recorded even when the block raises (the exception passes through unchanged)."""
        t0 = self.clock()
        try:
            yield
        finally:
            self.record(stage, self.clock() - t0)

    def summary(self) -> dict[str, dict[str, float | int]]:
        """{stage: {"count", "p50_ms", "p95_ms", "max_ms"}}. count is every record since start; the percentiles
        and max cover the last `window` of them."""
        with self._lock:
            snap = {k: (self._counts[k], list(v)) for k, v in self._stages.items()}
        out: dict[str, dict[str, float | int]] = {}
        for stage, (count, vals) in sorted(snap.items()):
            vals.sort()
            out[stage] = {"count": count,
                          "p50_ms": round(percentile(vals, P50) * MS_PER_S, MS_DECIMALS),
                          "p95_ms": round(percentile(vals, P95) * MS_PER_S, MS_DECIMALS),
                          "max_ms": round((vals[-1] if vals else 0.0) * MS_PER_S, MS_DECIMALS)}
        return out

    def reset(self) -> None:
        with self._lock:
            self._stages.clear()
            self._counts.clear()


__all__ = ["StageTimer", "percentile", "WINDOW", "MAX_STAGES", "MS_DECIMALS"]
