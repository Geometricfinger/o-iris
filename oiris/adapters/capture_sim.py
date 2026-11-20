"""Deterministic simulated capture ports (ADR-0063), for tests and for running the interlock without hardware.

SimIndicator is a recording light with injectable faults: stuck off, slow to confirm, failing after a time.
SimMuteSwitch is a mute switch on a scripted timeline. Both read time only from the injected clock."""
from __future__ import annotations
import time
from typing import Callable, Sequence


class SimIndicator:
    """set(on) / read_back(). Healthy by default: confirmed lit as soon as it is turned on.
    confirm_delay_s: reads False for this long after each turn-on (slow to confirm).
    stuck_off: always reads False (a dead LED or a broken driver).
    fail_after_s: from this long after the first turn-on, reads None (the light can no longer be confirmed)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, confirm_delay_s: float = 0.0, stuck_off: bool = False,
                 fail_after_s: float | None = None):
        self.clock, self.confirm_delay_s, self.stuck_off, self.fail_after_s = clock, confirm_delay_s, stuck_off, fail_after_s
        self.on = False
        self._on_at: float | None = None
        self._first_on: float | None = None
        self.sets: list[bool] = []               # every set() call, in order

    def set(self, on: bool) -> None:
        on = bool(on)
        self.sets.append(on)
        if on and not self.on:
            self._on_at = self.clock()
            if self._first_on is None:
                self._first_on = self._on_at
        self.on = on

    def read_back(self) -> bool | None:
        now = self.clock()
        if self.fail_after_s is not None and self._first_on is not None and now - self._first_on >= self.fail_after_s:
            return None
        if self.stuck_off or not self.on:
            return False
        return self._on_at is not None and now - self._on_at >= self.confirm_delay_s


class SimMuteSwitch:
    """muted() follows a scripted timeline [(t, state), ...] on the injected clock: the state of the last entry with
    t <= now, else `initial`. A state is True (muted), False (open) or None (unknown)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, timeline: Sequence[tuple[float, bool | None]] = (),
                 initial: bool | None = False):
        self.clock, self.initial = clock, initial
        self.timeline = sorted(((float(t), s) for t, s in timeline), key=lambda e: e[0])

    def muted(self) -> bool | None:
        now, state = self.clock(), self.initial
        for t, s in self.timeline:
            if t > now:
                break
            state = s
        return state


__all__ = ["SimIndicator", "SimMuteSwitch"]
