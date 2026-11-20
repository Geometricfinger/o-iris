"""The capture interlock (ADR-0063): the camera and the microphone run only while the rules allow it.

Pure logic over two ports (indicator.IndicatorPort, mute.MuteSense) and an injected clock; no hardware imports.

- acquire(kind), kind "camera" or "mic", returns a Grant, or a Refusal carrying one code: indicator_unconfirmed,
  indicator_failed, muted, mute_unknown, mute_breach or closed. Before the first grant the light is turned on and
  read back until it is confirmed (True) or confirm_timeout_s passes; unconfirmed means refused. The mic is refused
  while muted, and while the mute state is unknown when the mute sense is required. The camera ignores the mute.
- release(kind): when no grant is left, the light stays on for tail_s, then tick(now) turns it off.
- check(now), every step: a light whose read-back turns False or None while any grant is held revokes every grant
  (fail closed, indicator_failed); a mute switch turned on while the mic holds a grant revokes the mic.
- guard_samples(kind, block), every mic block: while muted, a non-zero sample on any channel is a mute_breach (the
  hardware cut failed). The breach revokes the mic, zeroes the block and latches until restart. A block is passed on
  only while the mic holds a grant and the switch is open; otherwise it is zeroed and dropped.
- state() returns codes only: {"indicator", "mute", "camera", "mic", "fault"}.

When the light is not required (the Mac) grants never touch it and it shows "not_required"; a mute sense that is
present still applies. The interlock is thread-safe: the audio callback, the live loop and the dashboard share it."""
from __future__ import annotations
import logging, math, threading, time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .indicator import FAILED, NOT_REQUIRED, OFF, ON, TAIL, IndicatorPort
from .mute import MUTED, NOT_PRESENT, OPEN, UNKNOWN, MuteSense

log = logging.getLogger(__name__)

CAMERA, MIC = "camera", "mic"
KINDS = (CAMERA, MIC)
# Refusal and revocation codes
INDICATOR_UNCONFIRMED, INDICATOR_FAILED = "indicator_unconfirmed", "indicator_failed"
REFUSE_MUTED, MUTE_UNKNOWN, MUTE_BREACH, CLOSED = "muted", "mute_unknown", "mute_breach", "closed"
REFUSAL_CODES = (INDICATOR_UNCONFIRMED, INDICATOR_FAILED, REFUSE_MUTED, MUTE_UNKNOWN, MUTE_BREACH, CLOSED)
# state()["camera"] / state()["mic"]: "active", "off" or REFUSED_PREFIX + code
ACTIVE, KIND_OFF, REFUSED_PREFIX = "active", "off", "refused:"
CONFIRM_POLL_S = 0.01          # read-back poll interval while confirming the light
CONFIRM_TIMEOUT_DEFAULT_S = 0.2
TAIL_DEFAULT_S = 2.0
# A mic block can hold audio from before the switch was sensed: up to one block (0.1 s, voice_link/capture.py BLOCK_S)
# plus the sense's 30 ms debounce. Blocks inside this settle time after the mute is first seen are zeroed and dropped
# but are not a breach; a non-zero sample after it is.
MUTE_SETTLE_S = 0.2


@dataclass(frozen=True)
class Grant:
    kind: str
    at: float

    def __bool__(self) -> bool:
        return True


@dataclass(frozen=True)
class Refusal:
    kind: str
    code: str

    def __bool__(self) -> bool:
        return False


class CaptureInterlock:
    def __init__(self, indicator: IndicatorPort | None = None, mute: MuteSense | None = None,
                 indicator_required: bool = False, mute_required: bool = False,
                 confirm_timeout_s: float = CONFIRM_TIMEOUT_DEFAULT_S, tail_s: float = TAIL_DEFAULT_S,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], Any] = time.sleep):
        """indicator: the recording light (None = none fitted; with indicator_required nothing is ever granted).
        mute: the mute sense (None = not present; with mute_required the mic is refused)."""
        self.indicator, self.mute = indicator, mute
        self.indicator_required, self.mute_required = bool(indicator_required), bool(mute_required)
        self.confirm_timeout_s, self.tail_s = float(confirm_timeout_s), float(tail_s)
        self.clock, self.sleep = clock, sleep
        self._lock = threading.RLock()
        self._held: dict[str, Grant] = {}
        self._refused: dict[str, str] = {}       # kind -> last refusal/revocation code, until the next grant or release
        self._light = False                      # the light was commanded on
        self._failed = False                     # the light failed to confirm, or failed while held
        self._tail_until: float | None = None
        self._muted_since: float | None = None
        self._fault: str | None = None           # MUTE_BREACH, latched until restart
        self._closed = False

    # ---- ports, each failure read as "unknown" ---------------------------------------------
    def _read_back(self) -> bool | None:
        if self.indicator is None:
            return None
        try:
            rb = self.indicator.read_back()
        except Exception as e:  # noqa: BLE001 - a broken read-back counts as unconfirmed (fail closed)
            log.warning("recording light read-back failed: %s", e)
            return None
        return rb if isinstance(rb, bool) else None

    def _set_light(self, on: bool) -> None:
        self._light = on
        if self.indicator is None:
            return
        try:
            self.indicator.set(on)
        except Exception as e:  # noqa: BLE001 - the read-back decides; a failed set reads back unconfirmed
            log.warning("recording light set(%s) failed: %s", on, e)

    def _mute_state(self, now: float) -> str:
        if self.mute is None:
            state = NOT_PRESENT
        else:
            try:
                m = self.mute.muted()
            except Exception as e:  # noqa: BLE001 - a broken sense reads as unknown
                log.warning("mute sense read failed: %s", e)
                m = None
            state = UNKNOWN if not isinstance(m, bool) else (MUTED if m else OPEN)
        if state == MUTED:
            if self._muted_since is None:
                self._muted_since = now
        else:
            self._muted_since = None
        return state

    def _mic_block_code(self, mute_state: str) -> str | None:
        """The code that refuses or revokes the mic for this mute state, or None."""
        if self._fault is not None:
            return self._fault
        if mute_state == MUTED:
            return REFUSE_MUTED
        if self.mute_required and mute_state in (UNKNOWN, NOT_PRESENT):
            return MUTE_UNKNOWN
        return None

    # ---- grants -----------------------------------------------------------------------------
    def acquire(self, kind: str) -> Grant | Refusal:
        if kind not in KINDS:
            raise ValueError(f"capture kind must be one of {list(KINDS)}")
        with self._lock:
            now = self.clock()
            if self._closed:
                return self._refuse(kind, CLOSED)
            if kind == MIC:
                code = self._mic_block_code(self._mute_state(now))
                if code is not None:
                    return self._refuse(kind, code)
            if kind in self._held:
                return self._held[kind]
            if self.indicator_required and not self._confirm_light():
                self._failed = True
                if not self._held:
                    self._set_light(False)
                    self._tail_until = None
                return self._refuse(kind, INDICATOR_UNCONFIRMED)
            g = self._held[kind] = Grant(kind, now)
            self._refused.pop(kind, None)
            self._tail_until = None
            return g

    def _confirm_light(self) -> bool:
        """Turn the light on and poll read_back until True or confirm_timeout_s. Bounded by a poll count as well,
        so a clock that does not move cannot hang it. A light already on and reading True needs no new wait."""
        if self._light and self._read_back() is True:
            self._failed = False
            return True
        self._set_light(True)
        deadline = self.clock() + self.confirm_timeout_s
        polls = max(1, math.ceil(self.confirm_timeout_s / CONFIRM_POLL_S)) + 1
        for _ in range(polls):
            if self._read_back() is True:
                self._failed = False
                return True
            if self.clock() >= deadline:
                break
            self.sleep(CONFIRM_POLL_S)
        return False

    def _refuse(self, kind: str, code: str) -> Refusal:
        self._refused[kind] = code
        return Refusal(kind, code)

    def _revoke(self, kind: str, code: str, now: float) -> None:
        """Caller holds the lock. Drops the grant and records why; the light then follows release's tail rule."""
        if self._held.pop(kind, None) is not None:
            log.warning("capture %s revoked: %s", kind, code)
        self._refused[kind] = code
        self._after_release(now)

    def release(self, kind: str) -> None:
        with self._lock:
            had = self._held.pop(kind, None)
            if had is not None:
                self._refused.pop(kind, None)
                self._after_release(self.clock())

    def _after_release(self, now: float) -> None:
        if self._held or not self._light or not self.indicator_required:
            return
        if self.tail_s <= 0:
            self._set_light(False)
            self._tail_until = None
        elif self._tail_until is None:
            self._tail_until = now + self.tail_s

    def held(self, kind: str) -> bool:
        with self._lock:
            return kind in self._held

    def refusal(self, kind: str) -> str | None:
        """The last refusal or revocation code for kind, or None (granted, released or never asked)."""
        with self._lock:
            return self._refused.get(kind)

    def mic_may_grant(self) -> bool:
        """A cheap pre-check that never touches the light: not closed, no latched fault and the mute permits the mic.
        Live mode uses it to decide when an ambient microphone may try again."""
        with self._lock:
            return not self._closed and self._mic_block_code(self._mute_state(self.clock())) is None

    # ---- every step -------------------------------------------------------------------------
    def check(self, now: float) -> list[str]:
        """Revoke what the rules no longer allow. Returns the kinds revoked by this call."""
        revoked: list[str] = []
        with self._lock:
            if self._held and self.indicator_required and self._read_back() is not True:
                self._failed = True
                for kind in list(self._held):
                    self._revoke(kind, INDICATOR_FAILED, now)
                    revoked.append(kind)
                self._set_light(False)
                self._tail_until = None
            mute_state = self._mute_state(now)
            if MIC in self._held:
                code = self._mic_block_code(mute_state)
                if code is not None:
                    self._revoke(MIC, code, now)
                    revoked.append(MIC)
        return revoked

    def tick(self, now: float) -> None:
        """Turn the light off once the tail after the last release has passed."""
        with self._lock:
            if self._tail_until is not None and not self._held and now >= self._tail_until:
                self._set_light(False)
                self._tail_until = None

    def guard_samples(self, kind: str, block: np.ndarray) -> bool:
        """True when the block may be used. Otherwise it has been zeroed in place and must be dropped. Mic blocks
        only; a block may be 1-D (mono) or (n, N) (a microphone array): every channel is checked."""
        if kind != MIC:
            return True
        with self._lock:
            now = self.clock()
            if self._fault is not None:
                block.fill(0)
                return False
            mute_state = self._mute_state(now)
            if mute_state == MUTED:
                settled = self._muted_since is not None and now - self._muted_since >= MUTE_SETTLE_S
                if settled and np.any(block):
                    self._fault = MUTE_BREACH
                    log.error("mute breach: non-zero microphone samples while the mute switch is on; mic off until restart")
                    self._revoke(MIC, MUTE_BREACH, now)
                elif MIC in self._held:
                    self._revoke(MIC, REFUSE_MUTED, now)
                block.fill(0)
                return False
            if MIC not in self._held:
                block.fill(0)
                return False
            return True

    def close(self) -> None:
        """Shutdown: every grant revoked, the light off at once, and every later acquire refused with "closed"."""
        with self._lock:
            self._closed = True
            for kind in list(self._held):
                self._held.pop(kind)
            self._refused.clear()
            self._tail_until = None
            if self.indicator_required and self._light:
                self._set_light(False)

    # ---- state ------------------------------------------------------------------------------
    def state(self) -> dict:
        """Codes only: indicator on|off|tail|failed|not_required, mute muted|open|unknown|not_present,
        camera/mic active|off|refused:<code>, fault null|mute_breach."""
        with self._lock:
            now = self.clock()
            if not self.indicator_required:
                light = NOT_REQUIRED
            elif self._failed:
                light = FAILED
            elif self._light:
                light = ON if self._held else TAIL
            else:
                light = OFF

            def kind_state(kind: str) -> str:
                if kind in self._held:
                    return ACTIVE
                code = self._refused.get(kind)
                return REFUSED_PREFIX + code if code else KIND_OFF

            return {"indicator": light, "mute": self._mute_state(now), "camera": kind_state(CAMERA),
                    "mic": kind_state(MIC), "fault": self._fault}


__all__ = ["ACTIVE", "CAMERA", "CLOSED", "CONFIRM_POLL_S", "INDICATOR_FAILED", "INDICATOR_UNCONFIRMED", "KINDS", "KIND_OFF",
           "MIC", "MUTE_BREACH", "MUTE_SETTLE_S", "MUTE_UNKNOWN", "REFUSAL_CODES", "REFUSED_PREFIX", "REFUSE_MUTED",
           "CaptureInterlock", "Grant", "Refusal"]
