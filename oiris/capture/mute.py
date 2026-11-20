"""The hardware microphone mute port (ADR-0063, spec D-18 and B-18).

The switch itself cuts the microphone's 3V3 in hardware (spec GPIO table: GPIO23 mic-mute sense, pull-up, 30 ms
debounce). A MuteSense only reports it: `muted()` is True (muted), False (open: the microphone may be used) or None
(unknown, for example a sense line that has not settled or a driver that failed). An absent sense is no port at all.
Adapters: oiris/adapters/capture_sim.py (a scripted timeline) and capture_gpio.py (GPIO23 on the Pi rig)."""
from __future__ import annotations
from typing import Protocol, runtime_checkable

# state()["mute"] codes (oiris/capture/interlock.py)
MUTED, OPEN, UNKNOWN, NOT_PRESENT = "muted", "open", "unknown", "not_present"
MUTE_STATES = (MUTED, OPEN, UNKNOWN, NOT_PRESENT)


@runtime_checkable
class MuteSense(Protocol):
    def muted(self) -> bool | None: ...


class UnknownMute:
    """A mute sense that was configured but could not open: always unknown, so a required sense refuses the mic."""

    def __init__(self, reason: str = "the mute sense could not open"):
        self.reason = reason

    def muted(self) -> bool | None:
        return None


__all__ = ["MUTED", "MUTE_STATES", "NOT_PRESENT", "OPEN", "UNKNOWN", "MuteSense", "UnknownMute"]
