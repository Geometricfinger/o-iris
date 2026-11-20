"""The bystander notice port (ADR-0063, ADR-0059 decision 3): an outward-facing recording light.

An IndicatorPort has `set(on)` and `read_back()`. read_back() is True when the light is confirmed lit, False when it is
confirmed dark, and None when it cannot be confirmed. None counts as unconfirmed: the capture interlock never grants on
it. Adapters: oiris/adapters/capture_sim.py (deterministic, with faults) and capture_gpio.py (the Pi rig's LED)."""
from __future__ import annotations
from typing import Protocol, runtime_checkable

# state()["indicator"] codes (oiris/capture/interlock.py)
ON, OFF, TAIL, FAILED, NOT_REQUIRED = "on", "off", "tail", "failed", "not_required"
INDICATOR_STATES = (ON, OFF, TAIL, FAILED, NOT_REQUIRED)


@runtime_checkable
class IndicatorPort(Protocol):
    def set(self, on: bool) -> None: ...

    def read_back(self) -> bool | None: ...


class NoIndicator:
    """No light fitted, or its driver could not open: set() does nothing and nothing can be confirmed."""

    def __init__(self, reason: str = "no recording light configured"):
        self.reason = reason

    def set(self, on: bool) -> None:
        return None

    def read_back(self) -> bool | None:
        return None


__all__ = ["FAILED", "INDICATOR_STATES", "NOT_REQUIRED", "OFF", "ON", "TAIL", "IndicatorPort", "NoIndicator"]
