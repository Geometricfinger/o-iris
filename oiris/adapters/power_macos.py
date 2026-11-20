"""Mac battery reading for the power governor (ADR-0061): `pmset -g batt` -> battery_pct and charging.

The Mac exposes no skin or device temperature here, so those stay None (unknown, never guessed); a Mac is a test
bench, not the wearable. No network. The command runner and the clock are injectable (tests use fixtures).
read() never raises into the caller: on any failure it returns an empty reading (every field None).

Typical output:
    Now drawing from 'Battery Power'
     -InternalBattery-0 (id=1234567)	64%; discharging; 4:12 remaining present: true
A desktop Mac prints only the first line ("AC Power") and has no battery: battery_pct and charging stay None."""
from __future__ import annotations
import logging
import re
import subprocess
import time
from typing import Callable

from ..power.governor import PowerReading

log = logging.getLogger(__name__)

PMSET = ["/usr/bin/pmset", "-g", "batt"]
PMSET_TIMEOUT_S = 5.0
MAX_OUTPUT_CHARS = 4096
PCT_MAX = 100
BATTERY_RE = re.compile(r"-InternalBattery-\d+[^\n]*?(\d{1,3})%;\s*([^;\n]+);")
SOURCE_RE = re.compile(r"Now drawing from '([^']+)'")
AC_SOURCE = "AC Power"
DISCHARGING_STATES = {"discharging"}
# "charging", "charged", "finishing charge" and "AC attached" all mean external power: the battery is not draining.
EXTERNAL_POWER_STATES = {"charging", "charged", "finishing charge", "AC attached"}


def _run_pmset() -> str:
    return subprocess.run(PMSET, capture_output=True, text=True, timeout=PMSET_TIMEOUT_S, check=False).stdout


def parse_pmset(text: object) -> tuple[float | None, bool | None]:
    """(battery_pct, charging) from `pmset -g batt` output. Unknown parts are None; never raises."""
    if not isinstance(text, str):
        return None, None
    text = text[:MAX_OUTPUT_CHARS]
    m = BATTERY_RE.search(text)
    if m is None:
        return None, None                      # no internal battery (desktop Mac) or unrecognised output
    pct = int(m.group(1))
    if pct > PCT_MAX:
        return None, None
    state = m.group(2).strip()
    src = SOURCE_RE.search(text)
    if state in DISCHARGING_STATES:
        charging: bool | None = False
    elif state in EXTERNAL_POWER_STATES:
        charging = True
    else:
        charging = None if src is None else src.group(1) == AC_SOURCE
    return float(pct), charging


class MacBattery:
    """read() -> PowerReading with battery_pct and charging (skin_c, device_c and thermal_status None)."""

    def __init__(self, runner: Callable[[], str] = _run_pmset, clock: Callable[[], float] = time.monotonic):
        self.runner, self.clock = runner, clock

    def read(self) -> PowerReading:
        try:
            t = self.clock()
        except Exception as e:  # python-brain: allow SILENT_EXCEPTION - a broken clock gives a stale empty reading, never an error
            log.warning("power clock failed: %s", e)
            return PowerReading(t=float("nan"))
        try:
            pct, charging = parse_pmset(self.runner())
        except Exception as e:  # python-brain: allow SILENT_EXCEPTION - no pmset, a timeout or odd output gives an empty reading
            log.warning("battery reading failed: %s", e)
            return PowerReading(t=t)
        return PowerReading(t=t, battery_pct=pct, charging=charging)


__all__ = ["MacBattery", "parse_pmset", "PMSET"]
