"""Pi rig capture ports on GPIO (ADR-0063). NOT YET TESTED ON HARDWARE: built against the RPi.GPIO API and tested
with a fake GPIO module only; the bench check (spec B-18) is still to do.

GpioIndicator: the recording LED on a configured BCM pin (the spec's GPIO map assigns none yet, so the config must name
one). read_back() is the pin level as GPIO reads it back: it proves the pin is driven, not that the LED is lit. A
photodiode read-back that proves the light is later hardware.

GpioMuteSense: the mic-mute sense on GPIO23 by default (spec GPIO table): input with the internal pull-up, 30 ms
debounce. The switch cuts the microphone's 3V3 in hardware; this only reports it. With the pull-up, an open sense line
reads high, so high means muted (MUTED_LEVEL_DEFAULT): a broken sense wire reads muted and fails closed. The
microphone-live position of the switch pulls the line to ground. Confirm against the wiring before use.

The RPi.GPIO module (MIT; on the Pi 5 the rpi-lgpio package provides the same API) is imported lazily, only when a port
opens; without it, or off a Pi, DriverUnavailable. Tests pass a fake `gpio` module and never import a real one."""
from __future__ import annotations
import importlib, time
from typing import Any, Callable

from .drivers import DriverUnavailable

MUTE_PIN_DEFAULT = 23            # spec GPIO table: GPIO22 / GPIO23 ack-panic button / mic-mute sense
DEBOUNCE_S = 0.030               # spec GPIO table: pull-ups, 30 ms debounce
HIGH, LOW = 1, 0
MUTED_LEVEL_DEFAULT = HIGH
NO_GPIO = "the RPi.GPIO module is not available (Pi rig only; on a Pi 5: pip install rpi-lgpio)"


def load_gpio(gpio: Any = None) -> Any:
    """The RPi.GPIO module, or the injected stand-in. DriverUnavailable when it is missing or not on a Pi."""
    if gpio is not None:
        return gpio
    try:
        return importlib.import_module("RPi.GPIO")
    except (ImportError, RuntimeError) as e:     # RuntimeError: RPi.GPIO imported on something that is not a Pi
        raise DriverUnavailable(NO_GPIO) from e


class GpioIndicator:
    """The recording LED on BCM `pin`, driven low (off) at open."""

    def __init__(self, pin: int, gpio: Any = None):
        self.pin = int(pin)
        self.gpio = load_gpio(gpio)
        self.gpio.setmode(self.gpio.BCM)
        self.gpio.setup(self.pin, self.gpio.OUT, initial=self.gpio.LOW)

    def set(self, on: bool) -> None:
        self.gpio.output(self.pin, self.gpio.HIGH if on else self.gpio.LOW)

    def read_back(self) -> bool | None:
        """The pin level read back from GPIO; None when it cannot be read."""
        try:
            return bool(self.gpio.input(self.pin))
        except Exception:  # python-brain: allow SILENT_EXCEPTION - an unreadable pin is "cannot be confirmed" (None), which fails closed
            return None

    def close(self) -> None:
        self.set(False)
        self.gpio.cleanup(self.pin)


class GpioMuteSense:
    """muted() -> True / False / None. A level is accepted only once it has read the same for DEBOUNCE_S; until the
    first level is accepted the state is unknown. open() reads the pin twice, DEBOUNCE_S apart, to settle it."""

    def __init__(self, pin: int = MUTE_PIN_DEFAULT, gpio: Any = None, muted_level: int = MUTED_LEVEL_DEFAULT,
                 debounce_s: float = DEBOUNCE_S, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Any] = time.sleep):
        self.pin, self.muted_level, self.debounce_s = int(pin), int(muted_level), float(debounce_s)
        self.clock = clock
        self.gpio = load_gpio(gpio)
        self.gpio.setmode(self.gpio.BCM)
        self.gpio.setup(self.pin, self.gpio.IN, pull_up_down=self.gpio.PUD_UP)
        self._stable: int | None = None          # the accepted (debounced) level
        self._raw: int | None = None             # the level being timed
        self._raw_since = 0.0
        self.muted()
        sleep(self.debounce_s)
        self.muted()

    def _read(self) -> int | None:
        try:
            return HIGH if self.gpio.input(self.pin) else LOW
        except Exception:  # python-brain: allow SILENT_EXCEPTION - an unreadable sense line is "unknown" (None), which fails closed when the sense is required
            return None

    def muted(self) -> bool | None:
        now, level = self.clock(), self._read()
        if level is None:
            self._raw = None
            return None
        if level != self._raw:
            self._raw, self._raw_since = level, now
        if level != self._stable and now - self._raw_since >= self.debounce_s:
            self._stable = level
        return None if self._stable is None else self._stable == self.muted_level

    def close(self) -> None:
        self.gpio.cleanup(self.pin)


__all__ = ["DEBOUNCE_S", "MUTED_LEVEL_DEFAULT", "MUTE_PIN_DEFAULT", "GpioIndicator", "GpioMuteSense", "load_gpio"]
