"""Capture adapters (ADR-0063): the deterministic sims and the Pi GPIO ports with a fake GPIO module. No real GPIO is
ever imported here."""
import sys

import pytest

from oiris.adapters import capture_gpio
from oiris.adapters.capture_gpio import DEBOUNCE_S, MUTE_PIN_DEFAULT, GpioIndicator, GpioMuteSense, load_gpio
from oiris.adapters.capture_sim import SimIndicator, SimMuteSwitch
from oiris.adapters.drivers import DriverUnavailable
from oiris.capture import IndicatorPort, MuteSense

STEP_S = 0.01


class Clock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


class FakeGPIO:
    """The slice of the RPi.GPIO API the adapters use. Output pins read back what was written unless `stuck` says
    otherwise; input pins read `levels`."""
    BCM, OUT, IN, HIGH, LOW, PUD_UP = "BCM", "OUT", "IN", 1, 0, "PUD_UP"

    def __init__(self):
        self.mode = None
        self.setups: dict = {}
        self.levels: dict = {}
        self.stuck: dict = {}
        self.cleaned: list = []
        self.broken = False

    def setmode(self, mode):
        self.mode = mode

    def setup(self, pin, direction, initial=None, pull_up_down=None):
        self.setups[pin] = (direction, initial, pull_up_down)
        if direction == self.OUT:
            self.levels[pin] = initial
        elif pull_up_down == self.PUD_UP:
            self.levels.setdefault(pin, self.HIGH)

    def output(self, pin, value):
        assert self.setups[pin][0] == self.OUT
        self.levels[pin] = value

    def input(self, pin):
        if self.broken:
            raise RuntimeError("gpio read failed")
        return self.stuck.get(pin, self.levels[pin])

    def cleanup(self, pin):
        self.cleaned.append(pin)


# ---- sims -------------------------------------------------------------------------------------------
def test_sim_indicator_faults():
    clk = Clock()
    ok = SimIndicator(clk)
    assert isinstance(ok, IndicatorPort) and ok.read_back() is False
    ok.set(True)
    assert ok.read_back() is True
    ok.set(False)
    assert ok.read_back() is False and ok.sets == [True, False]
    slow = SimIndicator(clk, confirm_delay_s=0.1)
    slow.set(True)
    assert slow.read_back() is False
    clk.t += 0.1
    assert slow.read_back() is True
    stuck = SimIndicator(clk, stuck_off=True)
    stuck.set(True)
    assert stuck.read_back() is False
    failing = SimIndicator(clk, fail_after_s=3.0)
    failing.set(True)
    assert failing.read_back() is True
    clk.t += 3.0
    assert failing.read_back() is None


def test_sim_mute_follows_its_timeline():
    clk = Clock()
    m = SimMuteSwitch(clk, timeline=[(2.0, True), (1.0, None), (5.0, False)])
    assert isinstance(m, MuteSense)
    seen = []
    for t in (0.0, 1.0, 2.0, 4.9, 5.0):
        clk.t = t
        seen.append(m.muted())
    assert seen == [False, None, True, True, False]


# ---- GPIO --------------------------------------------------------------------------------------------
def test_gpio_indicator_drives_and_reads_back_the_pin():
    g = FakeGPIO()
    led = GpioIndicator(24, gpio=g)
    assert g.mode == g.BCM and g.setups[24] == (g.OUT, g.LOW, None)
    assert isinstance(led, IndicatorPort) and led.read_back() is False
    led.set(True)
    assert g.levels[24] == g.HIGH and led.read_back() is True
    g.stuck[24] = g.LOW                                                  # the pin does not follow: unconfirmed
    assert led.read_back() is False
    g.broken = True
    assert led.read_back() is None
    g.broken = False
    led.close()
    assert g.levels[24] == g.LOW and g.cleaned == [24]


def test_gpio_mute_sense_pull_up_debounce_and_fail_closed_level():
    g, clk = FakeGPIO(), Clock()
    g.levels[MUTE_PIN_DEFAULT] = g.LOW                                    # the switch in the mic-live position
    m = GpioMuteSense(gpio=g, clock=clk, sleep=clk.sleep)
    assert g.setups[MUTE_PIN_DEFAULT] == (g.IN, None, g.PUD_UP)
    assert isinstance(m, MuteSense) and m.muted() is False and clk.t == pytest.approx(DEBOUNCE_S)
    g.levels[MUTE_PIN_DEFAULT] = g.HIGH                                   # switched to mute
    assert m.muted() is False                                            # not yet stable
    clk.t += DEBOUNCE_S / 2
    g.levels[MUTE_PIN_DEFAULT] = g.LOW                                    # a bounce
    assert m.muted() is False
    g.levels[MUTE_PIN_DEFAULT] = g.HIGH
    assert m.muted() is False
    clk.t += DEBOUNCE_S / 2
    assert m.muted() is False                                            # stable for less than the debounce
    clk.t += DEBOUNCE_S
    assert m.muted() is True
    g.broken = True
    assert m.muted() is None
    m.close()
    assert g.cleaned == [MUTE_PIN_DEFAULT]


def test_gpio_mute_sense_unknown_until_settled():
    g, clk = FakeGPIO(), Clock()
    m = GpioMuteSense(5, gpio=g, clock=clk, sleep=lambda s: None)         # no time passed at open
    assert m.muted() is None
    clk.t += DEBOUNCE_S
    assert m.muted() is True                                             # pull-up, nothing connected: reads muted


def test_without_the_gpio_library_the_ports_are_driver_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "RPi", None)                        # import fails as if not installed
    monkeypatch.setitem(sys.modules, "RPi.GPIO", None)
    with pytest.raises(DriverUnavailable, match="RPi.GPIO"):
        load_gpio()
    with pytest.raises(DriverUnavailable):
        GpioIndicator(24)
    with pytest.raises(DriverUnavailable):
        GpioMuteSense()


def test_off_a_pi_the_import_error_is_driver_unavailable(monkeypatch):
    def not_a_pi(name):
        raise RuntimeError("This module can only be run on a Raspberry Pi!")
    monkeypatch.setattr(capture_gpio.importlib, "import_module", not_a_pi)
    with pytest.raises(DriverUnavailable):
        load_gpio()
