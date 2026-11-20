"""Capture interlock (ADR-0063) and the config "capture" section. Synthetic only: simulated ports and a fake clock."""
import numpy as np
import pytest

from oiris.adapters.capture_sim import SimIndicator, SimMuteSwitch
from oiris.capture import INDICATOR_STATES, MUTE_STATES, REFUSAL_CODES
from oiris.capture.interlock import (CONFIRM_POLL_S, MUTE_SETTLE_S, CaptureInterlock, Grant, Refusal)
from oiris.config import (CAPTURE_PROFILES, Config, ConfigError, capture_settings, validate_capture)
from oiris.guard.vocabulary import violations

TIMEOUT_S = 0.2
TAIL_S = 2.0
STEP_S = 1.0


class Clock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


def make(clock, indicator=None, mute=None, indicator_required=True, mute_required=False, tail_s=TAIL_S):
    return CaptureInterlock(indicator, mute, indicator_required=indicator_required, mute_required=mute_required,
                            confirm_timeout_s=TIMEOUT_S, tail_s=tail_s, clock=clock, sleep=clock.sleep)


def voice_block(n=1600, channels=None):
    shape = (n,) if channels is None else (n, channels)
    return np.random.default_rng(1).standard_normal(shape).astype(np.float32) * np.float32(0.1)


# ---- the light ------------------------------------------------------------------------------------
def test_an_unconfirmed_light_refuses_and_turns_off():
    clk = Clock()
    led = SimIndicator(clk, stuck_off=True)
    il = make(clk, led)
    r = il.acquire("camera")
    assert isinstance(r, Refusal) and not r and r.code == "indicator_unconfirmed"
    assert led.sets[0] is True and led.on is False                      # tried, then turned off again
    assert TIMEOUT_S <= clk.t <= TIMEOUT_S + 2 * CONFIRM_POLL_S          # waited the timeout, no longer
    st = il.state()
    assert st["indicator"] == "failed" and st["camera"] == "refused:indicator_unconfirmed" and st["mic"] == "off"


def test_no_light_fitted_refuses_and_a_clock_that_never_moves_cannot_hang_it():
    clk = Clock()
    il = CaptureInterlock(None, None, indicator_required=True, confirm_timeout_s=TIMEOUT_S, clock=clk, sleep=lambda s: None)
    assert il.acquire("mic").code == "indicator_unconfirmed"


def test_a_light_confirming_inside_the_timeout_grants():
    clk = Clock()
    led = SimIndicator(clk, confirm_delay_s=TIMEOUT_S / 2)
    il = make(clk, led)
    g = il.acquire("mic")
    assert isinstance(g, Grant) and g and g.kind == "mic"
    assert TIMEOUT_S / 2 <= clk.t < TIMEOUT_S
    assert il.state() == {"indicator": "on", "mute": "not_present", "camera": "off", "mic": "active", "fault": None}
    t = clk.t
    assert il.acquire("camera") and clk.t == t                           # already lit and confirmed: no second wait
    assert il.acquire("mic") == g                                        # idempotent


def test_a_light_confirming_after_the_timeout_refuses():
    clk = Clock()
    il = make(clk, SimIndicator(clk, confirm_delay_s=TIMEOUT_S * 2))
    assert il.acquire("camera").code == "indicator_unconfirmed"


@pytest.mark.parametrize("led_kw", [{"fail_after_s": 5.0}])
def test_a_failing_light_revokes_every_grant_on_the_next_check(led_kw):
    clk = Clock()
    led = SimIndicator(clk, **led_kw)
    il = make(clk, led)
    assert il.acquire("camera") and il.acquire("mic")
    clk.t += STEP_S
    assert il.check(clk.t) == []
    clk.t = 5.0
    assert sorted(il.check(clk.t)) == ["camera", "mic"]
    st = il.state()
    assert st["indicator"] == "failed" and st["camera"] == st["mic"] == "refused:indicator_failed"
    assert led.on is False and not il.held("mic") and not il.held("camera")
    assert il.acquire("mic").code == "indicator_unconfirmed"            # it stays unconfirmable


def test_a_light_stuck_off_after_the_grant_revokes():
    clk = Clock()
    led = SimIndicator(clk)
    il = make(clk, led)
    assert il.acquire("mic")
    led.stuck_off = True
    assert il.check(clk.t) == ["mic"] and il.refusal("mic") == "indicator_failed"


def test_a_read_back_that_raises_counts_as_unconfirmed():
    clk = Clock()

    class Broken(SimIndicator):
        def read_back(self):
            raise OSError("bus error")
    assert make(clk, Broken(clk)).acquire("camera").code == "indicator_unconfirmed"


def test_the_tail_keeps_the_light_on_then_off():
    clk = Clock()
    led = SimIndicator(clk)
    il = make(clk, led)
    assert il.acquire("mic") and il.acquire("camera")
    il.release("mic")
    assert il.state()["indicator"] == "on" and led.on                   # the camera still holds it
    il.release("camera")
    assert il.state()["indicator"] == "tail" and led.on and il.state()["camera"] == "off"
    il.tick(clk.t + TAIL_S / 2)
    assert led.on
    il.tick(clk.t + TAIL_S)
    assert not led.on and il.state()["indicator"] == "off"


def test_a_grant_inside_the_tail_reuses_the_light_and_cancels_the_tail():
    clk = Clock()
    led = SimIndicator(clk)
    il = make(clk, led)
    assert il.acquire("mic")
    il.release("mic")
    assert il.acquire("mic") and il.state()["indicator"] == "on"
    il.tick(clk.t + TAIL_S * 2)
    assert led.on


def test_zero_tail_turns_the_light_off_at_once():
    clk = Clock()
    led = SimIndicator(clk)
    il = make(clk, led, tail_s=0.0)
    assert il.acquire("camera")
    il.release("camera")
    assert not led.on and il.state()["indicator"] == "off"


def test_light_not_required_never_touches_it():
    clk = Clock()
    led = SimIndicator(clk, stuck_off=True)
    il = make(clk, led, indicator_required=False)
    assert il.acquire("camera") and il.acquire("mic") and led.sets == []
    assert il.check(clk.t) == [] and il.state()["indicator"] == "not_required"


# ---- the mute -----------------------------------------------------------------------------------
def test_mute_refuses_the_mic_but_not_the_camera():
    clk = Clock()
    il = make(clk, SimIndicator(clk), SimMuteSwitch(clk, initial=True))
    assert il.acquire("mic").code == "muted"
    assert il.acquire("camera")
    assert il.state()["mute"] == "muted" and il.state()["mic"] == "refused:muted" and il.state()["camera"] == "active"


def test_muting_while_listening_revokes_the_mic_only():
    clk = Clock()
    il = make(clk, SimIndicator(clk), SimMuteSwitch(clk, timeline=[(10.0, True), (20.0, False)]))
    assert il.acquire("mic") and il.acquire("camera")
    clk.t = 10.0
    assert il.check(clk.t) == ["mic"]
    assert il.held("camera") and il.refusal("mic") == "muted" and il.state()["indicator"] == "on"
    assert not il.mic_may_grant()
    clk.t = 20.0
    assert il.mic_may_grant() and il.acquire("mic")


def test_unknown_mute_refuses_only_when_the_sense_is_required():
    clk = Clock()
    unknown = SimMuteSwitch(clk, initial=None)
    assert make(clk, SimIndicator(clk), unknown, mute_required=True).acquire("mic").code == "mute_unknown"
    assert make(clk, SimIndicator(clk), unknown, mute_required=False).acquire("mic")
    il = make(clk, SimIndicator(clk), None, mute_required=True)          # required but absent
    assert il.acquire("mic").code == "mute_unknown" and il.state()["mute"] == "not_present"
    assert il.acquire("camera")


def test_mute_going_unknown_while_listening_revokes_when_required():
    clk = Clock()
    il = make(clk, SimIndicator(clk), SimMuteSwitch(clk, timeline=[(3.0, None)]), mute_required=True)
    assert il.acquire("mic")
    clk.t = 3.0
    assert il.check(clk.t) == ["mic"] and il.refusal("mic") == "mute_unknown"


# ---- guard_samples ---------------------------------------------------------------------------------
@pytest.mark.parametrize("channels", [None, 4])
def test_blocks_pass_only_while_the_mic_is_granted(channels):
    clk = Clock()
    il = make(clk, SimIndicator(clk), SimMuteSwitch(clk))
    b = voice_block(channels=channels)
    assert not il.guard_samples("mic", b) and not b.any()                # no grant: zeroed, dropped
    assert il.acquire("mic")
    b = voice_block(channels=channels)
    keep = b.copy()
    assert il.guard_samples("mic", b) and np.array_equal(b, keep)        # granted, open: untouched
    assert il.guard_samples("camera", b)                                 # only mic blocks are checked


@pytest.mark.parametrize("channels", [None, 4])
def test_a_breach_latches_and_zeroes_the_block(channels):
    clk = Clock()
    il = make(clk, SimIndicator(clk), SimMuteSwitch(clk, timeline=[(10.0, True), (30.0, False)]))
    assert il.acquire("mic")
    clk.t = 10.0
    b = voice_block(channels=channels)
    assert not il.guard_samples("mic", b) and not b.any()                # first muted block: in the settle time
    assert il.refusal("mic") == "muted" and il.state()["fault"] is None
    clk.t = 10.0 + MUTE_SETTLE_S / 2
    assert not il.guard_samples("mic", np.zeros_like(b))                 # zeros while muted: the hardware cut works
    clk.t = 10.0 + MUTE_SETTLE_S * 2
    b = np.zeros_like(b)
    if channels is None:
        b[-1] = 1e-6
    else:
        b[7, 3] = 1e-6                                                   # one sample, on one channel
    assert not il.guard_samples("mic", b) and not b.any()
    st = il.state()
    assert st["fault"] == "mute_breach" and st["mic"] == "refused:mute_breach"
    clk.t = 30.0                                                         # switch open again: still latched
    assert il.acquire("mic").code == "mute_breach" and not il.mic_may_grant()
    b = voice_block(channels=channels)
    assert not il.guard_samples("mic", b) and not b.any()
    assert il.acquire("camera")                                          # the camera ignores the mute


def test_muted_from_a_check_already_counts_the_settle_time():
    clk = Clock()
    il = make(clk, SimIndicator(clk), SimMuteSwitch(clk, timeline=[(1.0, True)]))
    assert il.acquire("mic")
    clk.t = 1.0
    il.check(clk.t)
    clk.t = 1.0 + MUTE_SETTLE_S * 2
    assert not il.guard_samples("mic", voice_block()) and il.state()["fault"] == "mute_breach"


def test_close_revokes_everything_and_refuses_after():
    clk = Clock()
    led = SimIndicator(clk)
    il = make(clk, led)
    assert il.acquire("mic")
    il.close()
    assert not led.on and not il.held("mic") and il.acquire("camera").code == "closed"
    assert il.state()["indicator"] == "off" and il.state()["mic"] == "off"


def test_bad_kind_is_an_error():
    with pytest.raises(ValueError):
        make(Clock()).acquire("speaker")


def test_state_codes_are_plain_and_pass_the_vocabulary_guard():
    for code in (*INDICATOR_STATES, *MUTE_STATES, *REFUSAL_CODES, "active", "off"):
        assert not violations(code.replace("_", " "))
    assert set(REFUSAL_CODES) == {"indicator_unconfirmed", "indicator_failed", "muted", "mute_unknown", "mute_breach", "closed"}


# ---- config ------------------------------------------------------------------------------------------
def test_capture_defaults_are_the_mac_profile():
    s = capture_settings({})
    assert s["profile"] == "mac" and s["indicator"] == "off" and s["mute_sense"] == "if_present"
    assert s["indicator_adapter"] == s["mute_adapter"] == "none"
    assert s["confirm_timeout_ms"] == 200 and s["tail_s"] == 2 and s["mute_pin"] == 23
    assert Config().validate().capture == {}


def test_rig_profile_requires_the_mute_sense_and_a_light_pin():
    assert capture_settings({"profile": "rig"})["mute_sense"] == "required"
    errs = validate_capture({"profile": "rig"})
    assert any("indicator_pin is needed" in e for e in errs)
    assert validate_capture({"profile": "rig", "indicator_pin": 24}) == []


@pytest.mark.parametrize("profile", ["glasses", "helmet"])
def test_glasses_and_helmet_force_the_light_on(profile):
    assert capture_settings({"profile": profile})["indicator"] == "required"
    with pytest.raises(ConfigError, match=f"cannot be off for {profile}"):
        Config(capture={"profile": profile, "indicator": "off", "indicator_adapter": "sim"}).validate()
    with pytest.raises(ConfigError, match="set capture.indicator_adapter"):
        Config(capture={"profile": profile}).validate()
    Config(capture={"profile": profile, "indicator_adapter": "sim"}).validate()


@pytest.mark.parametrize("key,bad", [
    ("confirm_timeout_ms", 49), ("confirm_timeout_ms", 2001), ("confirm_timeout_ms", "200"), ("confirm_timeout_ms", float("nan")),
    ("tail_s", -0.1), ("tail_s", 10.5), ("tail_s", True),
    ("indicator_pin", 28), ("indicator_pin", -1), ("indicator_pin", 17), ("indicator_pin", 23), ("indicator_pin", True),
    ("mute_pin", 40), ("mute_pin", 18), ("mute_pin", "23"),
    ("profile", "phone"), ("indicator", "on"), ("mute_sense", "maybe"), ("indicator_adapter", "usb"), ("mute_adapter", "i2c"),
])
def test_capture_range_errors(key, bad):
    with pytest.raises(ConfigError, match=f"capture.{key}"):
        Config(capture={key: bad}).validate()


@pytest.mark.parametrize("key,ok", [("confirm_timeout_ms", 50), ("confirm_timeout_ms", 2000), ("tail_s", 0), ("tail_s", 10),
                                    ("indicator_pin", 24), ("mute_pin", 5)])
def test_capture_range_edges_pass(key, ok):
    Config(capture={key: ok}).validate()


def test_capture_rejects_unknown_keys_same_pins_and_required_without_adapter():
    with pytest.raises(ConfigError, match="unknown keys"):
        Config(capture={"led": 1}).validate()
    with pytest.raises(ConfigError, match="must differ"):
        Config(capture={"indicator_pin": 5, "mute_pin": 5}).validate()
    with pytest.raises(ConfigError, match="mute_sense is required"):
        Config(capture={"mute_sense": "required"}).validate()
    with pytest.raises(ConfigError, match="indicator is required"):
        Config(capture={"indicator": "required"}).validate()
    assert validate_capture([]) == ["capture must be an object"]


def test_every_profile_validates_with_sim_ports():
    for p in CAPTURE_PROFILES:
        Config(capture={"profile": p, "indicator_adapter": "sim", "mute_adapter": "sim"}).validate()
