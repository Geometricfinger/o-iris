"""Live mode under the capture interlock (ADR-0063), and the dashboard's Capture line. Synthetic only: fake microphone
captures, simulated light and mute ports, a fake clock for the interlock. No microphone, camera or GPIO."""
# ruff: noqa: F811
import sys

import numpy as np
import pytest

from oiris.adapters.capture_sim import SimIndicator, SimMuteSwitch
from oiris.capture.interlock import CaptureInterlock, Grant, Refusal
from oiris.config import Config
from oiris.dashboard.server import CAPTURE_REFUSALS, render
from oiris.guard.vocabulary import violations
from oiris.live import MIC_REFUSAL_REASONS, build_interlock
from tests.test_dashboard_live import _snap
from tests.test_voice_link import face_on, synth_voice, voice_opt_in
from tests.test_voice_link import env  # noqa: F401  (fixture reuse)
from tests.test_voice_link_live import FakeCapture, live_cfg, live_runner

CONFIRM_S = 0.05
TIMEOUT_S = 0.2
MUTE_AT, OPEN_AT = 10.0, 20.0


class Clock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


class LoggedIndicator(SimIndicator):
    def __init__(self, clock, log, **kw):
        super().__init__(clock, **kw)
        self.log = log

    def set(self, on):
        self.log.append(("set", bool(on)))
        super().set(on)

    def read_back(self):
        v = super().read_back()
        self.log.append(("read", v))
        return v


class LoggedCapture(FakeCapture):
    def __init__(self, log, **kw):
        super().__init__(**kw)
        self.log = log

    def start(self, cb):
        self.log.append(("start",))
        super().start(cb)

    def stop(self):
        self.log.append(("stop",))
        super().stop()


def interlock(clk, indicator, mute=None, mute_required=False):
    return CaptureInterlock(indicator, mute, indicator_required=True, mute_required=mute_required,
                            confirm_timeout_s=TIMEOUT_S, tail_s=2.0, clock=clk, sleep=clk.sleep)


def voice_ready(env):
    face_on(env)
    voice_opt_in(env)


def test_the_mic_opens_only_after_the_light_confirms(env):
    voice_ready(env)
    clk, log = Clock(), []
    cap = LoggedCapture(log)
    il = interlock(clk, LoggedIndicator(clk, log, confirm_delay_s=CONFIRM_S))
    r = live_runner(env, live_cfg(env), voice_capture=cap, interlock=il, clock=clk)
    r.start(background=False)
    try:
        assert cap.starts == 1
        first_true = log.index(("read", True))
        assert log[0] == ("set", True) and ("read", False) in log[:first_true]   # it had to wait for the light
        assert log.index(("start",)) > first_true                               # the mic opened only after it
        snap = r.snapshot()
        assert snap["capture"] == {"indicator": "on", "mute": "not_present", "camera": "off", "mic": "active",
                                   "fault": None, "profile": "mac"}
        assert snap["voice"]["state"] == "listening"
    finally:
        r.stop()
    assert cap.stops == 1 and ("set", False) in log
    assert r.snapshot()["capture"]["indicator"] == "off" and r.snapshot()["capture"]["mic"] == "off"


def test_an_unconfirmed_light_keeps_the_mic_closed(env):
    voice_ready(env)
    clk = Clock()
    cap = FakeCapture()
    r = live_runner(env, live_cfg(env), voice_capture=cap, interlock=interlock(clk, SimIndicator(clk, stuck_off=True)), clock=clk)
    r.start(background=False)
    try:
        assert cap.starts == 0
        snap = r.snapshot()
        assert snap["capture"]["mic"] == "refused:indicator_unconfirmed" and snap["capture"]["indicator"] == "failed"
        assert snap["voice"]["state"] == "unavailable" and snap["voice"]["reason"] == MIC_REFUSAL_REASONS["indicator_unconfirmed"]
        r.step()
        assert cap.starts == 0                                                   # retried at most every CAPTURE_RETRY_S
    finally:
        r.stop()


def test_a_revoked_grant_stops_listening_and_ambient_reopens_when_allowed(env):
    voice_ready(env)
    clk = Clock()
    cap = FakeCapture()
    il = interlock(clk, SimIndicator(clk), SimMuteSwitch(clk, timeline=[(MUTE_AT, True), (OPEN_AT, False)]))
    r = live_runner(env, live_cfg(env), voice_capture=cap, interlock=il, clock=clk)
    r.start(background=False)
    try:
        assert cap.starts == 1 and r.snapshot()["voice"]["state"] == "listening"
        clk.t = MUTE_AT
        cap.feed(synth_voice("A", seconds=0.3, seed=3))       # blocks after the switch, before the step: never passed on
        assert cap.handed and not any(b.any() for b in cap.handed)
        assert r.voice.dropped_while_paused == 0
        r.step()
        assert cap.stops == 1 and cap.cb is None                                 # the stream is closed
        snap = r.snapshot()
        assert snap["capture"]["mic"] == "refused:muted" and snap["capture"]["mute"] == "muted"
        assert snap["voice"]["state"] == "unavailable" and snap["voice"]["reason"] == MIC_REFUSAL_REASONS["muted"]
        r.step()
        assert cap.starts == 1                                                   # still muted: stays closed
        clk.t = OPEN_AT
        r.step()
        assert cap.starts == 2 and r.snapshot()["voice"]["state"] == "listening"
        assert r.snapshot()["capture"]["mic"] == "active"
    finally:
        r.stop()


def test_a_light_failing_while_listening_stops_the_mic(env):
    voice_ready(env)
    clk = Clock()
    cap = FakeCapture()
    led = SimIndicator(clk)
    r = live_runner(env, live_cfg(env), voice_capture=cap, interlock=interlock(clk, led), clock=clk)
    r.start(background=False)
    try:
        assert cap.starts == 1
        led.stuck_off = True
        r.step()
        assert cap.stops == 1 and r.snapshot()["capture"]["mic"] == "refused:indicator_failed"
        assert r.snapshot()["capture"]["indicator"] == "failed"
    finally:
        r.stop()


@pytest.mark.parametrize("channels", [1, 4])
def test_a_mute_breach_on_any_channel_latches_and_stops_the_mic(env, channels):
    voice_ready(env)
    clk = Clock()

    class MultiCapture(FakeCapture):
        def feed_block(self, block):
            cb = self.cb
            if cb is not None:
                self.handed.append(block)
                cb(block)
    cap = MultiCapture()
    cap.channels = channels
    il = interlock(clk, SimIndicator(clk), SimMuteSwitch(clk, timeline=[(MUTE_AT, True)]))
    r = live_runner(env, live_cfg(env), voice_capture=cap, interlock=il, clock=clk)
    r.start(background=False)
    try:
        clk.t = MUTE_AT
        shape = (1600,) if channels == 1 else (1600, channels)
        cap.feed_block(np.zeros(shape, np.float32))                              # muted, settling: dropped
        clk.t = MUTE_AT + 1.0
        b = np.zeros(shape, np.float32)
        b[(5,) if channels == 1 else (5, channels - 1)] = 0.25                    # one sample on the last channel
        cap.feed_block(b)
        assert not b.any()
        r.step()
        snap = r.snapshot()
        assert snap["capture"]["fault"] == "mute_breach" and snap["capture"]["mic"] == "refused:mute_breach"
        assert cap.stops == 1
    finally:
        r.stop()


def test_push_to_talk_follows_the_same_rule(env):
    voice_ready(env)
    clk = Clock()
    cap = FakeCapture()
    il = interlock(clk, SimIndicator(clk), SimMuteSwitch(clk, timeline=[(MUTE_AT, True), (OPEN_AT, False)]))
    r = live_runner(env, live_cfg(env, listening="push_to_talk"), voice_capture=cap, interlock=il, clock=clk)
    r.start(background=False)
    try:
        assert cap.starts == 0 and r.snapshot()["capture"]["mic"] == "off"
        clk.t = MUTE_AT
        assert r.voice_listen(5.0) is False and cap.starts == 0
        assert r.snapshot()["capture"]["mic"] == "refused:muted"
        assert MIC_REFUSAL_REASONS["muted"] in r.snapshot()["voice"]["reason"]
        r.step()
        assert cap.starts == 0                                                   # push-to-talk never reopens by itself
        clk.t = OPEN_AT
        assert r.voice_listen(5.0) is True and cap.starts == 1
        clk.t = OPEN_AT + 1.0
        il.mute.timeline.append((OPEN_AT + 1.0, True))
        r.step()
        assert cap.stops == 1 and r.snapshot()["voice"]["state"] == "idle"
    finally:
        r.stop()


def test_the_mac_profile_is_unchanged(env):
    voice_ready(env)
    cap = FakeCapture()
    r = live_runner(env, live_cfg(env), voice_capture=cap)                     # no interlock injected: from config
    r.start(background=False)
    try:
        assert cap.starts == 1 and r.snapshot()["voice"]["state"] == "listening"
        b = synth_voice("A", seconds=0.1, seed=4).astype(np.float32)
        assert r.interlock is not None and r.interlock.guard_samples("mic", b.copy())
        r.step()
        snap = r.snapshot()
        assert snap["capture"] == {"indicator": "not_required", "mute": "not_present", "camera": "off", "mic": "active",
                                   "fault": None, "profile": "mac"}
        g = r.camera_acquire()
        assert isinstance(g, Grant) and r.snapshot()["capture"]["camera"] == "active"
        r.camera_release()
        assert r.snapshot()["capture"]["camera"] == "off"
    finally:
        r.stop()
    assert cap.stops == 1
    assert isinstance(r.camera_acquire(), Refusal) and r.camera_acquire().code == "closed"


def test_without_voice_the_snapshot_still_carries_capture(env):
    cfg = Config(data_dir=str(env["dir"]))
    r = live_runner(env, cfg)
    r.start(background=False)
    try:
        r.step()
        assert "voice" not in r.snapshot()
        assert r.snapshot()["capture"]["mic"] == "off" and r.snapshot()["capture"]["indicator"] == "not_required"
    finally:
        r.stop()


def test_build_interlock_from_config_with_sim_ports():
    cfg = Config(capture={"indicator": "required", "indicator_adapter": "sim", "mute_adapter": "sim", "tail_s": 0}).validate()
    il = build_interlock(cfg)
    assert il.indicator_required and not il.mute_required and il.tail_s == 0
    assert il.acquire("mic") and il.state()["indicator"] == "on" and il.state()["mute"] == "open"
    il.release("mic")
    assert il.state()["indicator"] == "off"
    assert build_interlock(Config()).state()["indicator"] == "not_required"


def test_gpio_ports_without_the_library_fail_closed(monkeypatch):
    monkeypatch.setitem(sys.modules, "RPi", None)
    monkeypatch.setitem(sys.modules, "RPi.GPIO", None)
    cfg = Config(capture={"profile": "rig", "indicator_pin": 24, "confirm_timeout_ms": 50}).validate()
    il = build_interlock(cfg)
    assert il.acquire("camera").code == "indicator_unconfirmed"
    assert il.state()["mute"] == "unknown"
    off = Config(capture={"profile": "rig", "indicator": "off"}).validate()
    il = build_interlock(off)
    assert il.acquire("mic").code == "mute_unknown" and il.acquire("camera")


# ---- dashboard -----------------------------------------------------------------------------------------
def cap_snap(**c):
    base = {"indicator": "not_required", "mute": "not_present", "camera": "off", "mic": "active", "fault": None, "profile": "mac"}
    snap = _snap(None)
    snap["capture"] = {**base, **c}
    return snap


def _cap(page: str) -> str:
    return page.split("<p id=CAP")[1].split("</p>")[0]


def test_dashboard_capture_line_in_plain_words():
    line = _cap(render(None, cap_snap()))
    assert "Recording light: <b>not used on this Mac</b>" in line and "Mute switch: <b>none fitted</b>" in line
    assert "Microphone: <b>listening</b>" in line and "Camera" not in line
    line = _cap(render(None, cap_snap(indicator="failed", mute="muted", mic="refused:muted", profile="rig")))
    assert "Recording light: <b>FAILED</b>" in line and "Mute switch: <b>muted (microphone cut)</b>" in line
    assert "Microphone: <b>refused: the mute switch is set to mute</b>" in line
    line = _cap(render(None, cap_snap(indicator="not_required", profile="rig", camera="refused:indicator_failed")))
    assert "Recording light: <b>not used</b>" in line and "Camera: <b>refused: the recording light failed</b>" in line
    line = _cap(render(None, cap_snap(indicator="tail", mic="off")))
    assert "on (turning off shortly)" in line and "Microphone: <b>off</b>" in line
    assert "<p id=CAP" not in render(None, _snap(None))                       # no capture state: no line


def test_dashboard_capture_line_is_escaped():
    evil = "<script>alert(1)</script>"
    page = render(None, cap_snap(indicator=evil, mute=evil, mic="refused:" + evil, camera=evil, profile=evil))
    line = _cap(page)
    assert "<script" not in page and "unknown" in line and "refused: unknown reason" in line


def test_refusal_words_cover_every_code_and_pass_the_guard():
    from oiris.capture import REFUSAL_CODES
    assert set(CAPTURE_REFUSALS) == set(REFUSAL_CODES) == set(MIC_REFUSAL_REASONS)
    for text in (*CAPTURE_REFUSALS.values(), *MIC_REFUSAL_REASONS.values()):
        assert not violations(text)
