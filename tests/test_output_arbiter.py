"""The output arbiter (oiris/modules/output.py): priorities, barge-in, dedupe, the attention budget, the queue,
the motion gate, and its wiring into VoiceOutput, the voice-link Announcer and live mode. Fake players and
clocks only: no real sound, microphone or network."""
# ruff: noqa: F811
import threading
import time

import numpy as np
import pytest

from oiris.config import OUTPUT_DEFAULTS, Config, ConfigError, validate_output
from oiris.modules import output as O
from oiris.modules.motion import FAST, STILL, UNKNOWN, WALKING
from oiris.modules.output import (ATTENTION, CONFIRMED, DEDUPED, DROPPED, FYI, PANIC, PLAYED, QUEUED, SUPPRESSED,
                                  VOICE_LINK, NullOutputPlayer, OutputArbiter, OutputSettings, alert)
from oiris.modules.voice import NullTts, VoiceOutput
from oiris.modules.voice_link.announce import Announcer
from tests.test_live import SIM, _runner, _step_until, live_env  # noqa: F401  (fixture reuse)
from tests.test_voice_link import synth_voice
from tests.test_voice_link_live import FakeCapture, FakeVM, listener, quiet, wait_until


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class Gated(NullOutputPlayer):
    """say() blocks until released or stopped (barge-in), so a test can see speech "playing"."""

    def __init__(self):
        super().__init__()
        self.started, self.release = threading.Event(), threading.Event()

    def say(self, text):
        super().say(text)
        self.started.set()
        assert self.release.wait(10.0)

    def stop(self):
        super().stop()
        self.release.set()


class Boom:
    def chime(self):
        raise OSError("no audio device")

    def say(self, text):
        raise OSError("no audio device")


def arb(motion=STILL, start=False, **kw):
    clk = kw.pop("clock", None) or Clock()
    p = kw.pop("player", None) or NullOutputPlayer()
    s = OutputSettings(**{"speak_alerts": True, **kw.pop("settings", {})})
    a = OutputArbiter(p, settings=s, motion=(motion if callable(motion) else lambda: motion), clock=clk,
                      start=start, **kw)
    return a, p, clk


def said(p):
    return [t for k, t in p.played if k == "say"]


# ---- priorities, barge-in, dedupe --------------------------------------------------------------------
def test_priority_order_and_named_constants():
    assert PANIC > CONFIRMED > ATTENTION > VOICE_LINK > FYI
    a, p, _ = arb(queue_jobs=8)                                    # room for all (eviction is tested below)
    assert a.request("vl", VOICE_LINK, "v", text="sounds like Alex").outcome == PLAYED
    assert a.request("alert", ATTENTION, "a", text="Medium signal").outcome == PLAYED
    assert a.request("alert", CONFIRMED, "c", text="Confirmed signal").outcome == PLAYED
    assert a.request("alert", PANIC, "p", text="Locked").outcome == PLAYED
    d = a.request("alert", ATTENTION, "a2", text="Low signal")
    assert d.outcome == QUEUED and "attention budget" in d.reason and str(d) == QUEUED
    # attention a2 waits for its budget slot; everything else plays highest first
    assert a.pump() == 4 and said(p) == ["Locked", "Confirmed signal", "Medium signal", "sounds like Alex"]
    assert a.request("vl", VOICE_LINK, "v2", text="sounds like Blake").outcome == PLAYED
    assert a.request("alert", CONFIRMED, "c2", text="Confirmed signal").outcome == PLAYED
    assert a.request("vl", VOICE_LINK, "v3", text="sounds like Casey").outcome == QUEUED   # behind higher speech
    assert a.pump() == 3 and said(p)[4:] == ["Confirmed signal", "sounds like Blake", "sounds like Casey"]


def test_higher_priority_barges_in_on_lower_speech_equal_and_lower_wait():
    p = Gated()
    a, _, _ = arb(player=p, start=True, clock=lambda: 0.0)
    try:
        assert a.request("vl", VOICE_LINK, "v1", text="sounds like Alex").outcome == PLAYED
        assert p.started.wait(5.0) and a.playing() and a.state()["speaking"] == "voice_link"
        assert a.request("vl", VOICE_LINK, "v2", text="sounds like Blake").outcome == QUEUED     # equal: waits
        assert a.request("fyi", FYI, "f", haptic=True).outcome == DROPPED                          # no haptic device
        assert p.stops == 0
        d = a.request("alert", CONFIRMED, "c", text="Confirmed signal")
        assert d.outcome == PLAYED and "barged in on voice_link" in d.reason and p.stops == 1
        assert wait_until(lambda: said(p)[-1:] == ["sounds like Blake"] and not a.state()["speaking"])
        assert said(p) == ["sounds like Alex", "Confirmed signal", "sounds like Blake"]
        assert a.counts["interrupted"] == 1                         # the cut-off one is not replayed
    finally:
        p.release.set()
        a.close()


def test_dedupe_windows_pending_keys_and_exemptions():
    a, p, clk = arb(settings={"dedupe_s": 30.0})
    assert a.request("alert", CONFIRMED, "k", text="Confirmed signal").outcome == PLAYED
    d = a.request("alert", CONFIRMED, "k", text="Confirmed signal")
    assert d.outcome == DEDUPED and "already queued" in d.reason
    a.pump()
    clk.t += 29.9
    assert a.request("alert", CONFIRMED, "k", text="Confirmed signal").outcome == DEDUPED
    clk.t += 0.2
    assert a.request("alert", CONFIRMED, "k", text="Confirmed signal").outcome == PLAYED
    a.pump()
    for pr, key in ((PANIC, "p"), (VOICE_LINK, "v")):          # not windowed (the Announcer has its own limit)
        assert a.request("x", pr, key, text="Locked").outcome == PLAYED
        a.pump()
        assert a.request("x", pr, key, text="Locked").outcome == PLAYED
        a.pump()
    assert a.counts[DEDUPED] == 2


def test_attention_budget_one_utterance_per_interval():
    a, p, clk = arb(settings={"attention_interval_s": 10.0}, tail_s=0.0)
    assert a.request("alert", ATTENTION, "a1", text="Medium signal").outcome == PLAYED
    d = a.request("alert", ATTENTION, "a2", text="Low signal")
    assert d.outcome == QUEUED and "attention budget" in d.reason
    assert a.pump() == 1 and said(p) == ["Medium signal"]
    assert not a.playing()                                       # a job waiting for its slot does not pause input
    assert a.request("vl", VOICE_LINK, "v", text="sounds like Alex").outcome == PLAYED   # other priorities unaffected
    assert a.pump() == 1
    clk.t += 9.9
    assert a.pump() == 0
    clk.t += 0.1
    assert a.pump() == 1 and said(p)[-1] == "Low signal"
    # The popup channel is not budgeted: both attention pop-ups showed at once.
    assert a.counts[PLAYED] == 2 and a.counts[QUEUED] == 1


# ---- queue, worker, playing() ----------------------------------------------------------------------------
def test_queue_is_bounded_lowest_priority_evicted_or_new_dropped():
    a, p, _ = arb(queue_jobs=2)
    assert a.request("vl", VOICE_LINK, "v1", text="sounds like Alex").outcome == PLAYED
    assert a.request("vl", VOICE_LINK, "v2", text="sounds like Blake").outcome == QUEUED
    d = a.request("vl", VOICE_LINK, "v3", text="sounds like Casey")
    assert d.outcome == DROPPED and d.reason == "queue full"
    assert a.request("alert", CONFIRMED, "c", text="Confirmed signal", ).outcome == PLAYED   # evicts v2
    assert a.counts["evicted"] == 1 and a.pump() == 2
    assert said(p) == ["Confirmed signal", "sounds like Alex"]


def test_worker_plays_and_stops_cleanly_even_while_playing():
    p = Gated()
    a, _, _ = arb(player=p, start=True)
    assert a._worker.is_alive() and a._worker.daemon
    assert a.request("vl", VOICE_LINK, "v", text="sounds like Alex").outcome == PLAYED
    assert p.started.wait(5.0)
    a.request("vl", VOICE_LINK, "w", text="sounds like Blake")
    a.close(join_s=5.0)
    assert not a._worker.is_alive() and p.stops == 1 and a.closed and a.state()["queued"] == 0
    assert said(p) == ["sounds like Alex"]                         # the queued one was dropped, not played
    d = a.request("vl", VOICE_LINK, "x", text="sounds like Casey")
    assert d.outcome == DROPPED and d.reason == "closed"
    a.close()                                                      # idempotent


def test_playing_covers_queued_playing_and_the_tail():
    p, clk = Gated(), Clock()
    a, _, _ = arb(player=p, start=True, clock=clk, tail_s=O.TAIL_S)
    try:
        assert not a.playing()
        assert a.request("vl", VOICE_LINK, "v", text="sounds like Alex").outcome == PLAYED and a.playing()
        assert p.started.wait(5.0) and a.playing()
        p.release.set()
        assert wait_until(lambda: a.counts["completed"] == 1)
        assert a.playing()                                         # the tail
        clk.t += O.TAIL_S - 0.01
        assert a.playing()
        clk.t += 0.02
        assert not a.playing()
    finally:
        p.release.set()
        a.close()


def test_stale_jobs_expire_and_cancel_drops_a_key():
    a, p, clk = arb(settings={"attention_interval_s": 60.0})
    a.request("alert", ATTENTION, "a1", text="Medium signal")
    a.request("alert", ATTENTION, "a2", text="Low signal")         # its slot is 60 s away
    a.pump()
    clk.t += O.QUEUE_MAX_AGE_S + 31
    assert a.pump() == 0 and a.counts["expired"] == 1 and said(p) == ["Medium signal"]
    a.request("alert", CONFIRMED, "c", text="Confirmed signal")
    assert a.cancel("c") == 1 and a.pump() == 0 and a.popup() is None


def test_failures_are_logged_never_raised(caplog):
    def bad_haptic():
        raise RuntimeError("motor")

    def bad_motion():
        raise RuntimeError("imu")
    a, _, _ = arb(player=Boom(), motion=bad_motion, haptic=bad_haptic)
    d = a.request("alert", ATTENTION, "a", text="Medium signal", sound="chime", haptic=True)
    assert d.outcome == PLAYED                                      # broken motion counts as unknown, i.e. still
    assert "haptic: haptic failed" in d.reason or "haptic dropped (haptic failed)" in d.reason
    assert a.pump() == 1 and a.counts["failed"] == 1 and "could not be played" in caplog.text
    assert a.request("x", 99, "k", text="Locked").outcome == DROPPED                 # unknown priority
    assert a.request("x", PANIC, "k", sound="bell").outcome == DROPPED              # unknown sound
    assert a.request("x", PANIC, "k", text="robot detected").outcome == DROPPED     # vocabulary guard
    assert a.request("x", PANIC, "k", text=42).outcome == DROPPED                   # type: ignore[arg-type]


# ---- channels and the pop-up ------------------------------------------------------------------------------
def test_channels_per_priority_and_speak_alerts():
    hap = []
    a, p, _ = arb(settings={"speak_alerts": False}, haptic=lambda: hap.append(1))
    assert a.request("alert", ATTENTION, "a", text="Medium signal").outcome == PLAYED     # pop-up only
    assert a.pump() == 0 and said(p) == [] and a.popup()["text"] == "Medium signal"
    assert a.request("vl", VOICE_LINK, "v", text="sounds like Alex", sound="chime").outcome == PLAYED
    assert a.pump() == 1 and p.played == [("chime", ""), ("say", "sounds like Alex")]
    assert a.popup()["text"] == "Medium signal"                                          # a match never pops up
    assert a.request("fyi", FYI, "f", text="Low signal").outcome == DROPPED               # fyi is haptic only
    assert a.request("fyi", FYI, "f", haptic=True).outcome == PLAYED and hap == [1]


def test_popup_priority_and_expiry():
    a, _, clk = arb(settings={"popup_s": 6.0, "speak_alerts": False})
    a.request("alert", CONFIRMED, "c", text="Confirmed signal")
    d = a.request("alert", ATTENTION, "a", text="Medium signal")
    assert d.outcome == DROPPED and "higher pop-up" in d.reason
    st = a.state()
    assert st["popup"] == {"text": "Confirmed signal", "priority": "confirmed", "expires_in_s": 6.0}
    clk.t += 6.0
    assert a.popup() is None
    assert a.request("alert", ATTENTION, "a", text="Medium signal").outcome == PLAYED
    assert a.request("alert", PANIC, "p", text="Locked").outcome == PLAYED and a.popup()["priority"] == "panic"


# ---- the motion gate --------------------------------------------------------------------------------------
def test_motion_policy_still_and_unknown_allow_everything():
    for m in (STILL, UNKNOWN):
        a, p, _ = arb(motion=m)
        assert a.request("alert", ATTENTION, "a", text="Medium signal").outcome == PLAYED
        assert a.popup() is not None and a.pump() == 1


def test_motion_policy_walking_hides_popups_speech_allowed():
    a, p, _ = arb(motion=WALKING)
    d = a.request("alert", ATTENTION, "a", text="Medium signal")
    assert d.outcome == PLAYED and "no pop-ups while walking" in d.reason and a.popup() is None
    assert a.pump() == 1 and said(p) == ["Medium signal"]
    b, _, _ = arb(motion=WALKING, settings={"speak_alerts": False})
    d = b.request("alert", ATTENTION, "a", text="Medium signal")
    assert d.outcome == SUPPRESSED                                       # nothing else to play
    c, _, _ = arb(motion=WALKING, settings={"popups_while_walking": True})
    assert c.request("alert", ATTENTION, "a", text="Medium signal").outcome == PLAYED and c.popup() is not None


def test_motion_policy_fast_only_panic_and_confirmed_and_no_popups():
    a, p, _ = arb(motion=FAST)
    for pr in (ATTENTION, VOICE_LINK):
        d = a.request("x", pr, f"k{pr}", text="Medium signal")
        assert d.outcome == SUPPRESSED and d.reason == "moving fast"
    assert a.request("x", FYI, "f", haptic=True).outcome == SUPPRESSED
    assert a.request("x", CONFIRMED, "c", text="Confirmed signal").outcome == PLAYED
    assert a.request("x", PANIC, "p", text="Locked").outcome == PLAYED
    assert a.popup() is None and a.pump() == 2 and said(p) == ["Locked", "Confirmed signal"]
    b, _, _ = arb(motion=FAST, settings={"fast_min_priority": "panic"})
    assert b.request("x", CONFIRMED, "c", text="Confirmed signal").outcome == SUPPRESSED


def test_suppressed_items_are_dropped_not_replayed_later():
    m = {"s": STILL}
    a, p, _ = arb(motion=lambda: m["s"])
    a.request("vl", VOICE_LINK, "v", text="sounds like Alex")
    a.request("alert", ATTENTION, "a", text="Medium signal")
    assert a.popup() is not None
    m["s"] = WALKING
    assert a.popup() is None                                             # walking hides the pop-up at once
    m["s"] = FAST
    assert a.pump() == 0 and a.counts["suppressed_in_queue"] == 2
    m["s"] = STILL
    assert a.pump() == 0 and said(p) == [] and a.state()["queued"] == 0


def test_alert_helper_maps_engine_levels():
    hap = []
    a, p, _ = arb(settings={"speak_alerts": False}, haptic=lambda: hap.append(1))
    assert alert(a, "fyi", "A4").outcome == PLAYED and hap == [1] and a.popup() is None
    assert alert(a, "attention", "A1").outcome == PLAYED and a.popup()["text"] == "Medium signal"
    assert alert(a, "attention", "A1").outcome == DEDUPED
    assert alert(a, "confirmed", "A1").outcome == PLAYED and a.popup()["priority"] == "confirmed"
    assert alert(a, "loud", "A1").outcome == DROPPED and a.pump() == 0 and said(p) == []
    from oiris.ui import tokens as T
    levels = T.load()["alert_levels"]
    assert O.ALERT_WORDS == {"fyi": levels["low"]["word"], "attention": levels["medium"]["word"],
                             "confirmed": levels["confirmed"]["word"]}


# ---- config -------------------------------------------------------------------------------------------------
def test_output_config_defaults_validation_and_settings_in_sync():
    import dataclasses
    assert OUTPUT_DEFAULTS["speak_alerts"] is False and OUTPUT_DEFAULTS["popups_while_walking"] is False
    assert set(OUTPUT_DEFAULTS) == {f.name for f in dataclasses.fields(OutputSettings)} - {"attention_interval_s"}
    s = OutputSettings.from_config(Config(voice_attention_min_interval_s=12.0).validate())
    assert (s.speak_alerts, s.fast_min, s.attention_interval_s) == (False, CONFIRMED, 12.0)
    for bad in ({"speak_alerts": "yes"}, {"fast_min_priority": "attention"}, {"popup_s": 0.5}, {"dedupe_s": -1},
                {"popup_s": float("nan")}, {"loud": True}, {"popups_while_walking": 1}):
        assert validate_output(bad), bad
        with pytest.raises(ConfigError):
            Config(output=bad).validate()
        with pytest.raises(ValueError):
            OutputSettings.from_config(bad)
    assert validate_output("x") == ["output must be an object"]
    assert not validate_output({"speak_alerts": True, "fast_min_priority": "panic", "popup_s": 30, "dedupe_s": 0})
    with pytest.raises(ConfigError):
        Config(voice_attention_min_interval_s=-1).validate()


# ---- VoiceOutput -------------------------------------------------------------------------------------------
def test_voice_output_without_an_arbiter_is_unchanged():
    t, clk = NullTts(), Clock()
    v = VoiceOutput(t, clock=clk)
    assert v.arbiter is None
    assert v.announce("attention", "rf_near") == "spoken" and v.announce("attention", "eye") == "queued"
    assert v.announce("confirmed", "panic", confirm_token=True) == "barged_in" and t.stopped == 1


def test_voice_output_through_the_arbiter():
    hap = []
    a, p, clk = arb(haptic=lambda: hap.append(1), tail_s=0.0)
    t = NullTts()
    v = VoiceOutput(t, arbiter=a)
    assert v.announce("fyi", "rf_near") == PLAYED and hap == [1]
    assert v.announce("attention", "rf_near") == PLAYED
    assert v.announce("attention", "eye") == QUEUED                     # the M-06 budget, now in the arbiter
    with pytest.raises(PermissionError):
        v.announce("confirmed", "panic")
    assert v.announce("confirmed", "panic", confirm_token=True) == PLAYED
    a.pump()
    assert said(p) == ["Locked", "Radio device nearby"] and t.spoken == []     # the arbiter's player, not tts
    v.tick()                                                             # unacknowledged: repeat, paced by dedupe
    assert a.pump() == 0
    clk.t += a.settings.dedupe_s
    v.tick()
    assert a.pump() >= 1 and said(p).count("Locked") == 2
    v.acknowledge()
    clk.t += a.settings.dedupe_s
    v.tick()
    assert a.pump() == 0 and said(p).count("Locked") == 2


# ---- the voice-link Announcer -----------------------------------------------------------------------------
def test_announcer_through_the_arbiter_chime_name_and_repeat_limit():
    a, p, clk = arb(tail_s=0.0)
    ann = Announcer("chime_and_name", 60, clock=clk, arbiter=a)
    try:
        assert ann._worker is None and ann.player is p
        assert ann.announce("p1", "Alex") and ann.announce("p2", "Blake")
        a.pump()
        assert p.played == [("chime", ""), ("say", "sounds like Alex"), ("chime", ""), ("say", "sounds like Blake")]
        clk.t += 59.9
        assert not ann.announce("p1", "Alex")                          # the repeat limit stays in the Announcer
        clk.t += 0.1
        assert ann.announce("p1", "Robot")                             # name refused: the chime alone
        a.pump()
        assert p.played[-1] == ("chime", "")
        assert "person_id" not in str(a.state()) and "p1" not in str(a.state())
    finally:
        ann.close()
    assert not ann.announce("p3", "Casey") and not a.closed           # closing the Announcer leaves the arbiter
    chime = Announcer("chime", 60, arbiter=a)
    assert chime.announce("p9", "Alex") and a.pump() == 1 and p.played[-1] == ("chime", "")


def test_announcer_suppressed_while_fast_is_not_a_repeat():
    m = {"s": FAST}
    a, p, clk = arb(motion=lambda: m["s"])
    ann = Announcer("chime_and_name", 60, clock=clk, arbiter=a)
    assert not ann.announce("p1", "Alex") and a.counts[SUPPRESSED] == 1
    m["s"] = STILL
    assert ann.announce("p1", "Alex") and a.pump() == 1 and said(p) == ["sounds like Alex"]


def test_announcer_playing_drives_the_listener_input_pause():
    p = Gated()
    a, _, _ = arb(player=p, start=True, clock=time.monotonic)     # the real clock: the tail must run out
    ann = Announcer("chime_and_name", 60, arbiter=a)
    vm, cap = FakeVM(), FakeCapture()
    li = listener(vm, cap, paused=ann.playing)
    try:
        assert ann.announce("p1", "Alex") and p.started.wait(5.0) and ann.playing()
        cap.feed(np.concatenate([synth_voice("A", seed=3), quiet(1.5)]))
        assert li.dropped_while_paused == len(cap.handed) and all(not b.any() for b in cap.handed) and vm.calls == []
        p.release.set()
        assert wait_until(lambda: not ann.playing())
        cap.feed(np.concatenate([synth_voice("A", seed=4), quiet(1.5)]))
        assert wait_until(lambda: len(vm.calls) == 1)
    finally:
        li.stop()
        p.release.set()
        a.close()


# ---- live mode ----------------------------------------------------------------------------------------------
IMU = {"name": "imu", "args": {"profile": "walking", "period_s": 0.01}, "options": {}}


def test_live_snapshot_output_without_an_imu(live_env):
    p = NullOutputPlayer()
    r = _runner(live_env, [SIM], announce_player=p)
    r.start(background=False)
    try:
        _step_until(r, 3)
        assert wait_until(lambda: r.snapshot()["output"]["counts"][PLAYED] >= 1 or r.steps > 20, timeout=10)
        out = r.snapshot()["output"]
        assert out["motion"] == UNKNOWN and out["imu"] is False and r.output.player is p
        assert out["popup"]["text"] in O.ALERT_WORDS.values() and out["popup"]["expires_in_s"] > 0
        assert set(O.OUTCOMES) <= set(out["counts"]) and out["counts"][PLAYED] >= 1
        assert p.played == []                                       # speak_alerts is off: the Mac stays silent
    finally:
        r.stop()
    out = r.snapshot()["output"]
    assert r.output is None and out["popup"] is None and out["counts"][PLAYED] >= 1


def test_live_imu_feeds_motion_and_walking_hides_popups(live_env):
    p = NullOutputPlayer()
    r = _runner(live_env, [SIM, IMU], announce_player=p)
    r.start(background=False)
    try:
        assert r.adapters["imu"]["available"]
        assert wait_until(lambda: r.motion.state() == WALKING, timeout=10)
        _step_until(r, r.steps + 5)
        out = r.snapshot()["output"]
        assert out["motion"] == WALKING and out["imu"] is True and out["popup"] is None
        assert out["counts"][SUPPRESSED] >= 1 and out["counts"][PLAYED] == 0
    finally:
        r.stop()


def test_live_config_output_section_is_used(live_env):
    cfg, reg, tmp = live_env
    live_env2 = (Config(data_dir=cfg.data_dir, output={"speak_alerts": True}).validate(), reg, tmp)
    p = NullOutputPlayer()
    r = _runner(live_env2, [SIM], announce_player=p)
    r.start(background=False)
    try:
        assert r.output.settings.speak_alerts is True
        _step_until(r, 3)
        assert wait_until(lambda: said(p) != [] or r.steps > 50, timeout=10) and said(p)[0] in O.ALERT_WORDS.values()
    finally:
        r.stop()
