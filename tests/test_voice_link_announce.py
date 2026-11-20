"""ADR-0058: a clear voice-link match plays a chime and says "sounds like NAME". Synthetic audio and fake
players only: no real sound, microphone, model or network."""
# ruff: noqa: F811
import json, re, subprocess, threading
from types import SimpleNamespace

import numpy as np
import pytest

from oiris.config import VOICE_LINK_DEFAULTS, Config, validate_voice_link
from oiris.dashboard import server
from oiris.dashboard.server import render
from oiris.guard.phrases import MAX_WORDS
from oiris.guard.vocabulary import check_text
from oiris.modules.voice_link import announce, optin
from oiris.modules.voice_link.announce import (AFPLAY, CHIME_SOUND, SAY, Announcer, GenericPlayer, MacPlayer, NullPlayer,
                                               chime_tone, phrase, speakable_name)
from oiris.modules.voice_link.core import SELF_KEY, VoiceLinkConfig, VoiceLinkModule, self_person, speaker_slots
from oiris.registry import Registry
from oiris.slots import SlotManager
from tests.test_dashboard_live import _request
from tests.test_voice_link import (META, SR, FakeSpeaker, cli_args, cli_face_person, count, enroll_face, enroll_voice, face_on,
                                   synth_voice, voice_opt_in)
from tests.test_voice_link import env, store, two  # noqa: F401  (fixture reuse)
from tests.test_voice_link_live import (FakeCapture, FakeVM, enrolled, listener, live_cfg, live_runner, quiet,  # noqa: F401
                                        voice_snap, wait_until)


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class GatedPlayer(NullPlayer):
    """Holds each chime until released, so a test can see a job "playing"."""

    def __init__(self):
        super().__init__()
        self.started, self.release = threading.Event(), threading.Event()

    def chime(self) -> None:
        super().chime()
        self.started.set()
        assert self.release.wait(10.0)


class FailingPlayer:
    def chime(self) -> None:
        raise OSError("no audio device")

    def say(self, text: str) -> None:
        raise AssertionError("not reached")


def settle(a: Announcer) -> None:
    assert wait_until(lambda: a._q.empty() and not a._busy)


# ---- the name rules and the phrase ------------------------------------------------------------------
@pytest.mark.parametrize("name", ["Alex", "Mary-Jane O'Neil", "José", "Zoë Ann", "Иван Петров", "李小龍", "やまだ 太郎",
                                  "अनिल", "D’Arcy", "Ann Lee Van Dyke", "A" * 40])
def test_good_names_are_spoken_as_stored(name):
    assert speakable_name(name) == name


@pytest.mark.parametrize("name", [
    "A" * 41,                        # too many characters
    "Ann Lee Van Dyke Jones",        # five words
    "Robot", "Robo-Bot", "Not Human", "Fake Alex",   # vocabulary guard
    "Alex2", "R2D2",                 # digits
    "Аlex",                     # Cyrillic A + Latin: mixed scripts (a look-alike letter)
    "Alex Иван",                     # mixed scripts across words
    "", " Alex", "Alex ", "Alex  Lee", "-Alex", "Alex!", "Alex\nLee", "Alex_Lee", "Alex\tLee", None, 42,
])
def test_bad_names_give_the_chime_only(name):
    assert speakable_name(name) is None


def test_phrase_says_sounds_like_and_fits_the_budget():
    assert phrase("Alex") == "sounds like Alex"
    longest = "Ann Lee Van Dyke"
    assert speakable_name(longest) and len(phrase(longest).split()) <= MAX_WORDS
    check_text(phrase(longest))


# ---- modes ------------------------------------------------------------------------------------------
def test_mode_chime_and_name():
    p = NullPlayer()
    a = Announcer("chime_and_name", 60, p)
    try:
        assert a.announce("p1", "Alex")
        settle(a)
        assert p.played == [("chime", ""), ("say", "sounds like Alex")]
        assert a.announce("p2", "Robot")                       # name refused: the chime still plays
        settle(a)
        assert p.played[2:] == [("chime", "")]
    finally:
        a.close()


def test_mode_chime_only_and_off():
    p = NullPlayer()
    a = Announcer("chime", 60, p)
    try:
        assert a.announce("p1", "Alex")
        settle(a)
        assert p.played == [("chime", "")]
    finally:
        a.close()
    p = NullPlayer()
    a = Announcer("off", 60, p)
    assert a._worker is None and not a.announce("p1", "Alex") and not a.playing() and p.played == []
    a.close()
    with pytest.raises(ValueError):
        Announcer("loud")


def test_failures_are_logged_not_raised_and_closed_announcer_refuses(caplog):
    a = Announcer("chime_and_name", 60, FailingPlayer())
    try:
        assert a.announce("p1", "Alex")
        settle(a)
        assert "could not be played" in caplog.text
    finally:
        a.close()
    assert not a.announce("p2", "Blake") and not a._worker.is_alive()


# ---- repeat limit and queue -------------------------------------------------------------------------
def test_repeat_limit_per_person_with_injected_clock():
    clk, p = Clock(), NullPlayer()
    a = Announcer("chime_and_name", 60, p, clock=clk, tail_s=0)
    try:
        assert a.announce("p1", "Alex")
        assert a.announce("p2", "Blake")                       # different people back to back
        settle(a)
        clk.t += 59.9
        assert not a.announce("p1", "Alex") and not a.announce("p2", "Blake")
        clk.t += 0.1
        assert a.announce("p1", "Alex")
        settle(a)
        assert [t for k, t in p.played if k == "say"] == ["sounds like Alex", "sounds like Blake", "sounds like Alex"]
    finally:
        a.close()


def test_queue_is_bounded_and_a_dropped_job_is_not_a_repeat():
    p = GatedPlayer()
    a = Announcer("chime_and_name", 60, p, queue_jobs=2)
    try:
        assert a.announce("p1", "Alex")
        assert p.started.wait(5.0)                             # the worker is playing p1
        assert a.announce("p2", "Blake") and a.announce("p3", "Casey")
        assert not a.announce("p4", "Drew")                    # full: dropped
        p.release.set()
        settle(a)
        assert a.announce("p4", "Drew")                        # dropped earlier, so not limited now
        settle(a)
        assert [t for k, t in p.played if k == "say"] == ["sounds like Alex", "sounds like Blake", "sounds like Casey",
                                                           "sounds like Drew"]
    finally:
        p.release.set()
        a.close()


def test_playing_while_queued_playing_and_for_the_tail():
    clk, p = Clock(), GatedPlayer()
    a = Announcer("chime_and_name", 60, p, clock=clk, tail_s=announce.TAIL_S)
    try:
        assert not a.playing()
        assert a.announce("p1", "Alex") and a.playing()
        assert p.started.wait(5.0) and a.playing()
        p.release.set()
        settle(a)
        assert a.playing()                                     # the tail: same clock time as the end
        clk.t += announce.TAIL_S - 0.01
        assert a.playing()
        clk.t += 0.02
        assert not a.playing()
    finally:
        p.release.set()
        a.close()


# ---- players ----------------------------------------------------------------------------------------
class FakeProc:
    """A fake afplay/say: runs until finished, terminated or killed; wait() honours its timeout. With
    ignore_terminate, only kill() ends it. Nothing is started."""

    def __init__(self, argv, finish=True, rc=0, ignore_terminate=False):
        self.argv, self.ignore_terminate = argv, ignore_terminate
        self.done, self.rc = threading.Event(), None
        self.terminated = self.killed = 0
        self.waits = []
        if finish:
            self._exit(rc)

    def _exit(self, rc):
        if self.rc is None:
            self.rc = rc
        self.done.set()

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if not self.done.wait(timeout):
            raise subprocess.TimeoutExpired(self.argv, timeout)
        return self.rc

    def poll(self):
        return self.rc if self.done.is_set() else None

    def terminate(self):
        self.terminated += 1
        if not self.ignore_terminate:
            self._exit(-15)

    def kill(self):
        self.killed += 1
        self._exit(-9)


class FakePopen:
    """Records every start; `make(argv)` builds each FakeProc (default: finishes at once with exit 0)."""

    def __init__(self, make=None):
        self.make = make or (lambda argv: FakeProc(argv))
        self.calls, self.procs = [], []
        self.started = threading.Event()

    def __call__(self, argv, **kw):
        self.calls.append((argv, kw))
        p = self.make(argv)
        self.procs.append(p)
        self.started.set()
        return p


def test_mac_player_uses_argument_lists_absolute_paths_and_timeouts():
    fp = FakePopen()
    mp = MacPlayer(popen=fp)
    mp.chime(); mp.say("sounds like -o Alex")
    assert fp.calls[0][0] == [AFPLAY, CHIME_SOUND] and fp.calls[1][0] == [SAY, "sounds like -o Alex"]
    for cmd, kw in fp.calls:
        assert isinstance(cmd, list) and all(c.startswith("/") for c in cmd[:1]) and "shell" not in kw
        assert kw["stdin"] is kw["stdout"] is kw["stderr"] is subprocess.DEVNULL
    assert fp.procs[0].waits == [announce.CHIME_TIMEOUT_S] and fp.procs[1].waits == [announce.SAY_TIMEOUT_S]
    assert AFPLAY == "/usr/bin/afplay" and SAY == "/usr/bin/say" and CHIME_SOUND.startswith("/System/Library/Sounds/")
    mp.stop()                                                     # nothing playing: no-op
    assert all(p.terminated == 0 for p in fp.procs)


def test_mac_player_failed_command_raises_and_play_logs_it(caplog):
    mp = MacPlayer(popen=FakePopen(lambda argv: FakeProc(argv, rc=1)))
    with pytest.raises(subprocess.CalledProcessError):
        mp.chime()
    assert not announce.play(mp, "sounds like Alex") and "could not be played" in caplog.text


def test_mac_player_timeout_ends_the_command_and_raises(monkeypatch):
    monkeypatch.setattr(announce, "SAY_TIMEOUT_S", 0.05)
    fp = FakePopen(lambda argv: FakeProc(argv, finish=False))
    with pytest.raises(subprocess.TimeoutExpired):
        MacPlayer(popen=fp).say("sounds like Alex")
    p = fp.procs[0]
    assert p.waits[0] == 0.05 and p.terminated == 1 and p.killed == 0 and p.poll() == -15


def test_mac_player_stop_terminates_then_kills_after_the_grace(monkeypatch):
    monkeypatch.setattr(announce, "STOP_GRACE_S", 0.05)
    fp = FakePopen(lambda argv: FakeProc(argv, finish=False, ignore_terminate=True))
    mp, errors = MacPlayer(popen=fp), []

    def speak():
        try:
            mp.say("sounds like Alex")
        except Exception as e:  # noqa: BLE001 - recorded for the assertion below
            errors.append(e)
    t = threading.Thread(target=speak)
    t.start()
    assert fp.started.wait(5.0)
    mp.stop()
    t.join(5.0)
    p = fp.procs[0]
    assert not t.is_alive() and errors == []                       # a stopped sound returns quietly
    assert p.terminated == 1 and p.killed == 1 and p.poll() == -9 and announce.STOP_GRACE_S in p.waits
    assert mp._proc is None


def test_mac_player_barge_in_through_the_arbiter():
    """A higher-priority request stops lower speech on the Mac player; the cut-off phrase is not replayed."""
    from oiris.modules.output import CONFIRMED, PLAYED, VOICE_LINK, OutputArbiter, OutputSettings

    def make(argv):                                               # say runs until stopped; afplay finishes
        return FakeProc(argv, finish=argv[0] != SAY or argv[1] != "sounds like Alex")
    fp = FakePopen(make)
    mp = MacPlayer(popen=fp)
    a = OutputArbiter(mp, settings=OutputSettings(speak_alerts=True), clock=lambda: 0.0)
    try:
        assert a.request("voice_link", VOICE_LINK, "v", text="sounds like Alex", sound="chime").outcome == PLAYED
        assert wait_until(lambda: any(c[0] == [SAY, "sounds like Alex"] for c in fp.calls))
        d = a.request("alert", CONFIRMED, "c", text="Confirmed signal")
        assert d.outcome == PLAYED and "barged in on voice_link" in d.reason
        assert wait_until(lambda: a.counts["completed"] == 1 and not a.state()["speaking"])
        said_ = [c[0] for c in fp.calls if c[0][0] == SAY]
        assert said_ == [[SAY, "sounds like Alex"], [SAY, "Confirmed signal"]]
        alex = next(p for p in fp.procs if p.argv == [SAY, "sounds like Alex"])
        assert alex.terminated == 1 and alex.killed == 0
        assert a.counts["interrupted"] == 1 and a.counts["failed"] == 0   # a stopped sound is not a failure
    finally:
        a.close()


def test_standalone_announcer_close_stops_the_mac_player():
    fp = FakePopen(lambda argv: FakeProc(argv, finish=False))
    a = Announcer("chime", 60, MacPlayer(popen=fp))
    assert a.announce("p1", "Alex") and fp.started.wait(5.0)
    a.close()
    assert fp.procs[0].terminated == 1 and not a._worker.is_alive()


def test_generic_player_chime_and_tts():
    class Sd:
        def __init__(self):
            self.plays = []

        def play(self, x, rate, blocking):
            self.plays.append((x.size, rate, blocking, float(np.max(np.abs(x)))))

    class Tts:
        offline = True

        def __init__(self):
            self.spoken, self.waited = [], []

        def speak(self, text):
            self.spoken.append(text)

        def wait(self, timeout_s):
            self.waited.append(timeout_s)

        def stop(self):
            pass
    sd, tts = Sd(), Tts()
    g = GenericPlayer(sd, tts)
    g.chime(); g.say("sounds like Alex")
    assert sd.plays and sd.plays[0][2] is True and 0 < sd.plays[0][3] <= announce.CHIME_LEVEL
    assert tts.spoken == ["sounds like Alex"] and tts.waited
    GenericPlayer(None, None).chime(); GenericPlayer(sd, None).say("sounds like Alex")   # chime only / nothing: no error
    t = chime_tone()
    assert t.dtype == np.float32 and t.size == 2 * round(announce.CHIME_NOTE_S * announce.CHIME_RATE)


# ---- the listener ------------------------------------------------------------------------------------
def test_listener_zeroes_and_drops_blocks_while_paused():
    vm, cap = FakeVM(), FakeCapture()
    flag = {"on": True}
    li = listener(vm, cap, paused=lambda: flag["on"])
    try:
        cap.feed(np.concatenate([synth_voice("A", seed=3), quiet(1.5)]))
        assert all(not b.any() for b in cap.handed) and li.dropped_while_paused == len(cap.handed)
        assert li._q.empty() and vm.calls == []
        flag["on"] = False
        cap.feed(np.concatenate([synth_voice("A", seed=4), quiet(1.5)]))
        assert wait_until(lambda: len(vm.calls) == 1)
    finally:
        li.stop()


def test_on_heard_carries_the_person_id_but_cards_do_not(two):
    got, cap = [], FakeCapture()
    li = listener(two.vm, cap, on_heard=got.append)
    try:
        cap.feed(np.concatenate([synth_voice("A", seed=50), quiet(1.5)]))
        assert wait_until(lambda: len(got) == 1)
        assert got[0]["person_id"] == two.a and got[0]["name"] == "Alex"
        assert "person_id" not in li.cards()[0]
    finally:
        li.stop()


# ---- self -------------------------------------------------------------------------------------------
def test_self_is_never_logged_shown_counted_or_announced(two):
    db = two.env["db"]
    two.vm.set_self(two.a)
    assert self_person(db) == two.a and two.vm.self_id() == two.a
    p = NullPlayer()
    ann = Announcer("chime_and_name", 60, p)
    seen, cap = [], FakeCapture()
    li = listener(two.vm, cap, on_heard=lambda c: (seen.append(c), ann.announce(c["person_id"], c["name"])),
                  paused=ann.playing)
    try:
        cap.feed(np.concatenate([synth_voice("A", seed=50), quiet(1.5)]))     # self
        cap.feed(np.concatenate([synth_voice("B", seed=51), quiet(1.5)]))     # Blake
        assert wait_until(lambda: len(seen) == 1)
        settle(ann)
        assert [c["name"] for c in seen] == ["Blake"] and li.heard == 1 and [c["name"] for c in li.cards()] == ["Blake"]
        assert count(db, "vl_heard") == 1 and two.vm.stats["heard"] == 1
        assert db.execute("SELECT person_id FROM vl_heard").fetchone()[0] == two.b
        assert [t for k, t in p.played if k == "say"] == ["sounds like Blake"]
    finally:
        li.stop()
        ann.close()
    assert two.vm.process_audio(synth_voice("A", seed=52), SR)["heard"] is None
    assert two.vm.clear_self() and self_person(db) is None and not two.vm.clear_self()
    assert two.vm.process_audio(synth_voice("A", seed=53), SR)["heard"]["name"] == "Alex"


def test_self_needs_an_enrolled_person_and_is_removed_by_wipes(two):
    db, chain = two.env["db"], two.env["chain"]
    with pytest.raises(KeyError):
        two.vm.set_self("nobody")
    two.vm.set_self(two.a)
    assert [r for r in chain.records() if r.get("kind") == "self_set"] and "Alex" not in json.dumps(chain.records())
    two.vm.delete_voice(two.b)
    assert self_person(db) == two.a                            # someone else's voice: the mark stays
    two.vm.delete_voice(two.a)
    assert self_person(db) is None
    enroll_voice(two.vm, two.a, "A", seeds=(4, 5, 6))
    two.vm.set_self(two.a)
    assert two.vm.disable().verified and self_person(db) is None
    db.execute("INSERT INTO vl_state(k, v) VALUES(?, ?)", (SELF_KEY, two.a))
    voice_opt_in(two.env)                                      # a fresh opt-in starts without a self mark
    assert self_person(db) is None


# ---- anti-spoof ---------------------------------------------------------------------------------------
class FakeSpoof:
    name = "fake_spoof"

    def __init__(self, score: float = 0.9, model_dir: str = ""):
        self.score, self.seen = score, []

    def spoof_score(self, audio, sample_rate):
        self.seen.append(audio)
        return self.score


def spoof_vm(env, score: float | None, **cfg):
    reg = Registry()
    reg.register("speaker_embed", FakeSpeaker.name, FakeSpeaker, port_version="1.0", meta=META)
    reg.register("voice_antispoof", FakeSpoof.name, FakeSpoof, port_version="1.0", meta=META)
    face = face_on(env)
    a, b = enroll_face(face, 11, "Alex"), enroll_face(face, 12, "Blake")
    voice_opt_in(env)
    plain = SlotManager(reg, {"speaker_embed": FakeSpeaker.name}, {FakeSpeaker.name: {"model_dir": "unused"}})
    vcfg = VoiceLinkConfig(allow_test_backend=True, enroll_windows=3, **cfg)
    vm = VoiceLinkModule(env["db"], face, plain, vcfg, chain=env["chain"])
    enroll_voice(vm, a, "A"); enroll_voice(vm, b, "B")
    if score is not None:
        sm = SlotManager(reg, {"speaker_embed": FakeSpeaker.name, "voice_antispoof": FakeSpoof.name},
                         {FakeSpeaker.name: {"model_dir": "unused"}, FakeSpoof.name: {"score": score}})
        vm = VoiceLinkModule(env["db"], face, sm, vcfg, chain=env["chain"])
    return vm


def test_high_spoof_score_drops_the_match_without_a_trace(env):
    vm = spoof_vm(env, 0.9)
    res = vm.process_audio(synth_voice("A", seed=50), SR)
    assert res["heard"] is None and not res["unsure"] and count(env["db"], "vl_heard") == 0 and vm.stats["heard"] == 0
    spoof = vm.speaker.get("voice_antispoof")
    assert spoof.seen and all(not s.any() for s in spoof.seen)    # the speech it saw was zeroed afterwards


def test_spoof_score_at_the_threshold_drops_below_proceeds(env):
    vm = spoof_vm(env, 0.5, spoof_threshold=0.5)
    assert vm.process_audio(synth_voice("A", seed=50), SR)["heard"] is None
    vm.speaker.get("voice_antispoof").score = 0.2
    assert vm.process_audio(synth_voice("A", seed=51), SR)["heard"]["name"] == "Alex"
    assert count(env["db"], "vl_heard") == 1


def test_no_spoof_slot_proceeds(env):
    vm = spoof_vm(env, None)
    assert vm.process_audio(synth_voice("A", seed=50), SR)["heard"]["name"] == "Alex"


def test_speaker_slots_selects_the_antispoof_slot_only_when_set(tmp_path):
    reg = Registry()
    vcfg = VoiceLinkConfig()
    cfg = SimpleNamespace(data_dir=str(tmp_path), plugins={})
    assert "voice_antispoof" not in speaker_slots(cfg, vcfg, reg).slots
    cfg.plugins = {"slots": {"voice_antispoof": "aasist"}, "options": {"aasist": {"model_dir": "/m"}}}
    sm = speaker_slots(cfg, vcfg, reg)
    assert sm.slots == {"speaker_embed": "sherpa_campp", "voice_antispoof": "aasist"} and sm.options["aasist"] == {"model_dir": "/m"}


# ---- config and the opt-in notice ----------------------------------------------------------------------
def test_new_config_keys_defaults_and_ranges():
    assert (VOICE_LINK_DEFAULTS["announce"], VOICE_LINK_DEFAULTS["announce_repeat_s"], VOICE_LINK_DEFAULTS["spoof_threshold"]) == \
        ("chime_and_name", 60.0, 0.5)
    for k, ok, bad in (("announce", ["chime_and_name", "chime", "off"], ["loud", "", 1, None]),
                       ("announce_repeat_s", [5, 60, 3600], [4.9, 3601, True, "60"]),
                       ("spoof_threshold", [0.05, 0.5, 0.95], [0.04, 0.96, float("nan")])):
        for v in ok:
            assert not validate_voice_link({k: v}) and VoiceLinkConfig.from_dict({k: v})
        for v in bad:
            assert validate_voice_link({k: v}), (k, v)
            with pytest.raises(ValueError):
                VoiceLinkConfig.from_dict({k: v})


def test_optin_notice_says_names_are_spoken_and_recordings_can_trigger_it():
    s1 = ("When it recognises someone, this device plays a chime and says 'sounds like' and their name out loud; anyone near "
          "the speaker can hear it. You can change this to a chime only or turn it off.")
    s2 = "A recording or phone call of an enrolled person can sound like them: treat a match as a hint, not proof."
    for s in (s1, s2):
        assert s in optin.WARNING_TEXT and check_text(s) == s
    assert optin.WARNING_VERSION == "v2"


# ---- the dashboard card ---------------------------------------------------------------------------------
def test_card_says_sounds_like_name():
    page = render(None, voice_snap(cards=[{"name": "Alex", "score": 0.8, "face_agrees": False, "seconds_ago": 1.0}], heard=1))
    assert "Sounds like Alex</b>" in page and "Heard:" not in page


# ---- live runner wiring ------------------------------------------------------------------------------------
def test_live_runner_announces_with_the_injected_player_and_closes_it(enrolled):
    env, cap, p = enrolled.env, FakeCapture(), NullPlayer()
    r = live_runner(env, live_cfg(env), voice_capture=cap, announce_player=p)
    r.start(background=False)
    try:
        ann, arb = r.announcer, r.output
        assert ann is not None and ann.player is p and r.voice.paused == ann.playing
        # The announcement plays through the live output arbiter (no worker of its own), on the same player.
        assert ann.arbiter is arb and arb.player is p and ann._worker is None
        cap.feed(np.concatenate([synth_voice("A", seed=50), quiet(1.5)]))
        assert wait_until(lambda: ("say", "sounds like Alex") in p.played)
        assert p.played[0] == ("chime", "")
    finally:
        r.stop()
    assert r.announcer is None and ann._stop.is_set() and r.output is None and arb.closed and not arb._worker.is_alive()


def test_live_runner_announce_off_plays_nothing(enrolled):
    env, cap, p = enrolled.env, FakeCapture(), NullPlayer()
    r = live_runner(env, live_cfg(env, announce="off"), voice_capture=cap, announce_player=p)
    r.start(background=False)
    try:
        cap.feed(np.concatenate([synth_voice("A", seed=50), quiet(1.5)]))
        assert wait_until(lambda: r.snapshot()["voice"]["heard"] == 1)
        assert p.played == [] and r.announcer.mode == "off"
    finally:
        r.stop()


def test_listen_button_needs_an_explicit_on_listen(enrolled):
    env = enrolled.env
    r = live_runner(env, live_cfg(env, listening="push_to_talk"), voice_capture=FakeCapture())
    r.start(background=False)
    srv = server.make_server(Config(), port=0, live=r.snapshot, bind_and_activate=False, on_quit=lambda: None)
    try:
        page = _request(srv, b"GET / HTTP/1.0\r\nHost: 127.0.0.1:0\r\n\r\n")[1].decode()
        assert "/voice/listen" not in page                       # no bound-method lookup any more
    finally:
        srv.server_close()
        r.stop()


def test_run_live_wires_on_listen_explicitly(monkeypatch):
    from oiris import cli, live
    from oiris.dashboard import server as srv_mod
    got: dict = {}

    class FakeRunner:
        adapters: dict = {}
        steps, sweep_id, loop_alive_at_stop = 0, None, False

        def __init__(self, cfg, settings):
            self._stop = threading.Event()

        def start(self):
            return self

        def snapshot(self):
            return {}

        def voice_listen(self, seconds=None):
            return True

        def voice_photo(self, key):
            return None

        def wait(self, duration, install_signals=False):
            return "done"

    class FakeServer:
        server_address = ("127.0.0.1", 1)

        def serve_forever(self):
            pass

        def shutdown(self):
            pass

        def server_close(self):
            pass

    def fake_make_server(cfg, **kw):
        got.update(kw)
        return FakeServer()
    monkeypatch.setattr(live, "LiveRunner", FakeRunner)
    monkeypatch.setattr(srv_mod, "make_server", fake_make_server)
    a = SimpleNamespace(duration=0.0, config=None, no_dashboard=False, port=0)
    assert cli._run_live(a, Config()) == 0
    assert got["on_listen"].__func__ is FakeRunner.voice_listen and got["on_quit"].__self__ is not None
    assert got["on_photo"].__func__ is FakeRunner.voice_photo


# ---- CLI ---------------------------------------------------------------------------------------------------
def test_cli_test_sound_plays_once_and_records_nothing(store):
    from oiris.voice_link_cli import run as voice_run
    d, cfgp = store
    out: list[str] = []
    p = NullPlayer()
    assert voice_run(cli_args("test-sound", name="Alex"), Config.load(cfgp), cfgp, out=out.append, player=p) == 0
    assert p.played == [("chime", ""), ("say", "sounds like Alex")] and 'Played the chime and "sounds like Alex"' in out[-1]
    p, out = NullPlayer(), []
    assert voice_run(cli_args("test-sound", name="Robot 9"), Config.load(cfgp), cfgp, out=out.append, player=p) == 0
    assert p.played == [("chime", "")] and "not spoken" in out[0] and out[-1].startswith("Played the chime (")
    p, out = NullPlayer(), []
    assert voice_run(cli_args("test-sound", name=None), Config.load(cfgp), cfgp, out=out.append, player=p) == 0
    assert p.played == [("chime", "")]
    assert voice_run(cli_args("test-sound", name="Alex"), Config.load(cfgp), cfgp, out=out.append, player=FailingPlayer()) == 1
    assert not (d / "asb.db").exists() and not (d / "keys.json").exists()   # no store opened, nothing recorded


def test_cli_set_self_clear_self_and_status(store):
    from oiris.face_match_cli import run as face_run
    from oiris.voice_link_cli import run as voice_run
    from tests.test_voice_link import InjectedBackend, _inj_registry, slots
    d, cfgp = store
    sm = lambda: slots("inj", reg=_inj_registry())   # noqa: E731
    face_run(cli_args("enable", mode=["personal"]), Config.load(cfgp), cfgp, inp=lambda p: "I UNDERSTAND", out=[].append)
    voice_run(cli_args("enable"), Config.load(cfgp), cfgp, inp=lambda p: "I UNDERSTAND", out=[].append)
    face_run(cli_args("consent", subject="S-01", id="c-cli", scope="face_enrollment,voice_enrollment"), Config.load(cfgp), cfgp,
             inp=lambda p: "YES", out=[].append)
    pid = cli_face_person(d)
    out: list[str] = []
    kw = dict(out=out.append, face_backend=InjectedBackend())
    assert voice_run(cli_args("status"), Config.load(cfgp), cfgp, out=out.append, speaker=sm()) == 0
    st = json.loads(out[-1])
    assert st["announce"] == "chime_and_name" and st["self_set"] == "no"
    assert voice_run(cli_args("set-self"), Config.load(cfgp), cfgp, speaker=sm(), **kw) == 1
    assert voice_run(cli_args("set-self", person="zzzz"), Config.load(cfgp), cfgp, speaker=sm(), **kw) == 1 and out[-1] == "no such person"
    assert voice_run(cli_args("set-self", person=pid[:6]), Config.load(cfgp), cfgp, speaker=sm(), **kw) == 0
    assert out[-1].startswith("Self is set")
    assert voice_run(cli_args("status"), Config.load(cfgp), cfgp, out=out.append, speaker=sm()) == 0
    assert json.loads(out[-1])["self_set"] == "yes" and "Alex" not in out[-1] and pid not in out[-1]
    assert voice_run(cli_args("clear-self"), Config.load(cfgp), cfgp, speaker=sm(), **kw) == 0 and out[-1] == "Self cleared."
    assert voice_run(cli_args("clear-self"), Config.load(cfgp), cfgp, speaker=sm(), **kw) == 0 and out[-1] == "No self person was set."


def test_cli_parses_name_for_test_sound(monkeypatch, tmp_path):
    from oiris.cli import main
    got: dict = {}
    monkeypatch.setattr("oiris.voice_link_cli.run", lambda a, cfg, p: got.setdefault("a", a) and 0)
    main(["voice-link", "test-sound", "--name", "Mary-Jane", "--config", str(tmp_path / "none.json")])
    assert (got["a"].sub, got["a"].name) == ("test-sound", "Mary-Jane")


def test_announce_module_has_no_network_or_shell():
    src = (announce.__file__ and open(announce.__file__, encoding="utf-8").read())
    assert "shell=True" not in src and not re.search(r"\bimport (socket|urllib|requests|http)\b", src)
