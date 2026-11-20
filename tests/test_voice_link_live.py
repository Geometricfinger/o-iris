"""M-17 live listening (ADR-0057, docs/VOICE_LINK_MODULE_SPEC.md "Live listening"). Synthetic audio only
(the tones and fake speaker of tests/test_voice_link.py); the microphone is a fake capture that feeds
blocks by hand. No real microphone, model or network."""
# ruff: noqa: F811
import re, socket, sys, threading, time
from types import SimpleNamespace

import numpy as np
import pytest

from oiris.config import Config
from oiris.dashboard import server
from oiris.dashboard.server import render
from oiris.live import LiveRunner, LiveSettings, OutboundBlocked
from oiris.modules.voice_link import capture as vcap
from oiris.modules.voice_link.capture import (SILENCE_NOTICE, CaptureUnavailable, DigitalSilence, SoundDeviceCapture,
                                              resolve_device)
from oiris.modules.voice_link.announce import NullPlayer
from oiris.modules.voice_link.core import VoiceLinkConfig
from oiris.modules.voice_link.listener import END_QUIET_S, FRAME_S, MAX_UTTERANCE_S, PTT_DEFAULT_S, VoiceListener
from oiris.registry import default_registry
from tests.test_dashboard_live import _request, _snap
from tests.test_live import SIM, _step_until
from tests.test_voice_link import (SR, FakeSd, InjectedBackend, _inj_registry, cli_args, cli_face_person, count,
                                   enroll_face, enroll_voice, enter_skip_redo, face_on, open_store, slots, synth_voice,
                                   voice_on, voice_opt_in)
from tests.test_voice_link import env, store, two  # noqa: F401  (fixture reuse)

WAIT_S = 10.0


def quiet(seconds: float, seed: int = 99, sd: float = 1e-3) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, sd, int(seconds * SR)).astype(np.float32)


def wait_until(cond, timeout: float = WAIT_S) -> bool:
    t0 = time.monotonic()
    while not cond():
        if time.monotonic() - t0 > timeout:
            return False
        time.sleep(0.01)
    return True


def drained(listener: VoiceListener) -> None:
    """Wait until the worker has taken every queued block, then give it a moment to finish the last one."""
    assert wait_until(listener._q.empty)
    time.sleep(0.3)


class FakeCapture:
    """Stands in for the microphone: start/stop like SoundDeviceCapture; feed() hands blocks to the callback."""
    sample_rate = SR

    def __init__(self, fail: Exception | None = None, block: int = 1600):
        self.fail, self.block = fail, block
        self.cb = None
        self.starts = self.stops = 0
        self.handed: list[np.ndarray] = []

    def start(self, cb) -> None:
        if self.fail is not None:
            raise self.fail
        self.cb, self.starts = cb, self.starts + 1

    def stop(self) -> None:
        self.cb, self.stops = None, self.stops + 1

    def feed(self, audio: np.ndarray) -> None:
        for i in range(0, audio.size, self.block):
            cb = self.cb
            if cb is None:
                return
            b = audio[i:i + self.block].astype(np.float32).copy()
            self.handed.append(b)
            cb(b)


class FakeVM:
    """Records process_audio calls (length, lock held) and keeps the buffers to check they get zeroed."""

    def __init__(self, lock=None, **cfg):
        self.cfg = VoiceLinkConfig(allow_test_backend=True, **cfg)
        self.lock, self.calls, self.bufs, self.locked = lock, [], [], []

    def health(self):
        return "listening", ""

    def process_audio(self, audio, sample_rate, place=None, now=None):
        self.locked.append(self.lock.locked() if self.lock is not None else None)
        self.calls.append(audio.size / sample_rate)
        self.bufs.append(audio)
        return {"heard": None, "unsure": False, "speech_s": 0.0, "health": "listening"}


TEST_QUEUE_BLOCKS = 1000   # the fake feeds far faster than real time; the default queue would drop blocks


def listener(vm, cap=None, **kw) -> VoiceListener:
    kw.setdefault("queue_blocks", TEST_QUEUE_BLOCKS)
    return VoiceListener(vm, cap or FakeCapture(), **kw).start()


# ---- utterance cutting ------------------------------------------------------------------------
def test_two_utterances_separated_by_quiet_give_two_calls():
    vm, cap = FakeVM(), FakeCapture()
    li = listener(vm, cap)
    try:
        cap.feed(np.concatenate([synth_voice("A", seconds=3.5, seed=1), quiet(1.5), synth_voice("B", seconds=3.5, seed=2),
                                 quiet(1.5)]))
        assert wait_until(lambda: len(vm.calls) == 2)
        drained(li)
        assert len(vm.calls) == 2 and all(2.5 < s < 3.0 + END_QUIET_S + 0.8 for s in vm.calls), vm.calls
    finally:
        li.stop()


def test_long_utterance_is_split_at_the_maximum():
    vm, cap = FakeVM(), FakeCapture()
    li = listener(vm, cap)
    try:
        cap.feed(np.concatenate([synth_voice("A", seconds=20.0, seed=3), quiet(1.5)]))
        assert wait_until(lambda: len(vm.calls) == 3)
        drained(li)
        assert len(vm.calls) == 3
        assert all(MAX_UTTERANCE_S <= s <= MAX_UTTERANCE_S + FRAME_S for s in vm.calls[:2]), vm.calls
        assert vm.calls[2] < MAX_UTTERANCE_S
    finally:
        li.stop()


def test_quiet_room_and_clicks_are_not_processed():
    vm, cap = FakeVM(), FakeCapture()
    li = listener(vm, cap)
    try:
        click = quiet(2.0)
        click[SR:SR + int(0.06 * SR)] += 0.3                # 60 ms click: below MIN_VOICED_S
        cap.feed(np.concatenate([quiet(3.0), click, quiet(1.0)]))
        drained(li)
        assert vm.calls == []
    finally:
        li.stop()


def test_every_buffer_is_zeroed_after_processing():
    vm, cap = FakeVM(), FakeCapture()
    li = listener(vm, cap)
    try:
        cap.feed(np.concatenate([synth_voice("A", seconds=3.5, seed=4), quiet(1.5)]))
        assert wait_until(lambda: len(vm.calls) == 1)
        drained(li)
        assert vm.bufs and all(not b.any() for b in vm.bufs)                # utterances handed to the core
        assert cap.handed and all(not b.any() for b in cap.handed)          # every block the capture handed over
        assert not li._buf.any()                                             # the utterance buffer, between utterances
    finally:
        li.stop()
    assert not li._buf.any() and not li._pre.any() and not li._frame.any()  # the pre-roll ring too, once stopped


# ---- lock, stop, closed store -----------------------------------------------------------------
def test_process_audio_runs_under_the_lock_and_not_after_the_store_closes():
    lock = threading.Lock()
    closed = {"v": False}
    vm, cap = FakeVM(lock=lock), FakeCapture()
    li = listener(vm, cap, process_lock=lock, active=lambda: not closed["v"])
    try:
        cap.feed(np.concatenate([synth_voice("A", seconds=3.5, seed=5), quiet(1.5)]))
        assert wait_until(lambda: len(vm.calls) == 1)
        assert vm.locked == [True]
        closed["v"] = True                                                   # the store is closed
        cap.feed(np.concatenate([synth_voice("A", seconds=3.5, seed=6), quiet(1.5)]))
        drained(li)
        assert len(vm.calls) == 1
    finally:
        li.stop()


def test_nothing_processed_after_stop_and_queues_zeroed():
    vm, cap = FakeVM(), FakeCapture()
    li = listener(vm, cap)
    li.stop()
    assert cap.starts == 1 and cap.stops == 1 and cap.cb is None
    late = synth_voice("A", seed=7)
    li._on_block(late)                                                       # a block arriving after stop
    assert not late.any() and li._q.empty()
    assert not li._worker.is_alive() and vm.calls == []
    assert li.status()["state"] == "unavailable" and li.listen_for() is False
    vm2, cap2 = FakeVM(), FakeCapture()
    li2 = VoiceListener(vm2, cap2, queue_blocks=2)                           # not started: blocks stay queued
    blocks = [synth_voice("A", seconds=0.5, seed=s) for s in (1, 2, 3)]
    for b in blocks:
        li2._on_block(b)
    assert li2._q.qsize() == 2 and not blocks[2].any()                       # a full queue drops the block, zeroed
    li2.stop()
    assert all(not b.any() for b in blocks) and li2._q.empty()


# ---- heard cards ----------------------------------------------------------------------------------
def test_enrolled_voice_gives_a_card_that_expires_unknown_leaves_nothing(two):
    t = {"now": 1_700_000_000.0}
    calls = []
    real = two.vm.process_audio

    def spy(audio, sr, place=None, now=None):
        out = real(audio, sr, place, now)
        calls.append(out)
        return out
    two.vm.process_audio = spy
    heard_hook = []
    cap = FakeCapture()
    li = listener(two.vm, cap, clock=lambda: t["now"], place="Kitchen", on_heard=heard_hook.append)
    try:
        cap.feed(np.concatenate([synth_voice("A", seed=50), quiet(1.5)]))
        assert wait_until(lambda: len(calls) == 1)
        assert wait_until(lambda: li.heard == 1)
        (card,) = li.cards()
        assert card["name"] == "Alex" and card["face_agrees"] is False and card["seconds_ago"] == 0
        assert set(card) == {"name", "score", "face_agrees", "direction", "seconds_ago"} and heard_hook[0]["name"] == "Alex"
        assert card["direction"] is None and heard_hook[0]["direction"] is None   # mono: no direction (ADR-0065)
        t["now"] += 4
        assert li.cards()[0]["seconds_ago"] == 4
        t["now"] += two.vm.cfg.card_seconds                                   # older than card_seconds: gone
        assert li.cards() == [] and li.heard == 1
        before = count(two.env["db"], "vl_heard")
        cap.feed(np.concatenate([synth_voice("C", seed=52), quiet(1.5)]))   # never enrolled
        assert wait_until(lambda: len(calls) == 2)
        drained(li)
        assert calls[1]["heard"] is None and li.cards() == [] and li.heard == 1
        assert count(two.env["db"], "vl_heard") == before
        st = li.status()
        assert st["state"] == "listening" and st["heard"] == 1 and st["cards"] == []
    finally:
        li.stop()


# ---- digital silence ------------------------------------------------------------------------------
def test_digital_silence_gives_the_notice_low_noise_does_not():
    vm, cap = FakeVM(), FakeCapture()
    li = listener(vm, cap)
    try:
        cap.feed(np.zeros(int(2 * SR), np.float32))
        assert li.status()["notice"] == ""                                   # not yet SILENCE_NOTICE_S
        cap.feed(np.zeros(int(2 * SR), np.float32))
        st = li.status()
        assert st["notice"] == SILENCE_NOTICE and "Privacy & Security > Microphone" in st["notice"]
        cap.feed(quiet(4.0, sd=1e-4))                                        # a quiet room is not digital silence
        assert li.status()["notice"] == ""
    finally:
        li.stop()
    d = DigitalSilence(SR, notice_s=1.0)
    assert not d.feed(np.zeros(SR - 1, np.float32)) and d.feed(np.zeros(1, np.float32))
    assert not d.feed(np.array([0, 0, 1e-7], np.float32))


# ---- push-to-talk ---------------------------------------------------------------------------------
def test_push_to_talk_processes_only_inside_the_listen_window():
    vm, cap = FakeVM(listening="push_to_talk"), FakeCapture()
    li = listener(vm, cap)
    try:
        assert cap.starts == 0 and li.status()["state"] == "idle" and li.status()["mode"] == "push_to_talk"
        early = synth_voice("A", seed=8)
        li._on_block(early)                                                  # nothing before Listen
        assert not early.any() and li._q.empty()
        assert li.listen_for(3.0) and cap.starts == 1
        st = li.status()
        assert st["state"] == "listening" and st["listen_left_s"] == 3.0
        cap.feed(synth_voice("A", seconds=8.0, seed=9, lead=0.2))            # talks past the window
        assert wait_until(lambda: len(vm.calls) == 1)
        assert wait_until(lambda: cap.stops == 1)                            # microphone closed after the window
        drained(li)
        assert len(vm.calls) == 1 and vm.calls[0] <= 3.0 + FRAME_S
        assert li.status()["state"] == "idle" and cap.cb is None
        assert PTT_DEFAULT_S == pytest.approx(8.0)
    finally:
        li.stop()


def test_ambient_listen_for_changes_nothing_and_capture_failure_is_plain():
    li = listener(FakeVM(), FakeCapture())
    try:
        assert li.listen_for(3.0) is True and li.status()["listen_left_s"] == 0.0
    finally:
        li.stop()
    li = listener(FakeVM(), FakeCapture(fail=CaptureUnavailable(vcap.NO_SOUNDDEVICE)))
    try:
        st = li.status()
        assert st["state"] == "unavailable" and st["reason"] == "the sounddevice package is not installed"
    finally:
        li.stop()
    li = listener(FakeVM(), FakeCapture(fail=RuntimeError("Error opening InputStream")))
    try:
        assert li.status()["reason"] == "the microphone could not be opened (Error opening InputStream)"
    finally:
        li.stop()


# ---- the sounddevice backend, with a stand-in module -------------------------------------------------
class FakeStream:
    def __init__(self, **kw):
        self.kw, self.started, self.closed = kw, False, False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


class FakeSdStream(FakeSd):
    def __init__(self):
        super().__init__()
        self.streams: list[FakeStream] = []

    def InputStream(self, **kw):   # noqa: N802 - sounddevice's name
        s = FakeStream(**kw)
        self.streams.append(s)
        return s


def test_sounddevice_capture_copies_and_zeroes_the_stream_block():
    sd = FakeSdStream()
    cap = SoundDeviceCapture("iphone", sd=sd)
    assert cap.device == 2 and cap.sample_rate == SR
    got = []
    cap.start(got.append)
    (s,) = sd.streams
    assert s.started and s.kw["samplerate"] == SR and s.kw["channels"] == 1 and s.kw["dtype"] == "float32" and s.kw["device"] == 2
    indata = synth_voice("A", seconds=0.1, seed=1).reshape(-1, 1).copy()
    expect = indata[:, 0].copy()
    s.kw["callback"](indata, indata.shape[0], None, None)
    assert np.array_equal(got[0], expect) and got[0].dtype == np.float32 and not indata.any()
    cap.stop()
    assert s.closed and not s.started
    assert SoundDeviceCapture(None, sd=sd).device is None
    with pytest.raises(ValueError, match="no input device matches"):
        SoundDeviceCapture("USB headset", sd=sd)
    with pytest.raises(ValueError, match="no input device 1"):
        SoundDeviceCapture(1, sd=sd)                                         # an output-only device
    assert resolve_device("0", sd) == 0


def test_missing_sounddevice_is_a_plain_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)                   # import fails as if not installed
    with pytest.raises(CaptureUnavailable, match="^the sounddevice package is not installed$"):
        vcap.load_sounddevice()
    with pytest.raises(CaptureUnavailable):
        SoundDeviceCapture("")


def test_record_returns_a_view_that_zeroes_the_buffer():
    sd = FakeSd([synth_voice("A", seconds=1.0, seed=1)])
    buf = vcap.record(1.0, None, SR, sd=sd)
    assert buf.size == SR and buf.any()
    buf.fill(0)
    assert not sd.recorded[0].any()


# ---- live mode --------------------------------------------------------------------------------------
def live_cfg(env, **voice) -> Config:
    cfg = Config(data_dir=str(env["dir"]))
    cfg.flags["M-16"] = cfg.flags["M-17"] = True
    cfg.voice_link = dict(voice)
    return cfg


def live_runner(env, cfg, **kw) -> LiveRunner:
    reg = default_registry({"entry_points": False})
    kw.setdefault("face_backend", InjectedBackend())
    kw.setdefault("speaker", slots("inj", reg=_inj_registry()))
    kw.setdefault("announce_player", NullPlayer())   # never a real sound in tests (ADR-0058)
    return LiveRunner(cfg, LiveSettings(adapters=[SIM], hz=20.0, site="lab"), registry=reg, store=(env["ks"], env["db"]), **kw)


@pytest.fixture
def enrolled(env):
    """Alex enrolled in M-16 and M-17 (synthetic voice A), set up through the modules directly."""
    face = face_on(env)
    a = enroll_face(face, 11, "Alex")
    vm = voice_on(env, face)
    enroll_voice(vm, a, "A")
    return SimpleNamespace(env=env, a=a)


def test_live_runner_listens_and_shows_the_heard_card(enrolled):
    env = enrolled.env
    cap = FakeCapture()
    r = live_runner(env, live_cfg(env), voice_capture=cap)
    r.start(background=False)
    try:
        v = r.snapshot()["voice"]
        assert v["state"] == "listening" and v["mode"] == "ambient" and v["cards"] == [] and v["heard"] == 0 and v["reason"] == ""
        assert set(v) == {"state", "reason", "mode", "notice", "listen_left_s", "cards", "heard"}
        assert cap.starts == 1 and r.voice.process_lock is r._store_lock
        with socket.socket() as s, pytest.raises(OutboundBlocked):         # the local-only guard is unchanged
            s.connect(("192.0.2.1", 80))
        cap.feed(np.concatenate([synth_voice("A", seed=50), quiet(1.5)]))
        assert wait_until(lambda: r.snapshot()["voice"]["heard"] == 1)
        v = r.snapshot()["voice"]
        assert v["cards"][0]["name"] == "Alex" and v["cards"][0]["face_agrees"] is False
        _step_until(r, 2)
        assert r.steps >= 2 and count(env["db"], "vl_heard") == 1
    finally:
        r.stop()
    assert cap.stops == 1 and r.voice is None
    v = r.snapshot()["voice"]
    assert v["state"] == "unavailable" and v["reason"] == "live mode stopped"


def test_cli_withdrawal_removes_current_heard_cards_before_they_expire(env, monkeypatch):
    from oiris import face_match_cli
    from oiris.guard.consent import ConsentRecord

    face = face_on(env)
    vm = voice_on(env, face)
    for seed, name, voice in ((11, "Alex", "A"), (12, "Blake", "B")):
        cid = "consent-" + voice
        env["consents"].add(ConsentRecord(cid, "fixture-" + voice, {"face_enrollment", "voice_enrollment"}))
        pid = enroll_face(face, seed, name, cid)
        enroll_voice(vm, pid, voice, cid)

    cap = FakeCapture()
    cfg = live_cfg(env, announce="off")
    runner = live_runner(env, cfg, voice_capture=cap)
    runner.start(background=False)
    try:
        now = time.time()
        runner.voice.clock = lambda: now  # Neither card can expire during the test.
        for i, voice in enumerate(("A", "B"), 1):
            cap.feed(np.concatenate([synth_voice(voice, seed=50), quiet(1.5)]))
            assert wait_until(lambda: runner.snapshot()["voice"]["heard"] == i)
        assert [c["name"] for c in runner.snapshot()["voice"]["cards"]] == ["Blake", "Alex"]

        # The real CLI opens a separate store connection while the dashboard stays up.
        monkeypatch.setenv("OIRIS_PASSPHRASE", "pw")
        args = cli_args("consent", id="consent-A", withdraw=True)
        cfg.face_match = {"adapter": "store_only"}
        assert face_match_cli.run(args, cfg, env["dir"] / "config.json") == 0
        assert count(env["db"], "vl_voice") == 1
        snap = runner.snapshot()
        assert [c["name"] for c in snap["voice"]["cards"]] == ["Blake"]
        page = render(None, snap)
        assert "Sounds like Blake" in page and "Sounds like Alex" not in page
    finally:
        runner.stop()


@pytest.mark.parametrize("case,expect", [
    ("not_opted_in", "not opted in"),
    ("no_sounddevice", "the sounddevice package is not installed"),
    ("no_model", "the speaker model is unavailable"),
    ("capture_fails", "the microphone could not be opened"),
    ("face_model", "face matching could not load"),
])
def test_voice_failures_leave_live_mode_running(env, monkeypatch, case, expect):
    face_on(env)
    if case != "not_opted_in":
        voice_opt_in(env)
    kw: dict = {"voice_capture": FakeCapture()}
    if case == "no_sounddevice":
        kw.pop("voice_capture")
        monkeypatch.setitem(sys.modules, "sounddevice", None)
    elif case == "no_model":
        kw["speaker"] = slots("not_installed")
    elif case == "capture_fails":
        kw["voice_capture"] = FakeCapture(fail=OSError("Error querying device -1"))
    elif case == "face_model":
        class NoModels(InjectedBackend):
            def open(self, cfg):
                raise RuntimeError("face models not downloaded; run `oiris face-match fetch-models`")
        kw["face_backend"] = NoModels()
    r = live_runner(env, live_cfg(env), **kw)
    r.start(background=False)
    try:
        v = r.snapshot()["voice"]
        assert v["state"] == "unavailable" and expect in v["reason"] and v["cards"] == [] and v["heard"] == 0
        _step_until(r, 2)
        assert r.steps >= 2 and r.snapshot()["running"]
        page = render(r.snapshot()["result"], r.snapshot())
        assert "Voice link unavailable: " in page
    finally:
        r.stop()


def test_voice_is_not_loaded_while_m16_or_m17_is_off(env):
    face_on(env); voice_opt_in(env)
    for flags in ({"M-16": True, "M-17": False}, {"M-16": False, "M-17": True}):
        cfg = live_cfg(env)
        cfg.flags.update(flags)
        cap = FakeCapture()
        r = live_runner(env, cfg, voice_capture=cap)
        r.start(background=False)
        try:
            assert "voice" not in r.snapshot() and cap.starts == 0 and r.voice is None
        finally:
            r.stop()


def test_live_push_to_talk_through_voice_listen(enrolled):
    env = enrolled.env
    cap = FakeCapture()
    r = live_runner(env, live_cfg(env, listening="push_to_talk"), voice_capture=cap)
    r.start(background=False)
    try:
        assert r.snapshot()["voice"]["state"] == "idle" and cap.starts == 0
        assert r.voice_listen(5.0) and r.snapshot()["voice"]["state"] == "listening"
        cap.feed(np.concatenate([synth_voice("A", seconds=4.0, seed=50), quiet(1.0)]))
        assert wait_until(lambda: r.snapshot()["voice"]["heard"] == 1)
        assert wait_until(lambda: r.snapshot()["voice"]["state"] == "idle")
    finally:
        r.stop()
    assert r.voice_listen() is False


# The flake behind test_live_push_to_talk_through_voice_listen (once, under heavy load): the window closed only when
# its samples were used up, but a block dropped while the announcement played (paused) used none. On a loaded machine
# the worker could match the utterance and start the chime before the feeder handed over the last quiet blocks; those
# were dropped, the window kept 0.2-0.3 s forever, and the state stayed "listening". Dropped blocks now count.
def test_push_to_talk_window_counts_blocks_dropped_while_paused():
    vm, cap = FakeVM(), FakeCapture()
    paused = [False]
    li = listener(vm, cap, mode="push_to_talk", paused=lambda: paused[0])
    try:
        assert li.listen_for(1.0)
        cap.feed(quiet(0.5))
        paused[0] = True                                  # an announcement starts mid-window
        cap.feed(quiet(0.5))
        paused[0] = False
        assert li.dropped_while_paused > 0 and li.status()["listen_left_s"] == 0.0
        assert wait_until(lambda: li.status()["state"] == "idle" and cap.stops == 1)
    finally:
        li.stop()


FORCED_REPEATS = 20


def test_live_push_to_talk_ends_when_the_announcement_overlaps_the_window(enrolled):
    """The flaky interleaving forced on every run: the window's last blocks arrive while the announcement plays."""
    env = enrolled.env
    for i in range(FORCED_REPEATS):
        cap = FakeCapture()
        r = live_runner(env, live_cfg(env, listening="push_to_talk"), voice_capture=cap)
        r.start(background=False)
        try:
            assert r.voice_listen(5.0)
            cap.feed(np.concatenate([synth_voice("A", seconds=4.0, seed=50), quiet(0.8)]))
            assert wait_until(lambda: r.snapshot()["voice"]["heard"] == 1), i
            r.voice.paused = lambda: True                 # the chime and its tail: every later block is dropped
            cap.feed(quiet(0.2))                          # the window's last 0.2 s
            assert wait_until(lambda: r.snapshot()["voice"]["state"] == "idle"), i
            assert r.snapshot()["voice"]["listen_left_s"] == 0.0
            assert wait_until(lambda: cap.stops == 1), i  # the worker closes the stream on its next poll
        finally:
            r.stop()


# ---- dashboard --------------------------------------------------------------------------------------
def voice_snap(**v):
    base = {"state": "listening", "reason": "", "mode": "ambient", "notice": "", "listen_left_s": 0.0, "cards": [], "heard": 0}
    snap = _snap(None)
    snap["voice"] = {**base, **v}
    return snap


def _vl(page: str) -> str:
    return page.split("<section id=VL")[1].split("</section>")[0]


def test_dashboard_voice_states_and_cards_render_escaped():
    assert "Listening (microphone on)" in _vl(render(None, voice_snap()))
    vl = _vl(render(None, voice_snap(mode="push_to_talk", state="idle"), listen_token="tok"))
    assert "Push-to-talk: press Listen" in vl and "action=/voice/listen" in vl and 'value="tok"' in vl
    assert "action=/voice/listen" not in _vl(render(None, voice_snap(mode="push_to_talk", state="idle")))   # no token
    assert "action=/voice/listen" not in _vl(render(None, voice_snap(), listen_token="tok"))              # ambient
    vl = _vl(render(None, voice_snap(state="unavailable", reason="the sounddevice package is not installed")))
    assert 'Voice link unavailable: the sounddevice package is not installed</b> - Install it with: pip install -e &quot;.[voice]&quot;' in vl
    vl = _vl(render(None, voice_snap(state="unavailable", reason="<img src=x onerror=alert(1)>")))
    assert "<img" not in vl and "&lt;img src=x" in vl
    vl = _vl(render(None, voice_snap(notice=SILENCE_NOTICE)))
    assert "no sound from the microphone: check System Settings &gt; Privacy &amp; Security &gt; Microphone" in vl
    cards = [{"name": "<b>Al&ex</b>", "score": 0.81, "face_agrees": True, "seconds_ago": 3.2},
             {"name": "Blake", "score": 0.7, "face_agrees": False, "seconds_ago": 7.0}]
    vl = _vl(render(None, voice_snap(cards=cards, heard=2)))
    assert "Sounds like &lt;b&gt;Al&amp;ex&lt;/b&gt;</b> · Recent face match for this person · 3 s ago" in vl
    assert "Sounds like Blake</b> · 7 s ago" in vl and "2 heard this run" in vl and "<b>Al&ex" not in vl
    assert "id=VL" not in render(None, _snap(None))                                                         # no voice: unchanged


def _post_listen(srv, token, origin="http://127.0.0.1:0", host="127.0.0.1:0"):
    body = f"token={token}".encode() if token is not None else b""
    hdr = f"POST /voice/listen HTTP/1.0\r\nHost: {host}\r\nContent-Length: {len(body)}\r\n"
    hdr += f"Origin: {origin}\r\n" if origin else ""
    return _request(srv, hdr.encode() + b"\r\n" + body)


def test_listen_post_needs_token_same_origin_and_local_host():
    listens, quits = [], []
    snap = voice_snap(mode="push_to_talk", state="idle")
    srv = server.make_server(Config(), port=0, live=lambda: snap, bind_and_activate=False,
                             on_quit=lambda: quits.append(1), on_listen=lambda: listens.append(1))
    try:
        code, page = _request(srv, b"GET / HTTP/1.0\r\nHost: 127.0.0.1:0\r\n\r\n")
        assert code == 200 and b"<button>Listen</button>" in page
        tok = re.search(r'action=/voice/listen><input type=hidden name=token value="([^"]+)"', page.decode()).group(1)
        assert len(tok) >= 32
        assert _request(srv, b"GET /voice/listen HTTP/1.0\r\nHost: 127.0.0.1:0\r\n\r\n")[0] == 405
        assert _request(srv, b"GET /voice/listen HTTP/1.0\r\nHost: evil.example:0\r\n\r\n")[0] == 403
        assert _post_listen(srv, None)[0] == 403
        assert _post_listen(srv, ("y" if tok[0] == "x" else "x") + tok[1:])[0] == 403
        assert _post_listen(srv, tok, origin="http://evil.example")[0] == 403
        assert _post_listen(srv, tok, origin="http://127.0.0.1:9")[0] == 403
        assert _post_listen(srv, tok, origin=None)[0] == 403
        assert _post_listen(srv, tok, host="rebind.example:0", origin="http://rebind.example:0")[0] == 403
        assert listens == []
        assert _post_listen(srv, tok, origin="http://localhost:0", host="localhost:0")[0] == 303
        assert _post_listen(srv, tok)[0] == 303 and listens == [1, 1] and quits == []
    finally:
        srv.server_close()
    srv = server.make_server(Config(), port=0, live=lambda: snap, bind_and_activate=False, on_quit=lambda: None)
    try:
        assert _post_listen(srv, "anything")[0] == 404                       # no listener: no Listen
        assert b"/voice/listen" not in _request(srv, b"GET / HTTP/1.0\r\nHost: 127.0.0.1:0\r\n\r\n")[1]
    finally:
        srv.server_close()


def test_dashboard_uses_the_runner_voice_listen(enrolled):
    env = enrolled.env
    cap = FakeCapture()
    r = live_runner(env, live_cfg(env, listening="push_to_talk"), voice_capture=cap)
    r.start(background=False)
    srv = server.make_server(Config(), port=0, live=r.snapshot, bind_and_activate=False, on_quit=lambda: None,
                             on_listen=r.voice_listen)
    try:
        page = _request(srv, b"GET / HTTP/1.0\r\nHost: 127.0.0.1:0\r\n\r\n")[1].decode()
        tok = re.search(r'action=/voice/listen><input type=hidden name=token value="([^"]+)"', page).group(1)
        assert _post_listen(srv, tok)[0] == 303 and cap.starts == 1 and r.snapshot()["voice"]["state"] == "listening"
    finally:
        srv.server_close()
        r.stop()


# ---- CLI enroll -------------------------------------------------------------------------------------
def test_cli_enroll_still_works_with_injected_capture(store):
    from oiris.face_match_cli import run as face_run
    from oiris.voice_link_cli import run as voice_run
    d, cfgp = store
    face_run(cli_args("enable", mode=["personal"]), Config.load(cfgp), cfgp, inp=lambda p: "I UNDERSTAND", out=[].append)
    voice_run(cli_args("enable"), Config.load(cfgp), cfgp, inp=lambda p: "I UNDERSTAND", out=[].append)
    face_run(cli_args("consent", subject="S-07", id="c-cli", scope="face_enrollment,voice_enrollment"), Config.load(cfgp), cfgp,
             inp=lambda p: "YES", out=[].append)
    pid = cli_face_person(d)
    windows = [synth_voice("A", seed=s) for s in (1, 2, 3)]
    grabbed = []

    def grab(seconds, sample_rate, device):
        assert sample_rate == SR and device is None
        if seconds == vcap.MIC_CHECK_S:   # the microphone check before window 1: faint noise, not a window
            return np.random.default_rng(0).normal(0, 1e-2, int(seconds * sample_rate)).astype(np.float32)
        grabbed.append(windows.pop(0))
        return grabbed[-1]
    out: list[str] = []
    assert voice_run(cli_args("enroll", person=pid[:6], consent="c-cli"), Config.load(cfgp), cfgp, inp=enter_skip_redo,
                     out=out.append, face_backend=InjectedBackend(), speaker=slots("inj", reg=_inj_registry()), capture=grab) == 0
    assert out[-1] == "Voice template complete for Alex" and len(grabbed) == 3 and all(not g.any() for g in grabbed)
    ks, db, _ = open_store(d)
    try:
        assert count(db, "vl_voice") == 1
    finally:
        db.close()
