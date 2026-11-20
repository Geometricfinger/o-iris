"""M-17 says where a recognised voice came from (ADR-0065). Synthetic audio only: the harmonic tones of
tests/test_voice_link.py rendered per microphone with an exact fractional delay for a known head-frame azimuth.
Fake speaker and backend; no microphone, no real model, no network."""
# ruff: noqa: F811
import math
import time

import numpy as np
import pytest

from oiris.config import Config
from oiris.dashboard import server
from oiris.dashboard.server import render
from oiris.features.doa import DIRECTION_WORDS, SPEED_OF_SOUND_M_S
from oiris.guard.phrases import MAX_WORDS
from oiris.live import LiveRunner, LiveSettings
from oiris.modules.voice_link import announce
from oiris.modules.voice_link.announce import (CHIME, CHIME_AND_NAME, DIRECTION_PHRASES, OFF, Announcer, NullPlayer,
                                               phrase)
from oiris.modules.voice_link.capture import CaptureUnavailable
from oiris.modules.voice_link.direction import MAJORITY_SHARE, MIN_WINDOWS, WINDOW_S, VoiceDirection
from oiris.modules.voice_link.listener import REF_CHANNEL, VoiceListener
from oiris.registry import default_registry
from tests.test_live import SIM
from tests.test_voice_link import (SR, InjectedBackend, _inj_registry, count, enroll_face, enroll_voice, face_on, slots,
                                   synth_voice, voice_on)
from tests.test_voice_link import env, two  # noqa: F401  (fixture reuse)
from tests.test_voice_link_announce import spoof_vm
from tests.test_voice_link_live import FakeCapture, _vl, drained, quiet, voice_snap, wait_until

TEMPLE_Y_M = 0.07                 # smart-glasses temples, y = +-0.07 m
HELMET_HALF_M = 0.1               # four helmet mics, front/left/back/right at 0.1 m
TETRA_EDGE_M = 0.2
PAD = 1024                        # zero padding so the circular delay does not wrap
SENSOR_NOISE = 1e-3
TEST_QUEUE_BLOCKS = 1000
LEFT_AZ, RIGHT_AZ = 90.0, -90.0
HALF_SHARE = 0.5                  # an even split of the windows

TEMPLE = np.array([[0.0, TEMPLE_Y_M, 0.0], [0.0, -TEMPLE_Y_M, 0.0]])
HELMET = np.array([[HELMET_HALF_M, 0, 0], [0, HELMET_HALF_M, 0], [-HELMET_HALF_M, 0, 0], [0, -HELMET_HALF_M, 0]], float)
TETRA = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) * TETRA_EDGE_M / (2 * math.sqrt(2))


def _unit(az: float, el: float = 0.0) -> np.ndarray:
    a, e = math.radians(az), math.radians(el)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def render_mics(src: np.ndarray, pos: np.ndarray, az: float, el: float = 0.0, seed: int = 1) -> np.ndarray:
    """(M, N) float32: a plane wave from (az, el); mic p hears the source at t = -(p . u) / c (exact fractional delay)."""
    u = _unit(az, el)
    x = np.concatenate([np.asarray(src, float), np.zeros(PAD)])
    spec, f = np.fft.rfft(x), np.fft.rfftfreq(x.size, 1 / SR)
    out = np.array([np.fft.irfft(spec * np.exp(2j * np.pi * f * (p @ u) / SPEED_OF_SOUND_M_S), x.size)[:src.size] for p in pos])
    return (out + np.random.default_rng(seed).normal(0, SENSOR_NOISE, out.shape)).astype(np.float32)


def estimate(pos: np.ndarray, az: float, el: float = 0.0, seconds: float = 3.0, seed: int = 5) -> str | None:
    src = synth_voice("A", seconds=seconds, seed=seed)
    return VoiceDirection(pos, SR).estimate(render_mics(src, pos, az, el, seed), np.ones(src.size, bool))


def spoken(word: str | None) -> str | None:
    return None if word is None else DIRECTION_PHRASES[word]


# ---- the estimate ----------------------------------------------------------------------------------
@pytest.mark.parametrize("az,expect", [(90.0, "on your left"), (60.0, "on your left"), (-75.0, "on your right"),
                                       (0.0, None), (30.0, None)])
def test_temple_pair(az, expect):
    """60 deg: candidates 60 and 120 are both "at left". 0 and 30 deg: ahead or behind, so no word."""
    assert spoken(estimate(TEMPLE, az)) == expect


@pytest.mark.parametrize("az,expect", [(180.0, "behind you"), (0.0, "ahead of you"), (90.0, "on your left")])
def test_four_mic_helmet_array(az, expect):
    assert spoken(estimate(HELMET, az)) == expect


def test_elevation_only_from_a_non_coplanar_array():
    assert estimate(TETRA, 0.0, el=70.0) == "above"
    assert estimate(HELMET, 0.0, el=70.0) != "above"          # a horizontal array has no elevation


def test_too_few_speech_windows_is_none():
    src = synth_voice("A", seconds=3.0, seed=5)
    x = render_mics(src, TEMPLE, LEFT_AZ)
    vd = VoiceDirection(TEMPLE, SR)
    mask = np.zeros(src.size, bool)
    start = int(1.0 * SR)                                     # inside a syllable
    mask[start:start + (MIN_WINDOWS - 1) * vd.window] = True  # one window short, wherever they fall
    assert vd.estimate(x, mask) is None
    assert vd.estimate(x, np.ones(src.size, bool)) == "at left"   # the same audio with every window: a word


def test_a_split_vote_is_none():
    """Half the speech from the left, half from the right: no word reaches MAJORITY_SHARE."""
    a, b = synth_voice("A", seconds=2.0, seed=5), synth_voice("A", seconds=2.0, seed=6)
    x = np.concatenate([render_mics(a, TEMPLE, LEFT_AZ), render_mics(b, TEMPLE, RIGHT_AZ)], axis=1)
    assert MAJORITY_SHARE > HALF_SHARE
    assert VoiceDirection(TEMPLE, SR).estimate(x, np.ones(x.shape[1], bool)) is None


def test_silence_and_no_speech_are_none():
    vd = VoiceDirection(TEMPLE, SR)
    assert vd.estimate(np.zeros((2, 3 * SR), np.float32), np.ones(3 * SR, bool)) is None
    src = synth_voice("A", seconds=3.0, seed=5)
    assert vd.estimate(render_mics(src, TEMPLE, LEFT_AZ), np.zeros(src.size, bool)) is None   # the gate heard no speech


def test_estimate_checks_its_input():
    vd = VoiceDirection(TEMPLE, SR)
    assert vd.window == round(WINDOW_S * SR) and vd.channels == 2
    with pytest.raises(ValueError):
        vd.estimate(np.zeros((3, SR)), np.ones(SR, bool))
    with pytest.raises(ValueError):
        VoiceDirection([[0.0, 0.0, 0.0]], SR)


# ---- the listener ----------------------------------------------------------------------------------
class SpyDirection:
    """A VoiceDirection that counts its calls."""

    def __init__(self, pos: np.ndarray = TEMPLE):
        self.inner, self.calls = VoiceDirection(pos, SR), 0
        self.channels = self.inner.channels

    def estimate(self, channels, speech_mask):
        self.calls += 1
        return self.inner.estimate(channels, speech_mask)


class MultiCapture(FakeCapture):
    """FakeCapture for (M, N) audio: hands over (block, M) float32 copies, like SoundDeviceCapture with channels=M."""

    def __init__(self, channels: int = 2, **kw):
        super().__init__(**kw)
        self.channels = channels

    def feed(self, audio: np.ndarray) -> None:
        if audio.ndim == 1:
            return super().feed(audio)
        for i in range(0, audio.shape[1], self.block):
            cb = self.cb
            if cb is None:
                return
            b = np.ascontiguousarray(audio[:, i:i + self.block].T, dtype=np.float32)
            self.handed.append(b)
            cb(b)


def utterance(spk: str, az: float = LEFT_AZ, seed: int = 50, pos: np.ndarray = TEMPLE) -> np.ndarray:
    return render_mics(np.concatenate([synth_voice(spk, seed=seed), quiet(1.5)]), pos, az, seed=seed)


def multi_listener(vm, cap, direction, **kw) -> VoiceListener:
    kw.setdefault("queue_blocks", TEST_QUEUE_BLOCKS)
    return VoiceListener(vm, cap, direction=direction, **kw).start()


def spy_calls(vm):
    calls = []
    real = vm.process_audio

    def spy(audio, sr, place=None, now=None):
        out = real(audio, sr, place, now)
        calls.append(out)
        return out
    vm.process_audio = spy
    return calls


def test_a_clear_match_gets_its_direction_on_the_card_and_hook(two):
    heard, spy, cap = [], SpyDirection(), MultiCapture()
    li = multi_listener(two.vm, cap, spy, on_heard=heard.append)
    try:
        cap.feed(utterance("A"))
        assert wait_until(lambda: li.heard == 1)
        (card,) = li.cards()
        assert card["name"] == "Alex" and card["direction"] == "at left" and spy.calls == 1
        assert heard[0]["direction"] == "at left" and heard[0]["person_id"] == two.a
        assert li.status()["cards"][0]["direction"] == "at left"
    finally:
        li.stop()


def test_reference_channel_alone_is_matched(two):
    """Alex's voice on channel 0, Blake's on channel 1: the match is Alex, from channel 0 only."""
    calls, cap = spy_calls(two.vm), MultiCapture()
    li = multi_listener(two.vm, cap, SpyDirection())
    try:
        x = np.stack([np.concatenate([synth_voice(s, seed=50), quiet(1.5)]) for s in ("A", "B")]).astype(np.float32)
        assert REF_CHANNEL == 0
        cap.feed(x)
        assert wait_until(lambda: len(calls) == 1)
        drained(li)
        assert calls[0]["heard"]["name"] == "Alex" and li.cards()[0]["direction"] is None   # two voices: no direction
    finally:
        li.stop()


def test_unknown_voice_never_gets_a_direction(two):
    calls, spy, cap = spy_calls(two.vm), SpyDirection(), MultiCapture()
    li = multi_listener(two.vm, cap, spy)
    try:
        cap.feed(utterance("C", seed=52))                   # never enrolled
        assert wait_until(lambda: len(calls) == 1)
        drained(li)
        assert calls[0]["heard"] is None and spy.calls == 0 and li.cards() == []
    finally:
        li.stop()


def test_close_call_never_gets_a_direction(env):
    face = face_on(env)
    a, a2 = enroll_face(face, 11, "Alex"), enroll_face(face, 14, "Avery")
    vm = voice_on(env, face)
    enroll_voice(vm, a, "A", seeds=(1, 2, 3)); enroll_voice(vm, a2, "A", seeds=(4, 5, 6))   # two nearly identical voices
    calls, spy, cap = spy_calls(vm), SpyDirection(), MultiCapture()
    li = multi_listener(vm, cap, spy)
    try:
        cap.feed(utterance("A", seed=60))
        assert wait_until(lambda: len(calls) == 1)
        drained(li)
        assert calls[0]["unsure"] and spy.calls == 0 and li.cards() == []
    finally:
        li.stop()


def test_self_never_gets_a_direction(two):
    two.vm.set_self(two.a)
    calls, spy, cap = spy_calls(two.vm), SpyDirection(), MultiCapture()
    li = multi_listener(two.vm, cap, spy)
    try:
        cap.feed(utterance("A"))
        assert wait_until(lambda: len(calls) == 1)
        drained(li)
        assert calls[0]["heard"] is None and spy.calls == 0 and li.cards() == []
    finally:
        li.stop()


def test_spoof_veto_never_gets_a_direction(env):
    vm = spoof_vm(env, 0.9)
    calls, spy, cap = spy_calls(vm), SpyDirection(), MultiCapture()
    li = multi_listener(vm, cap, spy)
    try:
        cap.feed(utterance("A"))
        assert wait_until(lambda: len(calls) == 1)
        drained(li)
        assert calls[0]["heard"] is None and spy.calls == 0 and li.cards() == []
    finally:
        li.stop()


def test_paused_never_gets_a_direction(two):
    calls, spy, cap = spy_calls(two.vm), SpyDirection(), MultiCapture()
    li = multi_listener(two.vm, cap, spy, paused=lambda: True)
    try:
        cap.feed(utterance("A"))
        drained(li)
        assert calls == [] and spy.calls == 0 and li.dropped_while_paused > 0
        assert all(not b.any() for b in cap.handed)
    finally:
        li.stop()


def test_multi_channel_buffers_are_zeroed_after_matched_and_unmatched_utterances(two):
    calls, cap = spy_calls(two.vm), MultiCapture()
    li = multi_listener(two.vm, cap, SpyDirection())
    try:
        cap.feed(utterance("A"))
        assert wait_until(lambda: len(calls) == 1)
        drained(li)
        assert calls[0]["heard"] is not None
        assert not li._mbuf.any() and not li._speech.any() and not li._buf.any()          # after a match
        cap.feed(utterance("C", seed=52))
        assert wait_until(lambda: len(calls) == 2)
        drained(li)
        assert calls[1]["heard"] is None
        assert not li._mbuf.any() and not li._speech.any() and not li._buf.any()          # after an unknown voice
        assert all(not b.any() for b in cap.handed)
    finally:
        li.stop()
    assert not li._mbuf.any() and not li._mpre.any() and not li._mframe.any() and not li._speech.any()


def test_mono_blocks_with_a_direction_estimate_give_no_direction(two):
    """The capture fell back to mono: matching goes on, no direction, no estimate."""
    spy, cap = SpyDirection(), FakeCapture()
    li = multi_listener(two.vm, cap, spy)
    try:
        cap.feed(np.concatenate([synth_voice("A", seed=50), quiet(1.5)]))
        assert wait_until(lambda: li.heard == 1)
        assert li.cards()[0]["direction"] is None and spy.calls == 0
    finally:
        li.stop()


def test_no_direction_in_the_store(two):
    cap = MultiCapture()
    li = multi_listener(two.vm, cap, SpyDirection())
    try:
        cap.feed(utterance("A"))
        assert wait_until(lambda: li.heard == 1)
        drained(li)
        assert li.cards()[0]["direction"] == "at left"
    finally:
        li.stop()
    db = two.env["db"]
    assert count(db, "vl_heard") == 1
    cols = [r[1] for r in db.execute("PRAGMA table_info(vl_heard)").fetchall()]
    assert not any("direction" in c.lower() for c in cols)
    words = set(DIRECTION_WORDS) | set(DIRECTION_PHRASES.values())
    tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    for t in tables:
        for row in db.execute(f"SELECT * FROM {t}").fetchall():
            for cell in row:
                text = cell.decode("utf-8", "replace") if isinstance(cell, (bytes, bytearray)) else str(cell)
                assert "direction" not in text.lower() and not any(w in text for w in words), (t, text[:80])


# ---- the announcement ------------------------------------------------------------------------------
@pytest.mark.parametrize("word,text", [
    ("ahead", "sounds like Alex, ahead of you"), ("at left", "sounds like Alex, on your left"),
    ("at right", "sounds like Alex, on your right"), ("behind", "sounds like Alex, behind you"),
    ("above", "sounds like Alex, above you"), ("below", "sounds like Alex, below you"),
    (None, "sounds like Alex"), ("sideways", "sounds like Alex")])
def test_phrase_with_each_direction(word, text):
    assert phrase("Alex", word) == text and len(text.split()) <= MAX_WORDS
    assert set(DIRECTION_PHRASES) == set(DIRECTION_WORDS)


def test_the_word_budget_drops_the_direction_for_a_four_word_name():
    four = "Alex Blake Casey Drew"
    assert phrase(four, "at left") == f"sounds like {four}"            # 2 + 4 + 3 words > 8: the name wins
    assert phrase(four, "behind") == f"sounds like {four}, behind you"  # exactly 8 words
    assert phrase("Alex Blake", "at left") == "sounds like Alex Blake, on your left"


def test_vocabulary_refusal_falls_back_to_the_name_and_checks_it_too(monkeypatch):
    monkeypatch.setitem(announce.DIRECTION_PHRASES, "at left", "near a robot")
    assert phrase("Alex", "at left") == "sounds like Alex"
    checked = []

    def refuse(text):
        checked.append(text)
        raise announce.VocabularyViolation("refused")
    monkeypatch.setattr(announce, "check_text", refuse)
    assert phrase("Alex", "behind") is None
    assert checked == ["sounds like Alex, behind you", "sounds like Alex"]


def _played(a: Announcer, p: NullPlayer, n: int) -> list:
    assert wait_until(lambda: len(p.played) >= n and not a.playing())
    return list(p.played)


def test_announcer_speaks_the_direction_and_keeps_the_repeat_limit():
    t = {"now": 0.0}
    p = NullPlayer()
    a = Announcer(CHIME_AND_NAME, repeat_s=60.0, player=p, clock=lambda: t["now"], tail_s=0.0)
    try:
        assert a.announce("p1", "Alex", "at left")
        assert _played(a, p, 2) == [("chime", ""), ("say", "sounds like Alex, on your left")]
        t["now"] += 10
        assert not a.announce("p1", "Alex", "behind")              # a new direction is not a new announcement
        assert not a.announce("p1", "Alex")
        assert a.announce("p2", "Blake", "behind")
        assert _played(a, p, 4)[2:] == [("chime", ""), ("say", "sounds like Blake, behind you")]
        t["now"] += 60
        assert a.announce("p1", "Alex", "at right")
        assert _played(a, p, 6)[-1] == ("say", "sounds like Alex, on your right")
        assert a.announce("p3", "Robot", "at left")                 # name refused: chime alone, never a bare direction
        time.sleep(0.3)
        assert _played(a, p, 7)[6:] == [("chime", "")]
    finally:
        a.close()


def test_chime_and_off_are_unchanged():
    p = NullPlayer()
    a = Announcer(CHIME, player=p, tail_s=0.0)
    try:
        assert a.announce("p1", "Alex", "at left")
        assert _played(a, p, 1) == [("chime", "")]
        time.sleep(0.3)
        assert p.played == [("chime", "")]
    finally:
        a.close()
    p = NullPlayer()
    a = Announcer(OFF, player=p)
    assert not a.announce("p1", "Alex", "at left") and p.played == []
    a.close()


# ---- live mode --------------------------------------------------------------------------------------
TEMPLE_RIG = {"mounts": [{"id": "temples", "kind": "mic_array", "mic_positions_m": TEMPLE.tolist()}]}


class Factory:
    """A fake capture factory: records (device, channels) and hands out a MultiCapture, or refuses N channels."""

    def __init__(self, refuse_multi: bool = False):
        self.refuse_multi, self.calls, self.made = refuse_multi, [], []

    def __call__(self, device, channels):
        self.calls.append((device, channels))
        if channels > 1 and self.refuse_multi:
            raise RuntimeError("Invalid number of channels")
        cap = MultiCapture(channels) if channels > 1 else FakeCapture()
        self.made.append(cap)
        return cap


def live_cfg(env, rig: dict | None = None) -> Config:
    cfg = Config(data_dir=str(env["dir"]))
    cfg.flags["M-16"] = cfg.flags["M-17"] = True
    cfg.voice_link = {}
    if rig is not None:
        cfg.rig = rig
    return cfg


def runner(env, cfg, factory, player) -> LiveRunner:
    reg = default_registry({"entry_points": False})
    return LiveRunner(cfg, LiveSettings(adapters=[SIM], hz=20.0, site="lab"), registry=reg, store=(env["ks"], env["db"]),
                      face_backend=InjectedBackend(), speaker=slots("inj", reg=_inj_registry()), announce_player=player,
                      voice_capture_factory=factory)


@pytest.fixture
def alex(env):
    face = face_on(env)
    a = enroll_face(face, 11, "Alex")
    enroll_voice(voice_on(env, face), a, "A")
    return env


def said(p: NullPlayer) -> list[str]:
    return [t for k, t in p.played if k == "say"]


def test_live_rig_with_a_mic_array_opens_n_channels_and_announces_the_direction(alex):
    f, p = Factory(), NullPlayer()
    r = runner(alex, live_cfg(alex, TEMPLE_RIG), f, p)
    r.start(background=False)
    try:
        assert f.calls == [("", 2)] and r.voice.direction is not None
        f.made[0].feed(utterance("A"))
        assert wait_until(lambda: said(p) == ["sounds like Alex, on your left"])
        card = r.snapshot()["voice"]["cards"][0]
        assert card["name"] == "Alex" and card["direction"] == "at left"
        assert "Sounds like Alex, on your left</b>" in _vl(render(None, r.snapshot()))
    finally:
        r.stop()


def test_live_default_rig_is_mono_and_unchanged(alex):
    f, p = Factory(), NullPlayer()
    r = runner(alex, live_cfg(alex), f, p)
    r.start(background=False)
    try:
        assert f.calls == [("", 1)] and r.voice.direction is None
        f.made[0].feed(np.concatenate([synth_voice("A", seed=50), quiet(1.5)]))
        assert wait_until(lambda: said(p) == ["sounds like Alex"])
        assert r.snapshot()["voice"]["cards"][0]["direction"] is None
    finally:
        r.stop()


def test_live_capture_refusing_n_channels_falls_back_to_mono(alex, caplog):
    f, p = Factory(refuse_multi=True), NullPlayer()
    r = runner(alex, live_cfg(alex, TEMPLE_RIG), f, p)
    with caplog.at_level("WARNING"):
        r.start(background=False)
    try:
        assert f.calls == [("", 2), ("", 1)] and r.voice.direction is None
        assert "could not open 2 channels" in caplog.text and "listening in mono" in caplog.text
        assert r.snapshot()["voice"]["state"] == "listening"
        f.made[0].feed(np.concatenate([synth_voice("A", seed=50), quiet(1.5)]))
        assert wait_until(lambda: said(p) == ["sounds like Alex"])
    finally:
        r.stop()


def test_mount_pose_turns_mic_positions_into_the_head_frame(env):
    rig = {"mounts": [{"id": "m", "kind": "mic_array", "position_m": [0.0, 0.0, 0.1], "yaw_deg": 90.0,
                       "mic_positions_m": [[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]]}]}
    r = LiveRunner(live_cfg(env, rig), registry=default_registry({"entry_points": False}), store=(env["ks"], env["db"]))
    assert np.allclose(r._voice_mics_head_m(), [[0.0, 0.05, 0.1], [0.0, -0.05, 0.1]])   # mount forward = head left
    assert LiveRunner(live_cfg(env), registry=default_registry({"entry_points": False}),
                      store=(env["ks"], env["db"]))._voice_mics_head_m() is None


# ---- the sounddevice capture with several channels ----------------------------------------------------
class _Stream:
    def __init__(self, fail: bool = False, **kw):
        self.kw, self.fail, self.started, self.closed = kw, fail, False, False

    def start(self):
        if self.fail:
            raise RuntimeError("Invalid number of channels")
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


class _Sd:
    def __init__(self, max_channels: int = 4, fail_multi: bool = False):
        self.max_channels, self.fail_multi, self.streams = max_channels, fail_multi, []

    def query_devices(self):
        return [{"name": "Array Microphone", "max_input_channels": self.max_channels}]

    def InputStream(self, **kw):   # noqa: N802 - sounddevice's name
        s = _Stream(fail=self.fail_multi and kw["channels"] > 1, **kw)
        self.streams.append(s)
        return s


def test_sounddevice_capture_delivers_n_channel_blocks():
    from oiris.modules.voice_link.capture import SoundDeviceCapture
    sd = _Sd()
    cap = SoundDeviceCapture(0, sd=sd, channels=2)
    got = []
    cap.start(got.append)
    (s,) = sd.streams
    assert s.kw["channels"] == 2 and s.started
    indata = np.arange(20, dtype=np.float32).reshape(10, 2) + 1
    expect = indata.copy()
    s.kw["callback"](indata, 10, None, None)
    assert got[0].shape == (10, 2) and np.array_equal(got[0], expect) and not indata.any()
    cap.stop()
    with pytest.raises(CaptureUnavailable, match="has 1 channels, 2 needed"):
        SoundDeviceCapture(0, sd=_Sd(max_channels=1), channels=2)


def test_sounddevice_capture_falls_back_to_mono_when_n_channels_fail(caplog):
    from oiris.modules.voice_link.capture import SoundDeviceCapture
    sd = _Sd(fail_multi=True)
    cap = SoundDeviceCapture(None, sd=sd, channels=2)
    got = []
    with caplog.at_level("WARNING"):
        cap.start(got.append)
    first, second = sd.streams
    assert first.closed and second.kw["channels"] == 1 and second.started and cap.channels == 1
    assert "listening in mono" in caplog.text
    indata = np.ones((10, 1), np.float32)
    second.kw["callback"](indata, 10, None, None)
    assert got[0].shape == (10,)                                     # mono blocks, as before


def test_mic_check_measures_channel_0_of_multi_channel_input():
    from oiris.modules.voice_link.capture import MIC_CHECK_OK, MIC_CHECK_SILENT, mic_check
    x = np.zeros((SR, 2), np.float32)
    x[:, 1] = 0.1                                                    # only channel 1 has sound
    assert mic_check(lambda s, r: x.copy(), sample_rate=SR).status == MIC_CHECK_SILENT
    x[:, 0] = np.random.default_rng(0).normal(0, 0.05, SR)
    raw = x.copy()
    assert mic_check(lambda s, r: raw, sample_rate=SR).status == MIC_CHECK_OK and not raw.any()


# ---- the dashboard ----------------------------------------------------------------------------------
def test_dashboard_card_says_the_direction_escaped():
    assert server.VOICE_DIRECTIONS == DIRECTION_PHRASES
    cards = [{"name": "<b>Al&ex</b>", "score": 0.81, "face_agrees": True, "direction": "at left", "seconds_ago": 3.2},
             {"name": "Blake", "score": 0.7, "face_agrees": False, "direction": None, "seconds_ago": 7.0},
             {"name": "Casey", "score": 0.7, "face_agrees": False, "direction": "<script>x</script>", "seconds_ago": 1.0}]
    vl = _vl(render(None, voice_snap(cards=cards, heard=3)))
    assert "Sounds like &lt;b&gt;Al&amp;ex&lt;/b&gt;, on your left</b> · Recent face match for this person · 3 s ago" in vl
    assert "Sounds like Blake</b> · 7 s ago" in vl                   # no direction: unchanged
    assert "Sounds like Casey</b> · 1 s ago" in vl and "<script>" not in vl and "script" not in vl
