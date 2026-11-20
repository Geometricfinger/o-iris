"""Per-stage timing counters (oiris/timing.py) and their wiring into the engine, the voice listener, the output
arbiter and the live snapshot. Fake clocks, fakes and synthetic audio only: no real sound, microphone or network."""
# ruff: noqa: F811
import json, threading, time
from pathlib import Path

import numpy as np
import pytest

from oiris.adapters import sim
from oiris.config import Config
from oiris.engine import Engine
from oiris.headless import _round, run_sync
from oiris.modules.output import CONFIRMED, VOICE_LINK, NullOutputPlayer, OutputArbiter
from oiris.modules.voice_link.listener import VoiceListener, _TimedSpeaker
from oiris.timing import MAX_STAGES, MS_DECIMALS, StageTimer, percentile
from tests.test_live import SIM, _runner, _step_until, live_env  # noqa: F401  (fixture reuse)
from tests.test_voice_link import synth_voice
from tests.test_voice_link_live import FakeCapture, FakeVM, listener, quiet, wait_until

GOLDEN = Path(__file__).resolve().parent.parent / "golden" / "expected"
OVERHEAD_CALLS = 20000
OVERHEAD_BOUND_S = 20e-6      # mean cost per timed block; real cost is around a microsecond, generous for CI
THREADS, PER_THREAD = 8, 2000
TICK_S = 0.001
ENGINE_STAGES = {"step", "collect", "health", "scores", "fusion", "guard"}
SUMMARY_KEYS = {"count", "p50_ms", "p95_ms", "max_ms"}


class Clock:
    """A manual monotonic clock: tests and fakes advance it."""

    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


class Tick(Clock):
    """Advances TICK_S on every read: any code path gives non-zero, deterministic durations."""

    def __call__(self) -> float:
        self.t += TICK_S
        return self.t


def only_numbers(obj) -> bool:
    """Every leaf is an int or float (bool excluded) and every key a short code-chosen stage/field name."""
    if isinstance(obj, dict):
        return all(isinstance(k, str) and only_numbers(v) for k, v in obj.items())
    return isinstance(obj, (int, float)) and not isinstance(obj, bool)


# ---- StageTimer ---------------------------------------------------------------------------------------------
def test_percentiles_on_known_inputs():
    vals = [float(i) for i in range(1, 101)]
    assert percentile(vals, 50) == 50 and percentile(vals, 95) == 95 and percentile(vals, 100) == 100
    assert percentile([7.0], 50) == 7.0 and percentile([], 95) == 0.0
    t = StageTimer()
    for i in range(100, 0, -1):                     # order of arrival does not matter
        t.record("stage", i / 1000)
    assert t.summary() == {"stage": {"count": 100, "p50_ms": 50.0, "p95_ms": 95.0, "max_ms": 100.0}}
    t.record("tiny", 1.23456789e-3)
    assert t.summary()["tiny"]["max_ms"] == round(1.23456789, MS_DECIMALS)


def test_ring_buffer_is_bounded_and_count_keeps_total():
    t = StageTimer(window=10)
    for i in range(1000):
        t.record("s", float(i))
    assert len(t._stages["s"]) == 10
    s = t.summary()["s"]
    assert s["count"] == 1000 and s["max_ms"] == 999000.0 and s["p50_ms"] == 994000.0   # the last 10 only


def test_stage_names_and_values_are_bounded_and_never_raise():
    t = StageTimer()
    for bad in ("Alex Smith", "", "person:1234", "x" * 100, None, 3):
        t.record(bad, 0.1)                          # type: ignore[arg-type]
    for v in (-1.0, float("nan"), float("inf"), "1"):
        t.record("ok", v)                           # type: ignore[arg-type]
    assert t.summary() == {}
    for i in range(MAX_STAGES + 10):
        t.record(f"s{i}", 0.001)
    assert len(t.summary()) == MAX_STAGES
    t.reset()
    assert t.summary() == {}


def test_thread_safety_under_concurrent_records():
    t = StageTimer(window=THREADS * PER_THREAD)
    go = threading.Event()

    def work(k):
        go.wait()
        for i in range(PER_THREAD):
            t.record("shared", 0.001)
            t.record(f"own{k}", 0.002)
            if i % 100 == 0:
                t.summary()                         # reads interleave with writes

    th = [threading.Thread(target=work, args=(k,)) for k in range(THREADS)]
    for x in th:
        x.start()
    go.set()
    for x in th:
        x.join()
    s = t.summary()
    assert s["shared"]["count"] == THREADS * PER_THREAD and len(t._stages["shared"]) == THREADS * PER_THREAD
    assert all(s[f"own{k}"]["count"] == PER_THREAD for k in range(THREADS))


def test_context_manager_records_even_when_the_block_raises():
    clk = Clock()
    t = StageTimer(clock=clk)
    with t.time("block"):
        clk.advance(0.25)
    with pytest.raises(ValueError):
        with t.time("block"):
            clk.advance(0.5)
            raise ValueError("passes through")
    start = t.now()
    clk.advance(0.125)
    t.since("since", start)
    s = t.summary()
    assert s["block"] == {"count": 2, "p50_ms": 250.0, "p95_ms": 500.0, "max_ms": 500.0}
    assert s["since"]["max_ms"] == 125.0


def test_overhead_is_microseconds():
    t = StageTimer()
    t0 = time.perf_counter()
    for _ in range(OVERHEAD_CALLS):
        with t.time("x"):
            pass
    per_call = (time.perf_counter() - t0) / OVERHEAD_CALLS
    assert per_call < OVERHEAD_BOUND_S, per_call
    assert t.summary()["x"]["count"] == OVERHEAD_CALLS


# ---- engine ---------------------------------------------------------------------------------------------------
def test_engine_step_stages_with_injected_clock():
    clk = Clock()
    timer = StageTimer(clock=clk)
    e = Engine(Config(), timer=timer)
    for a in sim.make("camera_lens"):
        a.open({}); e.add_adapter(a)
    real_collect = e._collect
    e._collect = lambda: (clk.advance(0.004), real_collect())[1]          # type: ignore[method-assign]
    hits = []
    e.on_event = lambda ev: (clk.advance(0.030), hits.append(1))
    run_sync(e, 3)
    s = timer.summary()
    assert set(s) == ENGINE_STAGES | {"events"} and hits == [1, 1, 1]
    assert all(v["count"] == 3 for v in s.values())
    assert s["collect"]["max_ms"] == 4.0 and s["events"]["p50_ms"] == 30.0 and s["step"]["p50_ms"] == 34.0
    assert s["fusion"]["max_ms"] == 0.0 and only_numbers(s)


def test_engine_without_a_level_has_no_events_stage():
    e = Engine(Config(), timer=StageTimer(clock=Tick()))
    for a in sim.make("quiet"):
        a.open({}); e.add_adapter(a)
    run_sync(e, 2)
    assert set(e.timer.summary()) == ENGINE_STAGES


@pytest.mark.parametrize("scenario", sorted(p.stem for p in GOLDEN.glob("*.json")))
def test_golden_outputs_unchanged_with_timing(scenario):
    """The results do not depend on the timer: an odd (ticking) clock gives the golden outputs exactly."""
    e = Engine(Config(), timer=StageTimer(clock=Tick()))
    for a in sim.make(scenario, seed=0):
        a.open({}); e.add_adapter(a)
    got = json.loads(json.dumps(_round(run_sync(e, 3)), sort_keys=True))
    assert got == json.loads((GOLDEN / f"{scenario}.json").read_text(encoding="utf-8"))
    assert e.timer.summary()["step"]["count"] == 3


# ---- voice listener -------------------------------------------------------------------------------------------
class SlowVM(FakeVM):
    """process_audio advances the timer clock, and calls the speaker the way core.py does."""

    def __init__(self, clk, **kw):
        super().__init__(**kw)
        self.clk = clk
        self.speaker = FakeSpeaker(clk)

    def process_audio(self, audio, sample_rate, place=None, now=None):
        self.clk.advance(0.010)                         # VAD and the rest
        self.speaker.speaker_embedding(audio, sample_rate)
        self.speaker.spoof_score(audio, sample_rate)
        return super().process_audio(audio, sample_rate, place, now)


class FakeSpeaker:
    status = {"speaker_embed": "ok"}

    def __init__(self, clk):
        self.clk = clk

    def speaker_embedding(self, audio, sr):
        self.clk.advance(0.040)
        return np.ones(4, np.float32)

    def spoof_score(self, audio, sr):
        self.clk.advance(0.005)
        return 0.0


def test_listener_stages_with_injected_clock():
    clk = Clock()
    timer = StageTimer(clock=clk)
    vm, cap = SlowVM(clk), FakeCapture()
    inner = vm.speaker
    li = listener(vm, cap, timer=timer)
    try:
        assert isinstance(vm.speaker, _TimedSpeaker) and vm.speaker.status == inner.status   # passes through
        cap.feed(np.concatenate([synth_voice("A", seconds=2.0, seed=1), quiet(1.5)]))
        assert wait_until(lambda: len(vm.calls) == 1)
        assert wait_until(lambda: timer.summary().get("decision", {}).get("count") == 1)
        assert wait_until(li._q.empty)
    finally:
        li.stop()
    assert vm.speaker is inner                          # unwrapped at stop
    s = timer.summary()
    assert set(s) == {"queue_wait", "lock_wait", "process", "embed", "spoof", "decision"}
    assert s["embed"]["max_ms"] == 40.0 and s["spoof"]["max_ms"] == 5.0
    assert s["process"]["max_ms"] == 55.0 and s["decision"]["max_ms"] == 55.0
    assert s["queue_wait"]["count"] == len(cap.handed) and s["lock_wait"]["max_ms"] == 0.0
    assert only_numbers(s)


def test_listener_without_a_speaker_attribute_still_times_and_drops_nothing():
    vm, cap = FakeVM(), FakeCapture()
    li = listener(vm, cap, timer=StageTimer())
    try:
        cap.feed(np.concatenate([synth_voice("A", seconds=2.0, seed=1), quiet(1.5)]))
        assert wait_until(lambda: len(vm.calls) == 1)
    finally:
        li.stop()
    assert not hasattr(vm, "speaker") and all(not b.any() for b in cap.handed)   # every block still zeroed
    assert {"queue_wait", "process"} <= set(li.timer.summary())


def test_listener_default_timer_is_its_own():
    a, b = VoiceListener(FakeVM(), FakeCapture()), VoiceListener(FakeVM(), FakeCapture())
    assert isinstance(a.timer, StageTimer) and a.timer is not b.timer


# ---- output arbiter -------------------------------------------------------------------------------------------
class TimedPlayer(NullOutputPlayer):
    def __init__(self, clk):
        super().__init__()
        self.clk = clk

    def chime(self):
        super().chime()
        self.clk.advance(0.2)

    def say(self, text):
        super().say(text)
        self.clk.advance(1.0)


def test_arbiter_wait_and_play_per_channel():
    clk = Clock()
    timer = StageTimer(clock=clk)
    arb = OutputArbiter(TimedPlayer(clk), timer=timer, start=False, clock=clk)
    try:
        arb.request("voice_link", VOICE_LINK, "vl:1", text="sounds like Alex", sound="chime")
        clk.advance(0.3)                                # waits in the queue
        assert arb.pump() == 1
        arb.request("voice_link", VOICE_LINK, "vl:2", sound="chime")
        assert arb.pump() == 1
    finally:
        arb.close()
    s = timer.summary()
    assert set(s) == {"chime_wait", "chime_play", "speech_wait", "speech_play"}
    assert s["chime_wait"] == {"count": 2, "p50_ms": 0.0, "p95_ms": 300.0, "max_ms": 300.0}
    assert s["chime_play"]["p50_ms"] == 200.0 and s["speech_play"]["max_ms"] == 1000.0
    assert s["speech_wait"]["max_ms"] == 500.0          # speech after the chime: accepted to start of speech
    assert only_numbers(s)


def test_arbiter_popup_only_records_no_sound_timing():
    timer = StageTimer()
    arb = OutputArbiter(NullOutputPlayer(), timer=timer, start=False)
    try:
        assert str(arb.request("alert", CONFIRMED, "alert:A4:confirmed", text="Confirmed signal")) == "played"
        arb.pump()
    finally:
        arb.close()
    assert timer.summary() == {}


# ---- live snapshot ----------------------------------------------------------------------------------------------
def test_live_snapshot_timing_shape(live_env):
    r = _runner(live_env, [SIM])
    r.start(background=False)
    try:
        assert r.engine is not None and r.engine.timer is r.timers["engine"]
        assert r.output is not None and r.output.timer is r.timers["output"]
        _step_until(r, 3)
        timing = r.snapshot()["timing"]
    finally:
        r.stop()
    assert set(timing) == {"engine", "voice", "output"}
    assert ENGINE_STAGES <= set(timing["engine"]) and timing["voice"] == {}
    assert all(set(v) == SUMMARY_KEYS for part in timing.values() for v in part.values())
    assert timing["engine"]["step"]["count"] >= 3 and only_numbers(timing)
    after = r.snapshot()["timing"]                      # kept after stop
    assert after["engine"]["step"]["count"] >= timing["engine"]["step"]["count"]
