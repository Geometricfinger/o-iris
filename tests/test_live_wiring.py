"""Live wiring of the power governor, latency budgets, the record chain, scheduled retention, the heard card's photo and
the listener's public microphone API (ADR-0061, ADR-0066, ADR-0057 decision 4, ADR-0063). Synthetic only: fake clocks,
the power_sim scenes, fake captures, the TEST-ONLY face and speaker backends, a temp store. No microphone, camera,
network, real photo or real person."""
# ruff: noqa: F811
import dataclasses, inspect, json, re, socket, threading, time
from types import SimpleNamespace

import numpy as np
import pytest
from nacl.signing import SigningKey

from oiris import live as live_mod
from oiris.adapters import power_sim
from oiris.adapters.capture_sim import SimIndicator, SimMuteSwitch
from oiris.config import (POWER_INTERVAL_DEFAULT_S, POWER_SIM_SCENES, RETENTION_CHECK_DEFAULT_S, Config, ConfigError,
                          power_sources)
from oiris.dashboard import server
from oiris.dashboard.server import CSP, render
from oiris.guard import retention
from oiris.guard.vocabulary import violations
from oiris.live import LIVE_SIGHTING_ANCHOR, LIVE_SIGHTING_CHAIN, POWER_NO_CONSUMER, LiveRunner, LiveSettings, open_store
from oiris.power.duty import DUTY_FULL, PLAN_KEYS
from oiris.power.governor import PowerReading
from oiris.store import sightings as sightings_mod
from oiris.store.sightings import verify_log
from tests.test_dashboard_live import _snap
from tests.test_live import SIM, _Ev
from tests.test_live import live_env  # noqa: F401  (fixture reuse)
from tests.test_live_capture import MUTE_AT, OPEN_AT, Clock, interlock, voice_ready
from tests.test_voice_link import synth_voice
from tests.test_voice_link import env  # noqa: F401  (fixture reuse)
from tests.test_voice_link_live import (FakeCapture, FakeVM, drained, listener, live_cfg, live_runner, quiet,
                                        wait_until)
from tests.test_voice_link_live import enrolled  # noqa: F401  (fixture reuse)

STEP_S = POWER_INTERVAL_DEFAULT_S        # the fake clock moves one power interval per step
MAX_POWER_STEPS = 200                    # 2000 simulated seconds: hot_day turns hot near 365 s
HOT_DUTY, PERIOD_S = 0.25, 10.0          # DEFAULT_PLANS["hot"] listen duty and period
JPEG = b"\xff\xd8\xff\xe0" + b"synthetic-not-a-photo" + b"\xff\xd9"   # bytes only; never decoded, no image of anyone
SECONDS_PER_DAY = 86400.0
SHIFT_DAYS = 2


def runner(live_env, clock, **kw):
    cfg, reg, _ = live_env
    cfg = dataclasses.replace(cfg, **kw.pop("cfg", {})).validate()
    return LiveRunner(cfg, LiveSettings(adapters=[SIM], hz=20.0, site="lab"), registry=reg, clock=clock, **kw)


def power_cfg(env, source: str, **voice) -> Config:
    cfg = live_cfg(env, **voice)
    cfg.power = {"source": source}
    return cfg.validate()


def step_until(r, clk, cond, steps: int = MAX_POWER_STEPS) -> bool:
    for _ in range(steps):
        if cond():
            return True
        clk.t += STEP_S
        r.step()
    return cond()


class ScriptedPower:
    """A power reading source with a fixed battery level, timed in the runner's clock."""

    def __init__(self, clock, battery_pct: float, charging: bool = False):
        self.clock, self.battery_pct, self.charging = clock, battery_pct, charging
        self.plans: list = []

    def read(self, plan):
        self.plans.append(plan)
        return PowerReading(t=self.clock(), battery_pct=self.battery_pct, charging=self.charging)


# ---- config -------------------------------------------------------------------------------------------
def test_power_and_retention_config_keys_defaults_and_ranges():
    cfg = Config().validate()
    assert cfg.retention_check_s == RETENTION_CHECK_DEFAULT_S == 3600.0
    assert live_mod.power_live_settings(cfg.power) == ("none", POWER_INTERVAL_DEFAULT_S)
    assert set(POWER_SIM_SCENES) == set(power_sim.SCENES)            # the config's copy matches the simulation
    for ok in power_sources():
        Config(power={"source": ok, "interval_s": 5}).validate()
    Config(power={"source": "sim:hot_day", "skin_hot_enter_c": 40.0, "skin_hot_exit_c": 39.0}).validate()
    for bad in ({"source": "sim:volcano"}, {"source": "android"}, {"interval_s": 0.5}, {"interval_s": 601},
                {"interval_s": True}, {"bogus": 1}):
        with pytest.raises(ConfigError):
            Config(power=bad).validate()
    for bad_s in (59.0, 86401.0, float("nan")):
        with pytest.raises(ConfigError):
            Config(retention_check_s=bad_s).validate()


# ---- power governor -------------------------------------------------------------------------------------
def test_power_sources_built_from_config_without_reading():
    clk = Clock(0.0)
    assert live_mod.build_power_source(Config().validate(), clk) is None
    mac = live_mod.build_power_source(Config(power={"source": "macos"}).validate(), clk)
    assert isinstance(mac, live_mod._MacPower) and mac.battery.clock is clk          # pmset is not run here
    sim = live_mod.build_power_source(Config(power={"source": "sim:low_battery"}).validate(), clk)
    assert isinstance(sim, live_mod._SimPower) and sim.sim.scene_name == "low_battery"


def test_source_none_changes_nothing(enrolled):
    env = enrolled.env
    clk = Clock(100.0)
    r = live_runner(env, live_cfg(env).validate(), voice_capture=FakeCapture(), clock=clk)
    r.start(background=False)
    try:
        for _ in range(3):
            clk.t += STEP_S
            r.step()
        snap = r.snapshot()
        assert "power" not in snap and r._governor is None and r._power_source is None
        assert r.voice.duty() == {"duty": DUTY_FULL, "period_s": PERIOD_S, "matching": True}
        page = render(snap["result"], snap)
        assert "id=PWR" not in page and "Power:" not in page
    finally:
        r.stop()


def test_hot_day_reaches_hot_and_applies_the_listening_duty(enrolled):
    env = enrolled.env
    clk = Clock(0.0)
    r = live_runner(env, power_cfg(env, "sim:hot_day"), voice_capture=FakeCapture(), clock=clk)
    r.start(background=False)
    try:
        p = r.snapshot()["power"]
        assert p["source"] == "sim:hot_day" and p["level"] == "normal" and r.voice.duty()["duty"] == DUTY_FULL
        assert step_until(r, clk, lambda: r.snapshot()["power"]["level"] == "hot")
        p = r.snapshot()["power"]
        assert set(p) == {"source", "level", "reasons", "plan", "battery_pct", "charging", "skin_c", "no_consumer"}
        assert "skin_hot" in p["reasons"] and p["charging"] is False and isinstance(p["skin_c"], float)
        assert set(p["plan"]) == set(PLAN_KEYS) and p["plan"]["listen_duty"] == HOT_DUTY
        assert p["no_consumer"] == list(POWER_NO_CONSUMER)                  # shown for the camera and glasses paths
        assert p["plan"]["camera_max_fps"] < 30 and p["plan"]["face_match"] is False
        assert r.voice.duty() == {"duty": HOT_DUTY, "period_s": PERIOD_S, "matching": True}
        page = render(r.snapshot()["result"], r.snapshot())
        line = page.split("<p id=PWR")[1].split("</p>")[0]
        assert "Power: <b>hot, throttled (warm against the skin), battery" in line
        assert "voice listening 25 % of the time" in line and "paused" not in line
        assert not violations(re.sub(r"<[^>]+>", " ", line))
    finally:
        r.stop()


def test_the_closed_loop_simulation_is_fed_the_plan_in_force(live_env):
    clk = Clock(0.0)
    r = runner(live_env, clk, cfg={"power": {"source": "sim:hot_day", "interval_s": STEP_S}})
    r.start(background=False)
    try:
        assert step_until(r, clk, lambda: r.snapshot()["power"]["level"] == "hot")
        sim = r._power_source.sim
        normal = power_sim.plan_power_w(r._governor.settings.plans["normal"])
        assert sim.power_w == pytest.approx(normal)        # the reading that turned hot ran under the normal plan
        clk.t += STEP_S
        r.step()                                            # the next interval runs under the hot plan
        assert r.snapshot()["power"]["level"] == "hot"
        assert sim.power_w == pytest.approx(power_sim.plan_power_w(r._plan())) and sim.power_w < normal
    finally:
        r.stop()


def test_readings_come_every_interval_only(live_env):
    clk = Clock(0.0)
    src = ScriptedPower(clk, 80.0)
    r = runner(live_env, clk, power_source=src, cfg={"power": {"interval_s": 30.0}})
    r.start(background=False)
    try:
        assert len(src.plans) == 1                                          # the first reading at start
        for _ in range(5):
            clk.t += 5.0
            r.step()
        assert len(src.plans) == 1                                          # 25 s < 30 s
        clk.t += 5.0
        r.step()
        assert len(src.plans) == 2
        assert r.snapshot()["power"]["source"] == "injected"
    finally:
        r.stop()


def test_a_failing_reading_keeps_the_last_plan(live_env, caplog):
    clk = Clock(0.0)
    src = ScriptedPower(clk, 5.0)                                           # critical at once
    r = runner(live_env, clk, power_source=src)
    r.start(background=False)
    try:
        assert r.snapshot()["power"]["level"] == "critical"
        src.read = lambda plan: (_ for _ in ()).throw(OSError("sensor gone"))
        with caplog.at_level("ERROR", logger=live_mod.__name__):
            clk.t += STEP_S
            assert r.step() is not None                                     # the loop goes on
        assert r.snapshot()["power"]["level"] == "critical"
        assert any("power reading failed" in m for m in caplog.messages)
    finally:
        r.stop()


def test_critical_pauses_matching_and_says_so(enrolled):
    env = enrolled.env
    clk = Clock(0.0)
    cap = FakeCapture()
    r = live_runner(env, live_cfg(env).validate(), voice_capture=cap, clock=clk, power_source=ScriptedPower(clk, 5.0))
    r.start(background=False)
    try:
        p = r.snapshot()["power"]
        assert p["level"] == "critical" and p["reasons"] == ["battery_critical"] and p["plan"]["voice_match"] is False
        assert r.voice.duty()["matching"] is False
        cap.feed(np.concatenate([synth_voice("A", seconds=4.0, seed=50), quiet(1.5)]))
        drained(r.voice)
        assert r.snapshot()["voice"]["heard"] == 0 and r.voice.held_off > 0       # nothing started, nothing matched
        assert r.snapshot()["voice"]["state"] == "listening"                      # the microphone itself is unchanged
        page = render(r.snapshot()["result"], r.snapshot())
        line = page.split("<p id=PWR")[1].split("</p>")[0]
        assert "Power: <b>critical (battery critically low), battery 5 %</b>" in line
        assert "<b>voice matching paused</b>" in line
    finally:
        r.stop()


def test_matching_off_holds_new_utterances_and_on_again_resumes():
    vm, cap = FakeVM(), FakeCapture()
    li = listener(vm, cap)
    try:
        li.set_matching(False)
        cap.feed(np.concatenate([synth_voice("A", seconds=3.5, seed=1), quiet(1.5)]))
        drained(li)
        assert vm.calls == [] and li.held_off > 0
        li.set_matching(True)
        cap.feed(np.concatenate([synth_voice("A", seconds=3.5, seed=2), quiet(1.5)]))
        assert wait_until(lambda: len(vm.calls) == 1)
    finally:
        li.stop()


def test_an_utterance_in_progress_is_finished_not_cut():
    vm, cap = FakeVM(), FakeCapture()
    clk = Clock(0.0)
    li = listener(vm, cap, duty_clock=clk)
    try:
        voice = synth_voice("A", seconds=3.5, seed=4)
        head = int(2.0 * 16000)
        cap.feed(voice[:head])
        drained(li)
        assert li._in_utt                                     # the utterance has started
        li.set_duty(0.0, PERIOD_S)                            # listening off now (critical)
        cap.feed(np.concatenate([voice[head:], quiet(1.5)]))
        assert wait_until(lambda: len(vm.calls) == 1)
        drained(li)
        whole = vm.calls[0]
        assert whole > 3.0, whole                             # the whole utterance, not the part before the change
        cap.feed(np.concatenate([synth_voice("A", seconds=3.5, seed=5), quiet(1.5)]))
        drained(li)
        assert len(vm.calls) == 1 and li.held_off > 0         # no new utterance while off
    finally:
        li.stop()


def test_listening_duty_windows_follow_the_duty_clock():
    vm, cap = FakeVM(), FakeCapture()
    clk = Clock(7.0)                                          # 7 % 10 = 7: outside the first 2.5 s of the period
    li = listener(vm, cap, duty_clock=clk)
    try:
        li.set_duty(HOT_DUTY, PERIOD_S)
        cap.feed(np.concatenate([synth_voice("A", seconds=3.5, seed=6), quiet(1.5)]))
        drained(li)
        assert vm.calls == []
        clk.t = 21.0                                          # 21 % 10 = 1: inside the on window
        cap.feed(np.concatenate([synth_voice("A", seconds=3.5, seed=7), quiet(1.5)]))
        assert wait_until(lambda: len(vm.calls) == 1)
        with pytest.raises(ValueError):
            li.set_duty(1.5, PERIOD_S)
        assert li.duty() == {"duty": HOT_DUTY, "period_s": PERIOD_S, "matching": True}
    finally:
        li.stop()


# ---- latency budgets ------------------------------------------------------------------------------------
SLOW_DECISION_S = 2.0       # twice the voice.decision budget (1000 ms)
SAMPLES = 20                # BudgetMonitor's minimum before a p95 counts


def test_an_over_budget_stage_appears_in_the_snapshot_and_health(live_env):
    clk = Clock(0.0)
    r = runner(live_env, clk)
    r.start(background=False)
    try:
        r.step()
        snap = r.snapshot()
        assert snap["budgets"] == [] and "latency" not in snap["health"]
        assert "id=LAT" not in render(snap["result"], snap)
        for _ in range(SAMPLES):
            r.timers["voice"].record("decision", SLOW_DECISION_S)
        r.step()
        snap = r.snapshot()
        (o,) = [b for b in snap["budgets"] if b["part"] == "voice"]
        assert o["stage"] == "decision" and o["p95_ms"] == 2000.0 and o["budget_ms"] == 1000.0 and o["count"] == SAMPLES
        lat = snap["health"]["latency"]
        assert lat["health"] == "degraded" and "voice decision p95 2000 ms (budget 1000 ms)" in lat["reason"]
        assert "latency" not in r.engine.results[-1]["health"]                   # the engine's own result is untouched
        page = render(snap["result"], snap)
        assert "Latency over budget: voice decision p95 2000 ms (budget 1000 ms)" in page
        d5 = page.split("<section id=D5>")[1].split("</section>")[0]
        d1 = page.split("<section id=D1>")[1].split("</section>")[0]
        assert "latency" not in d5 and "latency" not in d1                       # a measurement, not a sensor
        r.timers["voice"].reset()
        r.step()
        assert r.snapshot()["budgets"] == [] and "latency" not in r.snapshot()["health"]
    finally:
        r.stop()


def test_configured_budgets_are_used(live_env):
    clk = Clock(0.0)
    r = runner(live_env, clk, cfg={"budgets": {"voice.decision": 5000.0}})
    r.start(background=False)
    try:
        for _ in range(SAMPLES):
            r.timers["voice"].record("decision", SLOW_DECISION_S)
        r.step()
        assert not [b for b in r.snapshot()["budgets"] if b["part"] == "voice"]
    finally:
        r.stop()


# ---- record chain (ADR-0066) ----------------------------------------------------------------------------
def chain_report(cfg) -> dict:
    ks, db = open_store(cfg)
    try:
        return verify_log(db, ks.subkey(LIVE_SIGHTING_CHAIN), SigningKey(ks.subkey(LIVE_SIGHTING_ANCHOR)).verify_key)
    finally:
        db.close()


def run_once(live_env, clk, events, **kw) -> LiveRunner:
    r = runner(live_env, clk, **kw)
    r.start(background=False)
    try:
        for i, (sig, level) in enumerate(events):
            r._on_event(_Ev(sig, level))
            if i % 2:
                clk.t += 1.0
    finally:
        r.stop()
    return r


def test_stop_anchors_the_tail(live_env):
    clk = Clock(0.0)
    r = runner(live_env, clk)
    r.start(background=False)
    try:
        assert r.snapshot()["chain"]["state"] == "new"
        r._on_event(_Ev("A4", "fyi")); r._on_event(_Ev("A2", "watch"))
        assert r._log.chain.has_unanchored()                    # no 60 s on the fake clock: not anchored yet
        anchors_before = r.db.execute("SELECT COUNT(*) FROM chain_anchor").fetchone()[0]
    finally:
        r.stop()
    rep = chain_report(live_env[0])
    assert anchors_before == 0 and rep["status"] == "ok" and rep["records"] == 2
    assert rep["unanchored_tail"] == 0 and rep["anchors"] == 1     # close() at stop anchored the tail


def test_tick_anchors_an_idle_tail_each_step(live_env):
    clk = Clock(0.0)
    r = runner(live_env, clk)
    r.start(background=False)
    try:
        r._on_event(_Ev("A4", "fyi"))
        r.step()
        assert r.db.execute("SELECT COUNT(*) FROM chain_anchor").fetchone()[0] == 0
        clk.t += 61.0                                           # past ANCHOR_EVERY_SECONDS on the fake clock
        r.step()
        assert r.db.execute("SELECT COUNT(*) FROM chain_anchor").fetchone()[0] == 1
    finally:
        r.stop()


def test_two_runs_continue_one_chain(live_env):
    clk = Clock(0.0)
    run_once(live_env, clk, [("A4", "fyi"), ("A2", "watch")])
    r = runner(live_env, clk)
    r.start(background=False)
    try:
        ch = r.snapshot()["chain"]
        assert ch["state"] == "resumed" and ch["records"] == 2 and ch["unanchored_tail"] == 0 and ch["segment"] == 1
        r._on_event(_Ev("A1", "watch"))
        page = render(r.snapshot()["result"], r.snapshot())
        assert "Record chain: <b>continued (2 records)</b>" in page
    finally:
        r.stop()
    rep = chain_report(live_env[0])
    assert rep["status"] == "ok" and rep["segments"] == 1 and rep["records"] == 3 and rep["last_seq"] == 3


def test_a_scheduled_purge_at_a_boundary_verifies_ok(live_env, monkeypatch):
    cfg = live_env[0]
    clk = Clock(0.0)
    run_once(live_env, clk, [("A4", "fyi"), ("A2", "watch")], cfg={"retention_days": 1})
    calls, real = [], retention.purge
    monkeypatch.setattr(retention, "purge", lambda db, days=30, now=None: (calls.append(days), real(db, days, now))[1])
    r = runner(live_env, clk, cfg={"retention_days": 1, "retention_check_s": 600.0})
    r.start(background=False)
    try:
        assert calls == [1] and r.snapshot()["records"]["purged_rows"] == 0     # at open: nothing old enough yet
        later = time.time() + SHIFT_DAYS * SECONDS_PER_DAY                       # two days on, for the sighting log
        monkeypatch.setattr(sightings_mod, "time", SimpleNamespace(time=lambda: later))
        clk.t += 599.0
        r.step()
        assert calls == [1]
        clk.t += 1.0
        r.step()                                                                 # retention_check_s on the fake clock
        assert calls == [1, 1] and r.snapshot()["records"]["purged_rows"] >= 2
        r._on_event(_Ev("A1", "watch"))
    finally:
        r.stop()
    rep = chain_report(cfg)
    assert rep["status"] == "ok" and rep["purged_gaps"] == 1 and rep["records"] >= 2, rep
    ks, db = open_store(cfg)
    try:
        assert db.execute("SELECT COUNT(*) FROM audit_log WHERE kind='chain_purge'").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM sighting").fetchone()[0] == 1
    finally:
        db.close()


def test_without_the_sighting_log_the_old_purge_path_runs(live_env, monkeypatch):
    calls, real = [], retention.purge
    monkeypatch.setattr(retention, "purge", lambda db, days=30, now=None: (calls.append(days), real(db, days, now))[1])
    cfg, _, _ = live_env
    flags = {**cfg.flags, "M-05": False}
    r = runner(live_env, Clock(0.0), cfg={"flags": flags})
    r.start(background=False)
    try:
        assert r._log is None and calls == [cfg.retention_days] and "chain" not in r.snapshot()
    finally:
        r.stop()


def test_the_dashboard_chain_line_in_plain_words_and_escaped():
    snap = _snap(None)
    snap["chain"] = {"state": "restarted", "reason": "legacy", "segment": 2, "records": 5}
    assert "Record chain: <b>restarted (records from before the resumable chain)</b>" in render(None, snap)
    snap["chain"] = {"state": "restarted", "reason": "<script>", "records": 1}
    page = render(None, snap)
    assert "restarted (unknown reason)" in page and "<script>" not in page
    snap["chain"] = {"state": "new", "records": 0}
    assert "Record chain: <b>new</b>" in render(None, snap)


# ---- scheduled retention --------------------------------------------------------------------------------
def test_scheduled_purge_runs_on_the_fake_clock_and_shows_on_the_dashboard(live_env, monkeypatch):
    calls, real = [], retention.purge
    monkeypatch.setattr(retention, "purge", lambda db, days=30, now=None: (calls.append(days), real(db, days, now))[1])
    clk = Clock(500.0)
    r = runner(live_env, clk, cfg={"retention_check_s": 600.0})
    r.start(background=False)
    try:
        assert len(calls) == 1
        rec = r.snapshot()["records"]
        assert isinstance(rec["last_purge_at"], float) and rec["retention_check_s"] == 600.0
        for _ in range(3):
            clk.t += 199.0
            r.step()
        assert len(calls) == 1
        clk.t += 3.0
        r.step()
        assert len(calls) == 2
        clk.t += 600.0
        r.step()
        assert len(calls) == 3
        d1 = render(r.snapshot()["result"], r.snapshot()).split("<section id=D1>")[1].split("</section>")[0]
        assert re.search(r"retention purge last run \d\d:\d\d, every 10 min", d1)
    finally:
        r.stop()
    clk.t += 6000.0
    assert r.step() is None and len(calls) == 3                              # never after close


def test_the_default_check_interval_is_an_hour(live_env, monkeypatch):
    calls, real = [], retention.purge
    monkeypatch.setattr(retention, "purge", lambda db, days=30, now=None: (calls.append(days), real(db, days, now))[1])
    clk = Clock(0.0)
    r = runner(live_env, clk)
    r.start(background=False)
    try:
        clk.t += RETENTION_CHECK_DEFAULT_S - 1
        r.step()
        assert len(calls) == 1
        clk.t += 1
        r.step()
        assert len(calls) == 2
    finally:
        r.stop()


# ---- photo on the heard card ----------------------------------------------------------------------------
def raw(srv, req: bytes) -> tuple[int, str, bytes]:
    a, b = socket.socketpair()
    b.settimeout(5)

    def handle():
        try:
            srv.RequestHandlerClass(a, ("127.0.0.1", 0), srv)
        finally:
            a.close()
    t = threading.Thread(target=handle, daemon=True)
    t.start()
    b.sendall(req)
    chunks = []
    while chunk := b.recv(65536):
        chunks.append(chunk)
    t.join(5); b.close()
    head, _, body = b"".join(chunks).partition(b"\r\n\r\n")
    return int(head.split()[1]), head.decode("latin-1").lower(), body


def get(srv, path: str, origin: str | None = "http://127.0.0.1:0", site: str | None = None,
        host: str = "127.0.0.1:0") -> tuple[int, str, bytes]:
    hdr = f"GET {path} HTTP/1.0\r\nHost: {host}\r\n"
    hdr += f"Origin: {origin}\r\n" if origin else ""
    hdr += f"Sec-Fetch-Site: {site}\r\n" if site else ""
    return raw(srv, (hdr + "\r\n").encode())


def card_snap(*cards) -> dict:
    snap = _snap(None)
    snap["voice"] = {"state": "listening", "reason": "", "mode": "ambient", "notice": "", "listen_left_s": 0.0,
                     "cards": list(cards), "heard": len(cards)}
    return snap


def page_token(page: str) -> str:
    return re.search(r'/voice/photo\?k=[^&"]+&amp;token=([^"&]+)"', page).group(1)


def test_photo_endpoint_serves_only_a_current_cards_photo_with_token_and_origin():
    served = []

    def photo(key):
        served.append(key)
        return JPEG if key == "K1" else None
    card = {"name": "Alex", "score": 0.9, "face_agrees": False, "direction": None, "seconds_ago": 1.0, "photo": "K1"}
    srv = server.make_server(Config(), port=0, live=lambda: card_snap(card), bind_and_activate=False, on_photo=photo)
    try:
        code, head, page = get(srv, "/", origin=None)
        assert code == 200 and "img-src 'self'" in head
        tok = page_token(page.decode())
        assert len(tok) >= 32
        path = f"/voice/photo?k=K1&token={tok}"
        code, head, body = get(srv, path)                                        # Origin names this dashboard
        assert code == 200 and body == JPEG
        assert "content-type: image/jpeg" in head and "cache-control: no-store" in head
        assert "x-content-type-options: nosniff" in head
        code, _, body = get(srv, path, origin=None, site="same-origin")          # what a browser's <img> sends
        assert code == 200 and body == JPEG
        n = len(served)
        for bad in (dict(path="/voice/photo?k=K1"),                              # no token
                    dict(path=f"/voice/photo?k=K1&token={'x' if tok[0] != 'x' else 'y'}{tok[1:]}"),
                    dict(path=path, origin="http://evil.example"),               # foreign Origin
                    dict(path=path, origin="http://127.0.0.1:9"),                # another local port
                    dict(path=path, origin=None, site="cross-site"),
                    dict(path=path, origin=None, site="same-site"),
                    dict(path=path, origin=None),                                # no proof of origin at all
                    dict(path=path, origin="http://127.0.0.1:0", site="cross-site"),
                    dict(path=path, host="rebind.example:0", origin="http://rebind.example:0")):
            assert get(srv, **bad)[0] == 403, bad
        assert len(served) == n                                                  # refused before any lookup
        code, head, body = get(srv, f"/voice/photo?k=K2&token={tok}")            # not a current card's key
        assert code == 404 and body == b"no photo" and "cache-control: no-store" in head
        assert get(srv, f"/voice/photo?token={tok}")[0] == 404
        assert get(srv, f"/voice/photos?token={tok}", origin=None)[0] == 200     # no listing: just the page
    finally:
        srv.server_close()


def test_no_photo_source_means_no_endpoint_and_no_image():
    card = {"name": "Alex", "score": 0.9, "face_agrees": False, "direction": None, "seconds_ago": 1.0, "photo": "K1"}
    srv = server.make_server(Config(), port=0, live=lambda: card_snap(card), bind_and_activate=False)
    try:
        code, _, page = get(srv, "/", origin=None)
        assert code == 200 and b"<img" not in page
        assert get(srv, "/voice/photo?k=K1&token=x")[0] == 404
    finally:
        srv.server_close()


def test_card_html_is_escaped_and_no_photo_means_no_image():
    evil = {"name": "<b>Al&ex</b>", "score": 0.9, "face_agrees": True, "direction": "at left", "seconds_ago": 2.0,
            "photo": 'K"><script>x</script>'}
    plain = {"name": "Sam", "score": 0.8, "face_agrees": False, "direction": None, "seconds_ago": 1.0}
    page = render(None, card_snap(evil, plain), photo_token='t"<')
    vl = page.split("<section id=VL")[1].split("</section>")[0]
    assert "<script>" not in vl and "<b>Al&ex" not in vl and "&lt;b&gt;Al&amp;ex&lt;/b&gt;" in vl
    assert vl.count("<img") == 1 and 'src="/voice/photo?k=K%22%3E%3Cscript%3Ex%3C%2Fscript%3E&amp;token=t%22%3C"' in vl
    assert "Sounds like Sam" in vl and "<img" not in vl.split("Sounds like Sam")[0].split("</p>")[-1]
    assert "<img" not in render(None, card_snap(evil))                       # no token on the page: no image
    assert "img-src 'self'" in CSP and "script-src" not in CSP


def test_live_runner_photo_only_for_a_current_cards_person(enrolled):
    env, alex = enrolled.env, enrolled.a
    cap = FakeCapture()
    r = live_runner(env, live_cfg(env).validate(), voice_capture=cap)
    r.start(background=False)
    try:
        real_face = r._face
        assert real_face.has_photo(alex) is False
        cap.feed(np.concatenate([synth_voice("A", seed=50), quiet(1.5)]))
        assert wait_until(lambda: r.snapshot()["voice"]["heard"] == 1)
        assert "photo" not in r.snapshot()["voice"]["cards"][0]                 # no photo stored: no key, no image
        assert "<img" not in render(None, r.snapshot(), photo_token="t")
        reads = []
        r._face = SimpleNamespace(has_photo=lambda pid: True,
                                  photo=lambda pid: (reads.append(pid), JPEG if pid == alex else None)[1])
        snap = r.snapshot()
        key = snap["voice"]["cards"][0]["photo"]
        assert alex not in json.dumps(snap)                                     # the person ID never reaches the page
        assert r.snapshot()["voice"]["cards"][0]["photo"] == key                # stable for the run
        assert r.voice_photo(key) == JPEG and reads == [alex]
        assert r.voice_photo("not-a-key") is None and r.voice_photo("") is None and reads == [alex]
        bare = server.make_server(r.cfg, port=0, live=r.snapshot, bind_and_activate=False)   # no on_photo: no photos
        try:
            assert get(bare, f"/voice/photo?k={key}&token=x", origin=None, site="same-origin")[0] == 404
        finally:
            bare.server_close()
        srv = server.make_server(r.cfg, port=0, live=r.snapshot, bind_and_activate=False,
                                 on_photo=r.voice_photo)                     # cli's call passes it explicitly
        try:
            code, _, page = get(srv, "/", origin=None)
            tok = page_token(page.decode())
            code, _, body = get(srv, f"/voice/photo?k={key}&token={tok}", origin=None, site="same-origin")
            assert code == 200 and body == JPEG
            later = time.time() + 3600.0                                        # the card has expired
            r.voice.clock = lambda: later
            assert r.snapshot()["voice"]["cards"] == []
            assert r.voice_photo(key) is None
            assert get(srv, f"/voice/photo?k={key}&token={tok}", origin=None, site="same-origin")[0] == 404
        finally:
            srv.server_close()
    finally:
        r.stop()
    assert r.voice_photo(key) is None                                            # never after stop


# ---- the listener's public microphone API (ADR-0063) ---------------------------------------------------
def test_live_mode_uses_the_public_pause_and_resume(env):
    voice_ready(env)
    clk = Clock()
    cap = FakeCapture()
    il = interlock(clk, SimIndicator(clk), SimMuteSwitch(clk, timeline=[(MUTE_AT, True), (OPEN_AT, False)]))
    r = live_runner(env, live_cfg(env), voice_capture=cap, interlock=il, clock=clk)
    r.start(background=False)
    calls = []
    li = r.voice
    real_pause, real_resume = li.pause_mic, li.resume_mic
    li.pause_mic = lambda: (calls.append("pause"), real_pause())[1]
    li.resume_mic = lambda: (calls.append("resume"), real_resume())[1]
    try:
        clk.t = MUTE_AT
        r.step()
        assert calls == ["pause"] and cap.stops == 1 and cap.cb is None
        r.step()
        assert calls == ["pause"]
        clk.t = OPEN_AT
        r.step()
        assert calls == ["pause", "resume"] and cap.starts == 2 and r.snapshot()["voice"]["state"] == "listening"
    finally:
        r.stop()
    src = inspect.getsource(live_mod)
    assert "._open(" not in src and "._close(" not in src                         # no private listener calls left
