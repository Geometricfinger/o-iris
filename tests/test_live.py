"""Live mode (`oiris run`) with sim/replay adapters only: no radios, no network, no people, no camera."""
import dataclasses, json, os, signal, socket, threading, time, urllib.request

import pytest

from oiris import cli
from oiris.adapters.base import AdapterBase
from oiris.adapters.drivers import DriverUnavailable
from oiris.config import RETENTION_CHECK_DEFAULT_S, Config
from oiris.headless import record_clip
from oiris import live as live_mod
from oiris.guard import retention
from oiris.live import (_local_target, LiveConfigError, LiveRunner, LiveSettings, OutboundBlocked,
                        SIGHTING_REPEAT_S, load_live_settings, open_store)
from oiris.registry import default_registry
from oiris.store.sightings import SweepTimeline

SIM = {"name": "sim", "args": {"scenario": "camera_lens", "period_s": 0.01}, "options": {}}


class NoDriver(AdapterBase):
    id, caps = "needs_driver", {"rf_adv"}

    def open(self, cfg):
        raise DriverUnavailable("pip install some-driver")

    def read(self, timeout_s):  # pragma: no cover - never opened
        raise AssertionError("must not be read")


@pytest.fixture
def live_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OIRIS_PASSPHRASE", "test-passphrase")
    cfg = Config(data_dir=str(tmp_path / "data")).validate()
    reg = default_registry({"entry_points": False})
    reg.unregister("sensor", "wifi_macos")      # may be absent on this branch anyway: same path
    reg.register("sensor", "needs_driver", NoDriver, port_version="1.0")
    return cfg, reg, tmp_path


def _runner(live_env, adapters, **kw):
    cfg, reg, _ = live_env
    return LiveRunner(cfg, LiveSettings(adapters=adapters, hz=kw.pop("hz", 20.0), site="lab"), registry=reg, **kw)


def _step_until(r, n, timeout=5.0):
    t0 = time.monotonic()
    while r.steps < n and time.monotonic() - t0 < timeout:
        time.sleep(0.02)
        r.step()


def test_settings_defaults_and_parsing(tmp_path):
    assert [a["name"] for a in load_live_settings(None).adapters] == ["ble_scan", "wifi_macos"]
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"retention_days": 7}))
    s = load_live_settings(p)
    assert [a["name"] for a in s.adapters] == ["ble_scan", "wifi_macos"] and s.hz == 1.0
    p.write_text(json.dumps({"live": {"adapters": [SIM, "replay"], "hz": 2}}))
    s = load_live_settings(p)
    assert [a["name"] for a in s.adapters] == ["sim", "replay"] and s.hz == 2.0
    assert s.adapters[0]["args"]["scenario"] == "camera_lens"
    for bad in ({"hz": 0}, {"hz": 99}, {"adapters": "sim"}, {"adapters": [{"options": {}}]}, {"bogus": 1}):
        p.write_text(json.dumps({"live": bad}))
        with pytest.raises(LiveConfigError):
            load_live_settings(p)


def test_adapters_resolved_by_name_unavailable_reported_not_fatal(live_env):
    r = _runner(live_env, [SIM, {"name": "wifi_macos", "options": {}}, {"name": "no_such_adapter"},
                           {"name": "needs_driver", "options": {}}])
    r.start(background=False)
    try:
        assert r.adapters["sim_nir"]["available"] and r.adapters["sim_thermal"]["available"]
        assert set(r.engine.bus.adapters) >= {"sim_nir", "sim_ble", "sim_radar"}
        for k in ("wifi_macos", "no_such_adapter", "needs_driver"):
            assert r.adapters[k]["available"] is False and r.adapters[k]["reason"]
        assert "pip install some-driver" in r.adapters["needs_driver"]["reason"]
        assert "no plugin sensor:wifi_macos" in r.adapters["wifi_macos"]["reason"]
        _step_until(r, 2)
        assert r.steps >= 2
    finally:
        r.stop()


def test_results_update_over_steps(live_env):
    r = _runner(live_env, [SIM])
    r.start(background=False)
    try:
        assert r.snapshot()["result"] is None
        _step_until(r, 1)
        first = r.snapshot()["result"]
        time.sleep(0.05)
        _step_until(r, 4)
        snap = r.snapshot()
        assert snap["steps"] >= 4 and snap["result"]["t_wall"] > first["t_wall"]
        assert snap["result"]["health"]["sim_nir"]["samples"] > first["health"]["sim_nir"]["samples"]
        assert len(r.engine.results) <= 1           # bounded memory over a long run
    finally:
        r.stop()


def test_background_loop_and_encrypted_sweep_record(live_env):
    cfg, _, _ = live_env
    r = _runner(live_env, [SIM, {"name": "no_such_adapter"}], hz=20.0)
    r.start()
    t0 = time.monotonic()
    while r.steps < 5 and time.monotonic() - t0 < 5:
        time.sleep(0.05)
    r.stop()
    assert r.outcome == "done" and r.steps >= 5
    ks, db = open_store(cfg)
    try:
        hist = SweepTimeline(db).history("lab")
        assert len(hist) == 1 and hist[0]["outcome"] == "done" and hist[0]["end"]
        assert hist[0]["counts"]["steps"] == r.steps
        assert hist[0]["counts"]["adapters_unavailable"] == ["no_such_adapter"]
        raw = db.execute("SELECT body FROM sweep").fetchone()[0]
        assert b"steps" not in raw and b"no_such_adapter" not in raw     # sealed, not plaintext
    finally:
        db.close()


def test_replay_adapter_by_name(live_env, tmp_path):
    clip = tmp_path / "clip"
    record_clip("quiet", clip, steps=5)
    r = _runner(live_env, [{"name": "replay", "args": {"adapter_id": "rep_thermal", "path": str(clip / "sim_thermal.jsonl"),
                                                        "caps": ["thermal_frame"]}}])
    r.start(background=False)
    try:
        assert r.adapters["rep_thermal"]["available"]
        _step_until(r, 2)
        assert r.snapshot()["result"]["health"]["rep_thermal"]["samples"] >= 1
    finally:
        r.stop()


def test_network_guard_blocks_outbound_allows_loopback(live_env):
    r = _runner(live_env, [SIM])
    r.start(background=False)
    try:
        with socket.socket() as s, pytest.raises(OutboundBlocked):
            s.connect(("192.0.2.1", 80))           # TEST-NET-1; refused before any packet
        with socket.socket() as c, pytest.raises(OSError):
            c.connect(("127.0.0.1", 9))             # loopback passes the guard (refused by the OS, not by us)
        with socket.socket(socket.AF_INET6) as s6:
            assert _local_target(s6, ("::1", 1, 0, 0))
        with socket.socket() as s4:
            assert not _local_target(s4, ("example.org", 443))
    finally:
        r.stop()
    assert socket.socket.connect.__qualname__ != "local_only_network.<locals>.connect"


def test_network_guard_refuses_unconnected_datagrams_except_loopback(live_env):
    r = _runner(live_env, [SIM])
    r.start(background=False)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as u:
            with pytest.raises(OutboundBlocked):
                u.sendto(b"x", ("192.0.2.1", 9))
            with pytest.raises(OutboundBlocked):
                u.sendto(b"x", 0, ("192.0.2.1", 9))           # flags form
            with pytest.raises(OutboundBlocked):
                u.sendmsg([b"x"], [], 0, ("192.0.2.1", 9))
            with pytest.raises(OutboundBlocked):
                u.sendto(b"x", ("example.org", 9))            # would need DNS
            for send in (lambda: u.sendto(b"x", ("127.0.0.1", 9)),
                         lambda: u.sendmsg([b"x"], [], 0, ("127.0.0.1", 9))):
                try:
                    assert send() == 1                        # loopback passes the guard ...
                # The guard already let the send through: a guard refusal is OutboundBlocked (a
                # RuntimeError, not caught here). An OS sandbox may still refuse the real datagram.
                except OSError:  # python-brain: allow SILENT_EXCEPTION - OS refusal, not the guard's
                    pass                                      # ... (a sandbox may still refuse it: not us)
        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        with a, b:
            assert a.sendmsg([b"hi"]) == 2 and b.recv(8) == b"hi"   # AF_UNIX stays allowed
    finally:
        r.stop()
    for n in ("connect", "connect_ex", "sendto", "sendmsg"):
        assert n not in socket.socket.__dict__ or "local_only_network" not in socket.socket.__dict__[n].__qualname__


class _Ev:
    def __init__(self, sig, level):
        self.d = {"signature": sig, "level": level}

    def to_dict(self):
        return dict(self.d)


def test_sightings_written_on_change_and_once_per_interval_repeats_counted(live_env):
    cfg, _, _ = live_env
    now = [1000.0]
    r = _runner(live_env, [SIM], clock=lambda: now[0])
    r.start(background=False)
    try:
        written = []
        real_add = r._log.add
        r._log.add = lambda ev, gps: (written.append((ev["signature"], ev["level"])), real_add(ev, gps))
        fyi, watch = _Ev("A4", "fyi"), _Ev("A4", "watch")
        r._on_event(fyi)                                     # first pair: written at once
        for _ in range(5):
            now[0] += 1.0
            r._on_event(fyi)                                 # unchanged within 60 s: suppressed
        r._on_event(watch)                                   # level changed: written at once
        r._on_event(_Ev("A2", "watch"))                      # signature changed: written at once
        r._on_event(_Ev("A2", "watch"))
        assert written == [("A4", "fyi"), ("A4", "watch"), ("A2", "watch")]
        now[0] += SIGHTING_REPEAT_S - 0.5
        r._on_event(_Ev("A2", "watch"))
        assert len(written) == 3                             # still inside the interval
        now[0] += 0.5
        r._on_event(_Ev("A2", "watch"))                      # 60 s since the last write
        assert written[-1] == ("A2", "watch") and len(written) == 4
        rec = r.snapshot()["records"]
        assert rec["events"] == 11 and rec["sightings"] == 4 and rec["sightings_suppressed"] == 7
    finally:
        r.stop()
    ks, db = open_store(cfg)
    try:
        (sw,) = SweepTimeline(db).history("lab")
        c = sw["counts"]
        assert (c["events"], c["sightings"], c["sightings_suppressed"]) == (11, 4, 7)
        assert db.execute("SELECT COUNT(*) FROM sighting").fetchone()[0] == 4
        assert "events" not in sw["new_since_last"] and "purged_rows" not in sw["new_since_last"]
    finally:
        db.close()


def test_engine_events_all_counted(live_env):
    now = [0.0]
    r = _runner(live_env, [SIM], clock=lambda: now[0])
    r.start(background=False)
    try:
        _step_until(r, 10)
        rec = r.snapshot()["records"]
        assert rec["events"] == rec["sightings"] + rec["sightings_suppressed"]
        assert rec["events"] <= r.steps                     # at most one event per step
    finally:
        r.stop()


def test_retention_purge_at_open_every_check_interval_never_after_close(live_env, monkeypatch):
    cfg, _, _ = live_env
    cfg = dataclasses.replace(cfg, retention_days=7)
    ks, db = open_store(cfg)                                 # an expired sweep from an earlier run
    old = SweepTimeline(db).start("lab")
    db.execute("UPDATE sweep SET t_wall=? WHERE sweep_id=?", (time.time() - 8 * 86400, old))
    db.close()
    calls, real = [], retention.purge
    monkeypatch.setattr(retention, "purge", lambda db, days=30, now=None: (calls.append(days), real(db, days, now))[1])
    now = [500.0]
    r = LiveRunner(cfg, LiveSettings(adapters=[SIM], hz=20.0, site="lab"), registry=live_env[1], clock=lambda: now[0])
    r.start(background=False)
    try:
        assert calls == [7]                                  # at store open, with the configured days
        assert r.snapshot()["records"]["purged_rows"] == 1
        assert [s["sweep_id"] for s in SweepTimeline(r.db).history("lab")] == [r.sweep_id]
        _step_until(r, 3)
        now[0] += RETENTION_CHECK_DEFAULT_S - 1
        r.step()
        assert calls == [7]                                  # not yet one check interval
        now[0] += 1
        r.step()
        assert calls == [7, 7]                               # one interval of faked time: purged again
        r.step()
        assert calls == [7, 7]
    finally:
        r.stop()
    now[0] += 10 * RETENTION_CHECK_DEFAULT_S
    assert r.step() is None and calls == [7, 7]              # never after close
    ks, db = open_store(cfg)
    try:
        (sw,) = SweepTimeline(db).history("lab")
        assert sw["counts"]["purged_rows"] == 1
    finally:
        db.close()


def test_retention_purge_failure_logged_not_fatal(live_env, monkeypatch, caplog):
    def boom(db, days=30, now=None):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(retention, "purge", boom)
    now = [0.0]
    r = _runner(live_env, [SIM], clock=lambda: now[0])
    with caplog.at_level("ERROR", logger=live_mod.__name__):
        r.start(background=False)
        try:
            _step_until(r, 2)
            now[0] += RETENTION_CHECK_DEFAULT_S
            assert r.step() is not None                      # second failure: still stepping
            assert r.steps >= 3 and r.snapshot()["records"]["purged_rows"] == 0
        finally:
            r.stop()
    assert sum("retention purge failed" in m for m in caplog.messages) == 2
    assert r.outcome == "done"


class SlowRead(AdapterBase):
    """A driver whose read ignores close() and outlasts stop()'s join timeout."""
    id, caps = "slow_read", {"rf_adv"}

    def __init__(self):
        super().__init__()
        self.release = threading.Event()

    def read(self, timeout_s):
        self.release.wait(10.0)
        return None


def test_stop_with_loop_stuck_in_slow_read_never_writes_after_close(live_env):
    cfg, reg, _ = live_env
    cfg = dataclasses.replace(cfg, adapter_timeout_s=0.05)     # bus read watchdog: keep bus.stop fast
    slow = SlowRead()
    reg.register("sensor", "slow_read", lambda: slow, port_version="1.0")
    r = LiveRunner(cfg, LiveSettings(adapters=[SIM, {"name": "slow_read"}], hz=50.0, site="lab"),
                   registry=reg, join_s=0.1)
    r.start()
    assert r._log is not None and r.adapters["slow_read"]["available"]
    real_step, real_add, real_close = r.engine.step, r._log.add, r.db.close
    writes, closes = [], []
    r._log.add = lambda *a, **k: (writes.append(not closes), real_add(*a, **k))
    r.db.close = lambda: (closes.append(threading.current_thread().name), real_close())
    ev = type("Ev", (), {"to_dict": lambda self: {"signature": "A4", "level": 1}})()

    entered = threading.Event()

    def stuck_step():                       # the loop's step is stuck inside the slow driver read
        entered.set()
        slow.read(1.0)
        res = real_step()
        r._on_event(ev)                     # an alert raised by the step that outlived stop()
        return res
    t0 = time.monotonic()
    while r.steps < 2 and time.monotonic() - t0 < 5:
        time.sleep(0.01)
    r.engine.step = stuck_step
    assert entered.wait(5)                  # the loop is now inside the slow read
    n = len(writes)                         # the sim scenario may already have raised alerts
    r._on_event(ev)                         # control: before stop the sighting is written
    assert len(writes) == n + 1 and all(writes)
    t0 = time.monotonic()
    r.stop()
    assert time.monotonic() - t0 < 2        # bounded: two 0.1 s joins plus the bus stop
    assert r.outcome == "done" and r.loop_alive_at_stop and r._thread.is_alive()
    assert closes == []                     # DB not closed under the running loop
    steps_at_stop = r.steps
    r.stop()                                # idempotent
    slow.release.set()
    r._thread.join(5)
    assert not r._thread.is_alive()
    assert len(writes) == n + 1 and all(writes)     # the late event was dropped, never written
    assert closes == ["oiris-live"]         # the loop thread closed the DB on its way out
    assert r.steps == steps_at_stop         # the late step was not counted
    ks, db = open_store(cfg)
    try:
        (sw,) = SweepTimeline(db).history("lab")
        assert sw["outcome"] == "done" and sw["counts"]["loop_alive_at_stop"] == 1
        assert sw["counts"]["sightings"] == len(writes)
    finally:
        db.close()


def _get(srv, path):
    """Drive the real request handler over a socketpair (no TCP port needed)."""
    a, b = socket.socketpair()
    b.settimeout(5)

    def handle():
        try:
            srv.RequestHandlerClass(a, ("127.0.0.1", 0), srv)
        finally:
            a.close()                                # what the server's shutdown_request would do
    t = threading.Thread(target=handle, daemon=True)
    t.start()
    b.sendall(f"GET {path} HTTP/1.0\r\nHost: 127.0.0.1:0\r\n\r\n".encode())
    chunks = []
    while chunk := b.recv(65536):
        chunks.append(chunk)
    t.join(5); b.close()
    head, _, body = b"".join(chunks).partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.0 200")
    return body


def test_dashboard_api_latest_returns_live_data(live_env, monkeypatch):
    from oiris.dashboard import server
    monkeypatch.setattr(server, "run_scenario", lambda *a, **k: (_ for _ in ()).throw(AssertionError("simulated")))
    r = _runner(live_env, [SIM, {"name": "needs_driver"}])
    r.start(background=False)
    srv = server.make_server(live_env[0], port=0, live=r.snapshot, bind_and_activate=False)
    try:
        assert srv.server_address[0] == "127.0.0.1"
        assert json.loads(_get(srv, "/api/latest"))["result"] is None      # before the first step: no simulation
        _step_until(r, 2)
        data = json.loads(_get(srv, "/api/latest"))
        assert data["mode"] == "live" and data["steps"] >= 2 and isinstance(data["result"]["signatures"], list)
        assert data["adapters"]["needs_driver"]["available"] is False
        assert data["result"]["health"]["sim_nir"]["samples"] >= 1
        page = _get(srv, "/").decode()
        assert "needs_driver: unavailable" in page and "live" in page and "id=D5" in page
    finally:
        srv.server_close(); r.stop()


def _write_cfg(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"data_dir": str(tmp_path / "data"), "live": {"adapters": [SIM, {"name": "no_such_adapter"}],
                                                                          "hz": 10}}))
    return p


def test_cli_run_duration_stops_and_records(tmp_path, monkeypatch):
    monkeypatch.setenv("OIRIS_PASSPHRASE", "test-passphrase")
    p = _write_cfg(tmp_path)
    t0 = time.monotonic()
    assert cli.main(["run", "--duration", "0.6", "--no-dashboard", "--config", str(p)]) == 0
    assert time.monotonic() - t0 < 5
    ks, db = open_store(Config.load(p))
    try:
        (sw,) = SweepTimeline(db).history("live")
        assert sw["outcome"] == "done" and sw["counts"]["steps"] >= 3
    finally:
        db.close()


def test_cli_run_signal_stops_cleanly(tmp_path, monkeypatch):
    monkeypatch.setenv("OIRIS_PASSPHRASE", "test-passphrase")
    p = _write_cfg(tmp_path)
    # Signal only once live mode has installed its own SIGTERM handler. A fixed delay raced slow
    # startups on a busy machine: the SIGTERM then hit the default handler and killed the whole
    # pytest run. A guard handler catches any stray early signal instead of dying.
    early = []

    def guard(signum, _frame):
        early.append(signum)

    original = signal.signal(signal.SIGTERM, guard)
    sent = {}

    def stop_when_installed():
        deadline = time.monotonic() + 30
        while signal.getsignal(signal.SIGTERM) is guard and time.monotonic() < deadline:
            time.sleep(0.02)
        sent["t"] = time.monotonic()
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=stop_when_installed, daemon=True).start()
    try:
        assert cli.main(["run", "--no-dashboard", "--config", str(p)]) == 0      # duration 0 = until stopped
        assert not early, "SIGTERM arrived before live mode installed its handler"
        assert time.monotonic() - sent["t"] < 5                                   # stops promptly once signalled
    finally:
        signal.signal(signal.SIGTERM, original)
    ks, db = open_store(Config.load(p))
    try:
        assert SweepTimeline(db).history("live")[0]["outcome"] == "interrupted"
    finally:
        db.close()


def test_cli_run_with_dashboard_and_signal_stop(tmp_path, monkeypatch):
    """Binds a local TCP port, like test_dashboard's server test (needs local binding allowed)."""
    monkeypatch.setenv("OIRIS_PASSPHRASE", "test-passphrase")
    p = _write_cfg(tmp_path)
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    got, done = {}, threading.Event()
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # never via an HTTP proxy

    def probe():
        for _ in range(50):
            try:
                got["d"] = json.loads(direct.open(f"http://127.0.0.1:{port}/api/latest", timeout=2).read())
                if got["d"]["steps"] >= 2:
                    break
            except OSError as e:               # dashboard not up yet: retry, keep the last error
                got["err"] = repr(e)
            time.sleep(0.1)
        if not done.is_set():
            os.kill(os.getpid(), signal.SIGINT)
    threading.Thread(target=probe, daemon=True).start()
    try:
        assert cli.main(["run", "--port", str(port), "--duration", "10", "--config", str(p)]) == 0
    finally:
        done.set()
    assert "d" in got, f"dashboard never answered: {got.get('err')}"
    assert got["d"]["mode"] == "live" and got["d"]["adapters"]["no_such_adapter"]["available"] is False
    ks, db = open_store(Config.load(p))
    try:
        assert SweepTimeline(db).history("live")[0]["outcome"] == "interrupted"
    finally:
        db.close()


def test_other_commands_unchanged():
    assert cli.main(["version"]) == 0
    assert cli.main(["--headless", "--scenario", "quiet", "--steps", "1", "--out", os.devnull]) == 0
