"""macOS sensors without radios: the app launcher's Wi-Fi feed through an in-process os.pipe()
(invented, locally-administered addresses) and fake bleak objects."""
import enum, inspect, json, os, subprocess, sys, time, types
from collections import namedtuple

import pytest

from oiris.adapters import drivers, macos
from oiris.guard.privacy import hash_identifier
from oiris.registry import default_registry
from oiris.schema import Health

KEY = b"k" * 32
B1, B2 = "02:00:00:00:00:01", "02:00:00:00:00:02"


def no_subprocess(*a, **kw):
    pytest.fail(f"wifi_macos started a subprocess: {a}")


@pytest.fixture
def fresh_feed_state(monkeypatch):
    """Process-wide feed state reset; any subprocess start fails the test (Wi-Fi never runs one)."""
    monkeypatch.setattr(macos, "_feed", None)
    monkeypatch.setattr(macos, "_feed_claimed", False)
    monkeypatch.setattr(macos, "_feed_ended", None)
    monkeypatch.setattr(subprocess, "Popen", no_subprocess)
    monkeypatch.setattr(subprocess, "run", no_subprocess)


# ---- Wi-Fi without the app launcher's feed (terminal `oiris run`, or off macOS) ----

def test_wifi_without_feed_fails_open_with_plain_reason(fresh_feed_state, monkeypatch):
    monkeypatch.delenv(macos.FEED_ENV, raising=False)
    a = macos.WifiMacosAdapter()
    with pytest.raises(drivers.DriverUnavailable) as e:
        a.open({"site_key": KEY})
    assert str(e.value) == macos.NO_FEED and a.health() == (Health.FAILED, macos.NO_FEED)
    assert "O Iris app" in macos.NO_FEED and "cached scan" in macos.NO_FEED and "Location" in macos.NO_FEED
    assert macos.NO_FEED.endswith("oiris run started from a terminal has no Wi-Fi source")


def test_wifi_system_profiler_fallback_is_gone():
    """system_profiler makes macOS start a live Wi-Fi scan on every run (macOS's own Wi-Fi log) and
    returned no usable BSSID inside the app: removed, with everything only it used."""
    for name in ("PROFILER", "PROFILER_ARGV", "parse_system_profiler", "profiler_available", "_check_argv", "subprocess"):
        assert not hasattr(macos, name), name
    assert not hasattr(macos.WifiMacosAdapter, "_profiler_age_s") and "close" not in vars(macos.WifiMacosAdapter)
    code = "\n".join(ln for ln in inspect.getsource(macos).splitlines() if not ln.lstrip().startswith("#"))
    assert "Popen" not in code and "/usr/sbin" not in code and "import subprocess" not in code
    doc = " ".join(macos.__doc__.split())
    assert "cachedScanResults()" in doc and "made no scan requests" in doc and "starts a scan, associates or transmits" in doc
    assert "TSF: never exposed" in doc and "live scan" in doc


# ---- Wi-Fi from the app launcher's feed (an in-process os.pipe(), no launcher) ----

def raw_line(**doc):
    return (json.dumps({"t": time.time(), "status": "ok", "aps": [], "withheld": 0, **doc}) + "\n").encode()


def feed_line(status="ok", aps=((B1, -48), (B2, -80)), withheld=0, t=None):
    """A line as the launcher writes it (synthetic, locally-administered BSSIDs)."""
    return raw_line(t=time.time() if t is None else t, status=status,
                    aps=[{"bssid": b, "rssi": r} for b, r in aps], withheld=withheld)


class Writer:
    """The launcher's end of the pipe; closed once (its fd number may be reused afterwards)."""

    def __init__(self, fd):
        self.fd = fd

    def write(self, data):
        assert os.write(self.fd, data) == len(data)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


@pytest.fixture
def feed(monkeypatch, fresh_feed_state):
    """-> (adapter on the feed, writer). Fresh process-wide feed state; a subprocess start fails."""
    r, w = os.pipe()
    monkeypatch.setenv(macos.FEED_ENV, str(r))
    a = macos.WifiMacosAdapter()
    a.open({"site_key": KEY})
    reader, writer = a._feed, Writer(w)
    yield a, writer
    writer.close()
    reader.close()   # the adapter never closes it; after an EOF it already did (fd None: no-op)


def test_feed_claimed_once_and_hidden_from_children(feed):
    a, w = feed
    assert a._feed is not None and macos.FEED_ENV not in os.environ     # nothing we start inherits it
    assert not os.get_inheritable(a._feed.fd) and not os.get_blocking(a._feed.fd)
    assert a.health() == (Health.DEGRADED, "waiting for the app launcher's Wi-Fi feed (one reading every 15 s)")
    assert a.read(0.01) is None                                          # no line yet: None
    a.close(); a.open({"site_key": KEY})                                # the bus reopens after a failure
    assert a._feed is macos._feed and a._feed.fd is not None


def test_feed_ok_line_is_sample_with_hashed_bssids(feed):
    a, w = feed
    w.write(feed_line())
    s = a.read(0.1)
    assert [set(ap) for ap in s.data["aps"]] == [{"key", "tsf_us", "rssi", "local_s", "class"}] * 2
    assert [ap["class"] for ap in s.data["aps"]] == ["randomized"] * 2   # locally administered
    assert [ap["key"] for ap in s.data["aps"]] == [hash_identifier(B1, KEY), hash_identifier(B2, KEY)]
    assert [ap["rssi"] for ap in s.data["aps"]] == [-48.0, -80.0] and all(ap["tsf_us"] is None for ap in s.data["aps"])
    assert B1 not in json.dumps(s.data) and B1.replace(":", "") not in json.dumps(s.data)
    assert a.health() == (Health.OK, macos.TSF_NOTE)
    assert a.read(0.01) is None                                          # no new line


def test_feed_partial_lines_across_reads_and_latest_wins(feed):
    a, w = feed
    line = feed_line(aps=[(B1, -50)])
    w.write(line[:17])
    assert a.read(0.01) is None
    w.write(line[17:] + feed_line(aps=[(B2, -70)])[:-1])             # first completes, second still partial
    assert [ap["key"] for ap in a.read(0.1).data["aps"]] == [hash_identifier(B1, KEY)]
    w.write(b"\n" + feed_line(aps=[(B1, -40)]) + feed_line(aps=[(B2.upper(), -60)]))
    assert [ap["key"] for ap in a.read(0.1).data["aps"]] == [hash_identifier(B2, KEY)]   # only the latest line
    assert a._feed.rejected == 0


def test_feed_stale_line_is_no_data(feed):
    a, w = feed
    w.write(feed_line(t=time.time() - 61))
    assert a.read(0.1) is None
    h, why = a.health()
    assert h == Health.DEGRADED and "older than 60s" in why
    w.write(feed_line(t=time.time() + 3600))                          # a future stamp is no better
    assert a.read(0.1) is None
    w.write(feed_line())
    assert a.read(0.1) is not None and a.health()[0] == Health.OK


def test_feed_silence_degrades_after_60s(feed):
    a, w = feed
    a._feed_last -= 61
    assert a.read(0.01) is None
    assert a.health()[0] == Health.DEGRADED and "no reading" in a.health()[1]


@pytest.mark.parametrize("bad", [
    b"not json\n", b"[1, 2]\n", b"\xff\xfe\n", b'{"t": 1}\n',
    feed_line(status="scanning"), feed_line(t="now"), feed_line(t=True), feed_line(withheld=-1),
    feed_line(withheld=1.5), raw_line(aps={"bssid": B1}), raw_line(withheld=True), raw_line(t=float("nan")),
    b"[" * 8000 + b"\n",   # nested past the recursion limit; small enough for the pipe buffer
])
def test_feed_malformed_lines_dropped_and_counted(feed, bad):
    a, w = feed
    w.write(bad)
    assert a.read(0.1) is None and a._feed.rejected == 1 and a._feed.fd is not None
    w.write(feed_line())
    s = a.read(0.1)
    assert len(s.data["aps"]) == 2
    h, why = a.health()
    assert h == Health.OK and "1 malformed Wi-Fi feed lines/entries dropped" in why


def test_feed_bad_entries_dropped_and_counted(feed):
    a, w = feed
    entries = [(B1, -50), ("02:00:00:00:00", -50), ("00:00:00:00:00:00", -50), ("zz:00:00:00:00:01", -50),
               (B2, "-50"), (B2, None), (B2, True), (12, -50)]
    w.write(feed_line(aps=entries) + raw_line(aps=["x", None]))
    assert a.read(0.1).data == {"aps": []}                               # the latest line: both entries bad
    assert a._feed.rejected == 7 + 2
    assert a.health()[0] == Health.DEGRADED                               # nothing usable in it


def test_feed_over_long_line_dropped_up_to_its_newline(feed, monkeypatch):
    monkeypatch.setattr(macos, "FEED_MAX_LINE", 64)
    a, w = feed
    w.write(b'{"t": ' + b" " * 100)
    assert a.read(0.01) is None and a._feed.rejected == 1
    w.write(b'1, "status": "ok"}\n' + feed_line())
    assert len(a.read(0.1).data["aps"]) == 2 and a._feed.rejected == 1   # the tail was skipped, not parsed


def test_feed_location_not_authorized_waits_then_recovers_at_once(feed):
    """Before the Location grant: a waiting state, not a failure (no exception, so the bus's circuit
    breaker never opens); the first OK line after the grant is a sample straight away."""
    a, w = feed
    for _ in range(6):                                                    # > the breaker threshold (5)
        w.write(feed_line("location_not_authorized", aps=[]))
        assert a.read(0.1) is None
    h, why = a.health()
    assert h == Health.DEGRADED and why.startswith("waiting for Location permission")
    assert "Location Services" in why and a._feed.fd is not None
    w.write(feed_line())                                                  # the user allowed Location
    assert len(a.read(0.1).data["aps"]) == 2 and a.health()[0] == Health.OK


def test_bus_never_opens_the_breaker_while_waiting_for_location(feed):
    from oiris.bus import SensorBus
    a, w = feed
    bus = SensorBus(timeout_s=0.5, breaker_threshold=2, breaker_cooldown_s=30.0)
    bus.add(a, {"site_key": KEY})
    for _ in range(4):
        w.write(feed_line("location_not_authorized", aps=[]))
    bus.start()
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not bus.health()[a.id]["reason"].startswith("waiting"):
            time.sleep(0.02)
        st = bus.health()[a.id]
        assert st["health"] == Health.DEGRADED.value and st["reason"].startswith("waiting for Location")
        assert st["restarts"] == 0
        w.write(feed_line())
        while time.monotonic() < deadline + 3 and bus.health()[a.id]["samples"] == 0:
            time.sleep(0.02)
        assert bus.health()[a.id]["samples"] == 1 and bus.health()[a.id]["health"] == Health.OK.value
    finally:
        bus.stop()


@pytest.mark.parametrize("kw,match", [
    ({"status": "wifi_off", "aps": []}, "^Wi-Fi is turned off$"),
    ({"status": "no_interface", "aps": []}, "no Wi-Fi interface"),
    ({"aps": [], "withheld": 25}, "withheld BSSIDs for all 25 visible networks.*Location Services"),
])
def test_feed_failed_states(feed, kw, match):
    a, w = feed
    # Built at run time: a line built at collection time is stamped then and can be past the
    # adapter's 60 s age limit by the time a long suite reaches this test (seen in a 65 s run).
    w.write(feed_line(**kw))
    with pytest.raises(drivers.DriverUnavailable, match=match):
        a.read(0.1)
    assert a.health()[0] == Health.FAILED and a._feed.fd is not None   # the feed carries on
    w.write(feed_line())
    assert a.read(0.1) is not None


def test_feed_partly_withheld_and_no_cache_degrade(feed):
    a, w = feed
    w.write(feed_line(aps=[(B1, -50)], withheld=3))
    assert [ap["key"] for ap in a.read(0.1).data["aps"]] == [hash_identifier(B1, KEY)]
    h, why = a.health()
    assert h == Health.DEGRADED and "3 of 4" in why and "Location Services" in why
    w.write(feed_line("no_cached_results", aps=[]))
    assert a.read(0.1).data == {"aps": []}
    assert a.health()[0] == Health.DEGRADED and "no cached Wi-Fi scan" in a.health()[1]


def test_feed_eof_fails_with_reason_and_starts_nothing(feed):
    a, w = feed
    fd = a._feed.fd
    w.write(feed_line(aps=[(B2, -70)]) + b'{"t": 1')                  # a last line, then a cut-off one
    w.close()
    with pytest.raises(drivers.DriverUnavailable) as e:
        a.read(0.1)
    assert str(e.value) == macos.feed_ended_reason(None)
    assert str(e.value).startswith("the app launcher's Wi-Fi feed ended; Wi-Fi stays unavailable")
    assert a.health() == (Health.FAILED, str(e.value))
    assert a._feed is None and macos.wifi_feed() is None
    with pytest.raises(OSError):
        os.fstat(fd)                                                      # closed
    with pytest.raises(drivers.DriverUnavailable, match="feed ended"):   # stays failed: no other source
        a.read(0.1)
    a.close()
    with pytest.raises(drivers.DriverUnavailable, match="feed ended"):   # the bus's reopen: never reclaimed
        a.open({"site_key": KEY})
    assert a._feed is None and a.health()[0] == Health.FAILED


def test_feed_max_age_spans_a_missed_line(feed):
    a, w = feed
    assert macos.FEED_READING_AGE_S == 2 * 15.0 + 5.0 and a.max_age_s == macos.FEED_READING_AGE_S
    assert macos.WifiMacosAdapter.max_age_s is None                     # per instance, set by open()


def test_feed_os_errors_are_recorded_logged_and_shown(feed, monkeypatch, caplog):
    a, w = feed
    real_close = os.close

    def broken_read(fd, n):
        raise OSError(5, "Input/output error")

    def broken_close(fd):
        real_close(fd)                                                    # still closed: no fd leak
        raise OSError(9, "Bad file descriptor")
    with monkeypatch.context() as m, caplog.at_level("WARNING", logger="oiris.adapters.macos"):
        m.setattr(macos.select, "select", lambda *a: (_ for _ in ()).throw(ValueError("fd out of range")))
        m.setattr(macos.os, "read", broken_read)
        m.setattr(macos.os, "close", broken_close)
        with pytest.raises(drivers.DriverUnavailable):                  # a broken pipe end counts as EOF
            a.read(0.01)
        assert a._feed is None
    msgs = [r.getMessage() for r in caplog.records]
    assert any("select failed: fd out of range" in m for m in msgs)
    assert any("read failed: [Errno 5] Input/output error" in m for m in msgs)
    assert any("close failed: [Errno 9] Bad file descriptor" in m for m in msgs)
    h, why = a.health()
    assert h == Health.FAILED and "feed ended (close failed: [Errno 9] Bad file descriptor)" in why
    a.close()
    with pytest.raises(drivers.DriverUnavailable, match=r"feed ended \(close failed"):   # kept for the reopen
        a.open({"site_key": KEY})


@pytest.mark.parametrize("value", ["", "abc", "1", "-4", "999999"])
def test_feed_env_without_valid_pipe_is_no_wifi_source(monkeypatch, fresh_feed_state, value):
    monkeypatch.setenv(macos.FEED_ENV, value)
    a = macos.WifiMacosAdapter()
    with pytest.raises(drivers.DriverUnavailable, match="terminal has no Wi-Fi source"):
        a.open({"site_key": KEY})
    assert a._feed is None and macos.FEED_ENV not in os.environ


def test_feed_env_regular_file_is_not_a_feed(monkeypatch, fresh_feed_state, tmp_path):
    with open(tmp_path / "f", "w") as f:
        monkeypatch.setenv(macos.FEED_ENV, str(f.fileno()))
        assert macos.wifi_feed() is None and macos.FEED_ENV not in os.environ


def test_wifi_open_runs_the_radio_guard(feed, monkeypatch):
    a, w = feed
    calls = []
    monkeypatch.setattr(macos.radio, "assert_passive", lambda *x: calls.append(x))
    a.close(); a.open({"site_key": KEY})
    assert calls == [()]


# ---- BLE with fake bleak ----

Adv = namedtuple("Adv", "local_name manufacturer_data service_data service_uuids tx_power rssi platform_data")


class Reason(enum.Enum):
    POWERED_OFF = 1
    DENIED_BY_USER = 2


class BleakError(Exception):
    pass


class BleakBluetoothNotAvailableError(BleakError):
    def __init__(self, msg, reason):
        super().__init__(msg, reason)
        self.reason = reason


@pytest.fixture
def fake_bleak(monkeypatch):
    state = {"found": {}, "exc": None, "timeouts": []}

    class BleakScanner:
        @staticmethod
        async def discover(timeout=5.0, *, return_adv=False, **kw):
            state["timeouts"].append(timeout)
            assert return_adv and not kw              # no BlueZ-only args (scanning_mode, bluez=...)
            if state["exc"]:
                raise state["exc"]
            return state["found"]

    state["auth"], state["requests"] = macos.BT_ALLOWED, 0   # never the real CoreBluetooth state

    def request():
        state["requests"] += 1
        return object()

    monkeypatch.setattr(macos, "ble_authorization", lambda: state["auth"])
    monkeypatch.setattr(macos, "ble_request_permission", request)
    mod = types.ModuleType("bleak"); mod.BleakScanner = BleakScanner
    exc = types.ModuleType("bleak.exc"); exc.BleakError = BleakError
    monkeypatch.setitem(sys.modules, "bleak", mod)
    monkeypatch.setitem(sys.modules, "bleak.exc", exc)
    return state


def ble(cfg=None):
    a = drivers.BleScanAdapter()
    a.open({"site_key": KEY, **(cfg or {})})
    return a


# CoreBluetooth identifiers are per-Mac random UUIDs, not MACs
U1, U2 = "1F2E3D4C-0000-4000-8000-00000000AAAA", "1F2E3D4C-0000-4000-8000-00000000BBBB"


def test_ble_macos_uuid_addresses_hashed_and_shape_unchanged(fake_bleak):
    fake_bleak["found"] = {U1: (object(), Adv("n", {0x004C: b"\x02\x15abc"}, {}, ["0000fd6f-0000-1000-8000-00805f9b34fb"], None, -52, ())),
                           U2: (object(), Adv(None, {}, {}, [], -8, -90, ()))}
    a = ble({"self_allowlist": [hash_identifier(U2, KEY)]})
    em = a.read(2.0).data["emitters"]
    assert [e["key"] for e in em] == [hash_identifier(U1, KEY), hash_identifier(U2, KEY)]
    assert em[0] == {"key": hash_identifier(U1, KEY), "rssi": -52, "tx_power": None, "company_ids": [0x004C],
                     "uuids": ["0000fd6f-0000-1000-8000-00805f9b34fb"], "payload_len": 5, "self": False}
    assert em[1]["self"] and em[1]["tx_power"] == -8
    assert U1 not in json.dumps(em) and U1.lower() not in json.dumps(em)


def test_ble_scan_window_fits_inside_bus_timeout(fake_bleak):
    a = ble()
    a.read(2.0); a.read(30.0)
    assert fake_bleak["timeouts"][0] == pytest.approx(1.2) and fake_bleak["timeouts"][1] == 5.0


@pytest.mark.parametrize("err,match", [
    (BleakBluetoothNotAvailableError("off", Reason.POWERED_OFF), "POWERED_OFF"),
    (BleakBluetoothNotAvailableError("no", Reason.DENIED_BY_USER), "DENIED_BY_USER.*Privacy & Security > Bluetooth"),
    (BleakError("Bluetooth device is turned off"), "turned off"),
])
def test_ble_unavailable_radio_or_permission(fake_bleak, err, match):
    fake_bleak["exc"] = err
    a = ble()
    with pytest.raises(drivers.DriverUnavailable, match=match):
        a.read(2.0)
    assert a.health()[0] == Health.FAILED
    fake_bleak["exc"] = None
    a.read(2.0)
    assert a.health()[0] == Health.OK


def test_ble_missing_bleak(monkeypatch):
    monkeypatch.setitem(sys.modules, "bleak", None)
    with pytest.raises(drivers.DriverUnavailable, match="pip install bleak"):
        drivers.BleScanAdapter().open({"site_key": KEY})


# ---- BLE: macOS Bluetooth permission (fake CBManager.authorization state) ----

@pytest.fixture
def darwin(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")


def test_ble_permission_allowed_scans(fake_bleak, darwin):
    a = ble()
    assert a.read(2.0) is not None and len(fake_bleak["timeouts"]) == 1
    assert a.health()[0] == Health.OK and fake_bleak["requests"] == 0


def test_ble_permission_undecided_prompts_once_then_never_scans(fake_bleak, darwin):
    fake_bleak["auth"] = macos.BT_NOT_DETERMINED
    a = ble()
    assert fake_bleak["requests"] == 1                       # open asks macOS once (shows the prompt)
    h, why = a.health()
    assert h == Health.DEGRADED and why == macos.BT_WAITING
    assert "answer the macOS prompt" in why and "Privacy & Security > Bluetooth" in why
    assert a.read(0.01) is None and a.read(0.01) is None
    assert fake_bleak["timeouts"] == [] and fake_bleak["requests"] == 1   # polls only: no scan, no new request
    assert a.health() == (Health.DEGRADED, macos.BT_WAITING)
    fake_bleak["auth"] = macos.BT_ALLOWED                   # the user clicked Allow
    assert a.read(2.0) is not None and len(fake_bleak["timeouts"]) == 1
    assert a.health()[0] == Health.OK


def test_ble_permission_reopen_asks_again(fake_bleak, darwin):
    fake_bleak["auth"] = macos.BT_NOT_DETERMINED
    ble(); ble()
    assert fake_bleak["requests"] == 2 and fake_bleak["timeouts"] == []


@pytest.mark.parametrize("auth,match", [
    (macos.BT_DENIED, "denied.*System Settings > Privacy & Security > Bluetooth"),
    (macos.BT_RESTRICTED, "restricted by device policy"),
])
def test_ble_permission_denied_or_restricted(fake_bleak, darwin, auth, match):
    fake_bleak["auth"] = auth
    with pytest.raises(drivers.DriverUnavailable, match=match):
        ble()
    fake_bleak["auth"] = macos.BT_ALLOWED
    a = ble()
    fake_bleak["auth"] = auth                               # revoked while running
    with pytest.raises(drivers.DriverUnavailable, match=match):
        a.read(2.0)
    assert a.health()[0] == Health.FAILED and len(fake_bleak["timeouts"]) == 0
    assert fake_bleak["requests"] == 0


def test_ble_permission_unknown_keeps_todays_behaviour(fake_bleak, darwin):
    fake_bleak["auth"] = None                               # PyObjC/CoreBluetooth not importable
    a = ble()
    assert a.read(2.0) is not None and len(fake_bleak["timeouts"]) == 1


def test_ble_off_macos_never_reads_permission(fake_bleak, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(macos, "ble_authorization", lambda: pytest.fail("permission read on Linux"))
    a = ble()
    assert a._bt is None and a.read(2.0) is not None and len(fake_bleak["timeouts"]) == 1


def test_ble_authorization_helper_without_pyobjc(monkeypatch):
    monkeypatch.setitem(sys.modules, "CoreBluetooth", None)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert macos.ble_authorization() is None and macos.ble_request_permission() is None
    monkeypatch.setattr(sys, "platform", "linux")
    assert macos.ble_authorization() is None


def test_ble_authorization_helper_maps_corebluetooth(monkeypatch):
    cb = types.ModuleType("CoreBluetooth")
    cb.CBManager = types.SimpleNamespace(authorization=lambda: 2)
    monkeypatch.setitem(sys.modules, "CoreBluetooth", cb)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert macos.ble_authorization() == macos.BT_DENIED


# ---- registry ----

def test_registry_names():
    names = default_registry({"entry_points": False}).names("sensor")
    assert {"wifi_macos", "ble_scan", "wifi_beacons"} <= set(names)
    assert macos.WifiMacosAdapter.caps == {"ap_scan"} and drivers.BleScanAdapter.caps == {"rf_adv"}
