"""Live dashboard: O Iris title, no index/level without active families, plain sensor lines,
and the token + same-origin guarded Quit. Sim/replay adapters only: no radios, no people."""
# ruff: noqa: F811
import json, re, socket, threading, time, urllib.error, urllib.request

from oiris import cli
from oiris.adapters.macos import BT_DENIED_REASON, BT_WAITING, LOCATION_HINT
from oiris.config import Config
from oiris.dashboard import server
from oiris.dashboard.server import KNOWN_REASONS, plain_reason, render
from oiris.guard.vocabulary import violations
from oiris.headless import run_scenario
from oiris.live import open_store
from oiris.store.sightings import SweepTimeline
from tests.test_live import SIM, _runner, _step_until, live_env  # noqa: F401  (fixture reuse)

WIFI_WITHHELD = ("circuit open: read error: macOS withheld BSSIDs and SSIDs for all 34 visible networks: "
                 + LOCATION_HINT)


def _snap(result, health=None, adapters=None):
    return {"mode": "live", "running": True, "started": 0.0, "steps": 3, "hz": 1.0, "site": "live",
            "sweep_id": None, "adapters": adapters or {}, "health": health or {}, "result": result}


def _empty_result():
    res = dict(run_scenario("quiet", 1)[-1])
    res.update(families_active=[], families_missing=["a", "b", "c", "d", "e", "f"], level=None, anomaly_index=0.08)
    return res


def test_title_is_o_iris_both_modes():
    assert "<title>O Iris dashboard</title>" in render(run_scenario("quiet", 1)[-1])
    page = render(None, _snap(None))
    assert "<title>O Iris dashboard</title>" in page and "ASB" not in page


def test_no_index_or_level_while_no_family_is_active():
    health = {"ble_scan": {"health": "degraded", "reason": BT_WAITING},
              "wifi_macos": {"health": "failed", "reason": WIFI_WITHHELD}}
    for res in (None, _empty_result()):
        page = render(res, _snap(res, health))
        d1 = page.split("<section id=D1>")[1].split("</section>")[0]
        assert "No live data yet" in d1
        assert "index" not in d1 and "Level" not in d1 and "0.08" not in d1
        assert "ble_scan (Waiting for Bluetooth permission)" in d1
        assert "wifi_macos (macOS is hiding Wi-Fi network identifiers)" in d1
        assert 'http-equiv="refresh" content="2"' in page   # 2 s auto-refresh kept


def _d1(res, health):
    return render(res, _snap(res, health)).split("<section id=D1>")[1].split("</section>")[0]


def test_no_live_data_when_no_adapter_has_a_sample():
    # Live snapshot seen on O Iris.app: S_rf scored "degraded / no samples", so rf was not listed
    # as missing, yet neither sensor had delivered anything.
    res = _empty_result()
    res.update(families_active=[], families_missing=["optical", "thermal", "vital", "eye_liveness", "voice"],
               scores=[{"name": "S_rf", "health": "degraded", "evidence": {"reason": "no samples"}}])
    health = {"ble_scan": {"samples": 0, "health": "degraded"}, "wifi_macos": {"samples": 0, "health": "degraded"}}
    d1 = _d1(res, health)
    assert "No live data yet" in d1 and "arriving" not in d1


def test_sensor_data_arriving_when_an_adapter_has_samples_but_no_family_is_active():
    health = {"ble_scan": {"samples": 4, "health": "ok"}, "wifi_macos": {"samples": 0, "health": "degraded"}}
    d1 = _d1(_empty_result(), health)
    assert "Sensor data arriving; no signal family active yet" in d1 and "No live data yet" not in d1


def test_index_and_level_shown_once_a_family_is_active(live_env):
    r = _runner(live_env, [SIM])
    r.start(background=False)
    try:
        _step_until(r, 3)
        snap = r.snapshot()
        assert snap["result"]["families_active"]            # camera_lens sim drives at least one family
        d1 = render(snap["result"], snap).split("<section id=D1>")[1].split("</section>")[0]
        assert "Level: <b>" in d1 and "index " in d1 and "No live data yet" not in d1
    finally:
        r.stop()


def test_plain_status_lines_for_known_reasons():
    cases = {
        BT_WAITING: ("Waiting for Bluetooth permission", "Answer the macOS prompt"),
        "DriverUnavailable: " + BT_DENIED_REASON: ("Bluetooth access denied", "Privacy & Security > Bluetooth"),
        "Bluetooth unavailable: POWERED_OFF": ("Bluetooth is off", "Turn Bluetooth on"),
        WIFI_WITHHELD: ("macOS is hiding Wi-Fi network identifiers", "Allow Location for O Iris"),
    }
    for raw, (status, action) in cases.items():
        st, act = plain_reason(raw)
        assert st == status and action in act and "circuit open" not in st
    # unknown reason: raw text with bus jargon stripped, no invented action
    assert plain_reason("circuit open: read error: system_profiler exited 1: boom") == ("system_profiler exited 1: boom", "")
    page = render(None, _snap(None, {"wifi_macos": {"health": "failed", "reason": WIFI_WITHHELD}},
                              {"ble_scan": {"name": "ble_scan", "available": False,
                                            "reason": "DriverUnavailable: " + BT_DENIED_REASON}}))
    d5 = page.split("<section id=D5>")[1].split("</section>")[0]
    wifi = re.search(r"<li>wifi_macos:(.*?)</li>", d5).group(1)
    plain, raw = wifi.split("<small>")
    assert "macOS is hiding Wi-Fi network identifiers" in plain and "Allow Location for O Iris" in plain
    assert "circuit open" not in plain and "circuit open: read error:" in raw     # raw kept, in small text
    assert "ble_scan: unavailable · Bluetooth access denied." in d5
    for _, status, action in KNOWN_REASONS:                                      # vocabulary guard
        assert not violations(status) and not violations(action)
    assert not violations(re.sub(r"<[^>]+>", " ", page))


def _request(srv, raw: bytes) -> tuple[int, bytes]:
    """Send one raw HTTP request to the real handler over a socketpair (no TCP port)."""
    a, b = socket.socketpair()
    b.settimeout(5)

    def handle():
        try:
            srv.RequestHandlerClass(a, ("127.0.0.1", 0), srv)
        finally:
            a.close()
    t = threading.Thread(target=handle, daemon=True)
    t.start()
    b.sendall(raw)
    chunks = []
    while chunk := b.recv(65536):
        chunks.append(chunk)
    t.join(5); b.close()
    head, _, body = b"".join(chunks).partition(b"\r\n\r\n")
    return int(head.split()[1]), body


def _post_quit(srv, token, origin="http://127.0.0.1:0", host="127.0.0.1:0"):
    body = f"token={token}".encode() if token is not None else b""
    hdr = f"POST /quit HTTP/1.0\r\nHost: {host}\r\nContent-Length: {len(body)}\r\n"
    hdr += f"Origin: {origin}\r\n" if origin else ""
    return _request(srv, hdr.encode() + b"\r\n" + body)


def _token(page: str) -> str:
    return re.search(r'name=token value="([^"]+)"', page.split("<form id=quit")[1]).group(1)


def test_quit_refused_without_token_and_same_origin():
    quits = []
    srv = server.make_server(Config(), port=0, live=lambda: _snap(None), bind_and_activate=False,
                             on_quit=lambda: quits.append(1))
    try:
        code, page = _request(srv, b"GET / HTTP/1.0\r\nHost: 127.0.0.1:0\r\n\r\n")
        assert code == 200 and b"Quit O Iris" in page
        tok = _token(page.decode())
        assert len(tok) >= 32
        assert _request(srv, b"GET /quit HTTP/1.0\r\nHost: 127.0.0.1:0\r\n\r\n")[0] == 405
        assert _post_quit(srv, None)[0] == 403                                   # missing token
        assert _post_quit(srv, ("y" if tok[0] == "x" else "x") + tok[1:])[0] == 403                          # wrong token
        assert _post_quit(srv, tok, origin="http://evil.example")[0] == 403      # foreign origin
        assert _post_quit(srv, tok, origin="http://127.0.0.1:9")[0] == 403       # other local port
        assert _post_quit(srv, tok, origin=None)[0] == 403                       # no Origin at all
        assert _post_quit(srv, tok, host="rebind.example:0", origin="http://rebind.example:0")[0] == 403
        assert quits == []
        code, body = _post_quit(srv, tok, origin="http://localhost:0", host="localhost:0")
        assert code == 200 and b"O Iris stopped" in body and quits == [1]
        assert _post_quit(srv, tok)[0] == 200 and quits == [1]                   # only once
        assert b"O Iris stopped" in _request(srv, b"GET / HTTP/1.0\r\nHost: 127.0.0.1:0\r\n\r\n")[1]
    finally:
        srv.server_close()


def test_every_get_needs_a_local_host_header():
    srv = server.make_server(Config(), port=0, live=lambda: _snap(None), bind_and_activate=False,
                             on_quit=lambda: None)
    try:
        for path in ("/", "/api/latest", "/face", "/nope", "/quit"):
            for host in ("rebind.example:0", "127.0.0.1:9", "127.0.0.1", "localhost", "127.0.0.2:0", ""):
                hdr = f"Host: {host}\r\n" if host else ""
                code, body = _request(srv, f"GET {path} HTTP/1.0\r\n{hdr}\r\n".encode())
                assert code == 403 and body == b"forbidden: bad Host", (path, host)
        for host in ("127.0.0.1:0", "localhost:0", "LOCALHOST:0"):          # case-insensitive; no Origin needed
            assert _request(srv, f"GET / HTTP/1.0\r\nHost: {host}\r\n\r\n".encode())[0] == 200
            code, body = _request(srv, f"GET /api/latest HTTP/1.0\r\nHost: {host}\r\n\r\n".encode())
            assert code == 200 and json.loads(body)["mode"] == "live"
    finally:
        srv.server_close()
    demo = server.make_server(Config(), port=0, bind_and_activate=False)     # demo mode: same rule
    try:
        assert _request(demo, b"GET / HTTP/1.0\r\nHost: evil.example:0\r\n\r\n")[0] == 403
        assert _request(demo, b"GET /face HTTP/1.0\r\nHost: localhost:0\r\n\r\n")[0] == 200
    finally:
        demo.server_close()


def test_dashboard_without_live_has_no_quit():
    srv = server.make_server(Config(), port=0, bind_and_activate=False)
    try:
        code, page = _request(srv, b"GET / HTTP/1.0\r\nHost: 127.0.0.1:0\r\n\r\n")
        assert code == 200 and b"Quit O Iris" not in page and b"/quit" not in page
        assert _post_quit(srv, "anything")[0] == 404
    finally:
        srv.server_close()


def test_cli_quit_button_stops_runner_and_returns_0(tmp_path, monkeypatch):
    """End to end over a real 127.0.0.1 port (needs local binding allowed)."""
    monkeypatch.setenv("OIRIS_PASSPHRASE", "test-passphrase")
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"data_dir": str(tmp_path / "data"), "live": {"adapters": [SIM], "hz": 10}}))
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    base = f"http://127.0.0.1:{port}"
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    got = {}

    def post(token, origin):
        req = urllib.request.Request(base + "/quit", data=f"token={token}".encode(), method="POST",
                                     headers={"Origin": origin} if origin else {})
        try:
            return direct.open(req, timeout=5).status, b""
        except urllib.error.HTTPError as e:
            return e.code, b""

    def drive():
        for _ in range(100):
            try:
                page = direct.open(base + "/", timeout=2).read().decode()
                break
            except OSError as e:
                got["err"] = repr(e); time.sleep(0.1)
        else:
            return
        tok = _token(page)
        got["bad"] = [post("wrong", base), post(tok, "http://evil.example"), post(tok, None)]
        try:
            direct.open(base + "/quit", timeout=2)
        except urllib.error.HTTPError as e:
            got["get"] = e.code
        got["alive_after_bad"] = json.loads(direct.open(base + "/api/latest", timeout=2).read())["running"]
        req = urllib.request.Request(base + "/quit", data=f"token={tok}".encode(), method="POST",
                                     headers={"Origin": base})
        got["ok"] = direct.open(req, timeout=5).read().decode()
    threading.Thread(target=drive, daemon=True).start()
    t0 = time.monotonic()
    assert cli.main(["run", "--port", str(port), "--duration", "20", "--config", str(p)]) == 0
    assert time.monotonic() - t0 < 15, "quit did not stop live mode (duration ran out)"
    assert "ok" in got, f"dashboard never answered: {got}"
    assert [c for c, _ in got["bad"]] == [403, 403, 403] and got["get"] == 405 and got["alive_after_bad"] is True
    assert "O Iris stopped" in got["ok"]
    ks, db = open_store(Config.load(p))
    try:
        (sw,) = SweepTimeline(db).history("live")
        assert sw["outcome"] == "done"          # sweep closed by the normal stop path, not left "open"
    finally:
        db.close()


# ---- outputs: the arbiter's pop-up and the motion state ---------------------------------------------
def _out_snap(output):
    return {**_snap(None), "output": output}


def _out(page):
    return page.split("<section id=OUT")[1].split("</section>")[0]


def test_pages_without_output_have_no_outputs_area():
    assert "id=OUT" not in render(None, _snap(None)) and "Outputs" not in render(None, _snap(None))
    assert "id=OUT" not in render(run_scenario("quiet", 1)[-1])


def test_popup_is_drawn_escaped_with_priority_and_hidden_at_expiry():
    popup = {"text": "<script>alert(1)</script> & Medium signal", "priority": "attention", "expires_in_s": 5.6}
    out = _out(render(None, _out_snap({"motion": "still", "imu": True, "popup": popup,
                                       "counts": {"played": 2, "suppressed": 1}})))
    assert "<script>" not in out and "&lt;script&gt;alert(1)&lt;/script&gt; &amp; Medium signal" in out
    assert "attention" in out and "6 s left" in out
    assert "animation:oiris-gone 0s linear 5.6s forwards" in out and "@keyframes oiris-gone" in out   # no script (CSP)
    assert "Motion: <b>still</b>" in out and "played 2" in out and "suppressed 1" in out and "dropped 0" in out


def test_expired_or_missing_popup_is_not_drawn():
    for popup in (None, {"text": "Medium signal", "priority": "attention", "expires_in_s": 0},
                  {"text": "Medium signal", "priority": "attention", "expires_in_s": -1.0},
                  {"text": "Medium signal", "priority": "attention", "expires_in_s": "soon"}, "Medium signal"):
        out = _out(render(None, _out_snap({"motion": "still", "imu": True, "popup": popup, "counts": {}})))
        assert "Medium signal" not in out and "No pop-up" in out


def test_motion_state_in_plain_words():
    want = {"still": "still", "walking": "walking", "fast": "moving fast", "unknown": "motion unknown",
            "warp": "motion unknown", None: "motion unknown"}
    for state, words in want.items():
        out = _out(render(None, _out_snap({"motion": state, "imu": True, "popup": None, "counts": {}})))
        assert f"Motion: <b>{words}</b>" in out
    out = _out(render(None, _out_snap({"motion": "unknown", "imu": False, "popup": None})))
    assert "Motion: <b>motion unknown (no motion sensor)</b>" in out
    for words in server.MOTION_WORDS.values():
        assert not violations(words)


def test_arbiter_state_renders_and_the_popup_goes_after_expiry():
    from oiris.modules.output import OutputArbiter, OutputSettings, alert

    class Clock:
        t = 1000.0

        def __call__(self):
            return self.t
    clk = Clock()
    a = OutputArbiter(settings=OutputSettings(popup_s=6.0), motion=lambda: "walking", clock=clk, start=False)
    try:
        assert alert(a, "confirmed", "A1").outcome == "suppressed"          # walking hides pop-ups
        out = _out(render(None, _out_snap({"motion": "walking", "imu": True, **a.state()})))
        assert "Motion: <b>walking</b>" in out and "No pop-up" in out and "suppressed 1" in out
        a.motion = lambda: "still"
        assert alert(a, "confirmed", "A2").outcome == "played"
        out = _out(render(None, _out_snap({"motion": "still", "imu": True, **a.state()})))
        assert "<b>Confirmed signal</b> · confirmed · 6 s left" in out
        clk.t += 6.0
        out = _out(render(None, _out_snap({"motion": "still", "imu": True, **a.state()})))
        assert "Confirmed signal" not in out and "No pop-up" in out
    finally:
        a.close()
