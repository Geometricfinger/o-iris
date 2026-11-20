"""One clock across devices (oiris/link/timesync.py) and the pod's time_sync op: the NTP math and its floor rule,
the estimate against a simulated remote clock (offset, drift, seeded link delays), refusals, quality states, the
op through PodService.handle and over the Noise/BLE loopback, and the Kotlin vector file being current."""
import importlib.util, json, random, threading
from pathlib import Path

import pytest

from oiris.link import ble, timesync as T
from oiris.link.pod_service import PodService, serve_once
from oiris.security.channel import Reassembler, fragment
from tests.test_link_ble import _paired

ROOT = Path(__file__).resolve().parents[1]
MS, S = 1_000_000, 1_000_000_000
BOOT = 3 * 86_400 * S                 # host uptime: realistic monotonic magnitudes
LEG = (7_500_000, 15_000_000)         # one BLE leg: one or two 7.5 ms connection events
TURNAROUND = (200_000, 2_000_000)     # pod t2 - t1
OFFSET_TOL_NS = 4 * MS                # half the largest leg asymmetry (3.75 ms) plus drift-carry slack
DRIFT_TOL_PPM = 5.0                   # Theil-Sen on 48 exchanges over 16 minutes
SEEDS = range(10)


class Remote:
    """A remote monotonic clock: remote = host + offset + ppm * (host - BOOT) / 1e6, integer ns."""

    def __init__(self, offset_ns: int, ppm: int):
        self.offset_ns, self.ppm = offset_ns, ppm

    def __call__(self, host_ns: int) -> int:
        return host_ns + self.offset_ns + self.ppm * (host_ns - BOOT) // T.PPM


def simulate(ts: T.TimeSync, remote: Remote, n: int, period_ns: int, seed: int) -> int:
    """n exchanges with seeded random leg delays; returns the host time after the last."""
    rng = random.Random(seed)
    for k in range(n):
        t0 = BOOT + k * period_ns
        h1 = t0 + rng.randint(*LEG)
        h2 = h1 + rng.randint(*TURNAROUND)
        t3 = h2 + rng.randint(*LEG)
        assert ts.add(t0, remote(h1), remote(h2), t3) is None
    return BOOT + n * period_ns


def test_offset_and_delay_on_known_clocks():
    R = 40 * 86_400 * S
    # symmetric 10 ms legs, 1 ms turnaround: exact
    assert T.offset_ns(BOOT, BOOT + R + 10 * MS, BOOT + R + 11 * MS, BOOT + 21 * MS) == R
    assert T.delay_ns(BOOT, BOOT + R + 10 * MS, BOOT + R + 11 * MS, BOOT + 21 * MS) == 20 * MS
    assert T.offset_ns(BOOT + R, BOOT + 10 * MS, BOOT + 11 * MS, BOOT + R + 21 * MS) == -R
    # asymmetric legs (40 ms up, 5 ms down): biased by half the asymmetry, still within delay / 2
    off = T.offset_ns(BOOT, BOOT + R + 40 * MS, BOOT + R + 41 * MS, BOOT + 46 * MS)
    assert off - R == 35 * MS // 2 and abs(off - R) <= T.delay_ns(BOOT, BOOT + R + 40 * MS, BOOT + R + 41 * MS, BOOT + 46 * MS) // 2


@pytest.mark.parametrize("t,expected", [((0, 1000, 1000, 1001), 499), ((1000, 0, 0, 1001), -1001), ((0, 0, 1, 2), -1),
                                        ((10, 5, 5, 11), -6)])
def test_offset_floors_odd_sums(t, expected):
    assert T.offset_ns(*t) == expected
    assert T.offset_ns(*t) == ((t[1] - t[0]) + (t[2] - t[3])) // 2


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("offset,ppm", [(40 * 86_400 * S, 40), (-(BOOT // 2), -25), (12_345_678_901, 0)])
def test_estimate_tracks_a_simulated_remote_clock(seed, offset, ppm):
    ts, remote = T.TimeSync(), Remote(offset, ppm)
    end = simulate(ts, remote, 48, 20 * S, seed)
    drift = ts.drift_ppm()
    assert drift is not None and abs(drift - ppm) <= DRIFT_TOL_PPM
    for at in (BOOT, BOOT + 400 * S, end, end + 60 * S):      # inside, at the end, and a minute past the last exchange
        assert abs(ts.offset_at(at) - (remote(at) - at)) <= OFFSET_TOL_NS
        assert abs(ts.remote_to_host(remote(at), at) - at) <= OFFSET_TOL_NS
    assert ts.quality(end) == T.OK


def test_drift_waits_for_enough_samples_and_span():
    remote = Remote(S, 40)
    few = T.TimeSync()
    simulate(few, remote, T.DRIFT_MIN_SAMPLES - 1, 5 * S, 1)
    assert few.drift_ppm() is None
    short = T.TimeSync()
    simulate(short, remote, 30, 100 * MS, 1)                 # 3 s: below DRIFT_MIN_SPAN_NS
    assert short.drift_ppm() is None
    assert short.offset_at(BOOT + 600 * S) == short.best().offset_ns   # no drift: the best offset as it is


@pytest.mark.parametrize("t,why", [
    ((BOOT, BOOT + MS, BOOT + 31 * MS, BOOT + 10 * MS), T.REJECT_NEGATIVE),     # pod turnaround longer than the round trip
    ((BOOT, BOOT, BOOT, BOOT + T.MAX_DELAY_NS + 1), T.REJECT_SLOW),
    ((BOOT, BOOT + 11 * MS, BOOT + 10 * MS, BOOT + 21 * MS), T.REJECT_ORDER),
    ((-1, 5, 6, 10), T.REJECT_RANGE),
    ((0, T.MAX_TIMESTAMP_NS, T.MAX_TIMESTAMP_NS, 10), T.REJECT_RANGE),
])
def test_refused_exchanges_are_not_kept(t, why):
    ts = T.TimeSync()
    assert ts.add(*t) == why
    assert ts.best() is None and ts.quality(BOOT) == T.UNSYNCED
    with pytest.raises(T.TimeSyncError):
        ts.remote_to_host(BOOT, BOOT)
    assert T.reject_reason(BOOT, BOOT, BOOT, BOOT + T.MAX_DELAY_NS) is None       # the maximum itself is allowed


def test_best_of_n_ties_and_window():
    def ex(t0, up, down, off):
        return t0, t0 + up + off, t0 + up + MS + off, t0 + up + MS + down

    ts = T.TimeSync()
    for k, (up, down) in enumerate([(30 * MS, 25 * MS), (5 * MS, 4 * MS), (12 * MS, 9 * MS)]):
        ts.add(*ex(BOOT + k * S, up, down, 7))
    assert ts.best() == T.Exchange(*ex(BOOT + S, 5 * MS, 4 * MS, 7))
    tie = T.TimeSync()
    tie.add(*ex(BOOT, 5 * MS, 5 * MS, 0))
    tie.add(*ex(BOOT + S, 4 * MS, 6 * MS, 0))
    assert tie.best().t0 == BOOT + S                         # equal delay: the most recent
    small = T.TimeSync(max_samples=2)
    for k, up in enumerate([MS, 9 * MS, 8 * MS]):
        small.add(*ex(BOOT + k * S, up, up, 0))
    assert len(small.samples) == 2 and small.best().t0 == BOOT + 2 * S      # the 1 ms sample fell out of the window
    with pytest.raises(ValueError):
        T.TimeSync(max_samples=0)


def test_quality_states():
    ts = T.TimeSync()
    assert ts.quality(BOOT) == T.UNSYNCED
    ts.add(BOOT, BOOT + 5 * MS, BOOT + 6 * MS, BOOT + 11 * MS)
    last = BOOT + 11 * MS
    assert ts.quality(last) == T.OK
    assert ts.quality(last + T.STALE_NS) == T.OK
    assert ts.quality(last + T.STALE_NS + 1) == T.STALE
    ts.add(BOOT, BOOT, BOOT, BOOT + T.MAX_DELAY_NS + 1)     # refused: does not refresh
    assert ts.quality(last + T.STALE_NS + 1) == T.STALE


class TickClock:
    """Injected pod clock: returns the next value of a list and records how many reads happened."""

    def __init__(self, *ticks: int):
        self.ticks, self.reads = list(ticks), 0

    def __call__(self) -> int:
        self.reads += 1
        return self.ticks.pop(0)


def test_pod_op_through_handle_with_an_injected_clock():
    clock = TickClock(5_000, 5_700)
    svc = PodService(clock=clock)
    r = json.loads(svc.handle(T.request(123)))
    assert r == {"ok": True, "t0": 123, "t1": 5_000, "t2": 5_700} and clock.reads == 2   # t1 first read, t2 last
    # an arrival time taken earlier by the caller (serve_once) is used as t1; t2 is still read last
    clock = TickClock(9_000)
    r = json.loads(PodService(clock=clock).handle(T.request(1), received_ns=8_500))
    assert (r["t1"], r["t2"]) == (8_500, 9_000)
    # host side: request -> pod reply -> estimate
    ts = T.TimeSync()
    t0, t3 = BOOT, BOOT + 19 * MS        # 9 ms each way, 1 ms on the pod, pod clock 1 s ahead
    reply = PodService(clock=TickClock(BOOT + S + 9 * MS, BOOT + S + 10 * MS)).handle(T.request(t0))
    assert ts.add_reply(reply, t0, t3) is None and ts.best().offset_ns == S and ts.best().delay_ns == 18 * MS


@pytest.mark.parametrize("req", ['{"op":"time_sync"}', '{"op":"time_sync","t0":null}', '{"op":"time_sync","t0":"5"}',
                                 '{"op":"time_sync","t0":5.0}', '{"op":"time_sync","t0":true}', '{"op":"time_sync","t0":-1}',
                                 f'{{"op":"time_sync","t0":{T.MAX_TIMESTAMP_NS}}}', '{"op":"time_sync","t0":[1]}'])
def test_pod_op_refuses_a_bad_t0(req):
    r = json.loads(PodService(clock=TickClock(1, 2)).handle(req))
    assert r == {"ok": False, "error": "bad time_sync t0"}
    assert T.TimeSync().add_reply(json.dumps(r), 5, 6) == T.REJECT_REPLY


@pytest.mark.parametrize("reply", ['{"ok":true,"t0":6,"t1":1,"t2":2}', '{"ok":true,"t0":5,"t1":1.0,"t2":2}',
                                   '{"ok":true,"t0":5,"t1":1,"t2":false}', '{"ok":true,"t0":5,"t1":1}', '{"t0":5,"t1":1,"t2":2}',
                                   f'{{"ok":true,"t0":5,"t1":1,"t2":{1 << 63}}}', "[5]", "{", ""])
def test_host_refuses_a_bad_reply(reply):
    ts = T.TimeSync()
    assert ts.add_reply(reply, 5, 6) == T.REJECT_REPLY and ts.best() is None


def test_time_sync_over_noise_and_ble_loopback():
    pod, phone = _paired()
    phone_link, pod_link = ble.loopback_pair(185)
    phone_link.send(phone.write_handshake()); pod.read_handshake(pod_link.receive(1))
    pod_link.send(pod.write_handshake()); phone.read_handshake(phone_link.receive(1))
    phone_link.send(phone.write_handshake()); pod.read_handshake(pod_link.receive(1))
    clock = TickClock(BOOT + 3 * S, BOOT + 3 * S + MS)
    svc = PodService(clock=clock)
    t = threading.Thread(target=serve_once, args=(pod, Reassembler(pod), pod_link, svc))
    t.start()
    for fr in fragment(phone, T.request(BOOT).encode()):
        phone_link.send(fr)
    rx, resp = Reassembler(phone), None
    while resp is None:
        resp = rx.feed(phone_link.receive(2))
    t.join(2)
    assert json.loads(resp) == {"ok": True, "t0": BOOT, "t1": BOOT + 3 * S, "t2": BOOT + 3 * S + MS}
    assert clock.reads == 2              # serve_once took t1 at arrival; handle read only t2


def test_time_sync_vectors_are_current():
    spec = importlib.util.spec_from_file_location("make_android_vectors", ROOT / "scripts/make_android_vectors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    on_disk = json.loads((ROOT / "tests/vectors/time_sync.json").read_text())
    assert mod.differences(on_disk, json.loads(json.dumps(mod.time_sync()))) == []
    names = {e["name"] for e in on_disk["exchanges"]}
    assert {e["reject"] for e in on_disk["exchanges"]} >= {None, T.REJECT_NEGATIVE, T.REJECT_SLOW, T.REJECT_ORDER, T.REJECT_RANGE}
    assert any("odd" in n for n in names) and any("asymmetric" in n for n in names)
    assert any(s["drift_ppm"] is not None for s in on_disk["sessions"])
    assert any(r["reject"] is None for r in on_disk["replies"]) and any(r["reject"] == T.REJECT_REPLY for r in on_disk["replies"])
