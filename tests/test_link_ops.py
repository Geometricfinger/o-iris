"""Link operations for glasses (ADR-0062): unlock from the phone, panic, signed wipe, HUD feed, and the existing
pod ops unchanged. Synthetic only: a temp KeyStore, a fake erase, a fake clock and generated keys."""
import json, logging, os

import pytest
from nacl.signing import SigningKey

from oiris.link import ops as O
from oiris.link.pod_service import PodService
from oiris.link.timesync import request as ts_request
from oiris.security.keys import KeyStore, KeyStoreError
from oiris.security.panic import PanicController
from oiris.ui.glasses import GlassesState, display_list

PASS = "owner passphrase"
SECRET = bytes(range(1, 33))
WRONG = bytes(32)


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def env(tmp_path):
    ks = KeyStore(tmp_path / "key.json")
    ks.create(PASS)
    ks.add_slot("phone", O.b64url(SECRET))
    ks.lock()
    erased: list[bool] = []
    clock = Clock()
    panic = PanicController(soft=ks.lock, erase=lambda: (erased.append(True), ks.crypto_erase()), clock=clock)
    phone = SigningKey.generate()
    svc = PodService(keystore=ks, panic=panic, phone_verify_key=phone.verify_key, hud=O.HudFeed(), now=clock)
    return svc, ks, panic, clock, erased, phone


def call(svc: PodService, req: str) -> dict:
    return json.loads(svc.handle(req))


def step(clock: Clock) -> None:
    clock.t += O.UNLOCK_MIN_INTERVAL_S


# ---- unlock ------------------------------------------------------------------------------------------------
def test_unlock_right_secret_unlocks_and_resets_the_count(env):
    svc, ks, panic, clock, _, _ = env
    assert call(svc, O.unlock_request("phone", WRONG)) == {"ok": False, "error": O.UNLOCK_REFUSED}
    assert svc.link.unlock_failures == 1 and panic.failed_unlocks == 0   # phone failures never count toward the erase
    step(clock)
    assert call(svc, O.unlock_request("phone", SECRET)) == {"ok": True}
    assert ks.master and panic.failed_unlocks == 0 and svc.link.unlock_failures == 0


def test_unlock_refusals_never_say_why(env):
    svc, _, panic, clock, _, _ = env
    bad = [O.unlock_request("phone", WRONG), O.unlock_request("watch", SECRET),
           json.dumps({"op": "unlock", "slot": "phone", "secret": "short"}),
           json.dumps({"op": "unlock", "slot": "phone", "secret": O.b64url(SECRET) + "="}),
           json.dumps({"op": "unlock", "slot": "phone"})]
    for req in bad:
        clock.t += O.UNLOCK_BACKOFF_MAX_S      # past any backoff, so every one is really tried
        assert call(svc, req) == {"ok": False, "error": O.UNLOCK_REFUSED}
    assert svc.link.unlock_failures == len(bad) and panic.failed_unlocks == 0


def test_phone_failures_back_off_and_never_erase(env):
    svc, ks, panic, clock, erased, _ = env
    tried = []
    real = ks.unlock_slot
    ks.unlock_slot = lambda *a: (tried.append(a), real(*a))[1]   # type: ignore[method-assign]
    waits = []
    for _ in range(3 * panic.fail_limit):            # far past the 10-failure erase limit
        clock.t += O.UNLOCK_BACKOFF_MAX_S
        assert call(svc, O.unlock_request("phone", WRONG)) == {"ok": False, "error": O.UNLOCK_REFUSED}
        waits.append(svc.link.unlock_wait_s())
    assert not erased and ks.exists() and panic.failed_unlocks == 0 and len(tried) == 3 * panic.fail_limit
    assert waits[:4] == [O.UNLOCK_MIN_INTERVAL_S, 2 * O.UNLOCK_MIN_INTERVAL_S, 4 * O.UNLOCK_MIN_INTERVAL_S,
                         8 * O.UNLOCK_MIN_INTERVAL_S]                       # doubles from the second failure
    assert waits == sorted(waits) and waits[-1] == O.UNLOCK_BACKOFF_MAX_S    # capped
    clock.t += O.UNLOCK_BACKOFF_MAX_S - 1                                   # inside the wait: not tried
    assert call(svc, O.unlock_request("phone", SECRET))["ok"] is False and len(tried) == 3 * panic.fail_limit
    clock.t += 1                                                           # the wait is over: the owner gets in
    assert call(svc, O.unlock_request("phone", SECRET)) == {"ok": True} and ks.master
    assert svc.link.unlock_failures == 0 and svc.link.unlock_wait_s() == O.UNLOCK_MIN_INTERVAL_S


def test_rate_limit_refuses_without_trying_or_counting(env):
    svc, ks, panic, clock, _, _ = env
    call(svc, O.unlock_request("phone", WRONG))
    clock.t += O.UNLOCK_MIN_INTERVAL_S / 2
    tried = []
    real = ks.unlock_slot
    ks.unlock_slot = lambda *a: (tried.append(a), real(*a))[1]   # type: ignore[method-assign]
    assert call(svc, O.unlock_request("phone", SECRET)) == {"ok": False, "error": O.UNLOCK_REFUSED}
    assert tried == [] and svc.link.unlock_failures == 1 and panic.failed_unlocks == 0
    with pytest.raises(KeyStoreError):
        ks.master
    clock.t += O.UNLOCK_MIN_INTERVAL_S / 2
    assert call(svc, O.unlock_request("phone", SECRET)) == {"ok": True} and len(tried) == 1


def test_add_unlock_slot_needs_the_window_and_an_unlocked_store(env):
    svc, ks, _, clock, _, _ = env
    new = os.urandom(32)
    req = O.add_unlock_slot_request("phone", new)
    refused = {"ok": False, "error": O.ADD_SLOT_REFUSED}
    assert call(svc, req) == refused                          # locked, no window
    svc.open_pairing_window()
    assert call(svc, req) == refused                          # window, still locked
    ks.unlock(PASS)
    svc.link._pairing_until = None
    assert call(svc, req) == refused                          # unlocked, no window
    svc.open_pairing_window(10)
    clock.t += 10
    assert call(svc, req) == refused                          # window expired
    svc.open_pairing_window()
    assert call(svc, O.add_unlock_slot_request("other", new)) == refused   # slot not whitelisted
    assert call(svc, req) == {"ok": True}
    assert call(svc, O.add_unlock_slot_request("phone", SECRET)) == refused   # the window lets one through
    ks.lock()
    step(clock)
    assert call(svc, O.unlock_request("phone", new)) == {"ok": True}


def test_pairing_window_is_local_and_bounded(env):
    svc = env[0]
    assert call(svc, json.dumps({"op": "open_pairing_window"}))["ok"] is False
    with pytest.raises(ValueError):
        svc.open_pairing_window(O.PAIRING_WINDOW_MAX_S + 1)
    with pytest.raises(ValueError):
        svc.open_pairing_window(0)


def test_secret_is_in_no_reply_and_no_log(env, caplog):
    svc, ks, _, clock, _, _ = env
    caplog.set_level(logging.DEBUG)
    replies = [svc.handle(O.unlock_request("phone", WRONG))]
    step(clock)
    replies.append(svc.handle(O.unlock_request("phone", SECRET)))
    svc.open_pairing_window()
    new = os.urandom(32)
    replies.append(svc.handle(O.add_unlock_slot_request("phone", new)))
    text = "".join(replies) + caplog.text
    for s in (SECRET, WRONG, new):
        assert O.b64url(s) not in text and s.hex() not in text
    assert caplog.records    # the outcomes were logged, without the secret


def test_unlock_ops_not_available_without_collaborators():
    svc = PodService()
    for req in (O.unlock_request("phone", SECRET), O.add_unlock_slot_request("phone", SECRET), O.panic_request(),
                O.wipe_challenge_request(), O.wipe_request(b"x", b"y"), O.hud_request(0)):
        assert call(svc, req) == {"ok": False, "error": O.NOT_AVAILABLE}


def test_strict_base64url():
    assert O.from_b64url(O.b64url(SECRET), 32) == SECRET
    for s in (O.b64url(SECRET) + "=", O.b64url(SECRET)[:-1] + "B", "A" * 42, "+" * 43, 7, None):
        assert O.from_b64url(s, 32) is None
    assert O.from_b64url("AAE", max_bytes=1) is None and O.from_b64url("AAE", max_bytes=2) == b"\x00\x01"


# ---- panic and wipe ----------------------------------------------------------------------------------------
def test_panic_drops_the_keys(env):
    svc, ks, _, _, erased, _ = env
    ks.unlock(PASS)
    assert call(svc, O.panic_request()) == {"ok": True}
    with pytest.raises(KeyStoreError):
        ks.master
    assert not erased and ks.exists()


def test_panic_locks_even_when_soft_is_wired_elsewhere(tmp_path):
    ks = KeyStore(tmp_path / "k.json")
    ks.create(PASS)
    soft = []
    svc = PodService(keystore=ks, panic=PanicController(soft=lambda: soft.append(1), erase=lambda: None))
    assert call(svc, O.panic_request()) == {"ok": True} and soft == [1]
    with pytest.raises(KeyStoreError):
        ks.master


def _challenge(svc) -> bytes:
    r = call(svc, O.wipe_challenge_request())
    return bytes(O.from_b64url(r["nonce"], O.NONCE_BYTES))


def test_good_signature_wipes(env):
    svc, ks, panic, _, erased, phone = env
    cmd = O.wipe_command(_challenge(svc))
    assert call(svc, O.wipe_request(cmd, phone.sign(cmd).signature)) == {"ok": True, "wiped": True}
    assert erased == [True] and panic.erased and not ks.exists()


def test_bad_signatures_and_replays_refuse(env):
    svc, _, _, clock, erased, phone = env
    refused = {"ok": False, "error": O.WIPE_REFUSED}
    cmd = O.wipe_command(_challenge(svc))
    other = SigningKey.generate()
    assert call(svc, O.wipe_request(cmd, other.sign(cmd).signature)) == refused      # someone else's key
    assert call(svc, O.wipe_request(cmd, phone.sign(cmd).signature)) == refused      # the nonce was used up
    n1 = _challenge(svc)
    n2 = _challenge(svc)                                                             # a newer challenge replaces it
    old = O.wipe_command(n1)
    assert call(svc, O.wipe_request(old, phone.sign(old).signature)) == refused
    _challenge(svc)
    cmd2 = O.wipe_command(n2)
    assert call(svc, O.wipe_request(cmd2, phone.sign(cmd2).signature)) == refused    # n2 superseded too
    n = _challenge(svc)
    bad = O.wipe_command(n)
    sig = bytearray(phone.sign(bad).signature)
    sig[0] ^= 1
    assert call(svc, O.wipe_request(bad, bytes(sig))) == refused                     # tampered signature
    n = _challenge(svc)
    clock.t += O.WIPE_NONCE_TTL_S
    late = O.wipe_command(n)
    assert call(svc, O.wipe_request(late, phone.sign(late).signature)) == refused    # expired
    assert call(svc, O.wipe_request(b"ASB-WIPE:" + bytes(32), phone.sign(b"x").signature)) == refused  # no challenge
    n = _challenge(svc)
    unprefixed = b"ASB-WIPX:" + n
    assert call(svc, O.wipe_request(unprefixed, phone.sign(unprefixed).signature)) == refused
    assert not erased


def test_wipe_challenge_is_fresh(env):
    svc = env[0]
    assert len({_challenge(svc) for _ in range(4)}) == 4


# ---- HUD ---------------------------------------------------------------------------------------------------
def test_hud_versions_only_on_change(env):
    svc = env[0]
    feed = svc.link.hud
    assert call(svc, O.hud_request(0)) == {"ok": True, "version": 0, "unchanged": True}
    a = display_list(GlassesState(level="low", alert_text="Signal nearby"))
    assert feed.set(a) == 1 and feed.set(json.loads(json.dumps(a))) == 1
    full = call(svc, O.hud_request(0))
    assert full["version"] == 1 and full["ops"] == a
    assert call(svc, O.hud_request(1)) == {"ok": True, "version": 1, "unchanged": True}
    assert feed.set(display_list(GlassesState(locked=True))) == 2
    assert call(svc, O.hud_request(1))["version"] == 2
    assert call(svc, json.dumps({"op": "hud"}))["ops"]
    assert call(svc, json.dumps({"op": "hud", "since": -1}))["ok"] is False
    assert call(svc, json.dumps({"op": "hud", "since": True}))["ok"] is False


def test_hud_caps_pod_side():
    feed = O.HudFeed()
    base = display_list(GlassesState())
    text = {"op": "text", "id": "t", "x": 1, "y": 2, "text": "a", "size": 12, "colour": "#fff", "anchor": "start"}
    with pytest.raises(ValueError):
        feed.set(base + [dict(text)] * O.MAX_HUD_OPS)
    with pytest.raises(ValueError):
        feed.set(base + [{**text, "text": "x" * (O.MAX_HUD_TEXT + 1)}])
    with pytest.raises(ValueError):
        feed.set(base + [{**text, "x": base[0]["w"] + 1}])
    with pytest.raises(ValueError):
        feed.set(base + [{**text, "op": "script"}])
    with pytest.raises(ValueError):
        feed.set(base[1:])
    with pytest.raises(ValueError):
        feed.set(base + [base[0]])
    assert feed.version == 0
    feed.set(base + [{**text, "text": "x" * O.MAX_HUD_TEXT}])
    assert feed.version == 1


def test_hud_parse_cleans_hostile_frames():
    canvas = display_list(GlassesState())[0]
    W, H = canvas["w"], canvas["h"]
    text = {"op": "text", "id": "t", "x": -5, "y": H + 9.5, "text": "y" * 200, "size": 12, "colour": "#fff",
            "anchor": "start", "onclick": "x"}
    frame = O.parse_hud(json.dumps({"ok": True, "version": 3, "ops": [canvas, text, {"op": "script"}, canvas]}))
    assert frame["dropped"] == 2 and frame["version"] == 3
    t = frame["ops"][1]
    assert t["x"] == 0 and t["y"] == float(H) and isinstance(t["y"], float) and len(t["text"]) == O.MAX_HUD_TEXT
    assert "onclick" not in t
    assert O.parse_hud(json.dumps({"ok": True, "version": 1, "ops": [text]}))["ops"] == []
    assert O.parse_hud('{"ok":true,"version":1,"ops":[NaN]}') == {"ok": False, "error": O.BAD_REPLY}
    many = O.parse_hud(json.dumps({"ok": True, "version": 1, "ops": [canvas] + [dict(text)] * 100}))
    assert len(many["ops"]) == O.MAX_HUD_OPS and many["dropped"] == 101 - O.MAX_HUD_OPS
    assert W == 640


# ---- existing ops unchanged --------------------------------------------------------------------------------
def test_time_sync_and_enrollment_unchanged(env):
    ticks = iter([5, 9])
    assert json.loads(PodService(clock=lambda: next(ticks)).handle(ts_request(3))) == {"ok": True, "t0": 3, "t1": 5, "t2": 9}

    class Enroll:
        def __init__(self):
            self.seen = []

        def handle(self, raw):
            self.seen.append(raw)
            return '{"ok":true,"from":"enrollment"}'

    e = Enroll()
    svc = PodService(enrollment=e)
    assert call(svc, '{"op":"status"}') == {"ok": True, "from": "enrollment"}
    assert call(svc, O.unlock_request("phone", SECRET)) == {"ok": False, "error": O.NOT_AVAILABLE}
    assert call(svc, '{"op":"model_status"}') == {"ok": False, "error": O.NOT_AVAILABLE}
    assert e.seen == ['{"op":"status"}']           # link ops never reach the enrollment server
    assert call(PodService(), '{"op":"nope"}') == {"ok": False, "error": "unknown op 'nope'"}
    assert call(PodService(), "[1]") == {"ok": False, "error": "bad request"}
    assert call(PodService(), '{"op":"thermal"}') == {"ok": True, "w": None, "h": None, "celsius": []}
