"""Link operations a pair of glasses needs from the owner's phone (ADR-0062): unlock with a phone-held secret,
panic, signed remote wipe and the HUD display-list feed. Model side-load is in sideload.py. PodService
(pod_service.py) routes the ops here; the phone is the client, over the paired Noise channel only.

Wire format: canonical JSON (sorted keys, compact). Binary values are base64url without padding (RFC 4648 section 5),
strict: a string that is not the canonical encoding of the expected number of bytes is refused.

  unlock           {"op","slot","secret"}    -> {"ok":true} | {"ok":false,"error":"unlock refused"} (never says why)
  add_unlock_slot  {"op","slot","secret"}    -> {"ok":true} | {"ok":false,"error":"add_unlock_slot refused"}
  panic            {"op"}                    -> {"ok":true}
  wipe_challenge   {"op"}                    -> {"ok":true,"nonce":b64url(32 bytes)}
  wipe             {"op","command","signature"} -> {"ok":true,"wiped":true} | {"ok":false,"error":"wipe refused"}
  hud              {"op","since"}            -> {"ok":true,"version":n,"ops":[...]} | {"ok":true,"version":n,"unchanged":true}

The request builders and reply parsers below are the reference android/core LinkOps.kt reproduces byte for byte
(tests/vectors/link_ops.json, scripts/make_android_vectors.py). Parsers never raise: a reply that is not what the op
returns parses as {"ok": false, "error": BAD_REPLY}.

Secrets: the unlock secret is never logged or echoed. Python cannot reliably erase memory: the decoded bytes are a
bytearray zeroed after use, but the request text, the decoded JSON string and the base64 text handed to the KDF are
immutable str objects that live until the garbage collector reuses them. That is best effort, not erasure."""
from __future__ import annotations
import base64, binascii, hmac, json, logging, math, os, re, time
from typing import Any, Callable

from ..security.keys import KeyStoreError
from ..ui.glasses import OP_KEYS

log = logging.getLogger(__name__)

SECRET_BYTES = 32                  # the phone's unlock secret: 32 random bytes from its hardware-backed vault
NONCE_BYTES = 32                   # wipe challenge
SIGNATURE_BYTES = 64               # Ed25519
UNLOCK_MIN_INTERVAL_S = 2.0        # an unlock attempt sooner than this after the previous tried one is not tried
# Phone-slot failures double the wait (2, 4, 8 ... s, from the second failure on) up to this cap. They never count
# toward PanicController's 10-failure erase: a 32-byte random secret cannot be guessed, so counting would add no
# protection and would let a buggy phone app erase the keys in about 20 s (ADR-0062).
UNLOCK_BACKOFF_MAX_S = 300.0
UNLOCK_BACKOFF_MAX_DOUBLINGS = 16  # keeps 2 ** n small; the cap is reached long before
PAIRING_WINDOW_S = 60.0            # default add_unlock_slot window, opened on the pod itself
PAIRING_WINDOW_MAX_S = 300.0
WIPE_NONCE_TTL_S = 60.0
WIPE_PREFIX = b"ASB-WIPE:"         # panic.PanicController.remote_wipe's command prefix
UNLOCK_SLOTS = ("phone",)          # slot names a request may name (whitelist)

# HUD feed caps (pod side refuses, phone side HudFrame.parse drops / truncates / clamps)
MAX_HUD_OPS = 64                   # draw ops in one frame
MAX_HUD_TEXT = 80                  # characters (code points) in any string field of an op
MAX_CANVAS_PX = 4096               # canvas width / height
NUMBER_KEYS = ("x", "y", "w", "h", "size", "stroke_width", "opacity", "font_size")
HUD_X_KEYS, HUD_Y_KEYS = ("x",), ("y",)
LONG_MIN, LONG_MAX = -(1 << 63), (1 << 63) - 1   # a reply integer outside a signed 64-bit long is malformed (Kotlin)

NOT_AVAILABLE = "not available"
BAD_REQUEST = "bad request"
BAD_REPLY = "bad reply"
UNLOCK_REFUSED = "unlock refused"
ADD_SLOT_REFUSED = "add_unlock_slot refused"
WIPE_REFUSED = "wipe refused"

_B64URL = re.compile(r"[A-Za-z0-9_-]*")


def dumps(d: Any) -> str:
    """Canonical JSON, as Json.encode in android/core (ASCII only, sorted keys, compact)."""
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def b64url(b: bytes | bytearray) -> str:
    return base64.urlsafe_b64encode(bytes(b)).rstrip(b"=").decode("ascii")


def b64url_len(n: int) -> int:
    """Unpadded base64url length of n bytes."""
    return (4 * n + 2) // 3


def from_b64url(s: Any, n: int | None = None, max_bytes: int | None = None) -> bytearray | None:
    """Strict decode: only the canonical unpadded encoding (of exactly n bytes, or at most max_bytes), else None."""
    if not isinstance(s, str) or not _B64URL.fullmatch(s) or len(s) % 4 == 1:
        return None
    if n is not None and len(s) != b64url_len(n):
        return None
    if max_bytes is not None and len(s) > b64url_len(max_bytes):
        return None
    try:
        b = bytearray(base64.urlsafe_b64decode(s + "=" * (-len(s) % 4)))
    except (ValueError, binascii.Error):   # python-brain: allow SILENT_EXCEPTION - undecodable input is refused as None
        return None
    if not hmac.compare_digest(b64url(b), s):   # unused trailing bits set: not canonical
        b[:] = bytes(len(b))
        return None
    return b


def _wipe(b: bytearray | None) -> None:
    if b is not None:
        b[:] = bytes(len(b))


# ---- request builders (LinkOps.kt builds the same strings) ------------------------------------------------
def unlock_request(slot: str, secret: bytes) -> str:
    return dumps({"op": "unlock", "slot": slot, "secret": b64url(secret)})


def add_unlock_slot_request(slot: str, secret: bytes) -> str:
    return dumps({"op": "add_unlock_slot", "slot": slot, "secret": b64url(secret)})


def panic_request() -> str:
    return dumps({"op": "panic"})


def wipe_challenge_request() -> str:
    return dumps({"op": "wipe_challenge"})


def wipe_command(nonce: bytes) -> bytes:
    """The bytes the phone signs with its Ed25519 key."""
    return WIPE_PREFIX + bytes(nonce)


def wipe_request(command: bytes, signature: bytes) -> str:
    return dumps({"op": "wipe", "command": b64url(command), "signature": b64url(signature)})


def hud_request(since: int) -> str:
    return dumps({"op": "hud", "since": since})


def _unlocked(keystore: Any) -> bool:
    try:
        return keystore.master is not None
    except KeyStoreError:  # python-brain: allow SILENT_EXCEPTION - a locked store is the answer, not an error
        return False


# ---- pod side: unlock, panic, wipe ----------------------------------------------------------------------
class LinkOps:
    """Unlock / panic / wipe / hud handlers. Each needs its collaborators; without them the op is NOT_AVAILABLE.
    `now`: monotonic seconds (rate limit, pairing window, nonce expiry). `rand`: random bytes (wipe nonces)."""

    def __init__(self, keystore: Any = None, panic: Any = None, phone_verify_key: Any = None, hud: HudFeed | None = None,
                 now: Callable[[], float] = time.monotonic, rand: Callable[[int], bytes] = os.urandom):
        self.keystore, self.panic, self.phone_verify_key, self.hud = keystore, panic, phone_verify_key, hud
        self.now, self.rand = now, rand
        self._last_unlock: float | None = None
        self.unlock_failures = 0           # phone-slot failures since the last success (backoff only, never an erase)
        self._pairing_until: float | None = None
        self._nonce: tuple[bytes, float] | None = None

    def open_pairing_window(self, seconds: float = PAIRING_WINDOW_S) -> None:
        """A local action on the pod (button / console), never an op: lets one add_unlock_slot through."""
        if not 0 < seconds <= PAIRING_WINDOW_MAX_S:
            raise ValueError(f"pairing window must be in (0, {PAIRING_WINDOW_MAX_S}] s")
        self._pairing_until = self.now() + seconds

    def pairing_open(self) -> bool:
        return self._pairing_until is not None and self.now() < self._pairing_until

    def handle(self, op: str, req: dict) -> dict:
        return getattr(self, "_" + op)(req)

    def _unlock(self, req: dict) -> dict:
        if self.keystore is None or self.panic is None:
            return {"ok": False, "error": NOT_AVAILABLE}
        now = self.now()
        if self._last_unlock is not None and now - self._last_unlock < self.unlock_wait_s():
            log.info("unlock: refused by the rate limit (not tried, not counted)")
            return {"ok": False, "error": UNLOCK_REFUSED}
        self._last_unlock = now
        slot = req.get("slot")
        secret = from_b64url(req.get("secret"), SECRET_BYTES)
        ok = False
        try:
            if slot in UNLOCK_SLOTS and secret is not None:
                try:
                    self.keystore.unlock_slot(slot, b64url(secret))
                    ok = True
                except Exception:  # noqa: BLE001  # python-brain: allow SILENT_EXCEPTION - any failure is a refused unlock, counted below
                    ok = False
        finally:
            _wipe(secret)
        if ok:
            self.unlock_failures = 0
            log.info("unlock: %s", self.panic.unlock_result(True))   # a success resets every failure count
            return {"ok": True}
        self.unlock_failures += 1
        log.info("unlock: refused (phone failure %d; next try after %.0f s)", self.unlock_failures, self.unlock_wait_s())
        return {"ok": False, "error": UNLOCK_REFUSED}

    def unlock_wait_s(self) -> float:
        """Seconds after the last tried attempt before the next one is tried: UNLOCK_MIN_INTERVAL_S, doubled for
        each phone failure after the first, capped at UNLOCK_BACKOFF_MAX_S."""
        doublings = min(max(self.unlock_failures - 1, 0), UNLOCK_BACKOFF_MAX_DOUBLINGS)
        return min(UNLOCK_MIN_INTERVAL_S * 2 ** doublings, UNLOCK_BACKOFF_MAX_S)

    def _add_unlock_slot(self, req: dict) -> dict:
        if self.keystore is None:
            return {"ok": False, "error": NOT_AVAILABLE}
        slot = req.get("slot")
        secret = from_b64url(req.get("secret"), SECRET_BYTES)
        try:
            unlocked = _unlocked(self.keystore)
            if not (unlocked and self.pairing_open() and slot in UNLOCK_SLOTS and secret is not None):
                log.info("add_unlock_slot: refused")
                return {"ok": False, "error": ADD_SLOT_REFUSED}
            self.keystore.add_slot(slot, b64url(secret))
            self._pairing_until = None          # one slot per window
            log.info("add_unlock_slot: added slot %s", slot)
            return {"ok": True}
        finally:
            _wipe(secret)

    def _panic(self, req: dict) -> dict:
        if self.panic is None:
            return {"ok": False, "error": NOT_AVAILABLE}
        self.panic.soft()                       # the soft-panic path (the phone's volume-down gesture does the same)
        if self.keystore is not None:
            self.keystore.lock()                # keys dropped even if `soft` was wired to something else
        log.info("panic: soft panic from the phone")
        return {"ok": True}

    def _wipe_challenge(self, req: dict) -> dict:
        if self.panic is None or self.phone_verify_key is None:
            return {"ok": False, "error": NOT_AVAILABLE}
        nonce = bytes(self.rand(NONCE_BYTES))
        got_len, want_len = len(nonce), NONCE_BYTES   # lengths, not secrets (the compare lint keys on names)
        if got_len != want_len:
            raise ValueError("nonce source returned the wrong length")
        self._nonce = (nonce, self.now() + WIPE_NONCE_TTL_S)   # replaces any earlier challenge
        return {"ok": True, "nonce": b64url(nonce)}

    def _wipe(self, req: dict) -> dict:
        if self.panic is None or self.phone_verify_key is None:
            return {"ok": False, "error": NOT_AVAILABLE}
        outstanding, self._nonce = self._nonce, None          # single use: any attempt consumes it
        command = from_b64url(req.get("command"), len(WIPE_PREFIX) + NONCE_BYTES)
        signature = from_b64url(req.get("signature"), SIGNATURE_BYTES)
        if outstanding is None or self.now() >= outstanding[1] or command is None or signature is None:
            log.info("wipe: refused (no live challenge or malformed)")
            return {"ok": False, "error": WIPE_REFUSED}
        if not self.panic.remote_wipe(bytes(command), bytes(signature), self.phone_verify_key, outstanding[0]):
            log.info("wipe: refused (nonce or signature)")
            return {"ok": False, "error": WIPE_REFUSED}
        log.warning("wipe: signed remote wipe from the paired phone: erased")
        return {"ok": True, "wiped": True}

    def _hud(self, req: dict) -> dict:
        if self.hud is None:
            return {"ok": False, "error": NOT_AVAILABLE}
        since = req.get("since")
        if since is not None and (type(since) is not int or since < 0):
            return {"ok": False, "error": BAD_REQUEST}
        return self.hud.reply(since)


# ---- HUD display-list feed ---------------------------------------------------------------------------------
def _number(v: Any) -> bool:
    return type(v) in (int, float) and math.isfinite(v)


class HudFeed:
    """The current glasses display list (oiris/ui/glasses.display_list) for glasses that are only a display.
    set() bumps the version only when the content changes; the glasses poll with `since` and get "unchanged"."""

    def __init__(self) -> None:
        self.version = 0
        self.ops: list[dict] = []
        self._text = dumps([])

    @staticmethod
    def check(ops: list[dict]) -> None:
        """The caps, pod side: a display list that breaks one is a bug here and is refused loudly."""
        if not isinstance(ops, list) or len(ops) > MAX_HUD_OPS:
            raise ValueError(f"a HUD frame has at most {MAX_HUD_OPS} ops")
        if not ops or ops[0].get("op") != "canvas":
            raise ValueError("a HUD frame starts with its canvas op")
        W, H = ops[0]["w"], ops[0]["h"]
        if not all(type(v) is int and 1 <= v <= MAX_CANVAS_PX for v in (W, H)):
            raise ValueError("canvas size out of range")
        for i, op in enumerate(ops):
            kind = op.get("op") if isinstance(op, dict) else None
            keys = OP_KEYS.get(kind) if isinstance(kind, str) else None
            if keys is None or set(op) != set(keys) or (kind == "canvas") != (i == 0):
                raise ValueError(f"unknown or malformed draw op {op.get('op')!r}")
            for k, v in op.items():
                if isinstance(v, str) and len(v) > MAX_HUD_TEXT:
                    raise ValueError(f"{k} longer than {MAX_HUD_TEXT} characters")
                if k in NUMBER_KEYS and not _number(v):
                    raise ValueError(f"{k} is not a finite number")
            if op["op"] != "canvas":
                if "x" in op and not 0 <= op["x"] <= W or "y" in op and not 0 <= op["y"] <= H:
                    raise ValueError("draw op outside the canvas")
                if "w" in op and op["x"] + op["w"] > W or "h" in op and op["y"] + op["h"] > H:
                    raise ValueError("draw op outside the canvas")

    def set(self, ops: list[dict]) -> int:
        self.check(ops)
        text = dumps(ops)
        if text != self._text:
            self.ops, self._text = json.loads(text), text
            self.version += 1
        return self.version

    def reply(self, since: int | None) -> dict:
        if since == self.version:
            return {"ok": True, "version": self.version, "unchanged": True}
        return {"ok": True, "version": self.version, "ops": json.loads(self._text)}


# ---- reply parsers (LinkOps.kt parses the same way) -------------------------------------------------------
class _Malformed(ValueError):
    pass


def _parse_int(s: str) -> int:
    v = int(s)
    if not LONG_MIN <= v <= LONG_MAX:
        raise _Malformed("integer beyond a long")
    return v


def _parse_constant(s: str) -> Any:
    raise _Malformed(f"{s} is not JSON")


def load_reply(reply: str | bytes) -> dict | None:
    """The reply object, or None when it is not a JSON object with a boolean "ok" (Json.decode in Kotlin refuses
    NaN / Infinity and integers beyond a long, so this does too)."""
    try:
        r = json.loads(reply, parse_int=_parse_int, parse_constant=_parse_constant)
    except (ValueError, TypeError):  # python-brain: allow SILENT_EXCEPTION - a malformed reply parses as BAD_REPLY
        return None
    if not isinstance(r, dict) or type(r.get("ok")) is not bool:
        return None
    if r["ok"] is False and not isinstance(r.get("error"), str):
        return None
    return r


def bad_reply() -> dict:
    return {"ok": False, "error": BAD_REPLY}


def refusal(r: dict) -> dict:
    return {"ok": False, "error": r["error"]}


def parse_ack(reply: str | bytes) -> dict:
    """unlock, add_unlock_slot, panic: {"ok": true} or the refusal."""
    r = load_reply(reply)
    if r is None:
        return bad_reply()
    return {"ok": True, "error": None} if r["ok"] else refusal(r)


def parse_wipe_challenge(reply: str | bytes) -> dict:
    r = load_reply(reply)
    if r is None:
        return bad_reply()
    if not r["ok"]:
        return refusal(r)
    nonce = from_b64url(r.get("nonce"), NONCE_BYTES)
    return bad_reply() if nonce is None else {"ok": True, "error": None, "nonce": bytes(nonce).hex()}


def parse_wipe(reply: str | bytes) -> dict:
    r = load_reply(reply)
    if r is None:
        return bad_reply()
    if not r["ok"]:
        return refusal(r)
    return {"ok": True, "error": None, "wiped": True} if r.get("wiped") is True else bad_reply()


def _clamp(v: int | float, lo: int | float, hi: int | float) -> int | float:
    """Clamp keeping the number's type (an int stays an int, a float stays a float). Explicit comparisons, not
    min / max, so an in-range value comes back unchanged in both languages."""
    c = lo if v < lo else hi if v > hi else v
    return float(c) if isinstance(v, float) else int(c)


def _hud_op(op: Any, W: int, H: int) -> dict | None:
    """One untrusted draw op, cleaned: None when its kind is unknown or a key is missing or mistyped; strings
    truncated to MAX_HUD_TEXT code points; x / y clamped into the canvas, w / h so the op ends inside it, sizes into
    [0, min(W, H)], opacity into [0, 1]. Extra keys are dropped."""
    if not isinstance(op, dict) or not isinstance(op.get("op"), str):
        return None
    keys = OP_KEYS.get(op["op"])
    if keys is None or op["op"] == "canvas":
        return None
    out: dict[str, Any] = {}
    for k in keys:
        v = op.get(k)
        if k in NUMBER_KEYS:
            if not _number(v):
                return None
            out[k] = v
        elif isinstance(v, str):
            out[k] = v[:MAX_HUD_TEXT]
        else:
            return None
    for k in keys:
        if k == "x":
            out[k] = _clamp(out[k], 0, W)
        elif k == "y":
            out[k] = _clamp(out[k], 0, H)
        elif k in ("size", "stroke_width"):
            out[k] = _clamp(out[k], 0, min(W, H))
    if "w" in out:
        out["w"] = _clamp(out["w"], 0, W - out["x"])
    if "h" in out:
        out["h"] = _clamp(out["h"], 0, H - out["y"])
    return out


def _hud_canvas(op: Any) -> dict | None:
    if not isinstance(op, dict) or op.get("op") != "canvas":
        return None
    keys = OP_KEYS["canvas"]
    W, H = op.get("w"), op.get("h")
    if not all(type(v) is int and 1 <= v <= MAX_CANVAS_PX for v in (W, H)):
        return None
    out: dict[str, Any] = {}
    for k in keys:
        v = op.get(k)
        if k in NUMBER_KEYS:
            if not _number(v):
                return None
            out[k] = v
        elif isinstance(v, str):
            out[k] = v[:MAX_HUD_TEXT]
        else:
            return None
    out["opacity"] = _clamp(out["opacity"], 0, 1)
    out["font_size"] = _clamp(out["font_size"], 0, min(out["w"], out["h"]))
    return out


def parse_hud(reply: str | bytes) -> dict:
    """HudFrame.parse: {"ok", "error", "version", "unchanged", "ops", "dropped"}. The frame's first op must be a valid
    canvas (else every op is dropped); only the first MAX_HUD_OPS input ops are looked at; ops that fail _hud_op and
    any further canvas op are dropped and counted."""
    r = load_reply(reply)
    if r is None:
        return bad_reply()
    if not r["ok"]:
        return refusal(r)
    version = r.get("version")
    if type(version) is not int or version < 0:
        return bad_reply()
    if r.get("unchanged") is True:
        return {"ok": True, "error": None, "version": version, "unchanged": True, "ops": [], "dropped": 0}
    raw = r.get("ops")
    if not isinstance(raw, list):
        return bad_reply()
    ops: list[dict] = []
    canvas = _hud_canvas(raw[0]) if raw else None
    if canvas is not None:
        ops.append(canvas)
        for op in raw[1:MAX_HUD_OPS]:
            clean = _hud_op(op, canvas["w"], canvas["h"])
            if clean is not None:
                ops.append(clean)
    return {"ok": True, "error": None, "version": version, "unchanged": False, "ops": ops, "dropped": len(raw) - len(ops)}
