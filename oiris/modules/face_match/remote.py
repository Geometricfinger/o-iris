"""EnrollmentPort over the pod<->phone SecureChannel (spec 7.3: "used by the phone app").

The phone only guides and captures; this pod-side handler runs every check (opt-in, consent,
quality, angle), builds and seals templates. Requests/responses are canonical JSON
(`sort_keys`, compact) inside Noise transport frames; frames are raw 8-bit pixels, `fmt` "bgr" (w*h*3) or "gray" (w*h), base64.
Messages over 60 kB travel as several Noise frames (`security.channel.fragment` / `Reassembler`).
Errors come back as {"ok": false, "error": ...}; nothing here raises to the transport.
There is no count limit anywhere (spec 7.3).

Phone-guided voice enrollment (ADR-0057; oiris/link/README.md): the ops voice_prompts, voice_start, voice_sample,
voice_finish and voice_cancel drive the M-17 module (`voice`) in the order `oiris voice-link enroll` does; every
check is the module's own. pcm16 is base64 of little-endian signed 16-bit mono. Each sample is decoded into
buffers this handler zeroes after use (the request text itself is an immutable str: best effort, as everywhere in
Python); audio is never stored, logged or echoed. Without a voice module every voice op replies VOICE_OFF. M-17
itself is imported only when a voice op runs with the module on. `voice_*_request` / `parse_voice_*` are the phone
side, ported to android/core Enrollment.kt (vectors tests/vectors/voice_enrollment.json)."""
from __future__ import annotations
import base64, json, math
from typing import Any, Callable

import numpy as np

from ...config import VOICE_ENROLL_MAX_WINDOWS
from ...link.ops import bad_reply, load_reply, refusal
from ...security.channel import MAX_MESSAGE
from . import optin
from .port import Angle

MAX_SIDE = 1280   # refuse absurd frames (memory), not people

VOICE_OPS = ("voice_prompts", "voice_start", "voice_sample", "voice_finish", "voice_cancel")
VOICE_OFF = "voice link is off"
VOICE_RATE_MIN, VOICE_RATE_MAX = 8000, 48000     # = voice_link.core.SAMPLE_RATE_MIN/MAX (a test keeps them equal)
MAX_SAMPLE_S = 12                                # one enrollment sample; enroll_window_s is at most 15 s, default 7 s
PCM_SAMPLE_BYTES = 2
INT16_SCALE = 32768.0
MAX_PCM_BYTES = MAX_SAMPLE_S * VOICE_RATE_MAX * PCM_SAMPLE_BYTES   # 1,152,000
MAX_PCM_B64 = 4 * math.ceil(MAX_PCM_BYTES / 3)                     # 1,536,000 characters
MAX_PCM_SHARE_OF_MESSAGE = 4                     # an encoded sample stays under a quarter of channel.MAX_MESSAGE
if MAX_PCM_B64 * MAX_PCM_SHARE_OF_MESSAGE > MAX_MESSAGE:
    raise ValueError("the voice sample cap must stay well under channel.MAX_MESSAGE")
B64_QUANTUM, B64_GROUP_BYTES, B64_BITS, BYTE_BITS = 4, 3, 6, 8
B64_ALPHABET = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
B64_MAX_PAD = 2
BYTE_MASK = 0xFF
_B64_BAD = 255
_B64_TABLE = np.full(256, _B64_BAD, np.uint8)
_B64_TABLE[np.frombuffer(B64_ALPHABET, np.uint8)] = np.arange(len(B64_ALPHABET), dtype=np.uint8)
_UNUSED_BITS_MASK = {1: 0x03, 2: 0x0F}           # padding count -> low bits of the last character that must be zero
MAX_PROMPTS, MAX_REPLY_TEXT = 64, 400            # phone-side reply caps (parse_voice_*)


def dumps(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def _wipe(bufs: list) -> None:
    for b in bufs:
        if isinstance(b, np.ndarray):
            b.fill(0)
        else:
            b[:] = bytes(len(b))


def _pcm16_bytes(text: Any) -> int:
    """The decoded length of a pcm16 text, from its shape alone (checked before anything is decoded)."""
    if not isinstance(text, str) or len(text) % B64_QUANTUM:
        raise ValueError("pcm16 is not valid base64")
    if len(text) > MAX_PCM_B64:
        raise ValueError(f"voice sample too large (at most {MAX_PCM_BYTES} bytes)")
    pad = len(text) - len(text.rstrip("="))
    if pad > B64_MAX_PAD:
        raise ValueError("pcm16 is not valid base64")
    return len(text) // B64_QUANTUM * B64_GROUP_BYTES - pad


def _decode_pcm16(text: str, n_bytes: int, scratch: list) -> np.ndarray:
    """Strict standard base64 (padded, canonical) of little-endian int16 -> a new float32 buffer in [-1, 1).
    Every intermediate buffer, and the result, is appended to `scratch` for the caller to zero."""
    if not text.isascii():
        raise ValueError("pcm16 is not valid base64")
    enc = bytearray(text, "ascii")
    scratch.append(enc)
    pad = len(text) - len(text.rstrip("="))
    vals = _B64_TABLE[np.frombuffer(enc, np.uint8)]                    # a new array
    scratch.append(vals)
    body = len(vals) - pad
    if np.any(vals[:body] == _B64_BAD) or (pad and int(vals[body - 1]) & _UNUSED_BITS_MASK[pad]):
        raise ValueError("pcm16 is not valid base64")
    vals[body:] = 0
    q = vals.reshape(-1, B64_QUANTUM).astype(np.uint32)
    scratch.append(q)
    word = (q[:, 0] << 3 * B64_BITS) | (q[:, 1] << 2 * B64_BITS) | (q[:, 2] << B64_BITS) | q[:, 3]
    scratch.append(word)
    raw = np.empty((len(word), B64_GROUP_BYTES), np.uint8)
    scratch.append(raw)
    for k in range(B64_GROUP_BYTES):
        raw[:, k] = (word >> BYTE_BITS * (B64_GROUP_BYTES - 1 - k)) & BYTE_MASK
    pcm = raw.reshape(-1)[:n_bytes].view("<i2")                         # a view of `raw`
    x = np.empty(pcm.size, np.float32)
    scratch.append(x)
    np.multiply(pcm, np.float32(1.0 / INT16_SCALE), out=x, casting="unsafe")
    return x


class EnrollmentServer:
    def __init__(self, db, chain, device_id: str, factory: Callable[[], Any], module: Any = None, voice: Any = None):
        """`voice`: the loaded M-17 VoiceLinkModule, or None (every voice op then replies VOICE_OFF)."""
        self.db, self.chain, self.device_id, self.factory, self.m = db, chain, device_id, factory, module
        self.voice = voice

    def handle(self, raw: bytes | str) -> str:
        try:
            req = json.loads(raw)
            return dumps({"ok": True, **self._op(req)})
        except Exception as e:  # noqa: BLE001  (fault isolation: report, never crash the link)
            return dumps({"ok": False, "error": f"{type(e).__name__}: {e}"[:200]})

    def _mod(self):
        if self.m is None:
            raise PermissionError("face matching is off; opt in first")
        return self.m

    def _op(self, r: dict) -> dict:
        op = r.get("op")
        if op in VOICE_OPS:
            return self._voice_op(op, r)
        if op == "status":
            rec = optin.opt_in_record(self.db)
            return {"opted_in": rec is not None and self.m is not None, "modes": rec["modes"] if rec else []}
        if op == "warning":
            return {"text": optin.warning_for(r.get("jurisdiction", "strictest"))}
        if op == "opt_in":
            rec = optin.record_opt_in(self.db, self.chain, r["typed"], list(r["modes"]), self.device_id,
                                      r.get("jurisdiction", "strictest"), warning_shown=r.get("warning_shown"))
            self.m = self.factory()
            return {"modes": rec["modes"]}
        m = self._mod()
        if op == "start_session":
            return {"session": m.start_session()}
        if op == "add_person":
            return {"person": m.add_person(r["session"], r["label"], r.get("consent_id"), bool(r.get("guardian", False)))}
        if op == "shot":
            w, h = int(r["w"]), int(r["h"])
            if not (0 < w <= MAX_SIDE and 0 < h <= MAX_SIDE):
                raise ValueError("frame size out of range")
            ch = {"bgr": 3, "gray": 1}[r.get("fmt", "bgr")]
            buf = bytearray(base64.b64decode(r["pixels_b64"], validate=True))
            if len(buf) != w * h * ch:
                raise ValueError("frame bytes do not match w*h*channels")
            frame = np.frombuffer(bytes(buf), np.uint8).reshape((h, w, 3) if ch == 3 else (h, w)).copy()
            buf[:] = b"\0" * len(buf)
            try:
                res = m.submit_shot(r["person"], frame, Angle(r["angle"]))
            finally:
                frame.fill(0)
            needed = sorted(getattr(a, "value", a) for a in res.needed_angles)
            return {"accepted": res.ok, "code": res.code, "reason": res.reason, "needed": needed}
        if op == "finish":
            st = m.finish_person(r["person"])
            return {"complete": st.complete, "message": st.message, "missing": sorted(getattr(a, "value", a) for a in st.missing_angles)}
        if op == "end_session":
            m.end_session(r["session"]); return {}
        raise ValueError(f"unknown op {op!r}")

    # ---- phone-guided voice enrollment (M-17) -----------------------------------------------
    def _voice_op(self, op: str, r: dict) -> dict:
        v = self.voice
        if v is None:
            raise PermissionError(VOICE_OFF)
        if op == "voice_prompts":
            from ..voice_link import prompts
            return {"prompts": [{"id": p["id"], "text": p["text"], "tone": p["tone"]} for p in prompts.prompt_list()]}
        if op == "voice_start":
            windows = v.start_voice(_text(r, "person"), _text(r, "consent"))
            return {"windows": windows, "target_speech_s": float(v.cfg.enroll_min_total_speech_s),
                    "window_s": float(v.cfg.enroll_window_s), "max_windows": VOICE_ENROLL_MAX_WINDOWS}
        if op == "voice_sample":
            return self._voice_sample(v, r)
        if op == "voice_finish":
            st = v.finish_voice(_text(r, "person"))
            return {"saved": st.ok, "code": st.code, "message": st.reason, "total_speech_s": st.total_speech_s,
                    "remaining": st.needed}
        v.cancel_voice(_text(r, "person"))   # voice_cancel
        return {}

    def _voice_sample(self, v: Any, r: dict) -> dict:
        person, prompt_id, rate = _text(r, "person"), r.get("prompt_id"), r.get("rate")
        if prompt_id is not None and not isinstance(prompt_id, str):
            raise ValueError("prompt_id must be text")
        if isinstance(rate, bool) or not isinstance(rate, int):
            raise ValueError("rate must be an integer")
        n_bytes = _pcm16_bytes(r.get("pcm16"))
        if n_bytes % PCM_SAMPLE_BYTES:
            raise ValueError("pcm16 must hold whole 16-bit samples (an even number of bytes)")
        scratch: list = []
        try:
            if VOICE_RATE_MIN <= rate <= VOICE_RATE_MAX:
                if n_bytes // PCM_SAMPLE_BYTES > MAX_SAMPLE_S * rate:
                    raise ValueError(f"voice sample longer than {MAX_SAMPLE_S} s")
                x = _decode_pcm16(r["pcm16"], n_bytes, scratch)
            else:   # not decoded: the module refuses the rate itself (code "rate") after its earlier checks
                x = np.zeros(0, np.float32)
            res = v.submit_window(person, x, rate, prompt_id)
        finally:
            _wipe(scratch)
        return {"accepted": res.ok, "code": res.code, "message": res.reason, "speech_s": res.speech_s,
                "total_speech_s": res.total_speech_s, "level_dbfs": res.level_dbfs, "remaining": res.needed}


def _text(r: dict, key: str) -> str:
    v = r.get(key)
    if not isinstance(v, str) or not v:
        raise ValueError(f"{key} must be non-empty text")
    return v


# ---- phone side: request builders and reply parsers (android/core Enrollment.kt) ------------------------
def voice_prompts_request() -> str:
    return dumps({"op": "voice_prompts"})


def voice_start_request(person: str, consent: str) -> str:
    return dumps({"op": "voice_start", "person": person, "consent": consent})


def voice_sample_request(person: str, prompt_id: str, rate: int, pcm: Any) -> str:
    """`pcm`: int16 samples, sent as standard base64 of their little-endian bytes."""
    le = np.asarray(pcm, dtype="<i2").tobytes()
    return dumps({"op": "voice_sample", "person": person, "prompt_id": prompt_id, "rate": rate,
                  "pcm16": base64.b64encode(le).decode("ascii")})


def voice_finish_request(person: str) -> str:
    return dumps({"op": "voice_finish", "person": person})


def voice_cancel_request(person: str) -> str:
    return dumps({"op": "voice_cancel", "person": person})


def _count(v: Any) -> bool:
    return type(v) is int and v >= 0


def _seconds(v: Any) -> bool:
    return type(v) in (int, float) and math.isfinite(v) and v >= 0


def _reply_text(v: Any) -> bool:
    return isinstance(v, str) and len(v) <= MAX_REPLY_TEXT


def _parse(reply: str | bytes, body: Callable[[dict], dict | None]) -> dict:
    """Never raises: a reply that is not what the op returns parses as {"ok": false, "error": "bad reply"}."""
    r = load_reply(reply)
    if r is None:
        return bad_reply()
    if not r["ok"]:
        return refusal(r)
    out = body(r)
    return bad_reply() if out is None else {"ok": True, "error": None, **out}


def _prompts_body(r: dict) -> dict | None:
    ps = r.get("prompts")
    if not isinstance(ps, list) or not 0 < len(ps) <= MAX_PROMPTS:
        return None
    for p in ps:
        if not isinstance(p, dict) or sorted(p) != ["id", "text", "tone"] or not all(_reply_text(x) and x for x in p.values()):
            return None
    return {"prompts": [{"id": p["id"], "text": p["text"], "tone": p["tone"]} for p in ps]}


def _start_body(r: dict) -> dict | None:
    w: Any = r.get("windows")
    mx: Any = r.get("max_windows")
    t: Any = r.get("target_speech_s")
    ws: Any = r.get("window_s")
    if not (_count(w) and _count(mx) and 0 < w <= mx and _seconds(t) and _seconds(ws) and ws > 0):
        return None
    return {"windows": w, "max_windows": mx, "target_speech_s": float(t), "window_s": float(ws)}


def _sample_body(r: dict) -> dict | None:
    lv = r.get("level_dbfs")
    if not (type(r.get("accepted")) is bool and _reply_text(r.get("code")) and _reply_text(r.get("message"))
            and _seconds(r.get("speech_s")) and _seconds(r.get("total_speech_s")) and _count(r.get("remaining"))
            and (lv is None or (type(lv) in (int, float) and math.isfinite(lv)))):
        return None
    return {"accepted": r["accepted"], "code": r["code"], "message": r["message"], "speech_s": float(r["speech_s"]),
            "total_speech_s": float(r["total_speech_s"]), "level_dbfs": None if lv is None else float(lv),
            "remaining": r["remaining"]}


def _finish_body(r: dict) -> dict | None:
    if not (type(r.get("saved")) is bool and _reply_text(r.get("code")) and _reply_text(r.get("message"))
            and _seconds(r.get("total_speech_s")) and _count(r.get("remaining"))):
        return None
    return {"saved": r["saved"], "code": r["code"], "message": r["message"],
            "total_speech_s": float(r["total_speech_s"]), "remaining": r["remaining"]}


def parse_voice_prompts(reply: str | bytes) -> dict:
    return _parse(reply, _prompts_body)


def parse_voice_start(reply: str | bytes) -> dict:
    return _parse(reply, _start_body)


def parse_voice_sample(reply: str | bytes) -> dict:
    return _parse(reply, _sample_body)


def parse_voice_finish(reply: str | bytes) -> dict:
    return _parse(reply, _finish_body)


def parse_voice_ack(reply: str | bytes) -> dict:
    """voice_cancel: {"ok": true} or the refusal."""
    return _parse(reply, lambda r: {})
