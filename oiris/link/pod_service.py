"""Pod-side request handler behind the Noise link (phone app = client). Adds the thermal live
view feed (P3) and the clock exchange (timesync.py) to the face-match EnrollmentServer ops. Requests/responses
are canonical JSON inside Noise frames (security/channel.fragment / Reassembler), carried over BLE by ble.py.

time_sync: {"op": "time_sync", "t0": <host ns>} -> {"ok": true, "t0": t0, "t1": <receive ns>, "t2": <send ns>} on
the pod's monotonic clock (the clock of its Sample.t_mono_ns). t1 is taken when the request's last frame arrives
(serve_once, before decryption) and t2 just before the reply is serialised, so the pod's turnaround (t2 - t1)
covers its own work and the host does not mistake it for link delay.

Link operations for glasses (ADR-0062): unlock / add_unlock_slot / panic / wipe_challenge / wipe / hud (ops.py) and
model_offer / model_chunk / model_commit / model_status (sideload.py). Each needs its collaborators (keystore, panic,
phone_verify_key, hud, pins_path + models_dir); without them the op replies {"ok": false, "error": "not available"}.
These op names are never passed to the enrollment server. The pairing window for add_unlock_slot is opened on the
pod itself (open_pairing_window), never by an op."""
from __future__ import annotations
import json, os, time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..security.channel import Reassembler, SecureChannel, fragment
from .ops import NOT_AVAILABLE, PAIRING_WINDOW_S, HudFeed, LinkOps
from .sideload import Sideload
from .timesync import MAX_TIMESTAMP_NS

LINK_OPS = ("unlock", "add_unlock_slot", "panic", "wipe_challenge", "wipe", "hud")
MODEL_OPS = ("model_offer", "model_chunk", "model_commit", "model_status")


def _dumps(d: dict) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class PodService:
    def __init__(self, thermal: Callable[[], np.ndarray | None] = lambda: None, enrollment: Any = None,
                 clock: Callable[[], int] = time.monotonic_ns, keystore: Any = None, panic: Any = None,
                 phone_verify_key: Any = None, pins_path: str | Path | None = None, models_dir: str | Path | None = None,
                 hud: HudFeed | None = None, now: Callable[[], float] = time.monotonic,
                 rand: Callable[[int], bytes] = os.urandom, sideload: Sideload | None = None):
        """`now`: monotonic seconds for the link ops (rate limit, windows, expiry); `rand`: wipe nonces; `sideload`:
        a prepared Sideload instead of one built from pins_path and models_dir (tests inject free space)."""
        self.thermal, self.enrollment, self.clock = thermal, enrollment, clock
        self.link = LinkOps(keystore, panic, phone_verify_key, hud, now=now, rand=rand)
        if sideload is None and pins_path is not None and models_dir is not None:
            sideload = Sideload(pins_path, models_dir)
        self.sideload = sideload

    def open_pairing_window(self, seconds: float = PAIRING_WINDOW_S) -> None:
        """A local action on the pod (never an op): add_unlock_slot is accepted for `seconds`, once."""
        self.link.open_pairing_window(seconds)

    def handle(self, raw: bytes | str, received_ns: int | None = None) -> str:
        """Answer one request. `received_ns`: the pod-clock arrival time, when the caller took it earlier (serve_once)."""
        t1 = self.clock() if received_ns is None else received_ns
        try:
            req = json.loads(raw)
            op = req.get("op")
        except (ValueError, AttributeError):
            return _dumps({"ok": False, "error": "bad request"})
        if op == "time_sync":
            t0 = req.get("t0")
            if type(t0) is not int or not 0 <= t0 < MAX_TIMESTAMP_NS:   # a JSON true is not a timestamp
                return _dumps({"ok": False, "error": "bad time_sync t0"})
            return _dumps({"ok": True, "t0": t0, "t1": t1, "t2": self.clock()})
        if op == "thermal":
            f = self.thermal()
            if f is None:
                return _dumps({"ok": True, "w": None, "h": None, "celsius": []})
            a = np.asarray(f, float)
            if a.ndim != 2:
                return _dumps({"ok": False, "error": "bad thermal frame"})
            return _dumps({"ok": True, "h": int(a.shape[0]), "w": int(a.shape[1]),
                           "celsius": [round(float(x), 1) for x in a.ravel()]})
        if op in LINK_OPS:
            return _dumps(self.link.handle(op, req))
        if op in MODEL_OPS:
            return _dumps(self.sideload.handle(op, req) if self.sideload is not None else {"ok": False, "error": NOT_AVAILABLE})
        if self.enrollment is not None:
            return str(self.enrollment.handle(raw))
        return _dumps({"ok": False, "error": f"unknown op {op!r}"})


def serve_once(ch: SecureChannel, rx: Reassembler, transport, svc: PodService, timeout_s: float = 5.0) -> bool:
    """Read one request from the transport, answer it. Returns False on timeout."""
    while True:
        f = transport.receive(timeout_s)
        if f is None:
            return False
        received_ns = svc.clock()
        msg = rx.feed(f)
        if msg is not None:
            for out in fragment(ch, svc.handle(msg, received_ns).encode()):
                transport.send(out)
            return True
