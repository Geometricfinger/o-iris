"""The face camera adapter (ADR-0057 decision 5) and live mode's gate around it. Synthetic only: a fake
VideoCapture handing out synth_face_frame arrays (no real camera, no real face) and a fake clock."""
from __future__ import annotations
import ast, logging, sys
from pathlib import Path

import numpy as np
import pytest

from oiris.adapters import camera as camera_mod
from oiris.adapters.camera import DEFAULT_FPS, MAX_FPS, MIN_FPS, FaceCameraAdapter, zero_frame
from oiris.adapters.drivers import DriverUnavailable
from oiris.capture.interlock import CAMERA, CLOSED, INDICATOR_FAILED, INDICATOR_UNCONFIRMED, Grant, Refusal
from oiris.live import (CAM_OFF, CAM_ON, CAM_PAUSE_HEAT, CAM_PAUSED, CAM_UNAVAILABLE, CAM_WAITING, CAMERA_RETRY_S,
                        _GatedCamera)
from oiris.modules.face_match.backends.test_deterministic import synth_face_frame
from oiris.registry import default_registry
from oiris.schema import Health, Sample

ROOT = Path(__file__).resolve().parents[1]
SECONDS = 5.0             # simulated seconds of reading in the rate test
READ_TIMEOUT_S = 2.0
FRAME_SEED = 11           # a synthetic "face" (no real person)
PIXEL = 7


class Clock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


class FakeVideoCapture:
    """cv2.VideoCapture stand-in: each read() hands out a fresh synthetic frame."""

    def __init__(self, device: int, log: list, opened: bool = True, frame=None):
        self.device, self.log, self.opened = device, log, opened
        self.frame = frame
        self.released = False
        self.handed: list[np.ndarray] = []
        log.append(("vc_open", device))

    def isOpened(self) -> bool:  # noqa: N802 - OpenCV's name
        return self.opened

    def read(self):
        f = synth_face_frame(FRAME_SEED) if self.frame is None else self.frame.copy()
        self.handed.append(f)
        return True, f

    def release(self) -> None:
        self.released = True
        self.log.append(("vc_release",))


def make(log, clock=None, **kw):
    made: list[FakeVideoCapture] = []

    def factory(device):
        vc = FakeVideoCapture(device, log, **kw)
        made.append(vc)
        return vc
    clk = clock or Clock()
    return FaceCameraAdapter(video_capture=factory, clock=clk, sleep=clk.sleep), made, clk


# ---- the adapter ------------------------------------------------------------------------------------------
def test_frames_are_uint8_arrays_at_the_fps_limit():
    log: list = []
    a, made, clk = make(log)
    a.open({"device": 1, "fps": DEFAULT_FPS, "site_key": b"ignored"})
    assert a.id == "face_camera" and a.caps == {"face_frame"} and made[0].device == 1
    got, at = [], []
    while clk.t < SECONDS:
        s = a.read(READ_TIMEOUT_S)
        if s is not None:
            got.append(s)
            at.append(clk.t)
    assert len(got) == int(SECONDS * DEFAULT_FPS) + 1                  # frames at 0, 0.5, ... 5.0 s: 2 a second
    assert all(b - a_ == pytest.approx(1.0 / DEFAULT_FPS) for a_, b in zip(at, at[1:]))   # never closer
    assert all(isinstance(s, Sample) and isinstance(s.data, np.ndarray) and s.data.dtype == np.uint8 for s in got)
    assert all(s.data is h for s, h in zip(got, made[0].handed))       # the array itself, no copy left behind
    assert a.wait_s() == pytest.approx(1.0 / DEFAULT_FPS) and a.read(0.0) is None   # read(0) never waits
    clk.t += 1.0 / DEFAULT_FPS
    assert a.wait_s() == 0.0 and a.read(0.0) is not None
    a.close()
    assert made[0].released and not a.is_open and a.read(READ_TIMEOUT_S) is None


def test_fps_bounds_and_options():
    log: list = []
    for fps in (MIN_FPS, MAX_FPS, 1):
        a, _, _ = make(log)
        a.open({"fps": fps})
        assert a.fps == float(fps)
        a.close()
    for bad in ({"fps": MIN_FPS / 2}, {"fps": MAX_FPS * 2}, {"fps": True}, {"fps": "2"}, {"device": -1},
                {"device": True}, {"device": "0"}):
        a, made, _ = make(log)
        with pytest.raises(ValueError):
            a.open(bad)
        assert made == []                                              # bad options: no device touched


def test_driver_unavailable_without_cv2(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)                      # import cv2 raises ImportError
    a = FaceCameraAdapter()
    with pytest.raises(DriverUnavailable, match="opencv"):
        a.open({})
    assert not a.is_open


def test_a_camera_that_does_not_open_is_released_and_unavailable():
    log: list = []
    a, made, _ = make(log, opened=False)
    with pytest.raises(DriverUnavailable, match="camera 0 could not be opened"):
        a.open({})
    assert made[0].released and not a.is_open


def test_an_unexpected_frame_format_is_zeroed_and_dropped():
    log: list = []
    odd = np.full((4, 4), 3.5, dtype=np.float32)
    a, made, _ = make(log, frame=odd)
    a.open({})
    assert a.read(READ_TIMEOUT_S) is None and not made[0].handed[0].any()
    assert a.health()[0] == Health.DEGRADED


def test_registered_as_a_builtin_sensor():
    reg = default_registry({"entry_points": False})
    assert isinstance(reg.create("sensor", "face_camera"), FaceCameraAdapter)


def test_the_adapter_never_writes_logs_or_records_a_frame(caplog):
    """RAM only: the module has no logger, no file or network calls and no image writer; reading logs nothing."""
    src = Path(camera_mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for banned in ("write", "write_bytes", "write_text", "imwrite", "save", "savez", "tofile", "dump",
                   "dumps", "pickle", "logging", "log", "getLogger", "print", "socket", "VideoWriter"):
        assert banned not in names, banned
    log: list = []
    a, _, clk = make(log)
    with caplog.at_level(logging.DEBUG):
        a.open({})
        for _ in range(3):
            a.read(READ_TIMEOUT_S)
            clk.t += 1.0
        a.close()
    assert caplog.records == []


def test_zero_frame_only_touches_writable_arrays():
    f = synth_face_frame(FRAME_SEED)
    zero_frame(f)
    assert not f.any()
    ro = synth_face_frame(FRAME_SEED)
    ro.flags.writeable = False
    zero_frame(ro)                                                     # never raises
    assert ro.any()
    zero_frame(None)
    zero_frame([1, 2])


# ---- live mode's gate around it (fakes for the interlock, the plan and the device) ------------------------
class FakeInner:
    id, caps, bus, fps = "face_camera", {"face_frame"}, "usb", DEFAULT_FPS

    def __init__(self, log, fail: Exception | None = None, during_read=None):
        self.log, self.fail, self.during_read = log, fail, during_read
        self.frames: list[np.ndarray] = []
        self.seq = 0

    def open(self, cfg):
        self.log.append("open")
        if self.fail is not None:
            raise self.fail

    def read(self, timeout_s):
        if self.during_read is not None:
            self.during_read()
        f = np.full((8, 8), PIXEL, dtype=np.uint8)
        self.frames.append(f)
        self.seq += 1
        return Sample(self.id, self.seq, 0.0, self.seq, "dn", 1.0, f)

    def close(self):
        self.log.append("close")

    def health(self):
        return Health.OK, ""


class Interlock:
    def __init__(self, log, allow: bool = True, code: str = INDICATOR_UNCONFIRMED):
        self.log, self.allow, self.code = log, allow, code
        self.held = False
        self.revoked_code: str | None = None

    def acquire(self):
        self.log.append("acquire")
        if not self.allow:
            return Refusal(CAMERA, self.code)
        self.held, self.revoked_code = True, None
        return Grant(CAMERA, 0.0)

    def release(self):
        self.log.append("release")
        self.held = False

    def revoked(self):
        return None if self.held else (self.revoked_code or CLOSED)

    def revoke(self, code: str = INDICATOR_FAILED):
        self.held, self.revoked_code = False, code


def gated(log, il, limits=(None, None, None), clock=None, **inner):
    lim = {"v": limits}
    clk = clock or Clock()
    g = _GatedCamera(FakeInner(log, **inner), il.acquire, il.release, il.revoked, lambda: lim["v"], clk, idle_s=0.0)
    return g, lim, clk


def test_the_device_never_opens_before_the_grant_and_a_refusal_retries_later():
    log: list = []
    il = Interlock(log, allow=False)
    g, _, clk = gated(log, il)
    g.open({})
    assert log == ["acquire"] and g.state == CAM_WAITING and g.code == INDICATOR_UNCONFIRMED
    assert g.health() == (Health.DEGRADED, "waiting for the recording light")
    assert g.read(READ_TIMEOUT_S) is None and log == ["acquire"]       # no new try before the retry interval
    clk.t += CAMERA_RETRY_S
    assert g.read(READ_TIMEOUT_S) is None and log == ["acquire", "acquire"]
    il.allow = True
    clk.t += CAMERA_RETRY_S
    s = g.read(READ_TIMEOUT_S)
    assert s is not None and log == ["acquire", "acquire", "acquire", "open"] and g.state == CAM_ON
    assert log.index("open") > log.index("acquire")


def test_a_revoked_grant_stops_frames_at_once_and_releases_the_device():
    log: list = []
    il = Interlock(log)
    g, _, clk = gated(log, il)
    g.open({})
    assert g.read(READ_TIMEOUT_S) is not None
    il.revoke()
    assert g.enforce() is False                                       # the live loop: closed at once
    assert log[-2:] == ["close", "release"] and g.state == CAM_WAITING and g.code == INDICATOR_FAILED
    assert g.read(READ_TIMEOUT_S) is None and "open" not in log[-2:]  # no frame, no reopen inside the retry interval
    clk.t += CAMERA_RETRY_S
    assert g.read(READ_TIMEOUT_S) is not None and log[-2:] == ["acquire", "open"]


def test_a_frame_read_across_a_revocation_is_zeroed_and_dropped():
    log: list = []
    il = Interlock(log)
    g, _, _ = gated(log, il, during_read=lambda: il.revoke())
    g.open({})
    assert g.read(READ_TIMEOUT_S) is None
    assert not g._inner.frames[-1].any() and log[-2:] == ["close", "release"]


def test_frame_gate_from_the_plan_caps_rate_and_stride():
    log: list = []
    il = Interlock(log)
    g, _, clk = gated(log, il, limits=(None, 1.0, 2))
    g.open({})
    passed = []
    for i in range(8):                                                 # a frame every 0.5 s for 4 s
        clk.t = i * 0.5
        s = g.read(READ_TIMEOUT_S)
        if s is not None:
            passed.append(clk.t)
    assert passed == [0.0, 2.0]                                        # 1 a second, then every 2nd of those
    dropped = [f for f in g._inner.frames if not f.any()]
    assert len(dropped) == len(g._inner.frames) - len(passed)          # every refused frame was zeroed


def test_the_plan_pausing_face_match_closes_the_camera_and_resumes():
    log: list = []
    il = Interlock(log)
    g, lim, _ = gated(log, il)
    g.open({})
    assert g.read(READ_TIMEOUT_S) is not None
    lim["v"] = (CAM_PAUSE_HEAT, None, None)
    assert g.read(READ_TIMEOUT_S) is None and log[-2:] == ["close", "release"]
    assert g.state == CAM_PAUSED and g.code == CAM_PAUSE_HEAT and not il.held
    n = len(log)
    assert g.read(READ_TIMEOUT_S) is None and len(log) == n            # no acquire while paused
    lim["v"] = (None, None, None)
    assert g.read(READ_TIMEOUT_S) is not None and log[-2:] == ["acquire", "open"] and g.state == CAM_ON


def test_a_pause_before_the_open_never_asks_for_the_grant():
    log: list = []
    il = Interlock(log)
    g, _, _ = gated(log, il, limits=(CAM_PAUSE_HEAT, None, None))
    g.open({})
    assert log == [] and g.state == CAM_PAUSED


def test_a_missing_driver_releases_the_grant_and_stays_unavailable():
    log: list = []
    il = Interlock(log)
    g, _, clk = gated(log, il, fail=DriverUnavailable("pip install opencv-python-headless"))
    g.open({})
    assert log == ["acquire", "open", "release"] and g.state == CAM_UNAVAILABLE and not il.held
    assert g.health()[0] == Health.FAILED and "opencv" in g.health()[1]
    clk.t += 10 * CAMERA_RETRY_S
    assert g.read(READ_TIMEOUT_S) is None and log == ["acquire", "open", "release"]   # not retried


def test_stop_and_close_release_the_device():
    log: list = []
    il = Interlock(log)
    g, _, _ = gated(log, il)
    g.open({})
    g.read(READ_TIMEOUT_S)
    g.stop()
    assert log[-2:] == ["close", "release"] and not il.held
    g.close()
    assert log.count("close") == 1                                     # idempotent
    assert g.read(READ_TIMEOUT_S) is None and log[-1] == "release"     # stopped: never reopens


def test_a_closed_interlock_shows_off():
    log: list = []
    il = Interlock(log, allow=False, code=CLOSED)
    g, _, _ = gated(log, il)
    g.open({})
    assert g.state == CAM_OFF and g.code == CLOSED and "open" not in log
