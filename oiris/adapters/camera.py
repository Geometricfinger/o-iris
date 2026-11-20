"""The face camera (ADR-0057 decision 5): camera frames for M-16 face matching in live mode.

FaceCameraAdapter reads one camera through OpenCV's cv2.VideoCapture (Apache-2.0), imported lazily inside open(), and
emits "face_frame" samples: uint8 arrays held in RAM only, at most `fps` a second. It never writes, logs or records a
frame; the engine hands each one to M-16's process_frame (counts only, spec section 6) and then zeroes it.

On its own it opens the camera whenever it is opened. Live mode never uses it bare: oiris/live.py refuses it while M-16
is off and wraps it so the device opens only after the capture interlock grants the camera (ADR-0063), frames pass
the power plan's FrameGate (ADR-0061), and a revoked grant or a hot plan closes the device.

Options (open cfg): "device" (camera index, default 0) and "fps" (DEFAULT_FPS, MIN_FPS to MAX_FPS). Other keys (such as
the site_key live mode passes to every adapter) are ignored."""
from __future__ import annotations
import time
from typing import Any, Callable

import numpy as np

from ..schema import Health, Sample
from .base import AdapterBase
from .drivers import DriverUnavailable

DEFAULT_FPS = 2.0          # frames a second; enough for "voice and face agree" within a few seconds
MIN_FPS, MAX_FPS = 0.2, 10.0
DEFAULT_DEVICE = 0
READ_WAIT_SHARE = 0.5      # a read waits at most this share of the bus timeout for the next frame time


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def zero_frame(frame: Any) -> None:
    """Overwrite a frame's pixels with zeros when it is a writable numpy array (RAM hygiene; never raises)."""
    if isinstance(frame, np.ndarray) and frame.flags.writeable:
        frame.fill(0)


class FaceCameraAdapter(AdapterBase):
    id, bus, caps, units = "face_camera", "usb", {"face_frame"}, "dn"

    def __init__(self, adapter_id: str | None = None, video_capture: Callable[[int], Any] | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], Any] = time.sleep):
        """video_capture(device): a cv2.VideoCapture stand-in (tests inject a fake); None imports cv2 in open()."""
        super().__init__(adapter_id)
        self._video_capture = video_capture
        self._clock, self._sleep = clock, sleep
        self._cap: Any = None
        self.device, self.fps = DEFAULT_DEVICE, DEFAULT_FPS
        self._next: float | None = None

    def open(self, cfg: dict) -> None:
        super().open(cfg)
        device, fps = self.cfg.get("device", DEFAULT_DEVICE), self.cfg.get("fps", DEFAULT_FPS)
        if isinstance(device, bool) or not isinstance(device, int) or device < 0:
            raise ValueError("face_camera option device must be a camera index (0, 1, ...)")
        if not _number(fps) or not MIN_FPS <= float(fps) <= MAX_FPS:
            raise ValueError(f"face_camera option fps must be a number from {MIN_FPS} to {MAX_FPS}")
        self.device, self.fps = device, float(fps)
        factory = self._video_capture
        if factory is None:
            try:
                import cv2  # type: ignore
            except ImportError as e:
                raise DriverUnavailable("pip install opencv-python-headless") from e
            factory = cv2.VideoCapture
        self.close()
        cap = factory(self.device)
        if not cap.isOpened():
            cap.release()
            raise DriverUnavailable(f"camera {self.device} could not be opened")
        self._cap, self._next = cap, None
        self.set_health(Health.OK, "")

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    def wait_s(self) -> float:
        """Seconds until the next frame may be taken (0 when one is due). Live mode's gate waits this long outside its
        lock and calls read(0), so a pause or a revocation never waits behind the frame pacing."""
        return 0.0 if self._next is None else max(0.0, self._next - self._clock())

    def read(self, timeout_s: float) -> Sample | None:
        """The next frame, at most fps a second: waits up to READ_WAIT_SHARE of timeout_s for its time, else None."""
        cap = self._cap
        if cap is None:
            return None
        now = self._clock()
        if self._next is not None and now < self._next:
            self._sleep(min(self._next - now, max(timeout_s, 0.0) * READ_WAIT_SHARE))
            now = self._clock()
            if now < self._next:
                return None
        ok, raw = cap.read()
        if not ok or raw is None:
            self.set_health(Health.DEGRADED, "frame grab failed")
            return None
        if not isinstance(raw, np.ndarray) or raw.dtype != np.uint8:
            zero_frame(raw)
            self.set_health(Health.DEGRADED, "the camera gave frames in an unexpected format")
            return None
        self._next = now + 1.0 / self.fps
        self.set_health(Health.OK, "")
        return self._sample(raw)   # the array itself: no copy is left behind to zero

    def close(self) -> None:
        cap, self._cap = self._cap, None
        self._running = False
        if cap is not None:
            cap.release()
