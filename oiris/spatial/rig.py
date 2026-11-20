"""The sensor rig: the spatial "ports" future cameras, microphone arrays and IMUs plug into (ADR-0060).

Config section "rig" (validated by oiris.config.validate_rig, which this module reuses):

    {"mounts": [{"id": "rear", "kind": "camera", "position_m": [-0.1, 0, 0.05], "yaw_deg": 180, "pitch_deg": 0,
                 "roll_deg": 0, "hfov_deg": 100, "vfov_deg": 75, "image_size": [640, 480]}, ...]}

kind is camera | thermal | nir | mic_array | imu. Imaging kinds take intrinsics {fx, fy, cx, cy} or
hfov_deg/vfov_deg, plus image_size; a mic_array takes mic_positions_m in its own mount frame. An absent mount
is no coverage, never an error; an empty config is the default rig, one forward camera "front"."""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

from ..config import (DEFAULT_HFOV_DEG, DEFAULT_IMAGE_SIZE, DEFAULT_RIG, DEFAULT_VFOV_DEG, IMAGING_KINDS, ConfigError,
                      validate_rig)
from .detections import SpatialDetection
from .frames import (Intrinsics, az_el_to_vec, fov_intrinsics, mount_ray_to_pixel, mount_to_head, pixel_to_mount_ray,
                     vec_to_az_el)


@dataclass(frozen=True)
class Mount:
    id: str
    kind: str
    position_m: tuple = (0.0, 0.0, 0.0)
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    intrinsics: Intrinsics | None = None
    mic_positions_m: tuple = ()
    ir_illuminator: bool = False
    ir_eye_safety_approved: bool = False
    rotation: np.ndarray = field(init=False, compare=False, repr=False)   # mount-to-head, set from yaw/pitch/roll

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", mount_to_head(self.yaw_deg, self.pitch_deg, self.roll_deg))

    @property
    def imaging(self) -> bool:
        return self.kind in IMAGING_KINDS

    @classmethod
    def from_dict(cls, m: dict) -> "Mount":
        k = None
        if m["kind"] in IMAGING_KINDS:
            w, h = m.get("image_size", DEFAULT_IMAGE_SIZE)
            if "intrinsics" in m:
                i = m["intrinsics"]
                k = Intrinsics(float(i["fx"]), float(i["fy"]), float(i["cx"]), float(i["cy"]), w, h)
            else:
                k = fov_intrinsics(w, h, m.get("hfov_deg", DEFAULT_HFOV_DEG), m.get("vfov_deg", DEFAULT_VFOV_DEG))
        return cls(m["id"], m["kind"], tuple(float(x) for x in m.get("position_m", (0.0, 0.0, 0.0))),
                   float(m.get("yaw_deg", 0.0)), float(m.get("pitch_deg", 0.0)), float(m.get("roll_deg", 0.0)), k,
                   tuple(tuple(float(x) for x in p) for p in m.get("mic_positions_m", ())),
                   bool(m.get("ir_illuminator", False)), bool(m.get("ir_eye_safety_approved", False)))

    def mount_vec_to_head(self, v) -> np.ndarray:
        return self.rotation @ np.asarray(v, dtype=np.float64)

    def head_vec_to_mount(self, v) -> np.ndarray:
        return self.rotation.T @ np.asarray(v, dtype=np.float64)

    def head_to_pixel(self, az_deg: float, el_deg: float) -> tuple[float, float] | None:
        """Where a head-frame bearing lands on this mount's image, or None when outside its field of view."""
        if self.intrinsics is None:
            return None
        return mount_ray_to_pixel(self.head_vec_to_mount(az_el_to_vec(az_deg, el_deg)), self.intrinsics)


class Rig:
    """The validated set of mounts. Rig.to_head resolves any SpatialDetection to a head-frame (az, el)."""

    def __init__(self, mounts: list[Mount] | tuple[Mount, ...]) -> None:
        self.mounts = {m.id: m for m in mounts}
        if len(self.mounts) != len(mounts):
            raise ConfigError("rig mounts must have unique ids")

    @classmethod
    def from_config(cls, d: dict | None) -> "Rig":
        d = d or DEFAULT_RIG
        errs = validate_rig(d)
        if errs:
            raise ConfigError("; ".join(errs))
        return cls([Mount.from_dict(m) for m in d.get("mounts", DEFAULT_RIG["mounts"])])

    def get(self, mount_id: str) -> Mount | None:
        return self.mounts.get(mount_id)

    def to_head(self, det: SpatialDetection) -> tuple[float, float]:
        """Head-frame (az, el) of a detection. Raises ValueError when its mount is absent or cannot place it."""
        if det.head_bearing is not None:
            az, el = det.head_bearing
            return vec_to_az_el(az_el_to_vec(az, el))
        m = self.mounts.get(det.mount_id)
        if m is None:
            raise ValueError(f"no mount {det.mount_id!r} in the rig")
        if det.mount_bearing is not None:
            ray = az_el_to_vec(*det.mount_bearing)
        else:
            if m.intrinsics is None or det.pixel is None:
                raise ValueError(f"mount {m.id!r} ({m.kind}) has no intrinsics to place a pixel")
            u, v = det.pixel
            if not (0 <= u < m.intrinsics.width and 0 <= v < m.intrinsics.height):
                raise ValueError(f"pixel {det.pixel} is off mount {m.id!r}'s image")
            ray = pixel_to_mount_ray(u, v, m.intrinsics)
        return vec_to_az_el(m.mount_vec_to_head(ray))


__all__ = ["Mount", "Rig"]
