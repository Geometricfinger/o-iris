"""SpatialDetection: one thing one sensor located, from any source (camera, thermal, NIR, a microphone array).

A detection carries exactly one location: a pixel (u, v) on its mount's image, a bearing (az, el) in its
mount's frame, or a bearing already in the head frame. oiris.spatial.rig.Rig.to_head resolves all three.
Pixels follow the image convention of oiris/features: u is the column (feature "x"), v the row ("y").

Sensor adapters put detections in a Sample's data as {"spatial_detections": [SpatialDetection | dict, ...]}
and declare the capability SPATIAL_CAP; the engine merges them (oiris/spatial/merge.py)."""
from __future__ import annotations
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..features.optical import _components, glint_score, nir_emitter_score
from ..features.thermal import skin_range_score

if TYPE_CHECKING:
    from .rig import Mount

SPATIAL_CAP = "spatial"                 # sensor capability of adapters whose samples carry detections
DETECTIONS_KEY = "spatial_detections"   # key in Sample.data
KINDS = ("glint", "ir_emitter", "thermal", "object", "sound")
# Family of each kind when the source does not say (signal families: docs/ARCHITECTURE.md).
DEFAULT_FAMILY = {"glint": "optical", "ir_emitter": "optical", "thermal": "thermal", "object": "optical", "sound": "voice"}
HOT_DELTA_C = 5.0      # a pixel this much above the frame's median is hot (as in oiris.features.thermal.roi_stats)
NIR_SATURATED = 250    # an LED-off pixel at or above this is saturated (as in nir_emitter_score's default)
NIR_MIN_PX = 2         # a saturated spot needs at least this many pixels (nir_emitter_score's default)


def _pair(v: object, name: str) -> tuple[float, float] | None:
    if v is None:
        return None
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        raise ValueError(f"{name} must be a pair of numbers")
    a, b = (float(x) for x in v)
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError(f"{name} must be finite")
    return a, b


@dataclass(frozen=True)
class SpatialDetection:
    kind: str
    family: str
    mount_id: str
    t_mono_ns: int
    confidence: float
    pixel: tuple[float, float] | None = None          # (u, v) on the mount's image
    mount_bearing: tuple[float, float] | None = None  # (az, el) degrees in the mount frame
    head_bearing: tuple[float, float] | None = None   # (az, el) degrees in the head frame
    range_m: float | None = None
    size: float | None = None                         # source-defined extent (pixels for blobs)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"detection kind must be one of {list(KINDS)}")
        if not isinstance(self.family, str) or not self.family or not isinstance(self.mount_id, str) or not self.mount_id:
            raise ValueError("detection family and mount_id must be non-empty strings")
        if isinstance(self.t_mono_ns, bool) or not isinstance(self.t_mono_ns, int) or self.t_mono_ns < 0:
            raise ValueError("detection t_mono_ns must be a non-negative integer")
        c = float(self.confidence)
        if not (math.isfinite(c) and 0.0 <= c <= 1.0):
            raise ValueError("detection confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", c)
        locs = {k: _pair(getattr(self, k), k) for k in ("pixel", "mount_bearing", "head_bearing")}
        if sum(v is not None for v in locs.values()) != 1:
            raise ValueError("a detection needs exactly one of pixel, mount_bearing, head_bearing")
        for k, v in locs.items():
            object.__setattr__(self, k, v)
        for k in ("range_m", "size"):
            x = getattr(self, k)
            if x is not None and not (math.isfinite(float(x)) and float(x) >= 0):
                raise ValueError(f"detection {k} must be a non-negative number")

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "SpatialDetection":
        if not isinstance(d, dict):
            raise ValueError("a detection must be an object")
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"detection: unknown keys {sorted(unknown)}")
        return cls(**d)


def glint_detections(cands: list[dict], mount: "Mount", t_mono_ns: int) -> list[SpatialDetection]:
    """A1 lens-glint candidates (oiris.features.optical.glint_candidates) as detections on an imaging mount.
    Each candidate's confidence is the A1 glint score of that candidate alone."""
    return [SpatialDetection("glint", DEFAULT_FAMILY["glint"], mount.id, t_mono_ns, glint_score([c])[0],
                             pixel=(c["x"], c["y"]), size=float(c["area"])) for c in cands]


def emitter_detections(frame_off: np.ndarray, mount: "Mount", t_mono_ns: int) -> list[SpatialDetection]:
    """A2 active NIR emitters: each saturated LED-off spot, scored by nir_emitter_score on its own pixels."""
    out = []
    for pix in _components(frame_off >= NIR_SATURATED):
        if len(pix) < NIR_MIN_PX:
            continue
        ys, xs = zip(*pix)
        vals = np.array([frame_off[y, x] for y, x in pix])
        out.append(SpatialDetection("ir_emitter", DEFAULT_FAMILY["ir_emitter"], mount.id, t_mono_ns,
                                    nir_emitter_score(vals, NIR_SATURATED, NIR_MIN_PX)[0],
                                    pixel=(float(np.mean(xs)), float(np.mean(ys))), size=float(len(pix))))
    return out


def hot_region_detections(frame_c: np.ndarray, mount: "Mount", t_mono_ns: int,
                          ambient_c: float | None = None) -> list[SpatialDetection]:
    """A3 hot regions (more than HOT_DELTA_C above ambient) at their centroids. Confidence is the A3
    skin-range score of the region: how far its hottest pixels lie outside a typical exposed-skin band."""
    amb = float(np.median(frame_c)) if ambient_c is None else ambient_c
    out = []
    for pix in _components(frame_c > amb + HOT_DELTA_C):
        ys, xs = zip(*pix)
        vals = np.array([frame_c[y, x] for y, x in pix])
        out.append(SpatialDetection("thermal", DEFAULT_FAMILY["thermal"], mount.id, t_mono_ns, skin_range_score(vals)[0],
                                    pixel=(float(np.mean(xs)), float(np.mean(ys))), size=float(len(pix))))
    return out


__all__ = ["SpatialDetection", "SPATIAL_CAP", "DETECTIONS_KEY", "KINDS", "DEFAULT_FAMILY", "glint_detections",
           "emitter_detections", "hot_region_detections"]
