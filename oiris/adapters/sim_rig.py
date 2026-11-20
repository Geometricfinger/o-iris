"""Simulated sensor rig (sensor adapter "sim_rig"), until the cameras exist (ADR-0059, ADR-0060).

A scripted scene places objects at head-frame bearings, from formulas: each has a kind, a start bearing, a
turn rate and an active time span. One adapter per mount reports what that mount would see, and only what
lies inside its field of view: imaging mounts (camera, thermal, nir) report pixels through their intrinsics,
a microphone array reports bearings in its own frame (it hears all round). Deterministic, no noise.

    data = {"mount_id": str, "spatial_detections": [SpatialDetection, ...]}

The default simulated rig is the owner's planned 360-degree layout: front, rear, left and right cameras
(hfov 100, so neighbours overlap by 10 degrees at +-45 and +-135), a forward thermal camera and a
four-microphone array. Real camera adapters plug into the same capability ("spatial") and sample shape."""
from __future__ import annotations
from dataclasses import dataclass

from ..spatial.detections import DEFAULT_FAMILY, DETECTIONS_KEY, SPATIAL_CAP, SpatialDetection
from ..spatial.frames import vec_to_az_el, wrap_az, az_el_to_vec
from ..spatial.rig import Mount, Rig
from .base import SimAdapter

NS_PER_S = 1_000_000_000
SIM_CAMERA_HFOV_DEG = 100.0
SIM_CAMERA_VFOV_DEG = 75.0
SIM_CAMERA_SIZE = [640, 480]
SIM_THERMAL_HFOV_DEG, SIM_THERMAL_VFOV_DEG = 56.0, 42.0
SIM_THERMAL_SIZE = [32, 24]      # the size of the sim thermal frames (oiris/adapters/sim.py)
SIM_MIC_SPACING_M = 0.07         # a square of four microphones, each 7 cm from the centre

# Which detection kinds each mount kind can report.
SEES = {"camera": ("glint", "ir_emitter", "object"), "nir": ("glint", "ir_emitter"), "thermal": ("thermal",),
        "mic_array": ("sound",), "imu": ()}


def _camera(mid: str, yaw: float) -> dict:
    return {"id": mid, "kind": "camera", "yaw_deg": yaw, "hfov_deg": SIM_CAMERA_HFOV_DEG, "vfov_deg": SIM_CAMERA_VFOV_DEG,
            "image_size": SIM_CAMERA_SIZE}


_S = SIM_MIC_SPACING_M
SIM_RIG: dict = {"mounts": [
    _camera("front", 0.0), _camera("left", 90.0), _camera("rear", 180.0), _camera("right", -90.0),
    {"id": "front_thermal", "kind": "thermal", "hfov_deg": SIM_THERMAL_HFOV_DEG, "vfov_deg": SIM_THERMAL_VFOV_DEG,
     "image_size": SIM_THERMAL_SIZE},
    {"id": "mics", "kind": "mic_array", "mic_positions_m": [[_S, 0.0, 0.0], [0.0, _S, 0.0], [-_S, 0.0, 0.0], [0.0, -_S, 0.0]]},
]}


@dataclass(frozen=True)
class SceneObject:
    """An object at head-frame bearing az(t) = az_deg + az_rate_deg_s * t, el_deg, active for t0_s <= t <= t1_s."""
    kind: str
    az_deg: float
    el_deg: float = 0.0
    confidence: float = 0.8
    az_rate_deg_s: float = 0.0
    t0_s: float = 0.0
    t1_s: float | None = None

    def bearing(self, t: float) -> tuple[float, float] | None:
        if t < self.t0_s or (self.t1_s is not None and t > self.t1_s):
            return None
        return wrap_az(self.az_deg + self.az_rate_deg_s * t), self.el_deg


ORBIT_RATE_DEG_S = 30.0
SCENES: dict[str, tuple[SceneObject, ...]] = {
    "quiet": (),
    "behind": (SceneObject("glint", 180.0),),
    "sides": (SceneObject("glint", 90.0), SceneObject("ir_emitter", -90.0, confidence=0.6)),
    "between": (SceneObject("glint", 45.0, confidence=0.7),),        # inside both the front and left fields of view
    "heat_and_sound_ahead": (SceneObject("thermal", 10.0, confidence=0.6), SceneObject("sound", 15.0, confidence=0.5)),
    "orbit": (SceneObject("ir_emitter", 0.0, az_rate_deg_s=ORBIT_RATE_DEG_S),),
    "above": (SceneObject("sound", 0.0, el_deg=70.0, confidence=0.5),),
}


def observe(mount: Mount, scene: tuple[SceneObject, ...], t: float) -> list[SpatialDetection]:
    """What one mount reports at scene time t."""
    out = []
    t_ns = int(round(t * NS_PER_S))
    for o in scene:
        b = o.bearing(t)
        if b is None or o.kind not in SEES.get(mount.kind, ()):
            continue
        fam = DEFAULT_FAMILY[o.kind]
        if mount.imaging:
            px = mount.head_to_pixel(*b)
            if px is not None:
                out.append(SpatialDetection(o.kind, fam, mount.id, t_ns, o.confidence, pixel=px))
        elif mount.kind == "mic_array":
            mb = vec_to_az_el(mount.head_vec_to_mount(az_el_to_vec(*b)))
            out.append(SpatialDetection(o.kind, fam, mount.id, t_ns, o.confidence, mount_bearing=mb))
    return out


def make(scenario: str = "behind", rig: Rig | dict | None = None, period_s: float = 0.0) -> list[SimAdapter]:
    """One adapter "rig_<mount id>" per mount that can report anything. rig: a Rig or a "rig" config section
    (default SIM_RIG). Each read advances the scene clock by period_s, or 0.1 s when period_s is 0."""
    if scenario not in SCENES:
        raise ValueError(f"sim_rig scenario must be one of {sorted(SCENES)}")
    r = rig if isinstance(rig, Rig) else Rig.from_config(rig if rig is not None else SIM_RIG)
    scene = SCENES[scenario]
    adapters = []
    for m in r.mounts.values():
        if not SEES.get(m.kind):
            continue

        def produce(t, _rng, m=m) -> dict:
            return {"mount_id": m.id, DETECTIONS_KEY: observe(m, scene, t)}
        adapters.append(SimAdapter(f"rig_{m.id}", {SPATIAL_CAP}, produce, period_s=period_s))
    return adapters


__all__ = ["make", "observe", "SceneObject", "SCENES", "SIM_RIG", "SEES"]
