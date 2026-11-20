"""Simulated head-worn IMU (sensor adapter "imu"), until the glasses exist. Deterministic formulas plus seeded
noise; the same C2 adapter interface and sample shape a real glasses IMU adapter will use:

    data = {"profile": str, "rate_hz": float, "t_s": [seconds], "accel": [[ax, ay, az] m/s^2], "speed_mps": x | None}

Profiles: "still" (worn, sitting: sensor noise and slight head sway), "walking" (about 1.8 steps a second,
vertical bounce and sideways sway) and "driving" (low road vibration; carries a simulated speed, standing in for
the GPS speed input a real rig would add: an IMU alone cannot tell a smooth car from sitting still)."""
from __future__ import annotations
import math

from .base import SimAdapter

G = 9.81
PROFILES = ("still", "walking", "driving")
RATE_HZ = 50.0                # accelerometer samples per second
BATCH = 10                    # samples per read (0.2 s at RATE_HZ)
NOISE = 0.03                  # m/s^2, sensor noise in every profile
STILL_SWAY = 0.08             # m/s^2, slow head sway while sitting
STILL_SWAY_HZ = 0.3
STEP_HZ = 1.8                 # walking cadence
STEP_BOUNCE = 2.5             # m/s^2, vertical bounce at the step frequency
STEP_SWAY = 0.8               # m/s^2, sideways sway at half the step frequency
ROAD_VIB = 0.15               # m/s^2, road and engine vibration
ROAD_VIB_HZ = 11.0
ROAD_VIB_Z_RATIO = 1.3        # the vertical vibration runs at a different frequency from the sideways one
DRIVE_SPEED_MPS = 13.9        # 50 km/h


def accel(profile: str, t: float, rng) -> list[float]:
    """One accelerometer reading [x, y, z] at time t (z up, gravity included)."""
    n = [rng.gauss(0, NOISE) for _ in range(3)]
    if profile == "walking":
        return [STEP_SWAY * math.sin(math.pi * STEP_HZ * t) + n[0], n[1],
                G + STEP_BOUNCE * math.sin(2 * math.pi * STEP_HZ * t) + n[2]]
    if profile == "driving":
        return [n[0], ROAD_VIB * math.sin(2 * math.pi * ROAD_VIB_HZ * t) + n[1],
                G + ROAD_VIB * math.sin(2 * math.pi * ROAD_VIB_HZ * ROAD_VIB_Z_RATIO * t) + n[2]]
    return [STILL_SWAY * math.sin(2 * math.pi * STILL_SWAY_HZ * t) + n[0], n[1], G + n[2]]


def make(profile: str = "still", seed: int = 0, period_s: float = BATCH / RATE_HZ, adapter_id: str = "imu",
         rate_hz: float = RATE_HZ, batch: int = BATCH) -> SimAdapter:
    """period_s: real seconds each read sleeps (0 = as fast as read). The sample clock always advances by
    batch / rate_hz per read, so a trace is the same whatever period_s is."""
    if profile not in PROFILES:
        raise ValueError(f"imu profile must be one of {list(PROFILES)}")
    if not rate_hz > 0 or batch < 1:
        raise ValueError("imu rate_hz must be positive and batch at least 1")
    state = {"k": 0}

    def produce(_t, rng) -> dict:
        k0 = state["k"]
        state["k"] = k0 + batch
        ts = [(k0 + i) / rate_hz for i in range(batch)]
        return {"profile": profile, "rate_hz": rate_hz, "t_s": ts, "accel": [accel(profile, t, rng) for t in ts],
                "speed_mps": DRIVE_SPEED_MPS if profile == "driving" else None}

    return SimAdapter(adapter_id, {"imu"}, produce, period_s=period_s, seed=seed, units="m/s^2")


__all__ = ["make", "accel", "PROFILES", "RATE_HZ", "BATCH"]
