"""Registers built-in adapters with the registry. Core resolves adapters by name only."""
from __future__ import annotations

from . import drivers, imu_sim, macos, sim, sim_rig
from .base import ReplayAdapter
from .camera import FaceCameraAdapter


def register(r) -> None:
    for cls in drivers.REAL + macos.MACOS:
        r.register("sensor", cls.id, cls, port_version="1.0")
    r.register("sensor", "replay", ReplayAdapter, port_version="1.0")
    # M-16 frames (ADR-0057 decision 5); live mode gates it on M-16, the capture interlock and the power plan
    r.register("sensor", FaceCameraAdapter.id, FaceCameraAdapter, port_version="1.0")
    r.register("sensor", "sim", sim.make, port_version="1.0")
    r.register("sensor", "imu", imu_sim.make, port_version="1.0")   # simulated until the glasses IMU exists
    r.register("sensor", "sim_rig", sim_rig.make, port_version="1.0")  # simulated 360-degree rig until the cameras exist
