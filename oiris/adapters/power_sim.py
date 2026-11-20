"""Simulated wearable power and heat (ADR-0061), until the glasses or helmet exist. Deterministic: no randomness.

Power model (watts), from the duty plan in force:
    P = BASE_W + CAMERA_W * (camera_max_fps / CAMERA_MAX_FPS) + COMPUTE_W (voice or face model on)
        + IR_W (illuminator) + MIC_W * listen_duty + DISPLAY_W * display_brightness_cap  [+ CHARGE_HEAT_W charging]
Battery: CAPACITY_WH, drained by P (or filled at CHARGE_W while charging).
Heat, first order: dT/dt = (P * R_C_PER_W - (T - ambient_c)) / TAU_S, integrated exactly over each step with P
held constant; skin rise above ambient = SKIN_FRACTION * the device's rise.

Every coefficient is a named placeholder, to be measured on hardware. The simulation proves the policy (levels,
hysteresis, dwell, the closed loop: throttling really cools the device), not any real device's numbers.

Closed loop: run() feeds readings to a Governor and applies the plan it last chose. make() is the C2 sensor
adapter shape ("power" capability) for the later wiring task; it is not registered by name yet."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable

from ..power.duty import NORMAL, DutyPlan, build_plans
from ..power.governor import Governor, PowerReading, PowerSettings
from .base import SimAdapter

# Placeholders, to be measured on hardware.
BASE_W = 0.25            # SoC idle, radios, sensors
CAMERA_W = 0.35          # camera and its pipeline at CAMERA_MAX_FPS
CAMERA_MAX_FPS = 30.0
COMPUTE_W = 0.45         # a voice or face model running
IR_W = 0.15              # IR illuminator
MIC_W = 0.03             # microphone and VAD at full duty
DISPLAY_W = 0.20         # display at full brightness
CAPACITY_WH = 2.0        # battery capacity
CHARGE_W = 2.0           # power into the battery while charging
CHARGE_HEAT_W = 0.6      # charging losses that heat the device
R_C_PER_W = 10.0         # device steady-state rise above ambient per watt
TAU_S = 300.0            # thermal time constant
SKIN_FRACTION = 0.6      # skin rise above ambient as a fraction of the device's rise
PCT_FULL, PCT_EMPTY = 100.0, 0.0
S_PER_H = 3600.0

DT_S = 1.0               # simulation step
READING_EVERY_S = 5.0    # a reading goes to the governor this often
DEFAULT_DURATION_S = 1800.0   # 30 simulated minutes


@dataclass(frozen=True)
class Scene:
    ambient_c: float
    battery_pct: float
    charging: bool


SCENES: dict[str, Scene] = {
    "cool_room": Scene(ambient_c=22.0, battery_pct=100.0, charging=False),     # stays normal
    "hot_day": Scene(ambient_c=35.0, battery_pct=100.0, charging=False),       # reaches hot and settles there
    "low_battery": Scene(ambient_c=22.0, battery_pct=30.0, charging=False),    # passes saver, then critical
    "charging_hot": Scene(ambient_c=32.0, battery_pct=15.0, charging=True),    # charging heat; battery axis cancelled
}


def plan_power_w(plan: DutyPlan, charging: bool = False) -> float:
    fps = min(max(plan.camera_max_fps, 0.0), CAMERA_MAX_FPS)
    p = (BASE_W + CAMERA_W * fps / CAMERA_MAX_FPS + (COMPUTE_W if plan.voice_match or plan.face_match else 0.0)
         + (IR_W if plan.ir_illuminator else 0.0) + MIC_W * plan.listen_duty + DISPLAY_W * plan.display_brightness_cap)
    return p + (CHARGE_HEAT_W if charging else 0.0)


class PowerSim:
    """State of one simulated device: battery, device temperature, scene. advance() moves it by dt under a plan."""

    def __init__(self, scene: str):
        if scene not in SCENES:
            raise ValueError(f"power scene must be one of {list(SCENES)}")
        self.scene_name, self.scene = scene, SCENES[scene]
        self.t = 0.0
        self.battery_pct = self.scene.battery_pct
        self.device_c = self.scene.ambient_c   # cold start
        self.power_w = 0.0

    @property
    def skin_c(self) -> float:
        return self.scene.ambient_c + SKIN_FRACTION * (self.device_c - self.scene.ambient_c)

    def advance(self, dt_s: float, plan: DutyPlan) -> None:
        sc = self.scene
        self.power_w = p = plan_power_w(plan, sc.charging)
        target = sc.ambient_c + p * R_C_PER_W
        self.device_c = target + (self.device_c - target) * math.exp(-dt_s / TAU_S)
        delta_wh = (CHARGE_W if sc.charging else -p) * dt_s / S_PER_H
        self.battery_pct = min(PCT_FULL, max(PCT_EMPTY, self.battery_pct + delta_wh / CAPACITY_WH * PCT_FULL))
        self.t += dt_s

    def reading(self) -> PowerReading:
        return PowerReading(t=self.t, battery_pct=self.battery_pct, charging=self.scene.charging,
                            skin_c=self.skin_c, device_c=self.device_c, thermal_status=None)


def run(scene: str, duration_s: float = DEFAULT_DURATION_S, settings: PowerSettings | None = None,
        dt_s: float = DT_S, reading_every_s: float = READING_EVERY_S) -> list[dict]:
    """Closed loop: every reading_every_s the governor sees a reading and its plan drives the next steps.
    Returns one row per governor update: t, level, reasons, battery_pct, skin_c, device_c, power_w."""
    sim, gov = PowerSim(scene), Governor(settings)
    every = max(1, round(reading_every_s / dt_s))
    plan = gov.settings.plans[NORMAL]
    rows = []
    for k in range(int(round(duration_s / dt_s)) + 1):
        if k % every == 0:
            d = gov.update(sim.reading())
            plan = d.plan
            rows.append({"t": sim.t, "level": d.level, "reasons": list(d.reasons), "battery_pct": sim.battery_pct,
                         "skin_c": sim.skin_c, "device_c": sim.device_c, "power_w": sim.power_w})
        sim.advance(dt_s, plan)
    return rows


def level_changes(rows: list[dict]) -> int:
    return sum(1 for a, b in zip(rows, rows[1:]) if a["level"] != b["level"])


def make(scene: str = "cool_room", seed: int = 0, period_s: float = READING_EVERY_S, adapter_id: str = "power",
         sim_step_s: float = READING_EVERY_S, plan_source: Callable[[], DutyPlan] | None = None) -> SimAdapter:
    """A "power" sensor adapter over PowerSim. Each read advances the simulation by sim_step_s under the plan
    plan_source() returns (default: the normal plan; the wiring task passes the governor's current plan).
    period_s: real seconds each read sleeps (0 = as fast as read). seed is accepted for the adapter
    interface; the model has no randomness."""
    sim = PowerSim(scene)
    normal = build_plans(None)[NORMAL]
    source = plan_source or (lambda: normal)

    def produce(_t, _rng) -> dict:
        sim.advance(sim_step_s, source())
        r = sim.reading()
        return {"scene": scene, "t_s": r.t, "battery_pct": r.battery_pct, "charging": r.charging,
                "skin_c": r.skin_c, "device_c": r.device_c, "thermal_status": r.thermal_status,
                "power_w": sim.power_w}

    return SimAdapter(adapter_id, {"power"}, produce, period_s=period_s, seed=seed, units="W")


__all__ = ["SCENES", "Scene", "PowerSim", "plan_power_w", "run", "level_changes", "make"]
