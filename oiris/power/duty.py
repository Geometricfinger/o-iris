"""Duty plans, one per power level, and the two pure duty-cycle helpers (ADR-0061).

A plan is a set of caps. It never turns on something the config or the owner has off: `ir_illuminator: true`
means "may be on" (the rig's eye-safety rule still applies), `camera_max_fps: 15` means "at most 15".

Order rule: each level is at least as restrictive as the one before it (normal, saver, hot, critical) on every
restricting field: lower or equal camera_max_fps, higher or equal frame_stride, lower or equal listen_duty,
longer or equal radio_scan_interval_s, lower or equal display_brightness_cap, and a feature that is off stays
off. listen_period_s only sets how the listening duty is spread over time, so it is not ordered.

critical keeps the guard's minimum: radio scanning stays on (at a slower interval), while camera processing
(camera_max_fps 0, face_match off), the voice model (voice_match off) and the IR illuminator are off.

Note for the wiring task: when ListenDuty turns listening off, an utterance already being heard is finished,
not cut. The listener checks the duty only between utterances; a plan change never truncates speech."""
from __future__ import annotations
import math
from dataclasses import asdict, dataclass

NORMAL, SAVER, HOT, CRITICAL = "normal", "saver", "hot", "critical"
LEVELS = (NORMAL, SAVER, HOT, CRITICAL)   # least to most restrictive

PLAN_KEYS = ("camera_max_fps", "frame_stride", "listen_duty", "listen_period_s", "radio_scan_interval_s",
             "ir_illuminator", "voice_match", "face_match", "display_brightness_cap")
# key -> (low, high). camera_max_fps 0 means camera processing off.
PLAN_RANGES: dict = {
    "camera_max_fps": (0.0, 60.0),
    "frame_stride": (1, 100),
    "listen_duty": (0.0, 1.0),
    "listen_period_s": (1.0, 600.0),
    "radio_scan_interval_s": (1.0, 600.0),   # never 0: radio scanning is the guard's minimum and never stops
    "display_brightness_cap": (0.0, 1.0),
}
PLAN_INTS = {"frame_stride"}
PLAN_BOOLS = ("ir_illuminator", "voice_match", "face_match")
LOWER_IS_TIGHTER = ("camera_max_fps", "listen_duty", "display_brightness_cap")
HIGHER_IS_TIGHTER = ("frame_stride", "radio_scan_interval_s")
CAMERA_OFF_FPS = 0.0
DUTY_OFF, DUTY_FULL = 0.0, 1.0
# Off in critical: camera processing (fps 0, face matching), the voice model and the IR illuminator.
CRITICAL_OFF = ("ir_illuminator", "voice_match", "face_match")

# Named, provisional defaults. Each step down cuts the largest power users first (camera rate and the IR
# illuminator, then the face and voice models), keeping radio scanning to the end. To be tuned on the device.
DEFAULT_PLANS: dict = {
    NORMAL: {"camera_max_fps": 30.0, "frame_stride": 1, "listen_duty": 1.0, "listen_period_s": 10.0,
             "radio_scan_interval_s": 15.0, "ir_illuminator": True, "voice_match": True, "face_match": True,
             "display_brightness_cap": 1.0},
    SAVER: {"camera_max_fps": 15.0, "frame_stride": 2, "listen_duty": 0.5, "listen_period_s": 10.0,
            "radio_scan_interval_s": 30.0, "ir_illuminator": True, "voice_match": True, "face_match": True,
            "display_brightness_cap": 0.6},
    HOT: {"camera_max_fps": 10.0, "frame_stride": 3, "listen_duty": 0.25, "listen_period_s": 10.0,
          "radio_scan_interval_s": 30.0, "ir_illuminator": False, "voice_match": True, "face_match": False,
          "display_brightness_cap": 0.5},
    CRITICAL: {"camera_max_fps": 0.0, "frame_stride": 3, "listen_duty": 0.0, "listen_period_s": 10.0,
               "radio_scan_interval_s": 60.0, "ir_illuminator": False, "voice_match": False, "face_match": False,
               "display_brightness_cap": 0.3},
}


@dataclass(frozen=True)
class DutyPlan:
    camera_max_fps: float
    frame_stride: int
    listen_duty: float
    listen_period_s: float
    radio_scan_interval_s: float
    ir_illuminator: bool
    voice_match: bool
    face_match: bool
    display_brightness_cap: float

    def to_dict(self) -> dict:
        return asdict(self)


def _finite_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def merged_plans(d: object) -> dict:
    """The config's power.plans laid over the defaults, per level and per field."""
    src: dict = d if isinstance(d, dict) else {}
    out = {}
    for lv in LEVELS:
        over = src.get(lv)
        out[lv] = {**DEFAULT_PLANS[lv], **(over if isinstance(over, dict) else {})}
    return out


def _validate_plan(level: str, p: dict) -> list[str]:
    where = f"power.plans.{level}"
    errs = []
    unknown = set(p) - set(PLAN_KEYS)
    if unknown:
        errs.append(f"{where}: unknown keys {sorted(unknown)}")
    for k, (lo, hi) in PLAN_RANGES.items():
        x = p[k]
        if not _finite_number(x) or (k in PLAN_INTS and int(x) != x) or not lo <= x <= hi:
            errs.append(f"{where}.{k} must be in [{lo}, {hi}]")
    for k in PLAN_BOOLS:
        if not isinstance(p[k], bool):
            errs.append(f"{where}.{k} must be true or false")
    return errs


def validate_plans(d: object) -> list[str]:
    """Ranges per field, the order rule between levels, and the critical minimum. Error strings, never raises."""
    if not isinstance(d, dict):
        return ["power.plans must be an object"]
    errs = []
    unknown = set(d) - set(LEVELS)
    if unknown:
        errs.append(f"power.plans: unknown levels {sorted(unknown)}")
    for lv in LEVELS:
        if lv in d and not isinstance(d[lv], dict):
            errs.append(f"power.plans.{lv} must be an object")
    if errs:
        return errs
    plans = merged_plans(d)
    for lv in LEVELS:
        errs += _validate_plan(lv, plans[lv])
    if errs:
        return errs
    for prev, cur in zip(LEVELS, LEVELS[1:]):
        a, b = plans[prev], plans[cur]
        for k in LOWER_IS_TIGHTER:
            if b[k] > a[k]:
                errs.append(f"power.plans.{cur}.{k} must be <= {prev}'s ({a[k]}): each level at least as restrictive")
        for k in HIGHER_IS_TIGHTER:
            if b[k] < a[k]:
                errs.append(f"power.plans.{cur}.{k} must be >= {prev}'s ({a[k]}): each level at least as restrictive")
        for k in PLAN_BOOLS:
            if b[k] and not a[k]:
                errs.append(f"power.plans.{cur}.{k} is off in {prev}, so it must stay off")
    crit = plans[CRITICAL]
    if crit["camera_max_fps"] != CAMERA_OFF_FPS:
        errs.append("power.plans.critical.camera_max_fps must be 0 (camera processing off)")
    for k in CRITICAL_OFF:
        if crit[k]:
            errs.append(f"power.plans.critical.{k} must be false")
    return errs


def build_plans(d: object) -> dict[str, DutyPlan]:
    """DutyPlan per level from an already validated power.plans (or None for the defaults)."""
    return {lv: DutyPlan(**{k: p[k] for k in PLAN_KEYS}) for lv, p in merged_plans(d).items()}


class FrameGate:
    """Which camera frames are processed: a rate cap, then every `stride`-th admitted frame.

    admit(t) is True for a frame at monotonic time t (seconds) when at least 1/max_fps has passed since the
    last frame that passed the rate cap, and that frame is the stride-th since the last processed one.
    max_fps 0 admits nothing. Time running backwards restarts the gate. Pure: no clock of its own."""

    TOLERANCE_S = 1e-9   # float slack, so frames exactly 1/max_fps apart are not refused by rounding

    def __init__(self, max_fps: float, stride: int = 1):
        if not _finite_number(max_fps) or max_fps < 0:
            raise ValueError("max_fps must be a number >= 0")
        if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
            raise ValueError("stride must be a whole number >= 1")
        self.max_fps, self.stride = float(max_fps), stride
        self.min_interval_s = 1 / self.max_fps if self.max_fps > CAMERA_OFF_FPS else math.inf
        self._last: float | None = None
        self._n = 0

    def admit(self, t: float) -> bool:
        if self.max_fps <= CAMERA_OFF_FPS:
            return False
        if self._last is not None and t < self._last:
            self._last, self._n = None, 0
        if self._last is not None and t - self._last < self.min_interval_s - self.TOLERANCE_S:
            return False
        self._last = t
        self._n += 1
        return (self._n - 1) % self.stride == 0


class ListenDuty:
    """Listening on for the first duty * period_s of every period_s (aligned to t = 0). duty 1 is always on,
    duty 0 always off. Pure: active(t) depends only on t. The wiring task finishes an utterance already being
    heard before honouring an off window (see the module note)."""

    def __init__(self, duty: float, period_s: float):
        if not _finite_number(duty) or not DUTY_OFF <= duty <= DUTY_FULL:
            raise ValueError("duty must be in [0, 1]")
        if not _finite_number(period_s) or period_s <= 0:
            raise ValueError("period_s must be a positive number")
        self.duty, self.period_s = float(duty), float(period_s)

    def active(self, t: float) -> bool:
        if self.duty >= DUTY_FULL:
            return True
        if self.duty <= DUTY_OFF:
            return False
        return t % self.period_s < self.duty * self.period_s


__all__ = ["NORMAL", "SAVER", "HOT", "CRITICAL", "LEVELS", "PLAN_KEYS", "PLAN_RANGES", "DEFAULT_PLANS", "DutyPlan",
           "validate_plans", "build_plans", "merged_plans", "FrameGate", "ListenDuty"]
