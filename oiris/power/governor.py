"""Power governor (ADR-0061): battery and heat readings -> a level and that level's duty plan. Pure and
deterministic: no clock, no I/O; the caller passes each reading and the current time.

Levels, least to most restrictive: normal, saver, hot, critical. Two axes; the more restrictive one wins.
- Battery axis -> saver or critical. Charging cancels it (heat still counts: charging makes heat).
- Heat axis -> hot or critical, from skin temperature, device temperature or the Android thermal status.

Every threshold has an enter and an exit value (hysteresis). Tightening is immediate. Relaxing goes one level
at a time and only when the exit thresholds are met and the governor has been at its level for min_dwell_s.
A reading older than stale_after_s relaxes nothing: the level holds and the reasons say "stale" (a stale reading
may still tighten, the safe direction). A value that is None is unknown: never guessed, and it keeps that
signal's last state, so a sensor that stops reporting cannot relax the level.

Decisions carry fixed reason codes and numbers only."""
from __future__ import annotations
import math
from dataclasses import dataclass, field

from .duty import CRITICAL, HOT, LEVELS, NORMAL, SAVER, DutyPlan, build_plans, validate_plans

RANK = {lv: i for i, lv in enumerate(LEVELS)}
REASONS = ("battery_low", "battery_critical", "charging", "skin_hot", "skin_critical", "device_hot",
           "device_critical", "thermal_status_hot", "thermal_status_critical", "stale")

# Android PowerManager.getCurrentThermalStatus (API 29+).
THERMAL_STATUS_NONE, THERMAL_STATUS_LIGHT, THERMAL_STATUS_MODERATE = 0, 1, 2
THERMAL_STATUS_SEVERE, THERMAL_STATUS_CRITICAL, THERMAL_STATUS_EMERGENCY, THERMAL_STATUS_SHUTDOWN = 3, 4, 5, 6

BATTERY_PCT_RANGE = (0.0, 100.0)
TEMP_PLAUSIBLE_C = (-40.0, 150.0)   # a temperature outside this is a sensor fault: treated as unknown

# Named, provisional defaults (config "power"). To be confirmed on the chosen glasses or helmet.
POWER_DEFAULTS: dict = {
    "battery_saver_enter_pct": 20.0, "battery_saver_exit_pct": 25.0,
    "battery_critical_enter_pct": 7.0, "battery_critical_exit_pct": 12.0,
    # About 43 C is the usual ceiling for long skin contact with worn devices (IEC 62368-1 touch-temperature
    # limits). Confirm against the standard and the chosen glasses' own thermal specification before shipping.
    "skin_hot_enter_c": 41.0, "skin_hot_exit_c": 39.5,
    "skin_critical_enter_c": 43.0, "skin_critical_exit_c": 41.5,
    # Device (SoC or battery) temperature: placeholders until the chosen device's limits are known.
    "device_hot_enter_c": 75.0, "device_hot_exit_c": 70.0,
    "device_critical_enter_c": 85.0, "device_critical_exit_c": 80.0,
    "min_dwell_s": 30.0,      # at a level at least this long before relaxing one step
    "stale_after_s": 60.0,    # a reading older than this relaxes nothing
}
POWER_KEYS = set(POWER_DEFAULTS) | {"plans"}
PCT_KEYS = ("battery_saver_enter_pct", "battery_saver_exit_pct", "battery_critical_enter_pct", "battery_critical_exit_pct")
SKIN_RANGE_C = (30.0, 60.0)
DEVICE_RANGE_C = (30.0, 130.0)
DWELL_RANGE_S = (0.0, 3600.0)
STALE_RANGE_S = (1.0, 3600.0)
POWER_RANGES: dict = {**{k: BATTERY_PCT_RANGE for k in PCT_KEYS},
                      **{k: SKIN_RANGE_C for k in POWER_DEFAULTS if k.startswith("skin_")},
                      **{k: DEVICE_RANGE_C for k in POWER_DEFAULTS if k.startswith("device_")},
                      "min_dwell_s": DWELL_RANGE_S, "stale_after_s": STALE_RANGE_S}
HEAT_SOURCES = ("skin", "device")


def _finite_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_power(d: object) -> list[str]:
    """Validate the "power" section (thresholds, hysteresis direction, plans). Error strings, never raises."""
    if not isinstance(d, dict):
        return ["power must be an object"]
    errs = []
    unknown = set(d) - POWER_KEYS
    if unknown:
        errs.append(f"power: unknown keys {sorted(unknown)}")
    v = {**POWER_DEFAULTS, **{k: x for k, x in d.items() if k in POWER_DEFAULTS}}
    for k, (lo, hi) in POWER_RANGES.items():
        if not _finite_number(v[k]) or not lo <= v[k] <= hi:
            errs.append(f"power.{k} must be in [{lo}, {hi}]")
    if not errs:
        for tier in ("saver", "critical"):   # battery: low is bad, so exit sits above enter
            if not v[f"battery_{tier}_exit_pct"] > v[f"battery_{tier}_enter_pct"]:
                errs.append(f"power.battery_{tier}_exit_pct must be above battery_{tier}_enter_pct (hysteresis)")
        if not v["battery_critical_enter_pct"] < v["battery_saver_enter_pct"]:
            errs.append("power.battery_critical_enter_pct must be below battery_saver_enter_pct")
        for src in HEAT_SOURCES:             # heat: high is bad, so exit sits below enter
            for tier in ("hot", "critical"):
                if not v[f"{src}_{tier}_exit_c"] < v[f"{src}_{tier}_enter_c"]:
                    errs.append(f"power.{src}_{tier}_exit_c must be below {src}_{tier}_enter_c (hysteresis)")
            if not v[f"{src}_hot_enter_c"] < v[f"{src}_critical_enter_c"]:
                errs.append(f"power.{src}_hot_enter_c must be below {src}_critical_enter_c")
    errs += validate_plans(d.get("plans", {}))
    return errs


@dataclass(frozen=True)
class PowerSettings:
    thresholds: dict = field(default_factory=lambda: dict(POWER_DEFAULTS))
    plans: dict = field(default_factory=lambda: build_plans(None))

    @classmethod
    def from_config(cls, d: dict | None = None) -> "PowerSettings":
        """From the config "power" section ({} or None = all defaults). Raises ConfigError when invalid."""
        d = d or {}
        errs = validate_power(d)
        if errs:
            from ..config import ConfigError   # here, not at module level: config imports this package
            raise ConfigError("; ".join(errs))
        return cls({**POWER_DEFAULTS, **{k: d[k] for k in POWER_DEFAULTS if k in d}}, build_plans(d.get("plans")))


@dataclass(frozen=True)
class PowerReading:
    """One reading. t: monotonic seconds when it was taken. Every other field may be None: unknown."""
    t: float
    battery_pct: float | None = None
    charging: bool | None = None
    skin_c: float | None = None
    device_c: float | None = None
    thermal_status: int | None = None


@dataclass(frozen=True)
class Decision:
    level: str
    plan: DutyPlan
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"level": self.level, "plan": self.plan.to_dict(), "reasons": list(self.reasons)}


def thermal_status_level(status: object) -> str | None:
    """Android thermal status integer -> heat level: MODERATE is hot, SEVERE and above critical, NONE and LIGHT
    normal. Anything else (None, a bool, out of range) is unknown: None. The Android side only passes the int."""
    if isinstance(status, bool) or not isinstance(status, int):
        return None
    if not THERMAL_STATUS_NONE <= status <= THERMAL_STATUS_SHUTDOWN:
        return None
    if status >= THERMAL_STATUS_SEVERE:
        return CRITICAL
    if status >= THERMAL_STATUS_MODERATE:
        return HOT
    return NORMAL


def _in_range(x: object, rng: tuple[float, float]) -> float | None:
    return float(x) if _finite_number(x) and rng[0] <= x <= rng[1] else None   # type: ignore[operator, arg-type]


def battery_state(prev: str, pct: float | None, s: dict) -> str:
    """Battery axis with hysteresis (low is bad). Unknown keeps prev."""
    if pct is None:
        return prev
    if pct <= s["battery_critical_enter_pct"]:
        return CRITICAL
    if prev == CRITICAL and pct < s["battery_critical_exit_pct"]:
        return CRITICAL
    if pct <= s["battery_saver_enter_pct"]:
        return SAVER
    if prev in (SAVER, CRITICAL) and pct < s["battery_saver_exit_pct"]:
        return SAVER
    return NORMAL


def heat_state(prev: str, temp_c: float | None, src: str, s: dict) -> str:
    """One temperature source with hysteresis (high is bad). Unknown keeps prev."""
    if temp_c is None:
        return prev
    if temp_c >= s[f"{src}_critical_enter_c"]:
        return CRITICAL
    if prev == CRITICAL and temp_c > s[f"{src}_critical_exit_c"]:
        return CRITICAL
    if temp_c >= s[f"{src}_hot_enter_c"]:
        return HOT
    if prev in (HOT, CRITICAL) and temp_c > s[f"{src}_hot_exit_c"]:
        return HOT
    return NORMAL


def _tighter(a: str, b: str) -> str:
    return a if RANK[a] >= RANK[b] else b


class Governor:
    """update(reading, now) -> Decision. One instance per device; state is the level, when it was entered,
    and each signal's own hysteresis state."""

    def __init__(self, settings: PowerSettings | None = None):
        self.settings = settings or PowerSettings()
        self.level = NORMAL
        self.since: float | None = None            # caller's clock when the current level was entered
        self.signals = {"battery": NORMAL, "skin": NORMAL, "device": NORMAL, "thermal_status": NORMAL}
        self.charging = False

    def decision(self, stale: bool = False) -> Decision:
        sig = self.signals
        r = set()
        if sig["battery"] == SAVER:
            r.add("battery_low")
        elif sig["battery"] == CRITICAL:
            r.add("battery_critical")
        if self.charging:
            r.add("charging")
        for name in ("skin", "device", "thermal_status"):
            if sig[name] in (HOT, CRITICAL):
                r.add(f"{name}_{sig[name]}")
        if stale:
            r.add("stale")
        return Decision(self.level, self.settings.plans[self.level], tuple(c for c in REASONS if c in r))

    def update(self, reading: PowerReading, now: float | None = None) -> Decision:
        """now: the caller's monotonic seconds (default reading.t). A reading with no finite t is stale."""
        s = self.settings.thresholds
        t = reading.t if _finite_number(reading.t) else None
        if now is None:
            now = t if t is not None else (self.since if self.since is not None else 0.0)
        stale = t is None or now - t > s["stale_after_s"]
        charging = reading.charging if isinstance(reading.charging, bool) else None
        pct = _in_range(reading.battery_pct, BATTERY_PCT_RANGE)
        sig = self.signals
        if charging is True:
            new = {"battery": NORMAL}               # charging cancels the battery axis, never the heat axis
        else:
            new = {"battery": battery_state(sig["battery"], pct, s)}
        new["skin"] = heat_state(sig["skin"], _in_range(reading.skin_c, TEMP_PLAUSIBLE_C), "skin", s)
        new["device"] = heat_state(sig["device"], _in_range(reading.device_c, TEMP_PLAUSIBLE_C), "device", s)
        ts = thermal_status_level(reading.thermal_status)
        new["thermal_status"] = ts if ts is not None else sig["thermal_status"]
        if stale:   # old data may tighten a signal, never relax it
            new = {k: _tighter(v, sig[k]) for k, v in new.items()}
        else:
            if charging is not None:
                self.charging = charging
        self.signals = new
        target = NORMAL
        for v in new.values():
            target = _tighter(target, v)
        if RANK[target] > RANK[self.level]:
            self.level, self.since = target, now
        elif (RANK[target] < RANK[self.level] and not stale and self.since is not None
              and now - self.since >= s["min_dwell_s"]):
            self.level, self.since = LEVELS[RANK[self.level] - 1], now
        if self.since is None:
            self.since = now
        return self.decision(stale)


__all__ = ["NORMAL", "SAVER", "HOT", "CRITICAL", "LEVELS", "REASONS", "POWER_DEFAULTS", "PowerSettings",
           "PowerReading", "Decision", "Governor", "thermal_status_level", "validate_power", "battery_state",
           "heat_state"]
