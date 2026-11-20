"""Power, heat and latency budgets for a wearable (ADR-0061). Pure and deterministic, stdlib only.

- governor: battery and heat readings -> a level (normal, saver, hot, critical) and the duty plan for it.
- duty: the per-level duty plans, their validation, and FrameGate / ListenDuty.
- budgets: per-stage latency budgets checked against the StageTimer summaries.

Built and simulated only (oiris/adapters/power_sim.py). Nothing here is wired into live mode yet; that is a
later task. This package never imports oiris.config (config imports it to validate the "power" and "budgets"
sections), so there is no import cycle."""
from __future__ import annotations

from .budgets import BudgetMonitor, DEFAULT_BUDGETS_MS, TIMING_PARTS, validate_budgets
from .duty import DEFAULT_PLANS, DutyPlan, FrameGate, ListenDuty, validate_plans
from .governor import (CRITICAL, HOT, LEVELS, NORMAL, POWER_DEFAULTS, REASONS, SAVER, Decision, Governor,
                       PowerReading, PowerSettings, thermal_status_level, validate_power)

__all__ = ["BudgetMonitor", "DEFAULT_BUDGETS_MS", "TIMING_PARTS", "validate_budgets",
           "DEFAULT_PLANS", "DutyPlan", "FrameGate", "ListenDuty", "validate_plans",
           "CRITICAL", "HOT", "LEVELS", "NORMAL", "POWER_DEFAULTS", "REASONS", "SAVER", "Decision", "Governor",
           "PowerReading", "PowerSettings", "thermal_status_level", "validate_power"]
