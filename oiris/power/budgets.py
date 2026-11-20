"""Per-stage latency budgets (ADR-0061, docs/LIVE.md "Timing").

Config "budgets": {"<part>.<stage>": milliseconds}, laid over DEFAULT_BUDGETS_MS. part is one of live mode's
StageTimer owners (TIMING_PARTS); stage follows oiris.timing.STAGE_RE. BudgetMonitor compares each budget with
the p95 of that stage's StageTimer summary and lists the stages over budget. Numbers and code-chosen names only.

Every default is provisional until measured on the device; the reasoning for each is in ADR-0061."""
from __future__ import annotations
import math

from ..timing import STAGE_RE

# Mirrors oiris.live.TIMING_PARTS (a test keeps them equal); defined here so oiris.config need not import live.
TIMING_PARTS = ("engine", "voice", "output")
BUDGET_MAX_MS = 60000.0
DEFAULT_MIN_SAMPLES = 20

# Defaults only for stages the code records today (engine.py, voice_link/listener.py, modules/output.py).
DEFAULT_BUDGETS_MS: dict = {
    "engine.step": 50.0,            # half the shortest live step interval (1 / live.MAX_HZ = 100 ms)
    "voice.queue_wait": 1000.0,     # ten 0.1 s microphone blocks; a tenth of the bounded queue (QUEUE_BLOCKS 100)
    "voice.decision": 1000.0,       # utterance end -> match result, after END_QUIET_S of quiet already waited
    "output.chime_wait": 1500.0,    # spec R-17 / E-20: attention output starts within 1.5 s p95
    "output.speech_wait": 1500.0,   # the same R-17 target; speech after a chime includes the chime
}


def _finite_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def split_key(key: object) -> tuple[str, str] | None:
    """"part.stage" -> (part, stage) when the part is known and the stage name is code-shaped, else None."""
    if not isinstance(key, str) or key.count(".") != 1:
        return None
    part, stage = key.split(".")
    if part not in TIMING_PARTS or not STAGE_RE.fullmatch(stage):
        return None
    return part, stage


def validate_budgets(d: object) -> list[str]:
    if not isinstance(d, dict):
        return ["budgets must be an object"]
    errs = []
    for k, v in d.items():
        if split_key(k) is None:
            errs.append(f"budgets: key {k!r} must be '<part>.<stage>' with part in {list(TIMING_PARTS)} "
                        f"and a stage of lowercase letters, digits and '_'")
        elif not _finite_number(v) or not 0 < v <= BUDGET_MAX_MS:
            errs.append(f"budgets.{k} must be a number of milliseconds in (0, {BUDGET_MAX_MS:g}]")
    return errs


def merged_budgets(d: dict | None = None) -> dict[str, float]:
    return {**DEFAULT_BUDGETS_MS, **{k: float(v) for k, v in (d or {}).items()}}


class BudgetMonitor:
    """check(timing_snapshot) -> the stages over budget. timing_snapshot is live mode's snapshot()["timing"]:
    {part: {stage: {"count", "p50_ms", "p95_ms", "max_ms"}}}. A stage is listed only when its count reaches
    min_samples and its p95 is above the budget. Missing or malformed entries are skipped, never raised."""

    def __init__(self, budgets: dict | None = None, min_samples: int = DEFAULT_MIN_SAMPLES):
        errs = validate_budgets(budgets or {})
        if errs:
            raise ValueError("; ".join(errs))
        if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples < 1:
            raise ValueError("min_samples must be a whole number >= 1")
        self.budgets = merged_budgets(budgets)
        self.min_samples = min_samples

    def check(self, timing_snapshot: object) -> list[dict]:
        out: list[dict] = []
        if not isinstance(timing_snapshot, dict):
            return out
        for key in sorted(self.budgets):
            part, stage = key.split(".")
            stages = timing_snapshot.get(part)
            entry = stages.get(stage) if isinstance(stages, dict) else None
            if not isinstance(entry, dict):
                continue
            count, p95 = entry.get("count"), entry.get("p95_ms")
            if isinstance(count, bool) or not isinstance(count, int) or not _finite_number(p95):
                continue
            p95_ms, budget = float(p95), self.budgets[key]   # type: ignore[arg-type]
            if count >= self.min_samples and p95_ms > budget:
                out.append({"part": part, "stage": stage, "p95_ms": p95_ms, "budget_ms": budget, "count": count})
        return out


__all__ = ["TIMING_PARTS", "DEFAULT_BUDGETS_MS", "BUDGET_MAX_MS", "BudgetMonitor", "validate_budgets",
           "merged_budgets", "split_key"]
