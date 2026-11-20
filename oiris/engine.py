"""The ASB runtime: one process, one thread per sensor (SensorBus), an asyncio score/fusion
loop at 1 Hz, a guard on every output. Core resolves adapters and score modules through
the registry; it never imports a concrete driver."""
from __future__ import annotations
import asyncio, logging, math, time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .bus import SensorBus
from .config import Config
from .fusion.stack import FusionModel, fuse
from .guard.vocabulary import check_output
from .modules.liveness import indicators
from .registry import Registry, default_registry
from .schema import Event, Health, Level, Score
from .scores.base import safe_update
from .spatial import DETECTIONS_KEY, SPATIAL_CAP, EventTracker, Rig, locate, merge
from .timing import StageTimer

log = logging.getLogger(__name__)


@dataclass
class Engine:
    config: Config
    registry: Registry = field(default_factory=default_registry)
    fusion_model: FusionModel = field(default_factory=FusionModel)
    bus: SensorBus = None  # type: ignore[assignment]  # built in __post_init__
    score_modules: list = field(default_factory=list)
    results: list = field(default_factory=list)
    events: list = field(default_factory=list)
    on_event: Any = None
    face_match: Any = None      # M-16 module when enabled (oiris.modules.face_match_gate.load); else None
    face_place: str | None = None
    timer: StageTimer = field(default_factory=StageTimer)   # per-stage step timing (oiris/timing.py)
    rig: Rig | None = None      # sensor mounts (ADR-0060); built from config.rig when spatial samples first arrive
    _spatial_tracker: EventTracker = field(default_factory=EventTracker, repr=False)

    def __post_init__(self) -> None:
        c = self.config
        self.bus = self.bus or SensorBus(c.queue_size, c.adapter_timeout_s, c.restart_backoff_max_s,
                                         c.breaker_threshold, c.breaker_cooldown_s,
                                         hung_disable_s=c.adapter_hung_disable_s)
        if not self.score_modules:
            for name in self.registry.names("score"):
                p = self.registry.get("score", name)
                if p.module_flag and not c.enabled(p.module_flag):
                    continue
                try:
                    self.score_modules.append(p.factory())
                except Exception as e:  # noqa: BLE001
                    log.error("score module %s failed to construct: %s", name, e)
        self._windows: dict[str, tuple[int, list]] = {}   # module -> (received_ns, last window)
        self._seen_ns: dict[str, int] = {}                  # adapter -> engine time of its last fresh sample

    def add_adapter(self, adapter: Any, cfg: dict | None = None) -> None:
        self.bus.add(adapter, cfg)

    def _collect(self) -> dict[str, list]:
        by_cap: dict[str, list] = {}
        now = time.monotonic_ns()
        for aid, a in self.bus.adapters.items():
            samples = self.bus.drain(aid)
            if samples:
                self._seen_ns[aid] = now
            for cap in getattr(a, "caps", ()):
                by_cap.setdefault(cap, []).extend(samples)
        return by_cap

    def _max_age_ns(self, aid: str) -> int:
        """Effective max age of one adapter's readings: max(config.sensor_max_age_s, its own
        max_age_s). A missing, non-numeric, non-finite or non-positive max_age_s is ignored, so such
        adapters keep the global value; read every step, so an adapter may change it (mode switch)."""
        own = getattr(self.bus.adapters.get(aid), "max_age_s", None)
        if isinstance(own, bool) or not isinstance(own, (int, float)) or not math.isfinite(own) or own <= 0:
            own = 0.0
        return int(max(self.config.sensor_max_age_s, own) * 1e9)

    def step(self) -> dict:
        """One fusion step. Timed as "step" and, inside it, "collect" (bus drain), "face_match" (when on), "health"
        (bus health and staleness), "scores", "fusion", "guard" (the result check) and "events" (only when a
        level is raised: the event, its check and on_event)."""
        with self.timer.time("step"):
            return self._step()

    def _step(self) -> dict:
        t = self.timer
        t0 = t.now()
        by_cap = self._collect()
        t.since("collect", t0)
        t0 = t.now()
        if self.face_match is not None:
            # M-16 consumes face frames off the bus. It returns counts only, writes to its own
            # review log and never emits an event, alert or score (spec section 6).
            for s in by_cap.get("face_frame", []):
                try:
                    self.face_match.process_frame(s.data, place=self.face_place)
                except Exception as e:  # noqa: BLE001 - M-16 can never stop the loop
                    log.error("face_match failed: %s", e)
            t.since("face_match", t0)
            t0 = t.now()
        # Spec section 6 and 7.2: a face frame lives in RAM for one frame cycle. Once M-16 has seen it (or with M-16
        # off) its pixels are zeroed and the reference dropped, so nothing later in the step, the results or a
        # record can hold it.
        for s in by_cap.pop("face_frame", ()):
            if isinstance(s.data, np.ndarray) and s.data.flags.writeable:
                s.data.fill(0)
        now = time.monotonic_ns()
        health = self.bus.health()
        for aid, seen in self._seen_ns.items():
            # Forge F1: a reading past the adapter's effective max age is stale; never show OK
            # health on stale data.
            age = now - seen
            if age > self._max_age_ns(aid) and aid in health and health[aid]["health"] == Health.OK.value:
                health[aid]["health"] = Health.DEGRADED.value
                health[aid]["reason"] = f"stale: no fresh reading for {age / 1e9:.1f}s"
        t.since("health", t0)
        t0 = t.now()
        scores: list[Score] = []
        for m in self.score_modules:
            src = [aid for aid, a in self.bus.adapters.items() if set(a.caps) & set(m.inputs)]
            if not src:
                continue   # capability absent: family masked (fallback-node behaviour)
            if all(health[s]["health"] == Health.FAILED.value for s in src):
                scores.append(Score(m.name, 0, 0, Health.FAILED, {"reason": "adapter failed"}))
                continue
            window = [s for cap in m.inputs for s in by_cap.get(cap, [])][-1:]
            if not window:
                got, cached = self._windows.get(m.name, (0, []))
                # Forge F1: reuse a cached reading only within max age; stale data is never scored.
                # The cache holds the last window; each reading is judged by the effective age of the
                # adapter that produced it, so a fast adapter keeps the global age even when it shares
                # a module with a slow one. Unknown producers get the global age.
                max_age_ns = min((self._max_age_ns(getattr(s, "adapter_id", "")) for s in cached),
                                 default=int(self.config.sensor_max_age_s * 1e9))
                window = cached if cached and now - got <= max_age_ns else []
                if not window:
                    self._windows.pop(m.name, None)
            else:
                self._windows[m.name] = (now, window)
            scores.append(safe_update(m, window))
        t.since("scores", t0)
        t0 = t.now()
        res = fuse(scores, self.fusion_model, self.config.min_families_for_alert, self.config.max_unhealthy_families)
        if self.config.enabled("M-02"):
            res.update(indicators(scores))
        res["scores"] = [s.to_dict() for s in scores]
        res["health"] = health
        res["t_wall"] = time.time()
        t.since("fusion", t0)
        self._add_spatial(res, by_cap)
        with t.time("guard"):
            check_output({k: v for k, v in res.items() if k != "scores"})
        self.results.append(res)
        if not res["level"]:
            return res
        with t.time("events"):
            sig = next((x for x in res["signatures"] if x != "A11"), "A11") if res["signatures"] else "A4"
            ev = Event(sig, Level(res["level"]), res["anomaly_index"],
                       [s.module for s in scores if s.value >= 0.5 and s.health != Health.FAILED],
                       res["families_active"], health_snapshot={k: v["health"] for k, v in health.items()})
            check_output(ev.to_dict())
            self.events.append(ev)
            if self.on_event:
                try:
                    self.on_event(ev)
                except Exception as e:  # noqa: BLE001
                    log.warning("on_event handler failed: %s", e)
        return res

    def _add_spatial(self, res: dict, by_cap: dict[str, list]) -> None:
        """ADR-0060: when samples from "spatial" adapters arrived this step, res["spatial"] lists the merged
        events (head-frame az/el, direction word, families, sensors). Timed as "spatial". Fusion, levels and
        the evidence rule are untouched; without such samples res is exactly as before."""
        carrying = [s.data[DETECTIONS_KEY] for s in by_cap.get(SPATIAL_CAP, [])
                    if isinstance(s.data, dict) and isinstance(s.data.get(DETECTIONS_KEY), list)]
        if not carrying:
            return
        dets = [d for batch in carrying for d in batch]
        with self.timer.time("spatial"):
            if self.rig is None:
                self.rig = Rig.from_config(self.config.rig)
            located, bad = locate(self.rig, dets)
            res["spatial"] = [e.to_dict() for e in self._spatial_tracker.update(merge(located))]
            if bad:
                res["spatial_unplaced"] = len(bad)
                log.debug("spatial: %d detections could not be placed: %s", len(bad), bad[:1])

    async def run(self, seconds: float | None = None, hz: float = 1.0, steps: int | None = None) -> list[dict]:
        self.bus.start()
        try:
            t0, n = time.monotonic(), 0
            while (seconds is None or time.monotonic() - t0 < seconds) and (steps is None or n < steps):
                await asyncio.sleep(1 / hz)
                try:
                    self.step()
                except Exception as e:  # noqa: BLE001 - the loop never dies
                    log.error("fusion step failed: %s", e)
                n += 1
        finally:
            self.bus.stop()
        return self.results
