"""Merge located detections from every sensor into spatial events (ADR-0060).

Two detections are linked when the angle between their head-frame unit vectors is at most MERGE_ANGLE_DEG
and their times are at most MERGE_WINDOW_S apart; each connected group is one SpatialEvent:
- bearing: the confidence-weighted mean of the unit vectors, normalised (equal weights if all are 0);
- families, sensors and kinds: united; confidence: noisy-OR 1 - prod(1 - c);
- evidence: the number of distinct families.
Grouping by connected components makes the result independent of input order; inputs are sorted by a
stable key first so the floating-point sums are too.

This adds a location to what the sensors saw. It does not change the fusion's evidence rule (an alert
still needs two families, oiris/fusion/stack.py)."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .detections import SpatialDetection
from .rig import Rig
from .frames import DirectionTracker, angle_between, az_el_to_vec, direction_word, normalize, vec_to_az_el

MERGE_ANGLE_DEG = 15.0
MERGE_WINDOW_S = 0.5
NS_PER_S = 1_000_000_000
_ROUND_DEG = 9   # sort keys compare bearings at this many decimals first, so float noise cannot reorder them


@dataclass(frozen=True)
class Located:
    """A detection resolved to the head frame (what merge() consumes)."""
    az: float
    el: float
    kind: str
    family: str
    mount_id: str
    t_mono_ns: int
    confidence: float

    def key(self) -> tuple:
        return (self.t_mono_ns, self.mount_id, self.kind, self.family, round(self.az, _ROUND_DEG),
                round(self.el, _ROUND_DEG), self.confidence, self.az, self.el)


@dataclass(frozen=True)
class SpatialEvent:
    az: float
    el: float
    direction: str
    families: tuple[str, ...]
    sensors: tuple[str, ...]
    kinds: tuple[str, ...]
    confidence: float
    evidence: int
    t_mono_ns: int          # the latest detection in the event
    count: int              # detections merged

    def to_dict(self) -> dict:
        return {"az": self.az, "el": self.el, "direction": self.direction, "families": list(self.families),
                "sensors": list(self.sensors), "kinds": list(self.kinds), "confidence": self.confidence,
                "evidence": self.evidence, "t_mono_ns": self.t_mono_ns, "count": self.count}


def locate(rig: Rig, detections) -> tuple[list[Located], list[str]]:
    """Resolve detections (SpatialDetection or its dict form) to the head frame. Returns the located ones
    and one reason per detection that could not be placed (bad shape, absent mount, pixel off the image)."""
    out, bad = [], []
    for d in detections:
        try:
            det = d if isinstance(d, SpatialDetection) else SpatialDetection.from_dict(d)
            az, el = rig.to_head(det)
        except (TypeError, ValueError) as e:   # python-brain: allow SILENT_EXCEPTION reported to the caller in `bad`
            bad.append(str(e))
            continue
        out.append(Located(az, el, det.kind, det.family, det.mount_id, det.t_mono_ns, det.confidence))
    return out, bad


def noisy_or(confidences) -> float:
    p = 1.0
    for c in confidences:
        p *= 1.0 - min(1.0, max(0.0, float(c)))
    return 1.0 - p


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def merge(detections: list[Located], angle_deg: float = MERGE_ANGLE_DEG, window_s: float = MERGE_WINDOW_S) -> list[SpatialEvent]:
    """Group located detections into events, sorted by (time, az, el). Deterministic and order-independent."""
    dets = sorted(detections, key=Located.key)
    vecs = [az_el_to_vec(d.az, d.el) for d in dets]
    window_ns = window_s * NS_PER_S
    parent = list(range(len(dets)))
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            if dets[j].t_mono_ns - dets[i].t_mono_ns > window_ns:
                break   # sorted by time: every later j is further away
            if angle_between(vecs[i], vecs[j]) <= angle_deg:
                ri, rj = _find(parent, i), _find(parent, j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[int]] = {}
    for i in range(len(dets)):
        groups.setdefault(_find(parent, i), []).append(i)
    events = []
    for idx in groups.values():
        ws = np.array([dets[i].confidence for i in idx])
        if not ws.sum() > 0:
            ws = np.ones(len(idx))
        mean = sum((w * vecs[i] for w, i in zip(ws, idx)), np.zeros(3))
        az, el = vec_to_az_el(normalize(mean))
        fams = tuple(sorted({dets[i].family for i in idx}))
        events.append(SpatialEvent(az, el, direction_word(az, el), fams, tuple(sorted({dets[i].mount_id for i in idx})),
                                   tuple(sorted({dets[i].kind for i in idx})), noisy_or(dets[i].confidence for i in idx),
                                   len(fams), max(dets[i].t_mono_ns for i in idx), len(idx)))
    return sorted(events, key=lambda e: (e.t_mono_ns, round(e.az, _ROUND_DEG), round(e.el, _ROUND_DEG), e.kinds, e.sensors))


class EventTracker:
    """Keeps each event's direction word steady across steps: an event within angle_deg of an event from the
    previous update continues that track (nearest first, each previous track used once) and its word follows
    DirectionTracker hysteresis. Tracks not continued are dropped."""

    def __init__(self, angle_deg: float = MERGE_ANGLE_DEG) -> None:
        self.angle_deg = angle_deg
        self._tracks: list[tuple[np.ndarray, DirectionTracker]] = []

    def update(self, events: list[SpatialEvent]) -> list[SpatialEvent]:
        pairs = []
        for i, e in enumerate(events):
            v = az_el_to_vec(e.az, e.el)
            for j, (pv, _) in enumerate(self._tracks):
                a = angle_between(v, pv)
                if a <= self.angle_deg:
                    pairs.append((a, i, j))
        taken_e: dict[int, int] = {}
        taken_t: set[int] = set()
        for _, i, j in sorted(pairs):
            if i not in taken_e and j not in taken_t:
                taken_e[i] = j
                taken_t.add(j)
        out, tracks = [], []
        for i, e in enumerate(events):
            tr = self._tracks[taken_e[i]][1] if i in taken_e else DirectionTracker()
            word = tr.update(e.az, e.el)
            tracks.append((az_el_to_vec(e.az, e.el), tr))
            out.append(e if word == e.direction else SpatialEvent(e.az, e.el, word, e.families, e.sensors, e.kinds,
                                                                  e.confidence, e.evidence, e.t_mono_ns, e.count))
        self._tracks = tracks
        return out


__all__ = ["Located", "SpatialEvent", "locate", "merge", "noisy_or", "EventTracker", "MERGE_ANGLE_DEG", "MERGE_WINDOW_S"]
