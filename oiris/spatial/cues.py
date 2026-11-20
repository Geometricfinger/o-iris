"""Direction cues: what the wearer hears and sees for a located signature (ADR-0060, "direction cues").

cues(result) pairs each engine signature whose phrase template has a {direction} slot with the spatial event
of the matching kind in the same step result (res["spatial"], oiris/engine.py):

    A1 lens glint     -> kind "glint"       -> phrase "glint"       ("Candidate camera lens {direction}, {distance}"),
                                               or "glint_bearing" ("Candidate camera lens {direction}") without a range
    A2 NIR emitter    -> kind "ir_emitter"  -> phrase "ir_emitter"  ("Infrared emitter {direction}")
    A3 heat pattern   -> kind "thermal"     -> phrase "thermal"     ("Unusual heat pattern {direction}")

- Only a raised level counts as an engine signature event (the engine emits an Event only then).
- Time: the reference is the newest event time in the result (t_ref_ns overrides it). Of the events carrying the
  kind and no older than MERGE_WINDOW_S before the reference, the nearest in time wins; ties go to the higher
  confidence, then more evidence, then the smaller azimuth (deterministic). An event without a time (headless
  output drops them) came with this step's result and counts as at the reference.
- The direction is the event's word (already hysteresis-tracked by the engine) and must be a whitelisted slot value.
- Distance: a whitelisted word only from a measured range ("range_m"). Without one the cue uses the phrase's
  no-distance variant (NO_RANGE_PHRASE), or makes no cue when it has none: an unknown distance is never spoken
  as "far". Merged events carry no range today, so a lens cue names the direction only.
- No matching event, an unknown word or a phrase the guard refuses: no cue. Every text goes through
  oiris.guard.phrases.render, so the vocabulary guard always applies.
- Signatures only: a cue says what kind of signature lies where, never who. Pure and deterministic."""
from __future__ import annotations
import math, string
from dataclasses import dataclass

from ..guard.phrases import SLOT_VALUES, TEMPLATES, PhraseError, render
from ..guard.vocabulary import VocabularyViolation
from .merge import MERGE_WINDOW_S, NS_PER_S

# Engine signature -> (spatial detection kind, phrase key).
SIGNATURE_CUES = {"A1": ("glint", "glint"), "A2": ("ir_emitter", "ir_emitter"), "A3": ("thermal", "thermal")}
# A phrase whose template needs a distance -> its variant without one, used when no range was measured.
NO_RANGE_PHRASE = {"glint": "glint_bearing"}
DIRECTION_SLOT, DISTANCE_SLOT = "direction", "distance"
FAR = "far"
# Upper bound (metres) of each distance word; beyond the last, "far".
DISTANCE_WORDS = ((0.75, "half a metre"), (1.5, "one metre"), (2.5, "two metres"), (3.5, "three metres"))


@dataclass(frozen=True)
class Cue:
    signature: str
    phrase_key: str
    slots: dict
    direction: str
    az: float
    el: float
    evidence: int
    level: str
    text: str           # render(phrase_key, **slots): guarded

    def to_dict(self) -> dict:
        return {"signature": self.signature, "phrase_key": self.phrase_key, "slots": dict(self.slots),
                "direction": self.direction, "az": self.az, "el": self.el, "evidence": self.evidence,
                "level": self.level, "text": self.text}


def _fields(key: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(TEMPLATES[key]) if f}


def distance_word(range_m) -> str | None:
    """The whitelisted distance word for a measured range in metres ("far" beyond the last bound). None when there
    is no range or it is not a usable number, so the caller never invents a distance."""
    if isinstance(range_m, bool) or not isinstance(range_m, (int, float)) or not math.isfinite(range_m) or range_m < 0:
        return None
    for bound, word in DISTANCE_WORDS:
        if range_m <= bound:
            return word
    return FAR


def _event_ns(e: dict) -> int | None:
    t = e.get("t_mono_ns")
    return t if isinstance(t, int) and not isinstance(t, bool) else None


def nearest_event(events: list, kind: str, t_ref_ns: int, window_s: float = MERGE_WINDOW_S) -> dict | None:
    """The event carrying `kind` nearest in time to t_ref_ns, at most window_s before or after it (None if none).
    An event without a time came with this step's result (headless output drops times) and counts as at t_ref_ns."""
    window_ns = window_s * NS_PER_S
    best, best_key = None, None
    for e in events:
        if not isinstance(e, dict) or kind not in (e.get("kinds") or ()):
            continue
        t = _event_ns(e)
        t = t_ref_ns if t is None else t
        if abs(t_ref_ns - t) > window_ns:
            continue
        key = (abs(t_ref_ns - t), -float(e.get("confidence") or 0.0), -int(e.get("evidence") or 0), float(e.get("az") or 0.0))
        if best_key is None or key < best_key:
            best, best_key = e, key
    return best


def cues(result: dict, t_ref_ns: int | None = None) -> list[Cue]:
    """Cues for one engine step result, in the result's signature order. See the module docstring."""
    level = result.get("level")
    events = [e for e in (result.get("spatial") or []) if isinstance(e, dict)]
    if not level or not events:
        return []
    times = [t for t in map(_event_ns, events) if t is not None]
    ref = t_ref_ns if t_ref_ns is not None else max(times, default=0)
    out = []
    for sig in result.get("signatures") or []:
        if sig not in SIGNATURE_CUES:
            continue
        kind, key = SIGNATURE_CUES[sig]
        fields = _fields(key)
        if DIRECTION_SLOT not in fields:
            continue
        e = nearest_event(events, kind, ref)
        if e is None or e.get("direction") not in SLOT_VALUES[DIRECTION_SLOT]:
            continue
        slots = {DIRECTION_SLOT: str(e["direction"])}
        if DISTANCE_SLOT in fields:
            word = distance_word(e.get("range_m"))
            if word is not None:
                slots[DISTANCE_SLOT] = word
            elif key in NO_RANGE_PHRASE:
                key = NO_RANGE_PHRASE[key]
            else:
                continue                        # the phrase needs a distance nobody measured: no cue
        try:
            text = render(key, **slots)
        except (PhraseError, VocabularyViolation):  # python-brain: allow SILENT_EXCEPTION - a refused phrase means no cue, by design
            continue
        out.append(Cue(sig, key, slots, slots[DIRECTION_SLOT], float(e.get("az") or 0.0), float(e.get("el") or 0.0),
                       int(e.get("evidence") or 0), str(level), text))
    return out


def current(cs: list[Cue]) -> Cue | None:
    """The cue the display shows: the most corroborated (evidence), then the first in signature order."""
    return max(cs, key=lambda c: c.evidence, default=None)


__all__ = ["Cue", "cues", "current", "nearest_event", "distance_word", "SIGNATURE_CUES", "DISTANCE_WORDS",
           "NO_RANGE_PHRASE"]
