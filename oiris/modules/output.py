"""One arbiter for every audible, visual and haptic output (docs/ARCHITECTURE.md, "Outputs and motion").

On a wearable, M-06 alerts, the voice-link announcement (ADR-0058), the glasses/dashboard pop-up and haptics
must never stack, overlap or distract while moving. Every output goes through OutputArbiter.request(), which
answers at once with played, queued, deduped, suppressed or dropped, and a reason.

- Priorities: panic > confirmed > attention > voice_link > fyi.
- Channels: speech (one utterance at a time) and chime, played in order by one daemon worker from a bounded
  queue; the pop-up (state only: the UI reads state()); haptic (a short pulse, called at once).
- Which channels each priority may use: alerts (panic, confirmed, attention) pop up, chime, buzz and speak
  only when output.speak_alerts is on; voice_link chimes and speaks but never pops up (the glasses show
  nothing on a match); fyi is haptic only (M-06).
- A higher priority barges in on lower speech (the player's stop(), when it has one; the interrupted job is
  dropped, never replayed). Equal or lower priority waits its turn.
- The same key again inside its dedupe window (output.dedupe_s) is deduped; so is a key already queued or
  playing. Panic and voice_link are not windowed: the Announcer keeps its own per-person repeat limit.
- M-06 attention budget: at most one attention utterance per voice_attention_min_interval_s; a later one is
  queued for its slot. Anything still queued after QUEUE_MAX_AGE_S is dropped as stale.
- Motion gate (oiris/modules/motion.py): walking hides pop-ups (unless output.popups_while_walking); fast
  allows only output.fast_min_priority and above, and no pop-ups. Unknown motion (no IMU) counts as still.
  Suppressed outputs are dropped, never replayed later; a queued job is checked again when its turn comes.
- playing() is true while a ready job waits or plays and for tail_s after, so the voice-link listener drops
  microphone input and never hears this device speak.
- Never raises into callers: player and haptic failures are logged. Nothing is sent over the network,
  nothing is written, and state() carries no person id.
- Timing (oiris/timing.py, docs/LIVE.md "Timing"): per sound channel, "<channel>_wait" is request accepted to the
  start of that sound (speech after a chime includes the chime) and "<channel>_play" is how long the player call
  took. Numbers only, in the StageTimer passed in."""
from __future__ import annotations
import collections, itertools, logging, threading, time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import OUTPUT_DEFAULTS, OUTPUT_FAST_MIN, OUTPUT_KEYS, validate_output
from ..guard.vocabulary import violations
from ..timing import StageTimer
from .motion import FAST, MOVING, UNKNOWN, WALKING, effective

log = logging.getLogger(__name__)

PANIC, CONFIRMED, ATTENTION, VOICE_LINK, FYI = 50, 40, 30, 20, 10
PRIORITY_NAMES = {PANIC: "panic", CONFIRMED: "confirmed", ATTENTION: "attention", VOICE_LINK: "voice_link", FYI: "fyi"}
PRIORITIES = {v: k for k, v in PRIORITY_NAMES.items()}
ALERT_PRIORITIES = frozenset({PANIC, CONFIRMED, ATTENTION})
DEDUPE_EXEMPT = frozenset({PANIC, VOICE_LINK})

PLAYED, QUEUED, DEDUPED, SUPPRESSED, DROPPED = "played", "queued", "deduped", "suppressed", "dropped"
OUTCOMES = (PLAYED, QUEUED, DEDUPED, SUPPRESSED, DROPPED)
SPEECH, CHIME, POPUP, HAPTIC = "speech", "chime", "popup", "haptic"
CHANNELS = (SPEECH, CHIME, POPUP, HAPTIC)
SOUNDS = ("chime",)

QUEUE_JOBS = 4                # bounded: beyond this the lowest-priority job (or the new one) is dropped
QUEUE_MAX_AGE_S = 30.0        # a job not started within this is stale and dropped
TAIL_S = 0.4                  # playing() stays true this long after a job (room echo, speaker ring-out)
WORKER_POLL_S = 0.2
CLOSE_JOIN_S = 2.0
DEFAULT_ATTENTION_INTERVAL_S = 10.0   # M-06; live passes config voice_attention_min_interval_s

# Engine alert level -> priority and the words shown/spoken (the G1 level words, design_tokens.json).
ALERT_LEVELS = {"fyi": FYI, "attention": ATTENTION, "confirmed": CONFIRMED}
ALERT_WORDS = {"fyi": "Low signal", "attention": "Medium signal", "confirmed": "Confirmed signal"}


@dataclass(frozen=True)
class Decision:
    outcome: str
    reason: str = ""

    def __str__(self) -> str:
        return self.outcome


@dataclass
class OutputSettings:
    speak_alerts: bool = OUTPUT_DEFAULTS["speak_alerts"]
    popups_while_walking: bool = OUTPUT_DEFAULTS["popups_while_walking"]
    fast_min_priority: str = OUTPUT_DEFAULTS["fast_min_priority"]
    popup_s: float = OUTPUT_DEFAULTS["popup_s"]
    dedupe_s: float = OUTPUT_DEFAULTS["dedupe_s"]
    attention_interval_s: float = DEFAULT_ATTENTION_INTERVAL_S

    @classmethod
    def from_config(cls, cfg: Any) -> "OutputSettings":
        """From a Config (its "output" section and voice_attention_min_interval_s) or a plain output dict."""
        d = getattr(cfg, "output", cfg) or {}
        errs = validate_output(d)
        if errs:
            raise ValueError("; ".join(errs))
        v = {**OUTPUT_DEFAULTS, **{k: x for k, x in d.items() if k in OUTPUT_KEYS}}
        interval = getattr(cfg, "voice_attention_min_interval_s", DEFAULT_ATTENTION_INTERVAL_S)
        return cls(**v, attention_interval_s=float(interval))

    @property
    def fast_min(self) -> int:
        return PRIORITIES[self.fast_min_priority]


@dataclass
class _Job:
    seq: int
    priority: int
    topic: str                         # the request key (dedupe and cancel identity)
    text: str | None
    sound: str | None
    created: float
    not_before: float
    person_id: str | None = None       # RAM only; never in state()
    interrupted: bool = field(default=False)
    accepted: float = 0.0              # timer clock when the request was accepted (queue-wait timing)


class NullOutputPlayer:
    """Plays nothing; records what it was asked to play (tests)."""

    def __init__(self) -> None:
        self.played: list[tuple[str, str]] = []
        self.stops = 0

    def chime(self) -> None:
        self.played.append(("chime", ""))

    def say(self, text: str) -> None:
        self.played.append(("say", text))

    def stop(self) -> None:
        self.stops += 1


class OutputArbiter:
    def __init__(self, player: Any = None, haptic: Callable[[], None] | None = None,
                 settings: OutputSettings | None = None, motion: Callable[[], str] = lambda: UNKNOWN,
                 clock: Callable[[], float] = time.monotonic, queue_jobs: int = QUEUE_JOBS, tail_s: float = TAIL_S,
                 start: bool = True, timer: StageTimer | None = None):
        """player: chime()/say(text), optionally stop() for barge-in (announce.MacPlayer, GenericPlayer wrapping
        the M-06 tts port, NullOutputPlayer). None plays nothing. haptic: one short pulse, or None (no device).
        motion: returns the motion state (MotionEstimator.state). start=False: no worker; tests call pump().
        timer: where per-channel wait and play times go (a fresh StageTimer when None)."""
        self.player = player if player is not None else NullOutputPlayer()
        self.haptic = haptic
        self.settings = settings or OutputSettings()
        self.motion, self.clock = motion, clock
        self.queue_jobs, self.tail_s = int(queue_jobs), float(tail_s)
        self.timer = timer if timer is not None else StageTimer()
        self._cv = threading.Condition()
        self._jobs: list[_Job] = []
        self._seq = itertools.count()
        self._playing: _Job | None = None
        self._quiet_at = float("-inf")
        self._attention_next = float("-inf")     # earliest clock time for the next attention utterance
        self._last: dict[str, float] = {}         # key -> last accepted (RAM only)
        self._popup: dict | None = None
        self.counts: collections.Counter = collections.Counter()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        if start:
            self._worker = threading.Thread(target=self._run, name="oiris-output", daemon=True)
            self._worker.start()

    # -- motion policy ---------------------------------------------------------------------------------
    def _motion(self) -> str:
        try:
            return effective(str(self.motion()))
        except Exception as e:  # noqa: BLE001 - a broken motion source counts as unknown (still), output goes on
            log.warning("motion state failed: %s", e)
            return effective(UNKNOWN)

    def _blocked(self, priority: int, motion: str) -> str:
        """The reason this priority may not play at all in this motion state, or ""."""
        if motion == FAST and priority < self.settings.fast_min:
            return "moving fast"
        return ""

    def _popups_allowed(self, motion: str) -> bool:
        return motion not in MOVING or (motion == WALKING and self.settings.popups_while_walking)

    # -- requests --------------------------------------------------------------------------------------
    def request(self, kind: str, priority: int, key: str, text: str | None = None, sound: str | None = None,
                haptic: bool = False, person_id: str | None = None) -> Decision:
        """Ask for an output. Never raises; the Decision says what happened and why."""
        try:
            d = self._request(str(kind), priority, str(key), text, sound, bool(haptic), person_id)
        except Exception as e:  # noqa: BLE001 - output must never break its caller
            log.error("output request failed: %s", e)
            d = Decision(DROPPED, "internal error")
        with self._cv:
            self.counts[d.outcome] += 1
        if d.outcome != PLAYED:
            log.debug("output %s %s: %s (%s)", kind, PRIORITY_NAMES.get(priority, priority), d.outcome, d.reason)
        return d

    def _request(self, kind: str, priority: int, key: str, text: str | None, sound: str | None, haptic: bool,
                 person_id: str | None) -> Decision:
        if self._stop.is_set():
            return Decision(DROPPED, "closed")
        if priority not in PRIORITY_NAMES:
            return Decision(DROPPED, "unknown priority")
        if sound is not None and sound not in SOUNDS:
            return Decision(DROPPED, "unknown sound")
        if text is not None and (not isinstance(text, str) or not text.strip() or violations(text)):
            return Decision(DROPPED, "text refused by the vocabulary guard")
        s = self.settings
        speech = bool(text) and (priority == VOICE_LINK or (priority in ALERT_PRIORITIES and s.speak_alerts))
        chime = bool(sound) and priority != FYI
        popup = bool(text) and priority in ALERT_PRIORITIES
        buzz = haptic and priority != VOICE_LINK
        if not (speech or chime or popup or buzz):
            return Decision(DROPPED, "no channel for this priority")
        now = self.clock()
        motion = self._motion()
        barge: _Job | None = None
        with self._cv:
            pending = [j.topic for j in self._jobs] + ([self._playing.topic] if self._playing else [])
            if key in pending:
                return Decision(DEDUPED, "already queued or playing")
            last = self._last.get(key)
            if priority not in DEDUPE_EXEMPT and last is not None and now - last < s.dedupe_s:
                return Decision(DEDUPED, f"same key within {s.dedupe_s:g} s")
            why = self._blocked(priority, motion)
            if why:
                return Decision(SUPPRESSED, why)
            results: dict[str, Decision] = {}
            if popup:
                results[POPUP] = self._show_popup(priority, key, text or "", now, motion)
            if speech or chime:
                results[SPEECH if speech else CHIME], barge = self._enqueue(
                    priority, key, text if speech else None, sound if chime else None, now, person_id)
        if buzz:
            results[HAPTIC] = self._buzz()
        if barge is not None:
            self._stop_player()
        d = _combine(results)
        if d.outcome in (PLAYED, QUEUED):
            with self._cv:
                self._last[key] = now
        return d

    def _show_popup(self, priority: int, key: str, text: str, now: float, motion: str) -> Decision:
        """Caller holds the lock."""
        if not self._popups_allowed(motion):
            return Decision(SUPPRESSED, f"no pop-ups while {motion}")
        cur = self._popup
        if cur is not None and cur["until"] > now and cur["priority"] > priority:
            return Decision(DROPPED, "a higher pop-up is showing")
        self._popup = {"topic": key, "text": text, "priority": priority, "until": now + self.settings.popup_s}
        return Decision(PLAYED, "pop-up shown")

    def _enqueue(self, priority: int, key: str, text: str | None, sound: str | None, now: float,
                 person_id: str | None) -> tuple[Decision, _Job | None]:
        """Caller holds the lock. Returns the decision and the playing job to barge in on, if any."""
        not_before = now
        if priority == ATTENTION and text:
            not_before = max(now, self._attention_next)
        if len(self._jobs) >= self.queue_jobs:
            low = min(self._jobs, key=lambda j: (j.priority, -j.seq))
            if low.priority >= priority:
                return Decision(DROPPED, "queue full"), None
            self._jobs.remove(low)
            self.counts["evicted"] += 1
        if priority == ATTENTION and text:
            self._attention_next = not_before + self.settings.attention_interval_s
        job = _Job(next(self._seq), priority, key, text, sound, now, not_before, person_id,
                   accepted=self.timer.now())
        self._jobs.append(job)
        self._cv.notify_all()
        cur = self._playing
        if cur is not None and cur.priority < priority and cur.text:
            cur.interrupted = True
            return Decision(PLAYED, f"barged in on {PRIORITY_NAMES[cur.priority]}"), cur
        if not_before > now:
            return Decision(QUEUED, "attention budget"), None
        ahead = [j for j in self._jobs if j is not job and j.not_before <= now and j.priority >= priority]
        if cur is not None or ahead:
            return Decision(QUEUED, "waiting for the speaker"), None
        return Decision(PLAYED, "playing"), None

    def _buzz(self) -> Decision:
        if self.haptic is None:
            return Decision(DROPPED, "no haptic device")
        try:
            self.haptic()
            return Decision(PLAYED, "haptic")
        except Exception as e:  # noqa: BLE001 - a failed motor must not break the caller
            log.warning("haptic failed: %s", e)
            return Decision(DROPPED, "haptic failed")

    def _stop_player(self) -> None:
        stop = getattr(self.player, "stop", None)
        if stop is None:
            return                              # not interruptible: the higher job plays next
        try:
            stop()
        except Exception as e:  # noqa: BLE001 - barge-in is best effort
            log.warning("output barge-in stop failed: %s", e)

    def cancel(self, key: str) -> int:
        """Drop queued jobs and the pop-up for key, and stop it if it is playing (M-06 acknowledge)."""
        topic = str(key)
        with self._cv:
            n = len(self._jobs)
            self._jobs = [j for j in self._jobs if j.topic != topic]
            n -= len(self._jobs)
            if self._popup is not None and self._popup["topic"] == topic:
                self._popup = None
            cur = self._playing
            hit = cur is not None and cur.topic == topic
            if hit and cur is not None:
                cur.interrupted = True
        if hit:
            self._stop_player()
        return n + int(hit)

    # -- the worker ------------------------------------------------------------------------------------
    def _next(self) -> tuple[_Job | None, float]:
        """Caller holds the lock. The next ready job (dropping stale and now-suppressed ones), and how long to
        wait when none is ready."""
        now = self.clock()
        wait = WORKER_POLL_S
        motion = self._motion()
        for j in list(self._jobs):
            if now - j.created > QUEUE_MAX_AGE_S:
                self._jobs.remove(j)
                self.counts["expired"] += 1
            elif self._blocked(j.priority, motion):
                self._jobs.remove(j)
                self.counts["suppressed_in_queue"] += 1
        ready = [j for j in self._jobs if j.not_before <= now]
        if not ready:
            if self._jobs:
                wait = min(wait, max(0.0, min(j.not_before for j in self._jobs) - now))
            return None, wait
        job = max(ready, key=lambda j: (j.priority, -j.seq))
        self._jobs.remove(job)
        return job, 0.0

    def _play(self, job: _Job) -> None:
        t = self.timer
        try:
            if job.sound:
                t.since(CHIME + "_wait", job.accepted)
                with t.time(CHIME + "_play"):
                    self.player.chime()
            if job.text and not job.interrupted:
                t.since(SPEECH + "_wait", job.accepted)
                with t.time(SPEECH + "_play"):
                    self.player.say(job.text)
        except Exception as e:  # noqa: BLE001 - a missing sound device or failed command must not stop output
            log.warning("output could not be played: %s", e)
            with self._cv:
                self.counts["failed"] += 1

    def _take(self) -> tuple[_Job | None, float]:
        """Mark the next ready job as playing. Returns it (or None) and how long to wait when none is ready."""
        with self._cv:
            if self._stop.is_set():
                return None, 0.0
            job, wait = self._next()
            self._playing = job
            if job is None and wait > 0:
                self._cv.wait(wait)
            return job, wait

    def _finish(self, job: _Job) -> None:
        try:
            self._play(job)
        finally:
            with self._cv:
                self._playing = None
                self._quiet_at = self.clock() + self.tail_s
                self.counts["interrupted" if job.interrupted else "completed"] += 1
                self._cv.notify_all()

    def pump(self) -> int:
        """Tests without a worker (start=False): play every ready job now. Returns how many played."""
        n = 0
        while True:
            with self._cv:
                job = None if self._stop.is_set() else self._next()[0]
                self._playing = job
            if job is None:
                return n
            self._finish(job)
            n += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            job, _wait = self._take()
            if job is not None:
                self._finish(job)

    # -- state -----------------------------------------------------------------------------------------
    def playing(self) -> bool:
        """True while a ready job waits or plays, and for tail_s after the last one ended."""
        with self._cv:
            now = self.clock()
            return (self._playing is not None or any(j.not_before <= now for j in self._jobs)
                    or now < self._quiet_at)

    def popup(self) -> dict | None:
        """The current pop-up for the glasses/dashboard: text, priority, expires_in_s; None when nothing shows."""
        motion = self._motion()
        with self._cv:
            p, now = self._popup, self.clock()
            if p is None or p["until"] <= now:
                self._popup = None
                return None
            if not self._popups_allowed(motion) or self._blocked(p["priority"], motion):
                return None
            return {"text": p["text"], "priority": PRIORITY_NAMES[p["priority"]],
                    "expires_in_s": round(p["until"] - now, 1)}

    def state(self) -> dict:
        """For snapshot(): the current pop-up, queue, speaking and counts by outcome. No person ids."""
        popup = self.popup()
        with self._cv:
            cur = self._playing
            return {"popup": popup, "speaking": PRIORITY_NAMES[cur.priority] if cur else None,
                    "queued": len(self._jobs),
                    "counts": {**{o: self.counts.get(o, 0) for o in OUTCOMES},
                               **{k: v for k, v in self.counts.items() if k not in OUTCOMES}}}

    def close(self, join_s: float = CLOSE_JOIN_S) -> None:
        """Stop the worker, drop queued jobs and the pop-up, stop current speech. Idempotent."""
        self._stop.set()
        with self._cv:
            self._jobs.clear()
            self._popup = None
            busy = self._playing is not None
            self._cv.notify_all()
        if busy:
            self._stop_player()
        if self._worker is not None:
            self._worker.join(join_s)
            if self._worker.is_alive():
                log.warning("output worker still playing %.1fs after close", join_s)

    @property
    def closed(self) -> bool:
        return self._stop.is_set()


def _combine(results: dict[str, Decision]) -> Decision:
    """One decision for a request from its channels. A single channel's decision stands as it is. The audible
    channel (speech or chime) decides when it played or queued, so a pop-up shown at once does not hide that the
    speech waits (the M-06 attention budget). Otherwise played > queued > suppressed > dropped."""
    if len(results) == 1:
        return next(iter(results.values()))
    audible = results.get(SPEECH) or results.get(CHIME)
    order: tuple[str, ...] = (PLAYED, QUEUED, SUPPRESSED, DROPPED)
    if audible is not None and audible.outcome in (PLAYED, QUEUED):
        order = (audible.outcome,)
    for outcome in order:
        hits = [f"{ch}: {d.reason}" for ch, d in results.items() if d.outcome == outcome]
        if hits:
            others = [f"{ch} {d.outcome} ({d.reason})" for ch, d in results.items() if d.outcome != outcome]
            return Decision(outcome, "; ".join(hits + others))
    return Decision(DROPPED, "no channel")


def alert(arbiter: OutputArbiter, level: str, signature: str) -> Decision:
    """An engine alert event: fyi buzzes, attention and confirmed pop up the level words (and speak them when
    output.speak_alerts is on). One key per (signature, level), so the dedupe window paces repeats."""
    level = str(level)
    if level not in ALERT_LEVELS:
        return Decision(DROPPED, "unknown level")
    pr = ALERT_LEVELS[level]
    if pr == FYI:
        return arbiter.request("alert", FYI, f"alert:{signature}:{level}", haptic=True)
    return arbiter.request("alert", pr, f"alert:{signature}:{level}", text=ALERT_WORDS[level], haptic=True)


__all__ = ["OutputArbiter", "OutputSettings", "Decision", "NullOutputPlayer", "alert", "PANIC", "CONFIRMED",
           "ATTENTION", "VOICE_LINK", "FYI", "PRIORITY_NAMES", "PRIORITIES", "OUTCOMES", "PLAYED", "QUEUED", "DEDUPED",
           "SUPPRESSED", "DROPPED", "CHANNELS", "OUTPUT_FAST_MIN", "QUEUE_JOBS", "TAIL_S"]
