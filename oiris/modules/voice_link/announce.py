"""Announcing a voice-link match (ADR-0058, docs/VOICE_LINK_MODULE_SPEC.md "Announcing a match").

A clear match plays a short chime and then says "sounds like NAME" on the device O Iris runs on. A voice
match is a hint, not proof, hence "sounds like". `voice_link.announce` chooses chime_and_name (default),
chime or off.

- The name is spoken only when it passes `speakable_name`: guard-clean, at most NAME_MAX_WORDS words and
  NAME_MAX_CHARS characters, letters of one script plus spaces, hyphens and apostrophes. Otherwise the
  chime plays alone. This is the one exception to M-06's "no free text is ever spoken".
- The same person is announced at most once per `repeat_s`; different people back to back. Jobs go into
  a bounded queue (a full queue drops the job) and one daemon worker plays them in order.
- `playing()` is true while a job waits or plays and for TAIL_S after it, so the listener can drop
  microphone input and never hears (and matches) its own announcement.
- Players are injectable: MacPlayer (afplay + say, absolute paths, argument lists, timeouts, stop() for
  barge-in),
  GenericPlayer (a generated chime through sounddevice, the phrase through the M-06 `tts` port) and
  NullPlayer (tests and "off"). Playback failures are logged, never raised into the listener.
- In live mode the jobs go through the output arbiter (oiris/modules/output.py) at voice_link priority instead
  of this module's own worker, so an announcement never overlaps an alert and obeys the motion gate.
Nothing is sent over the network and nothing is written."""
from __future__ import annotations
import importlib, logging, queue, subprocess, sys, threading, time, unicodedata
from typing import Any, Callable, Protocol

import numpy as np

from ...guard.phrases import MAX_WORDS
from ...guard.vocabulary import VocabularyViolation, check_text, violations

log = logging.getLogger(__name__)

CHIME_AND_NAME, CHIME, OFF = "chime_and_name", "chime", "off"
MODES = (CHIME_AND_NAME, CHIME, OFF)
PHRASE_PREFIX = "sounds like "
NAME_MAX_WORDS = 4
NAME_MAX_CHARS = 40
NAME_JOINERS = frozenset(" -'’")          # space, hyphen, apostrophe, typographic apostrophe
# Han, Hiragana and Katakana are written together in one Japanese name: one script for this rule.
SCRIPT_FAMILIES = {"CJK": "HAN", "HIRAGANA": "HAN", "KATAKANA": "HAN"}
assert len(PHRASE_PREFIX.split()) + NAME_MAX_WORDS <= MAX_WORDS   # M-06's 8-word budget
# ADR-0065: where the recognised voice came from, from the direction words of oiris/features/doa.py. Spoken only
# after the name; dropped when name plus direction would pass MAX_WORDS (the name always wins).
DIRECTION_PHRASES = {
    "ahead": "ahead of you",
    "at left": "on your left",
    "at right": "on your right",
    "behind": "behind you",
    "above": "above you",
    "below": "below you",
}

QUEUE_JOBS = 4                # bounded: a burst beyond this is dropped, not queued
TAIL_S = 0.4                  # input stays paused this long after playback (room echo, speaker ring-out)
WORKER_POLL_S = 0.2
CLOSE_JOIN_S = 2.0
DEFAULT_REPEAT_S = 60.0

# macOS players: absolute paths only.
AFPLAY = "/usr/bin/afplay"
SAY = "/usr/bin/say"
CHIME_SOUND = "/System/Library/Sounds/Glass.aiff"
CHIME_TIMEOUT_S = 5.0
SAY_TIMEOUT_S = 15.0
STOP_GRACE_S = 0.5            # after terminate, a player command gets this long to exit before it is killed
# Generated two-note chime for other hosts.
CHIME_RATE = 22050
CHIME_NOTES_HZ = (880.0, 1318.5)   # A5 then E6
CHIME_NOTE_S = 0.15
CHIME_LEVEL = 0.2
TTS_WAIT_S = 15.0


def _script(ch: str) -> str:
    first = unicodedata.name(ch, "UNKNOWN").split(" ")[0]
    return SCRIPT_FAMILIES.get(first, first)


def speakable_name(name: Any) -> str | None:
    """The name as stored when it may be spoken, else None (the chime still plays). Mixed scripts are
    refused so a look-alike letter from another script cannot slip a word past the vocabulary guard."""
    if not isinstance(name, str) or not name or len(name) > NAME_MAX_CHARS or violations(name):
        return None
    words = name.split(" ")
    if len(words) > NAME_MAX_WORDS or any(not w for w in words):
        return None
    scripts = set()
    for w in words:
        if not unicodedata.category(w[0]).startswith("L"):
            return None
        for ch in w:
            cat = unicodedata.category(ch)
            if cat.startswith("L"):
                scripts.add(_script(ch))
            elif not (cat.startswith("M") or ch in NAME_JOINERS):   # combining marks belong to their letter
                return None
    return name if len(scripts) == 1 else None


def phrase(name: str, direction: str | None = None) -> str | None:
    """"sounds like NAME", or "sounds like NAME, PHRASE" when a direction word is given (ADR-0065). The name
    always wins: past M-06's word budget, or when the guard refuses the full text, it is the name-only phrase.
    None only when the name-only phrase is refused too (then the chime plays alone)."""
    alone = PHRASE_PREFIX + name
    where = DIRECTION_PHRASES.get(direction) if direction is not None else None
    if where is not None:
        text = f"{alone}, {where}"
        if len(text.split()) <= MAX_WORDS:
            try:
                return check_text(text)
            except VocabularyViolation:
                log.info("announcement direction dropped: refused by the vocabulary guard")
    try:
        return check_text(alone)
    except VocabularyViolation:
        return None


class Player(Protocol):
    def chime(self) -> None: ...
    def say(self, text: str) -> None: ...


class NullPlayer:
    """Plays nothing; records what it was asked to play (tests, and announce = off)."""

    def __init__(self) -> None:
        self.played: list[tuple[str, str]] = []

    def chime(self) -> None:
        self.played.append(("chime", ""))

    def say(self, text: str) -> None:
        self.played.append(("say", text))


class MacPlayer:
    """macOS: the Glass system sound through afplay, then the phrase through say. The phrase always
    starts with "sounds like", so a name can never be read as a say option.

    chime() and say() start the command with Popen and wait for it to finish (or its timeout: then it is
    ended and TimeoutExpired raised). stop() (the output arbiter's barge-in) ends the sound playing now:
    terminate, then kill after STOP_GRACE_S. A stopped sound returns quietly; another non-zero exit raises
    CalledProcessError."""

    def __init__(self, popen: Callable[..., Any] = subprocess.Popen):
        self.popen = popen
        self._lock = threading.Lock()
        self._proc: Any = None          # the command playing now
        self._stopped: Any = None       # the command stop() ended

    def chime(self) -> None:
        self._play([AFPLAY, CHIME_SOUND], CHIME_TIMEOUT_S)

    def say(self, text: str) -> None:
        self._play([SAY, text], SAY_TIMEOUT_S)

    def _play(self, argv: list[str], timeout_s: float) -> None:
        with self._lock:
            p = self.popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._proc = p
        try:
            rc = p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _end(p)
            raise
        finally:
            with self._lock:
                if self._proc is p:
                    self._proc = None
                stopped, self._stopped = self._stopped is p, None
        if rc != 0 and not stopped:
            raise subprocess.CalledProcessError(rc, argv)

    def stop(self) -> None:
        """Barge-in: end the sound playing now, if any. Never blocks longer than twice STOP_GRACE_S."""
        with self._lock:
            p = self._proc
            if p is None:
                return
            self._stopped = p
        _end(p)


def _end(p: Any) -> None:
    """Terminate a player command, then kill it when it has not exited after STOP_GRACE_S."""
    if p.poll() is not None:
        return
    p.terminate()
    try:
        p.wait(timeout=STOP_GRACE_S)
    except subprocess.TimeoutExpired:
        log.warning("player command ignored terminate for %.1fs; killing it", STOP_GRACE_S)
        p.kill()
        p.wait(timeout=STOP_GRACE_S)


def chime_tone(rate: int = CHIME_RATE) -> np.ndarray:
    """Two short notes, each with a smooth envelope, float32 in [-CHIME_LEVEL, CHIME_LEVEL]."""
    n = int(round(CHIME_NOTE_S * rate))
    t = np.arange(n) / rate
    notes = [CHIME_LEVEL * np.hanning(n) * np.sin(2 * np.pi * f * t) for f in CHIME_NOTES_HZ]
    return np.concatenate(notes).astype(np.float32)


class GenericPlayer:
    """Other hosts and the glasses: the generated chime through sounddevice (when present) and the
    phrase through the M-06 `tts` port (FliteTts, when installed). Without a tts, the chime only."""

    def __init__(self, sd: Any = None, tts: Any = None, rate: int = CHIME_RATE):
        self.sd, self.tts, self.rate = sd, tts, rate
        self._tone = chime_tone(rate)

    def chime(self) -> None:
        if self.sd is not None:
            self.sd.play(self._tone, self.rate, blocking=True)

    def say(self, text: str) -> None:
        if self.tts is None:
            return
        self.tts.speak(text)
        wait = getattr(self.tts, "wait", None)
        if wait is not None:
            wait(TTS_WAIT_S)

    def stop(self) -> None:
        """Barge-in (the output arbiter): stop the phrase now."""
        if self.tts is not None:
            self.tts.stop()


def generic_player() -> GenericPlayer:
    """Whatever this host has: sounddevice for the chime, Flite for the phrase. Neither is required."""
    sd = tts = None
    try:
        sd = importlib.import_module("sounddevice")
    except (ImportError, OSError) as e:   # OSError: the PortAudio library itself is missing
        log.info("announce: no chime (sounddevice unavailable: %s)", e)
    try:
        from ..voice import FliteTts
        tts = FliteTts()
    except RuntimeError as e:
        log.info("announce: name not spoken (%s); chime only", e)
    return GenericPlayer(sd, tts)


def platform_player() -> Player:
    return MacPlayer() if sys.platform == "darwin" else generic_player()


def play(player: Player, text: str | None) -> bool:
    """Play the chime, then the phrase when given. Synchronous; failures logged. True when it all played."""
    try:
        player.chime()
        if text:
            player.say(text)
        return True
    except Exception as e:  # noqa: BLE001 - a missing sound device or a failed command must not stop listening
        log.warning("announcement could not be played: %s", e)
        return False


class Announcer:
    """Standalone (no arbiter): its own bounded queue and worker play the jobs. With an OutputArbiter
    (oiris/modules/output.py, as live mode builds it): the chime and phrase go through the arbiter at voice_link
    priority, which plays them on its player, after or before other outputs by priority and under the motion
    gate; playing() is the arbiter's. The name rules and the per-person repeat limit stay here either way."""

    def __init__(self, mode: str = CHIME_AND_NAME, repeat_s: float = DEFAULT_REPEAT_S, player: Player | None = None,
                 clock: Callable[[], float] = time.monotonic, queue_jobs: int = QUEUE_JOBS, tail_s: float = TAIL_S,
                 arbiter: Any = None):
        if mode not in MODES:
            raise ValueError(f"announce must be one of {list(MODES)}")
        self.mode, self.repeat_s, self.clock, self.tail_s = mode, float(repeat_s), clock, float(tail_s)
        self.arbiter = arbiter
        if player is None:
            player = arbiter.player if arbiter is not None else NullPlayer()
        self.player: Player = player
        self._q: queue.Queue = queue.Queue(maxsize=queue_jobs)
        self._lock = threading.Lock()
        self._last: dict[str, float] = {}       # person_id -> last announced (RAM only)
        self._busy = False
        self._quiet_at = float("-inf")          # playing() stays true until this clock time
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        if mode != OFF and arbiter is None:
            self._worker = threading.Thread(target=self._run, name="oiris-announce", daemon=True)
            self._worker.start()

    def announce(self, person_id: str, name: str, direction: str | None = None) -> bool:
        """Queue one announcement when allowed. False when off, closed, within repeat_s of the last one
        for this person (whatever the direction), or the queue is full (with an arbiter: when it did not play
        or queue it). direction: a direction word (ADR-0065), spoken only after a speakable name."""
        if self.mode == OFF or self._stop.is_set():
            return False
        now = self.clock()
        text = None
        if self.mode == CHIME_AND_NAME:
            ok = speakable_name(name)
            text = phrase(ok, direction) if ok else None
        with self._lock:
            last = self._last.get(person_id)
            if last is not None and now - last < self.repeat_s:
                return False
            if self.arbiter is not None:
                return self._via_arbiter(person_id, text, now)
            try:
                self._q.put_nowait(text)
            except queue.Full:
                log.info("announcement dropped: queue full")
                return False
            self._last[person_id] = now
        return True

    def _via_arbiter(self, person_id: str, text: str | None, now: float) -> bool:
        """Caller holds _lock. A dropped or suppressed request is not a repeat: the next match may announce."""
        from ..output import PLAYED, QUEUED, VOICE_LINK
        d = self.arbiter.request("voice_link", VOICE_LINK, f"voice_link:{person_id}", text=text, sound="chime",
                                 person_id=person_id)
        if d.outcome not in (PLAYED, QUEUED):
            log.info("announcement %s: %s", d.outcome, d.reason)
            return False
        self._last[person_id] = now
        return True

    def playing(self) -> bool:
        """True while a job waits or plays, and for tail_s after the last one ended."""
        if self.arbiter is not None:
            return bool(self.arbiter.playing())
        with self._lock:
            return self._busy or not self._q.empty() or self.clock() < self._quiet_at

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                text = self._q.get(timeout=WORKER_POLL_S)
            except queue.Empty:  # python-brain: allow SILENT_EXCEPTION - a poll timeout; the stop flag is checked again
                continue
            if self._stop.is_set():
                break
            with self._lock:
                self._busy = True
            try:
                play(self.player, text)
            finally:
                with self._lock:
                    self._busy = False
                    self._quiet_at = self.clock() + self.tail_s

    def close(self, join_s: float = CLOSE_JOIN_S) -> None:
        """Stop the worker and drop queued jobs. A sound already playing is stopped when the player has
        stop(); otherwise it ends at its own timeout. With an arbiter the player is the arbiter's: not touched."""
        self._stop.set()
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:  # python-brain: allow SILENT_EXCEPTION - an empty queue is the normal end of the drain
                break
        stop = getattr(self.player, "stop", None)
        if self._worker is not None and stop is not None:
            try:
                stop()
            except Exception as e:  # noqa: BLE001 - closing must go on even when the player cannot stop
                log.warning("announce player stop failed: %s", e)
        if self._worker is not None:
            self._worker.join(join_s)
            if self._worker.is_alive():
                log.warning("announce worker still playing %.1fs after close", join_s)


__all__ = ["Announcer", "MacPlayer", "GenericPlayer", "NullPlayer", "Player", "speakable_name", "phrase", "play",
           "platform_player", "generic_player", "chime_tone", "MODES", "CHIME_AND_NAME", "CHIME", "OFF", "TAIL_S"]
