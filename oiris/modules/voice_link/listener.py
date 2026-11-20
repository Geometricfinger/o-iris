"""Live listening for M-17 (ADR-0057, docs/VOICE_LINK_MODULE_SPEC.md "Live listening").

Microphone blocks go into a bounded RAM queue. A worker thread cuts them into utterances with an energy
gate over a slowly adapting noise floor: an utterance starts at the first loud frame (with a short
pre-roll) and ends after END_QUIET_S of quiet or at MAX_UTTERANCE_S. Each utterance goes to
`vm.process_audio` under `process_lock`; the core does its own VAD, matching and logging.

RAM only: every block, frame and utterance buffer is zeroed after use. A recognised (heard) person
becomes a card kept for `vm.cfg.card_seconds`; only heard people are counted. Unknown voices and close
calls leave no trace here.

Ambient mode listens continuously. Push-to-talk opens the microphone only for `listen_for(seconds)`.

Announcing (ADR-0058): `on_heard` gets the card plus the person_id (live mode passes it to the Announcer).
While `paused()` is true (the announcer is playing, plus a short tail) every incoming block is zeroed and
dropped, so O Iris never hears and matches its own announcement.

Direction (ADR-0065): with a VoiceDirection (live mode, a rig with a mic_array mount) the capture delivers (n, N)
blocks. The VAD, embedding and match use the reference channel REF_CHANNEL only. A parallel RAM-only
multi-channel buffer and the per-frame speech flags follow the mono buffer's lifecycle and are zeroed after every
utterance, matched or not. The direction is computed only after a clear match (res["heard"]); never for unknown
voices, close calls, self, a spoof-vetoed match or while paused. It goes on the card and to on_heard, and is never
written: the heard log is the core's, which never sees it.

Public control for live mode: pause_mic() / resume_mic() close and reopen the capture (the capture interlock,
ADR-0063), and set_duty(duty, period_s) / set_matching(on) apply the power governor's plan (ADR-0061). Duty and
matching are checked only when an utterance would start: while either is off no new utterance starts, and an
utterance already being heard is finished and matched, never cut. The microphone itself stays as it was.

Push-to-talk counts microphone time: a block dropped because an announcement is playing still uses up the window,
so the window always ends after the seconds asked for, however the blocks and the announcement interleave.

Timing (oiris/timing.py, docs/LIVE.md "Timing"), numbers only, in the StageTimer passed in: "queue_wait" (a block
queued until the worker takes it), "decision" (utterance end to process_audio returning: copy, lock wait and the
core), "lock_wait" (waiting for process_lock), "process" (the whole process_audio call), and "embed" / "spoof"
(the speaker model calls inside it, timed through a transparent wrapper on vm.speaker while listening). The
core's VAD is a direct call inside core.py and is not timed separately: it is inside "process"."""
from __future__ import annotations
import logging, queue, threading, time
from collections import deque
from typing import Any, Callable

import numpy as np

from ...power.duty import DUTY_FULL, ListenDuty
from ...timing import StageTimer
from .capture import SILENCE_NOTICE, CaptureUnavailable, DigitalSilence

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_S = 0.030               # gate frame
PRE_ROLL_S = 0.21             # frames kept before the first loud one, so a soft onset is not cut
END_QUIET_S = 0.7             # this much quiet ends an utterance
MAX_UTTERANCE_S = 8.0         # a longer stretch of speech is cut here and continues as a new utterance
MIN_VOICED_S = 0.3            # utterances with less gated speech (clicks, a cough) are dropped unprocessed
QUEUE_BLOCKS = 100            # bounded RAM queue (about 10 s of 0.1 s blocks); a full queue drops blocks
FLOOR_FALL = 0.2              # noise floor follows quieter frames quickly ...
FLOOR_RISE = 0.02             # ... and louder non-speech frames slowly
FLOOR_RISE_IN_SPEECH = 0.002  # very slowly inside an utterance, so a new steady noise cannot hold the gate open
POWER_EPS = 1e-12             # keeps log10 finite on digital silence
DB_PER_DECADE = 10.0          # power ratio to dB
PTT_DEFAULT_S = 8.0           # push-to-talk: listen this long per press
PTT_MAX_S = 60.0
WORKER_POLL_S = 0.2
STOP_JOIN_S = 5.0
SCORE_DECIMALS, SECONDS_DECIMALS = 2, 1
DUTY_PERIOD_DEFAULT_S = 10.0  # the full duty's period (it never matters at duty 1); the power plans' default
# Matching uses one channel of a microphone array. Averaging the channels would add copies of the voice delayed
# by up to a millisecond between microphones, which smears its spectrum (a comb filter) and changes the embedding.
REF_CHANNEL = 0
AMBIENT, PUSH_TO_TALK = "ambient", "push_to_talk"
LISTENING, IDLE, UNAVAILABLE = "listening", "idle", "unavailable"
_WAKE = None                  # queue marker: wakes the worker at stop


def plain_capture_reason(e: BaseException) -> str:
    """A plain reason for a microphone that cannot be opened (no exception class names)."""
    if isinstance(e, CaptureUnavailable):
        return str(e)
    if isinstance(e, ValueError):
        return str(e)
    return f"the microphone could not be opened ({e})"


class _TimedSpeaker:
    """Wraps the voice module's SlotManager so the listener can time the speaker model calls without changing
    core.py. Everything else passes straight through; results and exceptions are unchanged."""

    def __init__(self, inner: Any, timer: StageTimer):
        self._inner, self._timer = inner, timer

    def speaker_embedding(self, audio: Any, sr: int) -> Any:
        with self._timer.time("embed"):
            return self._inner.speaker_embedding(audio, sr)

    def spoof_score(self, audio: Any, sr: int) -> Any:
        with self._timer.time("spoof"):
            return self._inner.spoof_score(audio, sr)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class VoiceListener:
    def __init__(self, vm: Any, capture: Any, process_lock: Any = None, place: str | None = None,
                 clock: Callable[[], float] = time.time, active: Callable[[], bool] = lambda: True,
                 on_heard: Callable[[dict], object] | None = None, mode: str | None = None,
                 queue_blocks: int = QUEUE_BLOCKS, paused: Callable[[], bool] = lambda: False,
                 timer: StageTimer | None = None, direction: Any = None,
                 duty_clock: Callable[[], float] = time.monotonic):
        """vm: a VoiceLinkModule. capture: start(callback)/stop()/sample_rate (capture.py).
        process_lock: held around every process_audio call (the live store lock). active: checked under
        that lock; False (store closed) makes processing a no-op. on_heard(card + person_id): called for
        each heard person (the announcement). paused(): True drops incoming blocks (announcement playing).
        timer: where the stage timings go (a fresh StageTimer when None). direction: a VoiceDirection (direction.py)
        for (n, direction.channels) blocks, or None (mono, no direction). duty_clock: monotonic seconds for the
        listening duty's on/off windows (set_duty)."""
        self.vm, self.capture, self.place, self.clock = vm, capture, place, clock
        self.process_lock = process_lock if process_lock is not None else threading.Lock()
        self.active, self.on_heard, self.paused = active, on_heard, paused
        self.dropped_while_paused = 0          # blocks zeroed because an announcement was playing
        self.timer = timer if timer is not None else StageTimer()
        self._speaker_wrapped: _TimedSpeaker | None = None
        self.mode = mode or getattr(vm.cfg, "listening", AMBIENT)
        self.sample_rate = int(getattr(capture, "sample_rate", SAMPLE_RATE))
        sr = self.sample_rate
        self._frame_len = max(1, int(round(FRAME_S * sr)))
        self._pre_frames = max(1, int(round(PRE_ROLL_S / FRAME_S)))
        self._end_quiet = max(1, int(round(END_QUIET_S / FRAME_S)))
        self._max_n = int(round(MAX_UTTERANCE_S * sr))
        self._min_voiced = max(1, int(round(MIN_VOICED_S / FRAME_S)))
        # RAM buffers, preallocated and zeroed after each use
        self._frame = np.zeros(self._frame_len, np.float32)
        self._fill = 0
        self._pre = np.zeros((self._pre_frames, self._frame_len), np.float32)
        self._pre_count = self._pre_head = 0
        self._buf = np.zeros(self._max_n + self._frame_len, np.float32)
        self._n = self._voiced = self._quiet = 0
        self._in_utt = False
        self._floor: float | None = None
        # Multi-channel twins of _frame, _pre and _buf plus the per-frame speech flags (direction only, RAM only).
        self.direction = direction
        self._channels = int(direction.channels) if direction is not None else 0
        self._mframe: np.ndarray | None = None
        self._mpre: np.ndarray | None = None
        self._mpre_ok: np.ndarray | None = None
        self._mbuf: np.ndarray | None = None
        self._speech: np.ndarray | None = None
        if direction is not None:
            c = self._channels
            self._mframe = np.zeros((self._frame_len, c), np.float32)
            self._mpre = np.zeros((self._pre_frames, self._frame_len, c), np.float32)
            self._mpre_ok = np.zeros(self._pre_frames, bool)
            self._mbuf = np.zeros((self._buf.size, c), np.float32)
            self._speech = np.zeros(self._buf.size // self._frame_len + 1, bool)
        self._mframe_ok = True                 # every sample of the current frame came with all channels
        self._utt_multi = True                 # ... and of every frame of the current utterance
        self._frames = 0                       # frames in the current utterance
        self._q: queue.Queue = queue.Queue(maxsize=queue_blocks)
        self._lock = threading.Lock()          # capture state, push-to-talk window, cards
        self._cap_lock = threading.Lock()      # serialises capture start/stop
        self._ptt_left = 0                     # push-to-talk samples still to accept
        self._capturing = False
        self._stopped = False
        self._started = False
        self._reason = ""                      # plain reason when unavailable
        self._silence = DigitalSilence(sr)
        self._cards: deque = deque()
        self.heard = 0                         # heard (recognised) utterances only
        self._worker: threading.Thread | None = None
        # Power plan (ADR-0061): checked only when an utterance would start.
        self.duty_clock = duty_clock
        self._duty = ListenDuty(DUTY_FULL, DUTY_PERIOD_DEFAULT_S)
        self._matching = True
        self.held_off = 0                      # speech frames that started no utterance (duty or matching off)

    # ---- lifecycle ------------------------------------------------------------------------
    def start(self) -> "VoiceListener":
        if self._started:
            return self
        self._started = True
        self._wrap_speaker()
        self._worker = threading.Thread(target=self._run, name="oiris-voice", daemon=True)
        self._worker.start()
        if self.mode == AMBIENT:
            self._open()
        return self

    def _wrap_speaker(self) -> None:
        inner = getattr(self.vm, "speaker", None)
        if inner is None or not hasattr(inner, "speaker_embedding") or isinstance(inner, _TimedSpeaker):
            return
        self._speaker_wrapped = self.vm.speaker = _TimedSpeaker(inner, self.timer)

    def _unwrap_speaker(self) -> None:
        w, self._speaker_wrapped = self._speaker_wrapped, None
        if w is None:
            return
        inner = w._inner
        if getattr(self.vm, "speaker", None) is w:   # only undo our own wrapper
            self.vm.speaker = inner

    def _open(self) -> bool:
        """Start the capture. False, with the reason kept, when it fails. The capture is started and
        stopped under _cap_lock only: its callback takes _lock, and a stream stop waits for the callback."""
        with self._cap_lock:
            with self._lock:
                if self._capturing or self._stopped:
                    return self._capturing
                self._silence.reset()
            try:
                self.capture.start(self._on_block)
            except Exception as e:  # noqa: BLE001 - any capture failure leaves voice unavailable, never raises
                with self._lock:
                    self._reason = plain_capture_reason(e)
                log.warning("voice capture unavailable: %s", self._reason)
                return False
            with self._lock:
                self._capturing, self._reason = True, ""
            return True

    def _close(self, only_if_over: bool = False) -> None:
        """Stop the capture. only_if_over: not while a push-to-talk window is still open (a new press)."""
        with self._cap_lock:
            with self._lock:
                if not self._capturing or (only_if_over and self._ptt_left > 0):
                    return
                self._capturing = False
            try:
                self.capture.stop()
            except Exception as e:  # noqa: BLE001 - stopping must finish; the reason is logged
                log.warning("voice capture stop failed: %s", e)

    def pause_mic(self) -> None:
        """Close the microphone (the capture interlock revoked it). An open push-to-talk window is kept."""
        self._close()

    def resume_mic(self) -> bool:
        """Reopen the microphone after pause_mic(). False, with the plain reason kept for status(), when it fails."""
        return self._open()

    def set_duty(self, duty: float, period_s: float) -> None:
        """Listen for the first duty * period_s of every period_s (ListenDuty). Raises ValueError when out of range."""
        d = ListenDuty(duty, period_s)
        with self._lock:
            self._duty = d

    def set_matching(self, on: bool) -> None:
        """False: no new utterance starts (the voice model is off); one being heard is still finished and matched."""
        with self._lock:
            self._matching = bool(on)

    def duty(self) -> dict:
        """{"duty", "period_s", "matching"}: the plan in force. Numbers only."""
        with self._lock:
            return {"duty": self._duty.duty, "period_s": self._duty.period_s, "matching": self._matching}

    def _may_start(self) -> bool:
        with self._lock:
            d, matching = self._duty, self._matching
        return matching and d.active(self.duty_clock())

    def listen_for(self, seconds: float = PTT_DEFAULT_S) -> bool:
        """Push-to-talk: open the microphone for `seconds` of audio (at most PTT_MAX_S). In ambient mode
        it changes nothing. Returns True when listening."""
        if self._stopped or not self._started:
            return False
        if self.mode == AMBIENT:
            return self._capturing
        n = int(round(min(max(float(seconds), 0.0), PTT_MAX_S) * self.sample_rate))
        with self._lock:
            self._ptt_left = max(self._ptt_left, n)
        if n <= 0:
            return False
        if not self._open():
            with self._lock:
                self._ptt_left = 0
            return False
        return True

    def stop(self, join_s: float = STOP_JOIN_S) -> None:
        """Stop capture, end the worker and zero every queued block. Nothing is processed after this."""
        with self._lock:
            self._stopped = True
            self._ptt_left = 0
        self._close()
        try:
            self._q.put_nowait(_WAKE)        # wake the worker
        except queue.Full:  # python-brain: allow SILENT_EXCEPTION - the worker polls _stopped every WORKER_POLL_S anyway
            pass
        if self._worker is not None:
            self._worker.join(join_s)
            if self._worker.is_alive():
                log.warning("voice worker still running %.1fs after stop; it zeroes its buffers on exit", join_s)
        self._drain()
        self._unwrap_speaker()
        with self._lock:
            self._cards.clear()

    def _drain(self) -> None:
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:  # python-brain: allow SILENT_EXCEPTION - an empty queue is the normal end of the drain
                return
            if item is not None:
                item[1].fill(0)

    # ---- capture thread -------------------------------------------------------------------
    def _on_block(self, block: np.ndarray) -> None:
        """Called by the capture with a copy it hands over. Never blocks; zeroes what it does not keep."""
        if self._is_paused():
            n = block.shape[0]                 # samples per channel (== size for mono)
            block.fill(0)
            with self._lock:
                self.dropped_while_paused += 1
                if self.mode == PUSH_TO_TALK:  # the window counts microphone time, dropped or kept
                    self._ptt_left = max(0, self._ptt_left - n)
            return
        with self._lock:
            if self._stopped:
                block.fill(0)
                return
            self._silence.feed(block[:, REF_CHANNEL] if block.ndim == 2 else block)
            if self.mode == PUSH_TO_TALK:
                if self._ptt_left <= 0:
                    block.fill(0)
                    return
                if block.shape[0] > self._ptt_left:   # shape[0]: samples per channel (== size for mono)
                    keep = block[:self._ptt_left].copy()
                    block.fill(0)
                    block = keep
                self._ptt_left -= block.shape[0]
        try:
            self._q.put_nowait((self.timer.now(), block))
        except queue.Full:  # python-brain: allow SILENT_EXCEPTION - a full queue drops the block (zeroed) so RAM stays bounded
            block.fill(0)

    def _is_paused(self) -> bool:
        try:
            return bool(self.paused())
        except Exception as e:  # noqa: BLE001 - a broken pause check must not stop listening; input goes on
            log.warning("pause check failed: %s", e)
            return False

    # ---- worker thread --------------------------------------------------------------------
    def _run(self) -> None:
        try:
            while not self._stopped:
                try:
                    item = self._q.get(timeout=WORKER_POLL_S)
                except queue.Empty:  # python-brain: allow SILENT_EXCEPTION - a poll timeout; the push-to-talk check below still runs
                    item = None
                if item is not None:
                    queued_at, block = item
                    try:
                        self.timer.since("queue_wait", queued_at)
                        self._feed(block)
                    finally:
                        block.fill(0)
                if self.mode == PUSH_TO_TALK and self._q.empty():
                    with self._lock:
                        over = self._capturing and self._ptt_left <= 0
                    if over:
                        self._end_window()
        finally:
            self._reset_buffers()
            self._drain()

    def _end_window(self) -> None:
        """Push-to-talk window over: the current utterance is processed and the microphone closed."""
        if self._in_utt and not self._stopped:
            self._emit()
        self._reset_buffers()
        self._close(only_if_over=True)

    def _reset_buffers(self) -> None:
        self._frame.fill(0); self._fill = 0
        self._pre.fill(0); self._pre_count = self._pre_head = 0
        self._buf.fill(0); self._n = self._voiced = self._quiet = 0
        self._in_utt = False
        self._zero_multi()

    def _zero_multi(self, n: int | None = None, frames: int | None = None) -> None:
        """Zero the multi-channel buffer and speech flags (the first n samples / frames, or everything,
        including the multi-channel frame and pre-roll)."""
        self._frames, self._utt_multi = 0, True
        if self._mbuf is None or self._speech is None:
            return
        self._mbuf[:n].fill(0)
        self._speech[:frames].fill(False)
        if n is None and self._mframe is not None and self._mpre is not None and self._mpre_ok is not None:
            self._mframe.fill(0); self._mpre.fill(0); self._mpre_ok.fill(False)
            self._mframe_ok = True

    def _feed(self, block: np.ndarray) -> None:
        multi = None
        if block.ndim == 2:
            x = block[:, REF_CHANNEL]
            if self._mframe is not None and block.shape[1] == self._channels:
                multi = block
        else:
            x = block.reshape(-1)
        i = 0
        while i < x.size and not self._stopped:
            take = min(self._frame_len - self._fill, x.size - i)
            self._frame[self._fill:self._fill + take] = x[i:i + take]
            if self._mframe is not None:
                if multi is not None:
                    self._mframe[self._fill:self._fill + take] = multi[i:i + take]
                else:
                    self._mframe_ok = False      # a mono block (the capture fell back): no direction for this frame
            self._fill += take
            i += take
            if self._fill == self._frame_len:
                self._on_frame(self._frame)
                self._frame.fill(0)
                self._fill = 0
                if self._mframe is not None:
                    self._mframe.fill(0)
                    self._mframe_ok = True

    def _gate(self, frame: np.ndarray) -> tuple[float, bool]:
        level = float(DB_PER_DECADE * np.log10(float(np.dot(frame, frame)) / frame.size + POWER_EPS))
        if self._floor is None:
            self._floor = level
        cfg = self.vm.cfg
        return level, level >= max(self._floor + cfg.vad_snr_db, cfg.vad_floor_dbfs)

    def _adapt(self, level: float, rate: float) -> None:
        assert self._floor is not None
        self._floor += (FLOOR_FALL if level < self._floor else rate) * (level - self._floor)

    def _on_frame(self, frame: np.ndarray) -> None:
        level, speech = self._gate(frame)
        if not self._in_utt:
            held = speech and not self._may_start()
            if held:
                self.held_off += 1             # power plan: no new utterance; the floor does not learn the speech
            if not speech or held:
                if not held:
                    self._adapt(level, FLOOR_RISE)
                self._pre[self._pre_head] = frame
                if self._mpre is not None and self._mpre_ok is not None:
                    self._mpre[self._pre_head] = self._mframe
                    self._mpre_ok[self._pre_head] = self._mframe_ok
                self._pre_head = (self._pre_head + 1) % self._pre_frames
                self._pre_count = min(self._pre_count + 1, self._pre_frames)
                return
            self._in_utt, self._n, self._voiced, self._quiet = True, 0, 0, 0
            self._frames, self._utt_multi = 0, True
            for k in range(self._pre_count):   # oldest first
                j = (self._pre_head - self._pre_count + k) % self._pre_frames
                mpre = self._mpre[j] if self._mpre is not None else None
                self._append(self._pre[j], False, mpre, self._mpre_ok is not None and bool(self._mpre_ok[j]))
            self._pre.fill(0); self._pre_count = self._pre_head = 0
            if self._mpre is not None and self._mpre_ok is not None:
                self._mpre.fill(0); self._mpre_ok.fill(False)
        else:
            self._adapt(level, FLOOR_RISE_IN_SPEECH)
        self._append(frame, speech, self._mframe, self._mframe_ok)
        if speech:
            self._voiced += 1
            self._quiet = 0
        else:
            self._quiet += 1
        if self._quiet >= self._end_quiet or self._n >= self._max_n:
            self._emit()

    def _append(self, frame: np.ndarray, speech: bool = False, mframe: np.ndarray | None = None,
                mframe_ok: bool = False) -> None:
        self._buf[self._n:self._n + frame.size] = frame
        if self._mbuf is not None and self._speech is not None:
            if mframe is not None and mframe_ok:
                self._mbuf[self._n:self._n + frame.size] = mframe
            else:
                self._utt_multi = False
            self._speech[self._frames] = speech
            self._frames += 1
        self._n += frame.size

    def _emit(self) -> None:
        n, voiced, frames, multi = self._n, self._voiced, self._frames, self._utt_multi
        self._in_utt, self._n, self._voiced, self._quiet = False, 0, 0, 0
        try:
            if voiced < self._min_voiced:
                self._buf[:n].fill(0)
                return
            ended = self.timer.now()
            utt = self._buf[:n].copy()
            self._buf[:n].fill(0)
            try:
                self._process(utt, ended, (n, frames) if multi and self._mbuf is not None else None)
            finally:
                utt.fill(0)
        finally:
            self._zero_multi(n, frames)        # matched or not, the multi-channel buffer and flags are zeroed

    def _direction(self, n: int, frames: int) -> str | None:
        """The direction word of the utterance just matched (its first n samples / frames), or None.
        A failure means no direction, never a stopped listener."""
        if self.direction is None or self._mbuf is None or self._speech is None:
            return None
        mask = np.repeat(self._speech[:frames], self._frame_len)[:n]
        try:
            word = self.direction.estimate(self._mbuf[:n].T, mask)
        except Exception as e:  # noqa: BLE001 - a direction is an extra; the announcement goes ahead without it
            log.warning("voice direction failed: %s", e)
            return None
        finally:
            mask.fill(False)
        return str(word) if word else None

    def _process(self, utt: np.ndarray, ended: float | None = None, multi: tuple[int, int] | None = None) -> None:
        """multi: (samples, frames) of the multi-channel buffer holding this whole utterance, or None."""
        t = self.timer
        t0 = t.now()
        with self.process_lock:
            t.since("lock_wait", t0)
            if self._stopped or not self.active():
                return
            now = self.clock()
            try:
                with t.time("process"):
                    res = self.vm.process_audio(utt, self.sample_rate, self.place, now)
                if ended is not None:
                    t.since("decision", ended)
            except PermissionError as e:       # VoiceLinkLocked: turned off or opt-in gone meanwhile
                self._reason = f"voice link is off ({e})"
                log.warning("voice link stopped matching: %s", e)
                return
            except Exception as e:  # noqa: BLE001 - one bad utterance must not end listening; logged without audio
                log.error("voice utterance failed: %s", e)
                return
        heard = res.get("heard")
        if not heard:
            return                              # unknown voice or close call: no trace, not counted
        # Only now, after a clear match of an enrolled person (ADR-0065). Shown and spoken, never stored.
        direction = self._direction(*multi) if multi is not None else None
        card = {"name": str(heard["name"]), "score": round(float(heard["score"]), SCORE_DECIMALS),
                "face_agrees": bool(heard["face_agrees"]), "direction": direction, "at": now,
                "person_id": str(heard["person_id"])}
        with self._lock:
            self._cards.append(card)
            self.heard += 1
        if self.on_heard is not None:
            try:
                self.on_heard(dict(card))
            except Exception as e:  # noqa: BLE001 - a hook must not stop listening
                log.warning("on_heard hook failed: %s", e)

    # ---- state ----------------------------------------------------------------------------
    def cards(self, now: float | None = None, with_person: bool = False) -> list[dict]:
        """Heard cards younger than card_seconds, newest first, with seconds_ago. Older ones are dropped.
        with_person: each card also carries its person_id (live mode's photo lookup; never shown)."""
        now = self.clock() if now is None else now
        keep = float(self.vm.cfg.card_seconds)
        with self._lock:
            while self._cards and now - self._cards[0]["at"] > keep:
                self._cards.popleft()
            live = list(self._cards)
        out = []
        for c in reversed(live):
            card = {"name": c["name"], "score": c["score"], "face_agrees": c["face_agrees"], "direction": c["direction"],
                    "seconds_ago": round(max(0.0, now - c["at"]), SECONDS_DECIMALS)}
            if with_person:
                card["person_id"] = c["person_id"]
            out.append(card)
        return out

    def status(self, now: float | None = None, with_person: bool = False) -> dict:
        """{"state", "reason", "mode", "notice", "listen_left_s", "cards", "heard"} for the live snapshot.
        with_person: as for cards()."""
        health, why = self.vm.health()
        with self._lock:
            capturing, left, reason, silent = self._capturing, self._ptt_left, self._reason, self._silence.silent
        if self._stopped:
            state, reason = UNAVAILABLE, "stopped"
        elif health == UNAVAILABLE:
            state, reason = UNAVAILABLE, f"the speaker model is unavailable ({why})"
        elif reason:
            state = UNAVAILABLE
        elif self.mode == PUSH_TO_TALK:
            state = LISTENING if capturing and left > 0 else IDLE
        else:
            state = LISTENING if capturing else UNAVAILABLE
        return {"state": state, "reason": reason, "mode": self.mode,
                "notice": SILENCE_NOTICE if silent and capturing else "",
                "listen_left_s": round(left / self.sample_rate, SECONDS_DECIMALS) if self.mode == PUSH_TO_TALK else 0.0,
                "cards": self.cards(now, with_person), "heard": self.heard}


__all__ = ["VoiceListener", "REF_CHANNEL", "PTT_DEFAULT_S", "MAX_UTTERANCE_S", "END_QUIET_S", "AMBIENT", "PUSH_TO_TALK"]
