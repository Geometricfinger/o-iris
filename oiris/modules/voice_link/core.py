"""M-17 module core (ADR-0057, docs/VOICE_LINK_MODULE_SPEC.md): voice enrollment, matching against
enrolled voices only, "heard" review entries, retention and verified wipe.

- The speaker model is a `speaker_embed` slot plugin, always called through `SlotManager` (fault
  isolation: a failing model makes the voice link "unavailable" with the reason; nothing else breaks).
- A voice template is the unit mean of several unit speaker embeddings, sealed with the person's own
  M-16 data key (AAD b"voice:" + person_id). It is valid only with the model that made it.
- Every audio buffer received is zeroed after use, and so is every converted copy and embedding.
  Nothing is written except the sealed template and "heard" entries of enrolled people. Unknown
  voices are dropped at once: not stored, not counted.
- Capture (ambient or push-to-talk) is the caller's job; the core processes what it is given.
- One enrolled person can be marked as "self" (vl_state key "self"; ADR-0058 decision 8): a clear match of
  them is dropped like an unknown voice. A configured `voice_antispoof` slot can veto a clear match
  (decision 9); without that slot matching goes ahead.
- Enrollment windows may name the designed sentence read (prompts.py). The pitch of each accepted window
  is kept in RAM for an advisory range check only: never stored, never written to the chain log, never
  logged, and dropped with the capture."""
from __future__ import annotations
import logging, time, uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ...config import VOICE_ENROLL_MAX_WINDOWS
from ...config import VOICE_LINK_DEFAULTS as D
from ...config import validate_voice_link
from ...guard.consent import ConsentError
from ...guard.privacy import check_persist
from ...guard.vocabulary import check_output, check_text
from ...slots import SlotManager
from ..face_match.core import VOICE_AAD, WipeReport
from ..face_match.optin import opt_in_record as face_opt_in_record
from ..face_match.port import unit
from . import pitch, prompts, vad
from .optin import log_change, opt_in_record

log = logging.getLogger(__name__)

SLOT = "speaker_embed"
SPOOF_SLOT = "voice_antispoof"
SELF_KEY = "self"
SAMPLE_RATE_MIN, SAMPLE_RATE_MAX = 8000, 48000
CLIP_LEVEL = 0.99            # |x| at or above this counts as clipped
CLIP_FRACTION_MAX = 0.01     # more than 1% clipped samples => "too loud"
INT16_SCALE = 32768.0
SCORE_DECIMALS = 4
SECONDS_DECIMALS, LEVEL_DECIMALS, STORED_SPEECH_DECIMALS = 2, 1, 1
MINUTE_S, DAY_S = 60, 86400
VOICE_SCOPE, GUARDIAN_SCOPE = "voice_enrollment", "guardian"
REENROLL_NOTE = "re-enroll for the current model"
RANGE_HIGH_TONES, RANGE_LOW_TONES = ("high", "rising"), ("low",)
REDO_CUE = "a bit higher and livelier"
PITCH_DECIMALS = 0
SHORT_SPEECH_SHOWN_MIN_S = 0.1   # "still needed" never reads 0.0 s while the target is not met
REASONS = {
    "rate": f"the sample rate must be {SAMPLE_RATE_MIN} to {SAMPLE_RATE_MAX} Hz",
    "mono": "the audio must be one channel (mono)",
    "invalid": "the audio is empty or not valid numbers",
    "loud": "too loud: move back from the microphone or speak more softly",
    "speech": "not enough speech: keep speaking naturally for the whole window",
    "backend": "the voice model is unavailable, try again",
    "enough": "enough windows for this person, finish to save",
    "prompt": "unknown enrollment sentence",
    "ok": "window accepted",
}


class VoiceLinkLocked(PermissionError):
    pass


class VoiceLinkUnavailable(RuntimeError):
    pass


@dataclass
class VoiceLinkConfig:
    """Mirrors oiris.config.VOICE_LINK_DEFAULTS (validated by the same validate_voice_link); a test keeps
    the key sets in sync."""
    listening: str = D["listening"]
    match_threshold: float = D["match_threshold"]   # raw cosine; PROVISIONAL until the live check
    unsure_margin: float = D["unsure_margin"]
    min_speech_s: float = D["min_speech_s"]         # PROVISIONAL: from the first live check (oiris/config.py)
    enroll_windows: int = D["enroll_windows"]
    enroll_window_s: float = D["enroll_window_s"]
    enroll_min_speech_s: float = D["enroll_min_speech_s"]
    enroll_min_total_speech_s: float = D["enroll_min_total_speech_s"]
    enroll_range_min_rise: float = D["enroll_range_min_rise"]
    heard_retention_days: int = D["heard_retention_days"]
    agree_window_s: float = D["agree_window_s"]
    card_seconds: float = D["card_seconds"]
    vad_snr_db: float = D["vad_snr_db"]
    vad_floor_dbfs: float = D["vad_floor_dbfs"]
    speaker_model: str = D["speaker_model"]
    model_dir: str = D["model_dir"]
    input_device: Any = D["input_device"]
    allow_test_backend: bool = D["allow_test_backend"]   # tests only; the test speaker plugin is refused otherwise
    announce: str = D["announce"]                   # ADR-0058: chime_and_name | chime | off
    announce_repeat_s: float = D["announce_repeat_s"]
    spoof_threshold: float = D["spoof_threshold"]   # used only when a voice_antispoof slot is set

    def validate(self) -> "VoiceLinkConfig":
        errs = validate_voice_link(asdict(self), allow_test_backend=True)
        if errs:
            raise ValueError("; ".join(errs))
        return self

    @classmethod
    def from_dict(cls, d: dict | None) -> "VoiceLinkConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"voice_link: unknown keys {sorted(unknown)}")
        return cls(**d).validate()


def default_model_dir(data_dir: str | Path, vcfg: VoiceLinkConfig) -> str:
    return vcfg.model_dir or str(Path(data_dir).expanduser() / "models" / "voice_link")


def speaker_slots(cfg: Any, vcfg: VoiceLinkConfig, registry: Any) -> SlotManager:
    """The speaker plugin is chosen as M-16 chooses its backend: plugins.slots.speaker_embed, else
    voice_link.speaker_model. Its options are plugins.options[name]; model_dir defaults to
    voice_link.model_dir or <data_dir>/models/voice_link. plugins.slots.voice_antispoof, when set, is
    selected too (ADR-0058 decision 9), with its options from plugins.options."""
    plugins = getattr(cfg, "plugins", {}) or {}
    chosen = plugins.get("slots", {}) or {}
    options = plugins.get("options", {}) or {}
    name = chosen.get(SLOT) or vcfg.speaker_model
    opts = dict(options.get(name, {}))
    opts.setdefault("model_dir", default_model_dir(cfg.data_dir, vcfg))
    slots, all_opts = {SLOT: name}, {name: opts}
    spoof = chosen.get(SPOOF_SLOT)
    if spoof:
        slots[SPOOF_SLOT] = spoof
        all_opts.setdefault(spoof, dict(options.get(spoof, {})))
    return SlotManager(registry, slots, all_opts)


@dataclass
class WindowResult:
    ok: bool
    code: str
    reason: str
    done: bool
    needed: int
    speech_s: float = 0.0
    level_dbfs: float | None = None
    pitch_hz: float | None = None        # median pitch of the window's speech; None when unvoiced (RAM only)
    pitch_low_hz: float | None = None    # 10th percentile
    pitch_high_hz: float | None = None   # 90th percentile
    total_speech_s: float = 0.0          # speech over all accepted windows so far


@dataclass
class RangeReport:
    """Advisory: never blocks finish_voice. `ok` is True when the range is wide enough or not measured."""
    ok: bool
    measured: bool
    message: str
    low_hz: float | None = None          # median pitch of the "low" windows
    high_hz: float | None = None         # median pitch of the "high" and "rising" windows
    redo_prompt: str | None = None       # prompt id to record again, when not ok
    redo_cue: str = ""


@dataclass
class _Mark:
    """What the range check needs from one accepted window. RAM only."""
    prompt_id: str | None
    tone: str | None
    pitch_hz: float | None
    speech_s: float


@dataclass
class _VoiceCapture:
    person_id: str
    consent_id: str
    embeddings: list = field(default_factory=list)   # unit embeddings, RAM only
    marks: list = field(default_factory=list)        # one _Mark per accepted window, same order; RAM only
    speech_s: float = 0.0

    def wipe(self) -> None:
        for e in self.embeddings:
            e.fill(0.0)
        self.embeddings.clear()
        self.marks.clear()


def _prepare(audio: Any) -> tuple[np.ndarray | None, str]:
    """Mono float32 view or converted copy of `audio`, or (None, refusal code)."""
    a = audio if isinstance(audio, np.ndarray) else np.asarray(audio)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a.reshape(-1)
    if a.ndim != 1:
        return None, "mono"
    if a.dtype == np.float32:
        x = a
    elif np.issubdtype(a.dtype, np.floating):
        x = a.astype(np.float32)
    elif a.dtype == np.int16:
        x = a.astype(np.float32)
        x /= INT16_SCALE
    else:
        return None, "invalid"
    if x.size == 0 or not np.all(np.isfinite(x)):
        return None, "invalid"
    return x, ""


HEARD, UNSURE, NONE, SELF, SPOOFED = "heard", "unsure", "none", "self", "spoofed"


def decide(scores: list[tuple[float, str]], match_threshold: float, unsure_margin: float, self_id: str | None,
           spoof_score: float | None | Callable[[], float | None], spoof_threshold: float) -> tuple[str, str | None]:
    """The matching decision, pure (ported to android/core VoiceLink.kt; vectors tests/vectors/voice_link.json).
    `scores`: (cosine, person_id) per enrolled voice, any order; ties rank the larger person_id first.
    In order: best below match_threshold => none; a second voice within unsure_margin => unsure; the best is
    self => self; spoof score at or above spoof_threshold => spoofed; else heard with the best person.
    `spoof_score` None means no anti-spoof slot (or a failed one): the match goes ahead. A callable is
    called only when the decision reaches the spoof step."""
    if not scores:
        return NONE, None
    ranked = sorted(scores, reverse=True)
    best, pid = ranked[0]
    if best < match_threshold:
        return NONE, None
    if len(ranked) > 1 and best - ranked[1][0] < unsure_margin:
        return UNSURE, None
    if pid == self_id:
        return SELF, None
    s = spoof_score() if callable(spoof_score) else spoof_score
    if s is not None and s >= spoof_threshold:
        return SPOOFED, None
    return HEARD, pid


def _zero(audio: Any, x: np.ndarray | None) -> None:
    """Zero the caller's buffer in place (when writable) and any converted copy."""
    if isinstance(audio, np.ndarray) and audio.flags.writeable:
        audio.fill(0)
    if x is not None and x.flags.writeable:
        x.fill(0)


def _clipped(x: np.ndarray) -> bool:
    n = np.count_nonzero(x >= CLIP_LEVEL) + np.count_nonzero(x <= -CLIP_LEVEL)   # boolean masks: no audio copy
    return bool(n > CLIP_FRACTION_MAX * x.size)


class VoiceLinkModule:
    def __init__(self, db, face: Any, speaker: SlotManager, cfg: VoiceLinkConfig | None = None, chain=None,
                 clock: Callable[[], float] = time.time):
        self.cfg = (cfg or VoiceLinkConfig()).validate()
        self.db, self.face, self.speaker, self.chain, self.clock = db, face, speaker, chain, clock
        db.execute("PRAGMA secure_delete=ON")
        if opt_in_record(db) is None:
            raise VoiceLinkLocked("M-17 is off or not opted in; enable it with the typed opt-in first")
        self._require_face()
        self.listening = self.cfg.listening   # for the capture adapter (a later task); the core takes any buffer
        inst = speaker.get(SLOT)
        if inst is not None and getattr(inst, "test_only", False) and not self.cfg.allow_test_backend:
            raise VoiceLinkLocked("the deterministic test speaker plugin is test-only")
        self.consents = face.consents
        self._captures: dict[str, _VoiceCapture] = {}
        self._templates: list[tuple[str, np.ndarray]] | None = None
        self._tpl_consent: dict[str, str] = {}
        self._degraded = ""
        self._off = False
        self.stats = {"heard": 0}   # unknown voices and close calls are deliberately not counted
        self.consents.withdraw_hooks.append(self._on_consent_withdrawn)
        self._purge_revoked()

    # ---- gating -------------------------------------------------------------------------
    def _require_face(self) -> None:
        if self.face is None or face_opt_in_record(self.db) is None:
            raise VoiceLinkLocked("voice link needs face matching (M-16) opted in and loaded")
        if "personal" not in getattr(self.face, "modes", set()):
            raise VoiceLinkLocked("voice link needs face matching in personal mode")

    def _require_on(self) -> None:
        if self._off or opt_in_record(self.db) is None:
            raise VoiceLinkLocked("M-17 is off or not opted in")
        self._require_face()

    # ---- speaker slot -------------------------------------------------------------------
    def _model(self) -> Any | None:
        return self.speaker.get(SLOT)

    @property
    def model_id(self) -> str | None:
        m = self._model()
        return None if m is None else str(getattr(m, "model_id", m.name))

    @property
    def dim(self) -> int | None:
        m = self._model()
        return None if m is None else int(m.dim)

    def health(self) -> tuple[str, str]:
        """listening | degraded | unavailable, with the speaker slot's reason."""
        if self._model() is None:
            return "unavailable", self.speaker.status.get(SLOT, "empty")
        if self._degraded:
            return "degraded", self._degraded
        return "listening", ""

    def _embed(self, speech: np.ndarray, sample_rate: int) -> np.ndarray | None:
        v = self.speaker.speaker_embedding(speech, sample_rate)
        if v is None:
            self._degraded = self.speaker.status.get(SLOT, "speaker model failed")
            return None
        self._degraded = ""
        return v

    # ---- consent (Forge F3: rechecked whenever templates load) --------------------------
    def _consent_revoked(self, consent_id: str | None) -> bool:
        """True when a known consent record is withdrawn, expired or lacks voice_enrollment. Ids this
        registry has never seen are kept (listed for review), as M-16 does."""
        if not consent_id or self.consents.get(consent_id) is None:
            return False
        try:
            self.consents.require(consent_id, VOICE_SCOPE, now=self.clock())
        except ConsentError:
            return True
        return False

    def _purge_revoked(self) -> int:
        gone = [pid for pid, cid in self.db.execute("SELECT person_id, consent_id FROM vl_voice").fetchall()
                if self._consent_revoked(cid)]
        for pid in gone:
            rep = self.delete_voice(pid, kind="consent_withdrawn")
            if not rep.verified:
                log.error("voice wipe after consent withdrawal NOT verified")
        return len(gone)

    def _on_consent_withdrawn(self, consent_id: str) -> None:
        try:
            self._purge_revoked()
        except Exception as e:  # noqa: BLE001 - a hook must not break the withdrawing caller
            log.error("voice purge after consent withdrawal failed: %s", e)
            self._drop_templates()

    # ---- templates (RAM) ----------------------------------------------------------------
    def _load_templates(self) -> list[tuple[str, np.ndarray]]:
        if self._templates is not None and any(self._consent_revoked(c) for c in self._tpl_consent.values()):
            self._drop_templates()
        if self._templates is not None:
            # M-16 erasures (delete_person, consent withdrawal, expiry) remove rows behind this cache's back
            stored = {r[0] for r in self.db.execute("SELECT person_id FROM vl_voice")}
            if not set(self._tpl_consent) <= stored:
                self._drop_templates()
        if self._templates is None:
            self._purge_revoked()
            out, cons = [], {}
            mid, dim = self.model_id, self.dim
            for pid, blob, t_mid, t_dim, cid in self.db.execute(
                    "SELECT person_id, template, model_id, dim, consent_id FROM vl_voice").fetchall():
                if t_mid != mid or t_dim != dim:
                    continue   # a template is valid only with the model that made it
                try:
                    dek = self.face.person_sealer(pid)
                except KeyError:  # python-brain: allow SILENT_EXCEPTION - person left M-16 meanwhile; their row is being erased
                    continue
                out.append((pid, np.frombuffer(dek.open(blob, VOICE_AAD + pid.encode()), dtype=np.float32).copy()))
                cons[pid] = cid
            self._templates, self._tpl_consent = out, cons
        return self._templates

    def _drop_templates(self) -> None:
        for _, v in self._templates or []:
            v.fill(0.0)
        self._templates = None
        self._tpl_consent = {}

    # ---- enrollment ---------------------------------------------------------------------
    def start_voice(self, person_id: str, consent_id: str) -> int:
        """Open a voice capture for a person enrolled in M-16. Returns the number of windows needed."""
        self._require_on()
        row = self.db.execute("SELECT guardian_enrolled FROM fm_enrolled WHERE person_id=?", (person_id,)).fetchone()
        if row is None:
            raise KeyError("person is not enrolled in face matching")
        rec = self.consents.require(consent_id, VOICE_SCOPE, now=self.clock())
        if row[0] and GUARDIAN_SCOPE not in rec.scopes:
            raise ConsentError("a minor's voice enrollment needs the guardian's recorded agreement (scope 'guardian')")
        if self._model() is None:
            raise VoiceLinkUnavailable(f"speaker model unavailable: {self.speaker.status.get(SLOT, 'empty')}")
        self.cancel_voice(person_id)
        self._captures[person_id] = _VoiceCapture(person_id, consent_id)
        return self.cfg.enroll_windows

    def _cap(self, person_id: str) -> _VoiceCapture:
        if person_id not in self._captures:
            raise KeyError("person is not in an open voice capture")
        return self._captures[person_id]

    def _complete(self, c: _VoiceCapture) -> bool:
        """enroll_windows accepted and the total-speech target met, or VOICE_ENROLL_MAX_WINDOWS accepted."""
        n = len(c.embeddings)
        return n >= VOICE_ENROLL_MAX_WINDOWS or (n >= self.cfg.enroll_windows and c.speech_s >= self.cfg.enroll_min_total_speech_s)

    def _needed(self, c: _VoiceCapture) -> int:
        """Windows still needed: the rest of enroll_windows, else one more while the speech target is short."""
        if self._complete(c):
            return 0
        return max(1, self.cfg.enroll_windows - len(c.embeddings))

    def _result(self, c: _VoiceCapture, ok: bool, code: str, speech_s: float = 0.0, level: float | None = None,
                ps: pitch.PitchStats | None = None) -> WindowResult:
        hz = (None, None, None) if ps is None else tuple(round(f, PITCH_DECIMALS) for f in (ps.median_hz, ps.low_hz, ps.high_hz))
        return WindowResult(ok, code, REASONS[code], self._complete(c), self._needed(c),
                            round(speech_s, SECONDS_DECIMALS), None if level is None else round(level, LEVEL_DECIMALS),
                            pitch_hz=hz[0], pitch_low_hz=hz[1], pitch_high_hz=hz[2],
                            total_speech_s=round(c.speech_s, STORED_SPEECH_DECIMALS))

    def submit_window(self, person_id: str, audio: Any, sample_rate: int, prompt_id: str | None = None) -> WindowResult:
        """Check one enrollment window and embed its speech in RAM. The buffer is zeroed either way.
        `prompt_id` names the sentence read (prompts.PROMPTS); its tone feeds the advisory range_report()."""
        x = None
        try:
            self._require_on()
            c = self._cap(person_id)
            if self._complete(c):
                return self._result(c, False, "enough")
            tone = None
            if prompt_id is not None:
                try:
                    tone = prompts.by_id(prompt_id).tone
                except KeyError:
                    return self._result(c, False, "prompt")
            if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or not SAMPLE_RATE_MIN <= sample_rate <= SAMPLE_RATE_MAX:
                return self._result(c, False, "rate")
            x, code = _prepare(audio)
            if x is None:
                return self._result(c, False, code)
            if _clipped(x):
                return self._result(c, False, "loud")
            v = vad.detect(x, sample_rate, self.cfg.vad_snr_db, self.cfg.vad_floor_dbfs)
            try:
                if v.speech_s < self.cfg.enroll_min_speech_s:
                    return self._result(c, False, "speech", v.speech_s, v.level_dbfs)
                ps = pitch.estimate(v.speech, sample_rate)   # before the speech is handed over and zeroed
                e = self._embed(v.speech, sample_rate)
                if e is None:
                    return self._result(c, False, "backend", v.speech_s, v.level_dbfs, ps)
                c.embeddings.append(unit(e).copy())
                e.fill(0.0)
                c.marks.append(_Mark(prompt_id, tone, None if ps is None else ps.median_hz, v.speech_s))
                c.speech_s += v.speech_s
                return self._result(c, True, "ok", v.speech_s, v.level_dbfs, ps)
            finally:
                v.zero()
        finally:
            _zero(audio, x)

    def range_report(self, person_id: str) -> RangeReport:
        """Advisory tone-range check of the open capture: the median pitch of the accepted "high"/"rising"
        windows should be at least enroll_range_min_rise above that of the "low" windows. When it is not,
        names one high or rising sentence to redo. Too little pitch data => "not measured", ok."""
        self._require_on()
        c = self._cap(person_id)
        high = [m for m in c.marks if m.tone in RANGE_HIGH_TONES and m.pitch_hz is not None]
        low = [m.pitch_hz for m in c.marks if m.tone in RANGE_LOW_TONES and m.pitch_hz is not None]
        if not high or not low:
            return RangeReport(True, False, "voice range not measured (too little pitch data); that is fine")
        hi = float(np.median([m.pitch_hz for m in high]))
        lo = float(np.median(low))
        hi_r, lo_r = round(hi, PITCH_DECIMALS), round(lo, PITCH_DECIMALS)
        if hi >= lo * (1.0 + self.cfg.enroll_range_min_rise):
            return RangeReport(True, True, f"voice range ok: higher sentences about {hi_r:g} Hz, lower ones about {lo_r:g} Hz",
                               lo_r, hi_r)
        redo = min(high, key=lambda m: m.pitch_hz).prompt_id
        return RangeReport(False, True, f"the higher sentences (about {hi_r:g} Hz) were not much higher than the lower ones "
                                        f"(about {lo_r:g} Hz); one could be read again, {REDO_CUE}", lo_r, hi_r, redo, REDO_CUE)

    def redo_window(self, person_id: str, prompt_id: str) -> int:
        """Drop the accepted window recorded for `prompt_id` (the latest one), zeroing its embedding, so the
        sentence can be recorded again. Returns the number of windows now needed."""
        self._require_on()
        c = self._cap(person_id)
        idx = [i for i, m in enumerate(c.marks) if m.prompt_id == prompt_id]
        if not idx:
            raise KeyError("no accepted window for this sentence")
        i = idx[-1]
        c.embeddings.pop(i).fill(0.0)
        c.speech_s -= c.marks.pop(i).speech_s
        return self._needed(c)

    def finish_voice(self, person_id: str, now: float | None = None) -> WindowResult:
        self._require_on()
        c = self._cap(person_id)
        n, need = len(c.embeddings), self.cfg.enroll_windows
        total = round(c.speech_s, STORED_SPEECH_DECIMALS)
        target = self.cfg.enroll_min_total_speech_s
        if n < need:
            return WindowResult(False, "windows", f"still needed: {need - n} more good window{'s' if need - n != 1 else ''} ({n}/{need})",
                                False, need - n, total_speech_s=total)
        if not self._complete(c):
            short = max(SHORT_SPEECH_SHOWN_MIN_S, round(target - c.speech_s, STORED_SPEECH_DECIMALS))
            return WindowResult(False, "speech_total", f"still needed: {short:.1f} s more speech (total {total:.1f} of {target:g} s); "
                                                       "read another sentence", False, self._needed(c), total_speech_s=total)
        below = c.speech_s < target   # only at VOICE_ENROLL_MAX_WINDOWS windows
        self.consents.require(c.consent_id, VOICE_SCOPE, now=self.clock())   # withdrawn meanwhile => nothing stored
        check_persist("voice_template", True, source="M-17", m17_enabled=True)
        now = self.clock() if now is None else now
        vec = unit(np.mean(np.stack(c.embeddings), axis=0)).astype(np.float32)
        try:
            dek = self.face.person_sealer(person_id)
            replaced = self.db.execute("SELECT 1 FROM vl_voice WHERE person_id=?", (person_id,)).fetchone() is not None
            self.db.execute("INSERT OR REPLACE INTO vl_voice(person_id, template, model_id, dim, consent_id, windows, speech_s, enrolled_at) "
                            "VALUES(?,?,?,?,?,?,?,?)",
                            (person_id, dek.seal(vec.tobytes(), VOICE_AAD + person_id.encode()), self.model_id, int(vec.size),
                             c.consent_id, n, round(c.speech_s, STORED_SPEECH_DECIMALS), now))
        finally:
            vec.fill(0.0)
            c.wipe()
            del self._captures[person_id]
        self._drop_templates()
        log_change(self.db, self.chain, "voice_reenrolled" if replaced else "voice_enrolled", {"windows": n})
        msg = f"Voice template complete for {self.face.person_name(person_id)}"
        if below:
            msg += f" ({total:.1f} s of speech from {n} windows, below the {target:g} s target)"
        return WindowResult(True, "done", msg, True, 0, total_speech_s=total)

    def cancel_voice(self, person_id: str) -> None:
        c = self._captures.pop(person_id, None)
        if c:
            c.wipe()

    def capture_buffer_size(self) -> int:
        return sum(len(c.embeddings) for c in self._captures.values())

    # ---- matching -----------------------------------------------------------------------
    def process_audio(self, audio: Any, sample_rate: int, place: str | None = None, now: float | None = None) -> dict:
        """Match one buffer against enrolled voices. Returns {"heard": {...} | None, "unsure": bool,
        "speech_s", "health"}. Only a clear match of an enrolled person is logged; a close call is
        "unsure" and logs nothing; anything else leaves no trace. A clear match of the "self" person, or
        one the anti-spoof slot scores at or above spoof_threshold, is dropped like an unknown voice."""
        now = self.clock() if now is None else now
        out: dict = {"heard": None, "unsure": False, "speech_s": 0.0}
        x = None
        try:
            self._require_on()
            if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or not SAMPLE_RATE_MIN <= sample_rate <= SAMPLE_RATE_MAX:
                return self._out(out)
            x, _ = _prepare(audio)
            if x is None:
                return self._out(out)
            v = vad.detect(x, sample_rate, self.cfg.vad_snr_db, self.cfg.vad_floor_dbfs)
            try:
                out["speech_s"] = round(v.speech_s, SECONDS_DECIMALS)
                if v.speech_s < self.cfg.min_speech_s:
                    return self._out(out)
                templates = self._load_templates()
                if not templates:
                    return self._out(out)
                e = self._embed(v.speech, sample_rate)
                if e is None:
                    return self._out(out)
                q = unit(e)
                try:
                    scores = sorted(((float(np.dot(q, t)), pid) for pid, t in templates), reverse=True)
                finally:
                    e.fill(0.0); q.fill(0.0)
                best = scores[0][0]
                outcome, pid = decide(scores, self.cfg.match_threshold, self.cfg.unsure_margin, self.self_id(),
                                      lambda: self._spoof_score(v.speech, sample_rate), self.cfg.spoof_threshold)
                if outcome == UNSURE:
                    out["unsure"] = True
                # none: not an enrolled voice (dropped, not stored, not counted); self: the wearer/owner,
                # ignored like an unknown voice; spoofed: likely a recording or synthetic, dropped without a trace
                if outcome != HEARD or pid is None:
                    return self._out(out)
            finally:
                v.zero()
            out["heard"] = self._log_heard(pid, best, place, now)
            self.stats["heard"] += 1
            return self._out(out)
        finally:
            _zero(audio, x)

    def _spoof_score(self, speech: np.ndarray, sample_rate: int) -> float | None:
        """The voice_antispoof slot's score of the speech, or None when no slot is configured (or it
        failed): the match goes ahead. decide() compares it with spoof_threshold."""
        if not self.speaker.slots.get(SPOOF_SLOT):
            return None
        return self.speaker.spoof_score(speech, sample_rate)

    # ---- self (ADR-0058 decision 8) -----------------------------------------------------
    def self_id(self) -> str | None:
        return self_person(self.db)

    def set_self(self, person_id: str) -> None:
        """Mark one person enrolled in M-16 as self (the wearer or owner): never announced or logged."""
        self._require_on()
        if self.db.execute("SELECT 1 FROM fm_enrolled WHERE person_id=?", (person_id,)).fetchone() is None:
            raise KeyError("person is not enrolled in face matching")
        self.db.execute("INSERT OR REPLACE INTO vl_state(k, v) VALUES(?, ?)", (SELF_KEY, person_id))
        log_change(self.db, self.chain, "self_set", {})

    def clear_self(self) -> bool:
        n = self.db.execute("DELETE FROM vl_state WHERE k=?", (SELF_KEY,)).rowcount
        if n:
            log_change(self.db, self.chain, "self_cleared", {})
        return bool(n)

    def _out(self, out: dict) -> dict:
        out["health"] = self.health()[0]
        return check_output(out)

    def _log_heard(self, person_id: str, score: float, place: str | None, now: float) -> dict:
        heard_at = round(now / MINUTE_S) * MINUTE_S
        last = self.face.last_match_at(person_id)
        agrees = last is not None and abs(now - last) <= self.cfg.agree_window_s
        eid = uuid.uuid4().hex
        self.db.execute("INSERT INTO vl_heard(entry_id, person_id, heard_at, place, score, face_agrees, expires_at) VALUES(?,?,?,?,?,?,?)",
                        (eid, person_id, heard_at, self.face.seal_place(check_text(place or "unnamed place")),
                         round(score, SCORE_DECIMALS), int(agrees), heard_at + self.cfg.heard_retention_days * DAY_S))
        return {"person_id": person_id, "name": self.face.person_name(person_id), "score": round(score, SCORE_DECIMALS),
                "face_agrees": agrees, "entry_id": eid}

    # ---- review and deletes -------------------------------------------------------------
    def review(self, days: int | None = None, now: float | None = None) -> dict:
        now = self.clock() if now is None else now
        since = now - (days or self.cfg.heard_retention_days) * DAY_S
        mid, dim = self.model_id, self.dim
        people = []
        for pid, t_mid, t_dim, windows, enrolled_at in self.db.execute(
                "SELECT person_id, model_id, dim, windows, enrolled_at FROM vl_voice ORDER BY enrolled_at").fetchall():
            current = t_mid == mid and t_dim == dim
            entries = [{"entry_id": eid, "heard_at": t, "place": self.face.open_place(pl), "score": s, "face_agrees": bool(fa)}
                       for eid, t, pl, s, fa in self.db.execute(
                           "SELECT entry_id, heard_at, place, score, face_agrees FROM vl_heard WHERE person_id=? AND heard_at>=? "
                           "ORDER BY heard_at", (pid, since)).fetchall()]
            people.append({"person_id": pid, "name": self.face.person_name(pid), "windows": windows, "enrolled_at": enrolled_at,
                           "model_current": current, "note": "" if current else REENROLL_NOTE, "entries": entries})
        return check_output({"people": people})

    def delete_heard(self, entry_id: str) -> bool:
        n = self.db.execute("DELETE FROM vl_heard WHERE entry_id=?", (entry_id,)).rowcount
        if n:
            log_change(self.db, self.chain, "heard_deleted", {"entries": n})
        return bool(n)

    def delete_voice(self, person_id: str, kind: str = "voice_deleted") -> WipeReport:
        """Delete one voice template and its "heard" entries; the person's face enrollment stays. Their
        data key is replaced, so the old ciphertext is undecryptable even if a copy survived."""
        row = self.db.execute("SELECT template FROM vl_voice WHERE person_id=?", (person_id,)).fetchone()
        if row is None:
            raise KeyError("no voice template for this person")
        self.cancel_voice(person_id)
        self._drop_templates()
        self.db.execute("DELETE FROM vl_voice WHERE person_id=?", (person_id,))
        self.db.execute("DELETE FROM vl_heard WHERE person_id=?", (person_id,))
        self.db.execute("DELETE FROM vl_state WHERE k=? AND v=?", (SELF_KEY, person_id))
        self._rekey(person_id)
        chk = self.face.erasure_checks(row[0], VOICE_AAD + person_id.encode())
        gone = self.db.execute("SELECT 1 FROM vl_voice WHERE person_id=?", (person_id,)).fetchone() is None
        ok = gone and chk["undecryptable"] and chk["ciphertext_absent"]
        log_change(self.db, self.chain, kind, {"voice_templates_destroyed": 1, "verified": ok})
        return WipeReport(1, ok, {"row_absent": gone, **chk}, "1 voice template destroyed, verified" if ok else "wipe NOT verified")

    def _rekey(self, person_id: str) -> None:
        _rekey(self.face, person_id)

    # ---- disable + verified wipe (M-16 templates stay) ---------------------------------
    def disable(self) -> WipeReport:
        self._drop_templates()
        for c in self._captures.values():
            c.wipe()
        self._captures.clear()
        self._off = True
        return wipe_voice_data(self.db, self.face, self.chain)


def self_person(db) -> str | None:
    """The person marked as self, or None."""
    r = db.execute("SELECT v FROM vl_state WHERE k=?", (SELF_KEY,)).fetchone()
    return str(r[0]) if r else None


def _rekey(face: Any, person_id: str) -> None:
    try:
        face.rekey_person(person_id)
    except KeyError:  # python-brain: allow SILENT_EXCEPTION - no longer enrolled in M-16: the data key is already destroyed
        return


def _compact(db) -> None:
    db.execute("VACUUM")
    if db.path != ":memory:":
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def _in_db_files(db, blob: bytes) -> bool:
    if db.path == ":memory:":
        return False
    return any(p.exists() and blob in p.read_bytes() for p in (Path(db.path + s) for s in ("", "-wal", "-journal")))


def wipe_voice_data(db, face: Any, chain=None) -> WipeReport:
    """Delete every M-17 row and the M-17 opt-in, verified; M-16 templates stay. With the M-16 module
    loaded (`face`), each affected person's data key is replaced so the deleted voice ciphertext is
    undecryptable. Without it (CLI fallback) the ciphertext is still removed, but undecryptability is
    verified only for people no longer enrolled in M-16 (their key is already gone)."""
    sample = db.execute("SELECT person_id, template FROM vl_voice").fetchall()
    db.execute("DELETE FROM vl_voice")
    db.execute("DELETE FROM vl_heard")
    db.execute("DELETE FROM vl_state WHERE k IN ('opt_in', ?)", (SELF_KEY,))
    if face is not None:
        for pid, _ in sample:
            _rekey(face, pid)
        checks = [face.erasure_checks(t, VOICE_AAD + pid.encode()) for pid, t in sample]
    else:
        checks = []
    if face is None or not sample:
        _compact(db)
    if face is None:
        checks = [{"undecryptable": db.execute("SELECT 1 FROM fm_enrolled WHERE person_id=?", (pid,)).fetchone() is None,
                   "ciphertext_absent": not _in_db_files(db, t)} for pid, t in sample]
    rows = db.execute("SELECT COUNT(*) FROM vl_voice").fetchone()[0]
    heard = db.execute("SELECT COUNT(*) FROM vl_heard").fetchone()[0]
    undecryptable = all(c["undecryptable"] for c in checks)
    absent = all(c["ciphertext_absent"] for c in checks)
    opt_in_gone = opt_in_record(db) is None
    ok = rows == 0 and heard == 0 and undecryptable and absent and opt_in_gone
    n = len(sample)
    log_change(db, chain, "disabled_wiped", {"voice_templates_destroyed": n, "verified": ok})
    return WipeReport(n, ok, {"rows": rows, "heard_rows": heard, "undecryptable": undecryptable, "ciphertext_absent": absent,
                              "opt_in_absent": opt_in_gone},
                      f"{n} voice templates destroyed, verified" if ok else "wipe NOT verified")


__all__ = ["VoiceLinkModule", "VoiceLinkConfig", "VoiceLinkLocked", "VoiceLinkUnavailable", "WindowResult", "RangeReport", "speaker_slots",
           "default_model_dir", "wipe_voice_data", "self_person", "decide", "HEARD", "UNSURE", "NONE", "SELF", "SPOOFED"]
