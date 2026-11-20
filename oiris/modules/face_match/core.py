"""M-16 module core: storage, modes, enrollment, review log, retention and wipe.
Backends only compute (port.py); everything that touches disk is here.

Keys (spec section 9): a random face key is kept in `<data_dir>/face_match.key`, sealed with
`keys.subkey("face-template")`. Each enrolled person has a random data key (DEK) sealed
with the face key; their name and template are sealed with the DEK. Deleting a person
destroys their DEK row; disabling destroys the face key file and every row, then VACUUMs.
Logs (places) are sealed with a separate log key in `<data_dir>/face_match_log.key`.

M-17 voice link (ADR-0057) is an add-on: a person's voice template (vl_voice) is sealed with the same
DEK (AAD b"voice:" + person_id), so every M-16 erasure path (delete, consent withdrawal or expiry,
disable) erases the voice too, verified. M-17 cannot outlive M-16.

Face photo (ADR-0057 decision 4, photo.py): for a person whose consent record has the "photo" scope, one
JPEG face crop from their own best FRONT enrollment shot is sealed with the same DEK (AAD b"photo:" +
person_id) in fm_photo. It is erased, verified, by delete_person, withdrawing the photo scope or the
consent, delete_photo and disable. `photo()` is the only read path; there is no list or export."""
from __future__ import annotations
import logging, os, statistics, time, uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ...config import POSE_DELTA_MAX, POSE_DELTA_MIN
from ...guard.consent import ConsentError, ConsentRegistry
from ...guard.privacy import PrivacyError, check_persist, round_gps
from ...guard.vocabulary import check_output, check_text
from ...schema import Health
from ...security.aead import Sealer
from .optin import STRICT, OptInError, log_change, opt_in_record
from .photo import PHOTO_AAD, PHOTO_SCOPE, PHOTO_TABLE, PhotoCandidate, encode_jpeg

from .port import REQUIRED_ANGLES, Angle, Embedding, FaceBox, Template, TemplateStatus, unit
from .quality import QualityLimits, check_shot
from .supervisor import Guarded

log = logging.getLogger(__name__)

KEY_FILE, LOG_KEY_FILE = "face_match.key", "face_match_log.key"
VOICE_AAD = b"voice:"   # + person_id: M-17 voice template sealed with the person's DEK
VOICE_TABLES = ("vl_voice", "vl_heard")
REENROLL_MARGIN = 0.02
HINT = "recognition getting weaker, re-enroll?"


class FaceMatchLocked(PermissionError):
    pass


class ExportBlocked(PermissionError):
    pass


@dataclass
class FaceMatchConfig:
    """Spec section 13. There is deliberately no setting for the number of enrolled people."""
    modes: list = field(default_factory=list)
    jurisdiction: str = "strictest"
    require_enrollment_consent: bool = True
    match_threshold: float = 0.8
    encounter_retention_days: int = 30
    place_source: str = "named"
    time_bucket_minutes: int = 60
    allow_log_export: bool = False
    enroll_min_shots: int = 5
    enroll_max_shots: int = 10
    enroll_required_angles: list = field(default_factory=lambda: [a.value for a in REQUIRED_ANGLES])
    # UP/DOWN are judged against the person's own FRONT pitch (backend scale) when the backend gives one.
    # Provisional: set from one real webcam test (2026-11-15) where deliberate tilts reached only about
    # -0.08 (up) and +0.12 (down) from level. Proposals until measured on more people and cameras.
    pose_up_delta: float = 0.06
    pose_down_delta: float = 0.08
    reenroll_confidence_drop: float = 0.1
    reenroll_window_matches: int = 20
    adapter: str = "opencv_zoo"
    model_dir: str = ""
    allow_test_backend: bool = False   # tests only; the test backend is refused otherwise

    def validate(self) -> "FaceMatchConfig":
        errs = []
        if any(m not in ("personal", "research") for m in self.modes): errs.append("face_match.modes: personal/research only")
        if not 0.5 <= self.match_threshold < 1.0: errs.append("face_match.match_threshold must be in [0.5, 1)")
        if not 1 <= self.encounter_retention_days <= 365: errs.append("face_match.encounter_retention_days must be 1..365")
        if self.place_source not in ("named", "gps_0.01"): errs.append("face_match.place_source must be named|gps_0.01")
        if not 1 <= self.time_bucket_minutes <= 1440: errs.append("face_match.time_bucket_minutes must be 1..1440")
        if not 5 <= self.enroll_min_shots <= self.enroll_max_shots: errs.append("face_match.enroll_min_shots must be >= 5 and <= enroll_max_shots")
        if any(a not in [x.value for x in Angle] for a in self.enroll_required_angles): errs.append("face_match.enroll_required_angles: unknown angle")
        for k in ("pose_up_delta", "pose_down_delta"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not POSE_DELTA_MIN <= v <= POSE_DELTA_MAX:
                errs.append(f"face_match.{k} must be in [{POSE_DELTA_MIN}, {POSE_DELTA_MAX}]")
        if not 0 < self.reenroll_confidence_drop < 1 or self.reenroll_window_matches < 1: errs.append("face_match.reenroll_* out of range")
        if errs:
            raise ValueError("; ".join(errs))
        return self

    @classmethod
    def from_dict(cls, d: dict | None) -> "FaceMatchConfig":
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"face_match: unknown keys {sorted(unknown)}")
        return cls(**d).validate()


@dataclass
class ShotResult:
    ok: bool
    code: str
    reason: str
    needed_angles: list = field(default_factory=list)


@dataclass
class WipeReport:
    templates_destroyed: int
    verified: bool
    checks: dict = field(default_factory=dict)
    message: str = ""
    photos_destroyed: int = 0


@dataclass
class _Capture:
    session_id: str
    label: str
    consent_id: str | None
    guardian: bool
    consent_note: str
    replaces: bool = False
    shots: list = field(default_factory=list)   # (Angle, Embedding)
    front_pitches: list[float] = field(default_factory=list)   # pitch of accepted FRONT shots (RAM only)
    photo: PhotoCandidate | None = None   # best FRONT face crop (RAM only); None unless the consent has scope "photo"

    def front_pitch(self) -> float | None:
        return sum(self.front_pitches) / len(self.front_pitches) if self.front_pitches else None

    def wipe(self) -> None:
        for _, e in self.shots:
            e.zero()
        self.shots.clear()
        self.front_pitches.clear()
        if self.photo is not None:
            self.photo.wipe()


def _overwrite_unlink(p: Path) -> None:
    if p.exists():
        n = p.stat().st_size
        with open(p, "r+b") as f:
            f.write(os.urandom(max(n, 1))); f.flush(); os.fsync(f.fileno())
        p.unlink()


class FaceMatchModule:
    def __init__(self, db, keys, data_dir: str | Path, backend: Any, chain=None, cfg: FaceMatchConfig | None = None,
                 consents: ConsentRegistry | None = None, clock: Callable[[], float] = time.time,
                 timeout_s: float = 2.0, breaker_threshold: int = 5, breaker_cooldown_s: float = 30.0):
        self.cfg = (cfg or FaceMatchConfig()).validate()
        if getattr(backend, "test_only", False) and not self.cfg.allow_test_backend:
            raise FaceMatchLocked("the deterministic test backend is test-only")
        self.db, self.keys, self.chain, self.clock = db, keys, chain, clock
        db.execute("PRAGMA secure_delete=ON")   # deleted rows are zeroed in the file, not just unlinked
        self.data_dir = Path(data_dir).expanduser()
        self.consents = consents if consents is not None else ConsentRegistry(db)   # durable by default
        rec = opt_in_record(db)
        if rec is None:
            raise FaceMatchLocked("M-16 is off or not opted in; enable it with the typed opt-in first")
        self.modes = set(rec["modes"])
        self.strict = rec.get("jurisdiction", "strictest") in STRICT
        self.limits = QualityLimits()
        backend.open({"model_dir": self.cfg.model_dir or str(self.data_dir / "models" / "face_match")})
        self.backend = backend
        self.guard = Guarded(backend, timeout_s, breaker_threshold, breaker_cooldown_s)
        self._fk = Sealer(self._face_key(KEY_FILE, "face-template"))
        self._lk = Sealer(self._face_key(LOG_KEY_FILE, "face-log"))
        self._captures: dict[str, _Capture] = {}
        self._sessions: dict[str, set] = {}
        self._templates: list[Template] | None = None
        self._tpl_consent: dict[str, str | None] = {}
        self._last_match: dict[str, float] = {}   # RAM only: person -> time of latest face match (M-17 agreement)
        self.consents.withdraw_hooks.append(self._on_consent_withdrawn)
        self._withdrawn_purged = 0
        self._purge_revoked()   # load-time: templates whose (persisted) consent is withdrawn/expired are erased
        self.stats = {"frames": 0, "matched": 0, "unrecognized": 0, "faces": 0}

    # ---- keys --------------------------------------------------------------------------
    def _face_key(self, fname: str, purpose: str) -> bytes:
        p = self.data_dir / fname
        wrap = Sealer(self.keys.subkey(purpose))
        aad = b"m16:" + purpose.encode()
        if p.exists():
            return wrap.open(p.read_bytes(), aad)
        k = Sealer.new_key()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(wrap.seal(k, aad)); os.chmod(tmp, 0o600); tmp.replace(p)
        return k

    def _dek(self, person_id: str, blob: bytes) -> Sealer:
        return Sealer(self._fk.open(blob, person_id.encode()))

    @contextmanager
    def _tx(self):
        with self.db.lock:
            self.db.execute("BEGIN")
            try:
                yield
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            self.db.execute("COMMIT")

    # ---- public helpers for the M-17 voice link (ADR-0057) -----------------------------
    def person_sealer(self, person_id: str) -> Sealer:
        """The enrolled person's own data key (DEK). KeyError if the person is not enrolled."""
        row = self.db.execute("SELECT dek FROM fm_enrolled WHERE person_id=?", (person_id,)).fetchone()
        if row is None:
            raise KeyError("person is not enrolled in face matching")
        return self._dek(person_id, row[0])

    def person_name(self, person_id: str) -> str:
        if self.db.execute("SELECT 1 FROM fm_enrolled WHERE person_id=?", (person_id,)).fetchone() is None:
            raise KeyError("person is not enrolled in face matching")
        return self._person_meta(person_id)[0]

    def seal_place(self, place: str) -> bytes:
        return self._lk.seal(place.encode(), b"place")

    def open_place(self, blob: bytes) -> str:
        return self._open_place(blob)

    def last_match_at(self, person_id: str) -> float | None:
        """RAM-only time of the person's latest face match in this process (None if none)."""
        return self._last_match.get(person_id)

    def erasure_checks(self, blob: bytes, aad: bytes) -> dict:
        """After a deletion: compact the store, then report whether any key still on this device opens
        `blob` and whether its ciphertext is still in the database files."""
        self._compact()
        return {"undecryptable": not self._can_open_with_any(blob, aad), "ciphertext_absent": not self._bytes_on_disk(blob)}

    def rekey_person(self, person_id: str) -> None:
        """Crypto-erase the person's current DEK: name, face template, voice template and photo (if any)
        are re-sealed under a fresh DEK in one transaction. M-17 uses it so a deleted voice template is
        undecryptable while the person stays enrolled; delete_photo uses it the same way."""
        row = self.db.execute("SELECT dek, display_name, template FROM fm_enrolled WHERE person_id=?", (person_id,)).fetchone()
        if row is None:
            raise KeyError("person is not enrolled in face matching")
        old, p = self._dek(person_id, row[0]), person_id.encode()
        name, tpl = old.open(row[1], b"name:" + p), old.open(row[2], b"tpl:" + p)
        voice = self._open_voice(person_id, old)
        photo = self._open_photo(person_id, old)
        raw = Sealer.new_key()
        new = Sealer(raw)
        try:
            with self._tx():
                self.db.execute("UPDATE fm_enrolled SET display_name=?, template=?, dek=? WHERE person_id=?",
                                (new.seal(name, b"name:" + p), new.seal(tpl, b"tpl:" + p), self._fk.seal(raw, p), person_id))
                if voice is not None:
                    self._reseal_voice(person_id, new, voice)
                if photo is not None:
                    self.db.execute(f"UPDATE {PHOTO_TABLE} SET photo=? WHERE person_id=?",
                                    (new.seal(photo, PHOTO_AAD + p), person_id))
        finally:
            if voice is not None:
                voice.fill(0)

    # ---- face photo (ADR-0057 decision 4) ----------------------------------------------
    def _photo_blob(self, person_id: str) -> bytes | None:
        if not self.db.has_table(PHOTO_TABLE):
            return None
        row = self.db.execute(f"SELECT photo FROM {PHOTO_TABLE} WHERE person_id=?", (person_id,)).fetchone()
        return None if row is None else row[0]

    def _open_photo(self, person_id: str, dek: Sealer) -> bytes | None:
        blob = self._photo_blob(person_id)
        return None if blob is None else dek.open(blob, PHOTO_AAD + person_id.encode())

    def _photo_consent_ok(self, consent_id: str | None) -> bool:
        """A photo needs a known, current consent record with the "photo" scope. Stricter than templates:
        an unknown record is not enough to keep or show a photo."""
        if not consent_id:
            return False
        try:
            self.consents.require(consent_id, PHOTO_SCOPE, now=self.clock())
        except ConsentError:
            return False
        return True

    def has_photo(self, person_id: str) -> bool:
        """Whether a photo is stored for this person (for `face-match list`). Never the image."""
        return self._photo_blob(person_id) is not None

    def photo(self, person_id: str) -> bytes | None:
        """The enrolled person's photo as JPEG bytes, or None: not enrolled, no photo, the module is off,
        or the photo consent is no longer current (the photo is then erased). The only read path; the
        dashboard calls it to show who was recognised. Nothing is logged about the image."""
        if self._fk is None or opt_in_record(self.db) is None:
            return None
        row = self.db.execute("SELECT dek, consent_id FROM fm_enrolled WHERE person_id=?", (person_id,)).fetchone()
        if row is None or self._photo_blob(person_id) is None:
            return None
        if not self._photo_consent_ok(row[1]):
            self.delete_photo(person_id, kind="photo_consent_withdrawn")
            return None
        return self._open_photo(person_id, self._dek(person_id, row[0]))

    def delete_photo(self, person_id: str, kind: str = "photo_deleted") -> WipeReport:
        """Erase the person's photo, keep the enrollment: the row goes and the DEK is replaced (rekey_person),
        so the old ciphertext opens with no key on this device; then the store is compacted and checked."""
        if self.db.execute("SELECT 1 FROM fm_enrolled WHERE person_id=?", (person_id,)).fetchone() is None:
            raise KeyError("unknown person")
        blob = self._photo_blob(person_id)
        if blob is None:
            return WipeReport(0, True, {}, "no photo stored")
        with self._tx():
            self.db.execute(f"DELETE FROM {PHOTO_TABLE} WHERE person_id=?", (person_id,))
        self.rekey_person(person_id)
        gone = self._photo_blob(person_id) is None
        chk = self.erasure_checks(blob, PHOTO_AAD + person_id.encode())
        ok = gone and chk["undecryptable"] and chk["ciphertext_absent"]
        log_change(self.db, self.chain, kind, {"photos_destroyed": 1, "verified": ok})
        return WipeReport(0, ok, {"photo_row_absent": gone, "photo_undecryptable": chk["undecryptable"],
                                  "photo_ciphertext_absent": chk["ciphertext_absent"]},
                          "1 photo destroyed, verified" if ok else "photo wipe NOT verified", photos_destroyed=1)

    def withdraw_photo_scope(self, consent_id: str) -> WipeReport:
        """Remove the "photo" scope from a consent record (the rest of the record stays) and erase, verified,
        the photo of everyone enrolled under it. KeyError if the record is unknown."""
        rec = self.consents.get(consent_id)
        if rec is None:
            raise KeyError("no such consent record")
        rec.scopes.discard(PHOTO_SCOPE)
        self.consents.add(rec)   # durable before anything is erased
        pids = [r[0] for r in self.db.execute("SELECT person_id FROM fm_enrolled WHERE consent_id=?", (consent_id,)).fetchall()]
        return _merge_photo_reports([self.delete_photo(pid, kind="photo_consent_withdrawn") for pid in pids if self.has_photo(pid)])

    def _purge_photo_consent(self) -> int:
        """Load-time: photos whose consent record is not (or no longer) current with scope "photo" are erased."""
        if not self.db.has_table(PHOTO_TABLE):
            return 0
        rows = self.db.execute(f"SELECT e.person_id, e.consent_id FROM {PHOTO_TABLE} p JOIN fm_enrolled e "
                               "ON e.person_id = p.person_id").fetchall()
        gone = [pid for pid, cid in rows if not self._photo_consent_ok(cid)]
        for pid in gone:
            if not self.delete_photo(pid, kind="photo_consent_withdrawn").verified:
                log.error("photo wipe after photo consent withdrawal NOT verified")
        return len(gone)

    def _photo_for(self, pid: str, c: _Capture, dek: Sealer) -> tuple[bytes | None, str]:
        """(sealed photo or None, plain status) for finish_person."""
        if not self._photo_consent_ok(c.consent_id):
            return None, "No photo kept: there is no current consent record with the 'photo' scope."
        if c.photo is None or c.photo.pixels is None:
            return None, "No photo kept: no front shot to take it from."
        try:
            check_persist("face_photo", True, source="M-16", m16_enabled=opt_in_record(self.db) is not None, photo_consent=True)
        except PrivacyError as e:
            return None, f"No photo kept: {e}."
        try:
            jpeg = encode_jpeg(c.photo.pixels)
        except (ImportError, ValueError) as e:
            log.warning("face photo not kept: %s", type(e).__name__)
            return None, "No photo kept: the JPEG encoder (OpenCV) is not available."
        return dek.seal(jpeg, PHOTO_AAD + pid.encode()), "One face photo kept (consent scope 'photo')."

    def _open_voice(self, person_id: str, dek: Sealer | None = None) -> np.ndarray | None:
        if not self.db.has_table("vl_voice"):
            return None
        row = self.db.execute("SELECT template FROM vl_voice WHERE person_id=?", (person_id,)).fetchone()
        if row is None:
            return None
        if dek is None:
            d = self.db.execute("SELECT dek FROM fm_enrolled WHERE person_id=?", (person_id,)).fetchone()
            if d is None:
                return None
            dek = self._dek(person_id, d[0])
        return np.frombuffer(dek.open(row[0], VOICE_AAD + person_id.encode()), dtype=np.float32).copy()

    def _reseal_voice(self, person_id: str, dek: Sealer, vec: np.ndarray) -> None:
        self.db.execute("UPDATE vl_voice SET template=? WHERE person_id=?",
                        (dek.seal(vec.tobytes(), VOICE_AAD + person_id.encode()), person_id))

    # ---- health ------------------------------------------------------------------------
    def health(self) -> tuple[Health, str]:
        return self.guard.health, self.guard.reason

    # ---- templates (RAM) ---------------------------------------------------------------
    def _consent_revoked(self, consent_id: str | None) -> bool:
        """Forge F3: True when a known consent record is withdrawn or expired. Templates enrolled under a
        recorded no-consent override (consent_id None) and ids this registry has never seen are kept;
        the registry is not persisted, so absence is not evidence of withdrawal."""
        if not consent_id:
            return False
        rec = self.consents.get(consent_id)
        if rec is None:
            return False
        try:
            self.consents.require(consent_id, "face_enrollment", now=self.clock())
        except ConsentError:
            return True
        return False

    def _purge_revoked(self) -> int:
        rows = self.db.execute("SELECT person_id, consent_id FROM fm_enrolled").fetchall()
        gone = [pid for pid, cid in rows if self._consent_revoked(cid)]
        for pid in gone:
            rep = self.delete_person(pid)   # same crypto-erase + verification path as a user deletion
            log_change(self.db, self.chain, "consent_withdrawn", {"templates_destroyed": 1, "verified": rep.verified})
        self._withdrawn_purged += len(gone)
        self._purge_photo_consent()   # people who stay enrolled but no longer have a current photo scope
        return len(gone)

    def consent_review(self) -> dict:
        """Templates whose consent ID has no record here are kept but listed for review."""
        missing = [{"person_id": pid, "consent_id": cid, "status": "consent record missing"}
                   for pid, cid in self.db.execute("SELECT person_id, consent_id FROM fm_enrolled ORDER BY enrolled_at")
                   if cid and not self.consents.known(cid)]
        return {"missing": missing, "withdrawn_purged": self._withdrawn_purged}

    def _on_consent_withdrawn(self, consent_id: str) -> None:
        try:
            self._purge_revoked()
        except Exception as e:  # noqa: BLE001 - a hook must not break the withdrawing caller
            log.error("purge after consent withdrawal failed: %s", e)
            self._drop_templates()

    def _load_templates(self) -> list[Template]:
        # Forge F3: consent is rechecked on every load/match, not only at enrollment.
        if self._templates is not None and any(self._consent_revoked(c) for c in self._tpl_consent.values()):
            self._purge_revoked()
        if self._templates is None:
            self._purge_revoked()
            out, cons = [], {}
            for pid, dek, tpl, mid, cid in self.db.execute(
                    "SELECT person_id, dek, template, model_id, consent_id FROM fm_enrolled"):
                if mid != self.backend.model_id:
                    continue   # a template is only valid with the model that made it
                vec = np.frombuffer(self._dek(pid, dek).open(tpl, b"tpl:" + pid.encode()), dtype=np.float32).copy()
                out.append(Template(pid, vec)); cons[pid] = cid
            self._templates, self._tpl_consent = out, cons
        return self._templates

    def _drop_templates(self) -> None:
        for t in self._templates or []:
            t.vec.fill(0)
        self._templates = None

    # ---- live processing (no output about anyone; review log only) ---------------------
    def _place(self, place: str | None, gps: tuple | None) -> str:
        if self.cfg.place_source == "gps_0.01" and gps:
            lat, lon = round_gps(gps[0], gps[1], 2)
            return f"{lat:.2f},{lon:.2f}"
        return check_text(place or "unnamed place")

    def process_frame(self, frame: Any, place: str | None = None, gps: tuple | None = None, now: float | None = None) -> dict:
        """Returns counts only. Never names, never an event, never a notification (spec 6)."""
        now = self.clock() if now is None else now
        self.stats["frames"] += 1
        ok, boxes = self.guard.call("detect", frame, self.guard.timeout_s,
                                    validate=lambda v: isinstance(v, list) and all(isinstance(b, FaceBox) for b in v))
        if not ok:
            return {"faces": 0, "matched": 0, "unrecognized": 0, "health": self.guard.health.value}
        where = self._place(place, gps)
        templates = self._load_templates() if "personal" in self.modes else []
        matched = unrec = 0
        for b in boxes:
            ok, e = self.guard.call("embed", frame, b, validate=lambda v: isinstance(v, Embedding) and v.valid())
            if not ok:
                continue
            try:
                m = None
                if templates:
                    ok, m = self.guard.call("compare", e, templates, validate=lambda v: v is None or (0.0 <= v.confidence <= 1.0))
                    m = m if ok else None
                if m is not None and m.confidence >= self.cfg.match_threshold:
                    self._log_encounter(m.person_id, m.confidence, where, now); matched += 1
                    self._last_match[m.person_id] = now
                elif "research" in self.modes:
                    self._count_unrecognized(where, now); unrec += 1
            finally:
                e.zero()   # embeddings never outlive the frame cycle
        self.stats["faces"] += len(boxes); self.stats["matched"] += matched; self.stats["unrecognized"] += unrec
        return {"faces": len(boxes), "matched": matched, "unrecognized": unrec, "health": self.guard.health.value}

    def _log_encounter(self, person_id: str, conf: float, place: str, now: float) -> None:
        seen = round(now / 60) * 60
        exp = seen + self.cfg.encounter_retention_days * 86400
        self.db.execute("INSERT INTO fm_encounter(entry_id, person_id, seen_at, place, confidence, expires_at) VALUES(?,?,?,?,?,?)",
                        (uuid.uuid4().hex, person_id, seen, self._lk.seal(place.encode(), b"place"), round(conf, 4), exp))

    def _place_tag(self, place: str) -> str:
        import hashlib, hmac
        return hmac.new(self._lk_tag_key(), place.encode(), hashlib.sha256).hexdigest()[:24]

    def _lk_tag_key(self) -> bytes:
        return self.keys.subkey("face-log-tag")

    def _count_unrecognized(self, place: str, now: float) -> None:
        """Research mode: (place, date, time bucket) -> count. No face data of any kind."""
        b = self.cfg.time_bucket_minutes * 60
        bucket = int(now // b) * b
        date = time.strftime("%Y-%m-%d", time.gmtime(now))
        exp = bucket + self.cfg.encounter_retention_days * 86400
        tag = self._place_tag(place)
        cur = self.db.execute("UPDATE fm_unrecognized_count SET count = count + 1 WHERE place_tag=? AND date=? AND time_bucket=?",
                              (tag, date, bucket))
        if cur.rowcount == 0:
            self.db.execute("INSERT INTO fm_unrecognized_count(place_tag, place, date, time_bucket, count, expires_at) VALUES(?,?,?,?,1,?)",
                            (tag, self._lk.seal(place.encode(), b"place"), date, bucket, exp))

    # ---- enrollment (EnrollmentPort, spec 7.3) -----------------------------------------
    def _require_personal(self) -> None:
        if opt_in_record(self.db) is None:
            raise FaceMatchLocked("M-16 is off or not opted in")
        if "personal" not in self.modes:
            raise FaceMatchLocked("enrollment needs personal mode")

    def start_session(self) -> str:
        self._require_personal()
        sid = uuid.uuid4().hex
        self._sessions[sid] = set()
        return sid

    def add_person(self, session_id: str, label: str, consent_id: str | None = None, guardian: bool = False,
                   no_consent_override: bool = False, person_id: str | None = None) -> str:
        self._require_personal()
        if session_id not in self._sessions:
            raise KeyError("unknown session")
        label = check_text(label.strip())
        if not label:
            raise ValueError("give the person a name or label")
        # the previous person's buffer in this session is cleared when the next one starts
        for prev in list(self._sessions[session_id]):
            cap = self._captures.pop(prev, None)
            if cap:
                cap.wipe()   # an unfinished previous person is discarded, never mixed in
            self._sessions[session_id].discard(prev)
        must = self.cfg.require_enrollment_consent or self.strict or guardian
        note = "recorded"
        if consent_id is not None or must and not no_consent_override:
            rec = self.consents.require(consent_id, "face_enrollment")
            if guardian and "guardian" not in rec.scopes:
                raise ConsentError("a minor's enrollment needs the guardian's recorded agreement (scope 'guardian')")
        elif must and no_consent_override:
            if self.strict:
                raise ConsentError("your jurisdiction setting requires a consent record for enrollment")
            note = "none (user override)"
            log_change(self.db, self.chain, "consent_override", {"session": session_id})
        else:
            note = "none"
        pid = person_id or uuid.uuid4().hex
        self._captures[pid] = _Capture(session_id, label, consent_id, guardian, note, replaces=person_id is not None,
                                       photo=self._photo_candidate(consent_id))
        self._sessions[session_id].add(pid)
        return pid

    def _photo_candidate(self, consent_id: str | None) -> PhotoCandidate | None:
        """A RAM photo candidate only for a capture whose consent has the photo scope: without it no crop is kept."""
        return PhotoCandidate() if self._photo_consent_ok(consent_id) else None

    def _cap(self, pid: str) -> _Capture:
        if pid not in self._captures:
            raise KeyError("person is not in an open capture")
        return self._captures[pid]

    def needed_angles(self, pid: str) -> set:
        have = {a for a, _ in self._cap(pid).shots}
        return {Angle(a) for a in self.cfg.enroll_required_angles} - have

    def submit_shot(self, pid: str, frame: Any, angle: Angle | str) -> ShotResult:
        self._require_personal()
        c, angle = self._cap(pid), Angle(angle)
        if len(c.shots) >= self.cfg.enroll_max_shots:
            return ShotResult(False, "enough", "enough shots for this person, finish to save", sorted(a.value for a in self.needed_angles(pid)))
        ok, boxes = self.guard.call("detect", frame, self.guard.timeout_s,
                                    validate=lambda v: isinstance(v, list) and all(isinstance(b, FaceBox) for b in v))
        if not ok:
            return ShotResult(False, "backend", "face engine unavailable, try again", sorted(a.value for a in self.needed_angles(pid)))
        good, code, reason = check_shot(frame, boxes, angle, self.limits, c.front_pitch(),
                                        self.cfg.pose_up_delta, self.cfg.pose_down_delta)
        if good:
            ok, e = self.guard.call("embed", frame, boxes[0], validate=lambda v: isinstance(v, Embedding) and v.valid())
            if not ok:
                good, code, reason = False, "backend", "face engine unavailable, try again"
            else:
                c.shots.append((angle, e))
                if angle is Angle.FRONT and boxes[0].pitch is not None:
                    c.front_pitches.append(float(boxes[0].pitch))
                if angle is Angle.FRONT and c.photo is not None:
                    c.photo.offer(frame, boxes[0])   # this person's own accepted FRONT shot only; best one kept
        return ShotResult(good, code, reason, sorted(a.value for a in self.needed_angles(pid)))

    def finish_person(self, pid: str, now: float | None = None) -> TemplateStatus:
        c = self._cap(pid)
        missing = sorted(a.value for a in self.needed_angles(pid))
        need = self.cfg.enroll_min_shots
        if missing or len(c.shots) < need:
            return TemplateStatus(False, missing, len(c.shots), need,
                                  f"still needed: {', '.join(missing) or 'more shots'} ({len(c.shots)}/{need} good shots)")
        check_persist("face_template", True, source="M-16", m16_enabled=True)
        vec = unit(np.mean([unit(e.vec) for _, e in c.shots], axis=0)).astype(np.float32)
        now = self.clock() if now is None else now
        # A face re-enrollment keeps the person's M-17 voice template: re-sealed under the new DEK in the same step.
        voice = self._open_voice(pid) if c.replaces else None
        dek_raw = Sealer.new_key()
        dek = Sealer(dek_raw)
        row = (pid, dek.seal(c.label.encode(), b"name:" + pid.encode()), dek.seal(vec.tobytes(), b"tpl:" + pid.encode()),
               self._fk.seal(dek_raw, pid.encode()), self.backend.model_id, self.backend.model_version, c.consent_id,
               c.consent_note, int(c.guardian), len(c.shots), ",".join(sorted({a.value for a, _ in c.shots})), now, None)
        # The photo (ADR-0057 decision 4): only with a current "photo" consent scope; a re-enrollment replaces it.
        photo, photo_note = self._photo_for(pid, c, dek)
        try:
            with self._tx():
                if c.replaces:
                    self._erase_person_key(pid)   # old template and photo crypto-erased; encounter history kept
                self.db.execute("INSERT INTO fm_enrolled(person_id, display_name, template, dek, model_id, model_version, consent_id, "
                                "consent_note, guardian_enrolled, shot_count, angles_covered, enrolled_at, expires_at) "
                                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
                if voice is not None:
                    self._reseal_voice(pid, dek, voice)
                if photo is not None:
                    self.db.execute(f"INSERT INTO {PHOTO_TABLE}(person_id, photo, created_at) VALUES(?,?,?)", (pid, photo, now))
        finally:
            if voice is not None:
                voice.fill(0)
        vec.fill(0)
        n = len(c.shots)
        c.wipe(); del self._captures[pid]
        self._sessions.get(c.session_id, set()).discard(pid)
        self._drop_templates()
        detail: dict = {"shots": n}
        if photo is not None:
            detail["photo"] = True
        log_change(self.db, self.chain, "reenrolled" if c.replaces else "enrolled", detail)
        return TemplateStatus(True, [], n, need, f"Template complete for {c.label}. {photo_note}")

    def end_session(self, session_id: str) -> None:
        for pid in self._sessions.pop(session_id, set()):
            cap = self._captures.pop(pid, None)
            if cap:
                cap.wipe()

    def capture_buffer_size(self) -> int:
        return sum(len(c.shots) for c in self._captures.values())

    def reenroll(self, pid: str) -> str:
        self._require_personal()
        if self.db.execute("SELECT 1 FROM fm_enrolled WHERE person_id=?", (pid,)).fetchone() is None:
            raise KeyError("unknown person")
        label, consent_id, guardian = self._person_meta(pid)
        sid = self.start_session()
        self._captures[pid] = _Capture(sid, label, consent_id, guardian, "recorded" if consent_id else "none", replaces=True,
                                       photo=self._photo_candidate(consent_id))
        self._sessions[sid].add(pid)
        return sid

    def _person_meta(self, pid: str) -> tuple[str, str | None, bool]:
        dek, name, consent_id, guardian = self.db.execute(
            "SELECT dek, display_name, consent_id, guardian_enrolled FROM fm_enrolled WHERE person_id=?", (pid,)).fetchone()
        return self._dek(pid, dek).open(name, b"name:" + pid.encode()).decode(), consent_id, bool(guardian)

    def _erase_person_key(self, pid: str) -> None:
        self.db.execute("DELETE FROM fm_enrolled WHERE person_id=?", (pid,))
        if self.db.has_table(PHOTO_TABLE):   # foreign keys are not enforced, so the photo row is deleted explicitly
            self.db.execute(f"DELETE FROM {PHOTO_TABLE} WHERE person_id=?", (pid,))
        self._drop_templates()

    def delete_person(self, pid: str, keep_history: bool = False) -> WipeReport:
        row = self.db.execute("SELECT template FROM fm_enrolled WHERE person_id=?", (pid,)).fetchone()
        if row is None:
            raise KeyError("unknown person")
        old_tpl = row[0]
        vrow = self.db.execute("SELECT template FROM vl_voice WHERE person_id=?", (pid,)).fetchone() if self.db.has_table("vl_voice") else None
        old_photo = self._photo_blob(pid)
        self._erase_person_key(pid)   # the photo row goes with it
        self._last_match.pop(pid, None)
        if not keep_history:
            self.db.execute("DELETE FROM fm_encounter WHERE person_id=?", (pid,))
        for t in VOICE_TABLES:   # the voice template and its "heard" entries go with the person (ADR-0057)
            if self.db.has_table(t):
                self.db.execute(f"DELETE FROM {t} WHERE person_id=?", (pid,))
        self._compact()
        gone = self.db.execute("SELECT 1 FROM fm_enrolled WHERE person_id=?", (pid,)).fetchone() is None
        undecryptable = not self._can_open_with_any(old_tpl, b"tpl:" + pid.encode())
        absent = not self._bytes_on_disk(old_tpl)
        v_gone = not self.db.has_table("vl_voice") or self.db.execute("SELECT 1 FROM vl_voice WHERE person_id=?", (pid,)).fetchone() is None
        v_undecryptable = vrow is None or not self._can_open_with_any(vrow[0], VOICE_AAD + pid.encode())
        v_absent = vrow is None or not self._bytes_on_disk(vrow[0])
        p_gone = self._photo_blob(pid) is None
        p_undecryptable = old_photo is None or not self._can_open_with_any(old_photo, PHOTO_AAD + pid.encode())
        p_absent = old_photo is None or not self._bytes_on_disk(old_photo)
        ok = gone and undecryptable and absent and v_gone and v_undecryptable and v_absent and p_gone and p_undecryptable and p_absent
        detail: dict = {"templates_destroyed": 1, "verified": ok}
        if vrow is not None:
            detail["voice_templates_destroyed"] = 1
        if old_photo is not None:
            detail["photos_destroyed"] = 1
        log_change(self.db, self.chain, "person_deleted", detail)
        what = _destroyed(["1 template"] + (["1 voice template"] if vrow is not None else []) + (["1 photo"] if old_photo is not None else []))
        return WipeReport(1, ok, {"row_absent": gone, "undecryptable": undecryptable, "ciphertext_absent": absent,
                                  "voice_row_absent": v_gone, "voice_undecryptable": v_undecryptable, "voice_ciphertext_absent": v_absent,
                                  "photo_row_absent": p_gone, "photo_undecryptable": p_undecryptable, "photo_ciphertext_absent": p_absent},
                          f"{what}, verified" if ok else "wipe NOT verified", photos_destroyed=int(old_photo is not None))

    def _compact(self) -> None:
        self.db.execute("VACUUM")
        if self.db.path != ":memory:":
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def _bytes_on_disk(self, blob: bytes) -> bool:
        if self.db.path == ":memory:":
            return False
        for suffix in ("", "-wal", "-journal"):
            p = Path(self.db.path + suffix)
            if p.exists() and blob in p.read_bytes():
                return True
        return False

    def _candidate_keys(self) -> list[bytes]:
        """Every key still reachable from what is stored on this device."""
        ks: list[bytes] = []
        for purpose in ("face-template", "face-log", "face-log-tag"):
            try:
                ks.append(self.keys.subkey(purpose))
            except Exception:  # noqa: BLE001 - locked/erased key store
                pass
        try:
            ks.append(self.keys.master)
        except Exception:  # noqa: BLE001
            pass
        face_keys = []
        for f, purpose in ((KEY_FILE, "face-template"), (LOG_KEY_FILE, "face-log")):
            p = self.data_dir / f
            if p.exists():
                try:
                    face_keys.append(Sealer(self.keys.subkey(purpose)).open(p.read_bytes(), b"m16:" + purpose.encode()))
                except Exception:  # noqa: BLE001
                    pass
        ks += face_keys
        for pid, dek in self.db.execute("SELECT person_id, dek FROM fm_enrolled").fetchall():
            for fk in face_keys:
                try:
                    ks.append(Sealer(fk).open(dek, pid.encode()))
                except Exception:  # noqa: BLE001
                    pass
        return ks

    def _can_open_with_any(self, blob: bytes, aad: bytes) -> bool:
        for k in self._candidate_keys():
            try:
                Sealer(k).open(blob, aad)
                return True
            except Exception:  # noqa: BLE001
                pass
        return False

    # ---- review log --------------------------------------------------------------------
    def _open_place(self, blob: bytes) -> str:
        return self._lk.open(blob, b"place").decode()

    def review_log(self, days: int | None = None, now: float | None = None) -> dict:
        now = self.clock() if now is None else now
        window_days = days or self.cfg.encounter_retention_days
        since = now - window_days * 86400
        people = []
        for pid, dek, name, consent_note, guardian, enrolled_at in self.db.execute(
                "SELECT person_id, dek, display_name, consent_note, guardian_enrolled, enrolled_at FROM fm_enrolled ORDER BY enrolled_at").fetchall():
            nm = self._dek(pid, dek).open(name, b"name:" + pid.encode()).decode()
            places: dict[str, int] = {}
            entries = []
            for eid, seen, place, conf in self.db.execute(
                    "SELECT entry_id, seen_at, place, confidence FROM fm_encounter WHERE person_id=? AND seen_at>=? ORDER BY seen_at",
                    (pid, since)).fetchall():
                pl = self._open_place(place)
                places[pl] = places.get(pl, 0) + 1
                entries.append({"entry_id": eid, "seen_at": seen, "place": pl, "confidence": conf})
            summary = [
                f"{n} possible face {'match' if n == 1 else 'matches'} for {nm} in processed frames "
                f"at {pl} over the last {window_days} {'day' if window_days == 1 else 'days'}"
                for pl, n in sorted(places.items())
            ]
            people.append({"person_id": pid, "name": nm, "consent": consent_note, "guardian_enrolled": bool(guardian),
                           "no_consent_badge": consent_note != "recorded", "hint": HINT if self._weaker(pid, enrolled_at) else "",
                           "summary": summary, "entries": entries})
        counts = [{"place": self._open_place(p), "date": d, "time_bucket": tb, "count": c}
                  for p, d, tb, c in self.db.execute(
                      "SELECT place, date, time_bucket, count FROM fm_unrecognized_count WHERE time_bucket>=? ORDER BY time_bucket", (since,)).fetchall()]
        out = {"people": people, "unrecognized_counts": counts}
        return check_output(out)

    def _weaker(self, pid: str, enrolled_at: float) -> bool:
        confs = [r[0] for r in self.db.execute("SELECT confidence FROM fm_encounter WHERE person_id=? ORDER BY seen_at", (pid,))]
        n = self.cfg.reenroll_window_matches
        if len(confs) < n:
            return False
        first_week = [c for (c,) in self.db.execute("SELECT confidence FROM fm_encounter WHERE person_id=? AND seen_at<?",
                                                     (pid, enrolled_at + 7 * 86400))]
        recent = statistics.median(confs[-n:])
        if first_week and recent <= statistics.median(first_week) - self.cfg.reenroll_confidence_drop:
            return True
        return recent < self.cfg.match_threshold + REENROLL_MARGIN

    def delete_entry(self, entry_id: str) -> bool:
        n = self.db.execute("DELETE FROM fm_encounter WHERE entry_id=?", (entry_id,)).rowcount
        if n:
            log_change(self.db, self.chain, "entry_deleted", {"entries": n})
        return bool(n)

    def export_log(self, explicit_user_action: bool = False) -> dict:
        """Encounter logs/counts only, by explicit per-export action. Templates: never."""
        if not (self.cfg.allow_log_export and explicit_user_action):
            raise ExportBlocked("log export is off; enable allow_log_export and confirm the export")
        log = self.review_log()
        for p in log["people"]:
            p.pop("person_id", None)
        log_change(self.db, self.chain, "log_exported", {"people": len(log["people"])})
        return log

    def purge_expired(self, now: float | None = None) -> int:
        return purge(self.db, self.clock() if now is None else now)

    # ---- disable + verified wipe (spec 14) ---------------------------------------------
    def disable(self, keep_logs: bool = False) -> WipeReport:
        self._drop_templates()
        for c in self._captures.values():
            c.wipe()
        self._captures.clear(); self._sessions.clear()
        try:
            self.backend.close()
        except Exception:  # noqa: BLE001
            pass
        sample = self.db.execute("SELECT person_id, dek, template FROM fm_enrolled").fetchall()
        voices = self.db.execute("SELECT person_id, template FROM vl_voice").fetchall() if self.db.has_table("vl_voice") else []
        photos = self.db.execute(f"SELECT person_id, photo FROM {PHOTO_TABLE}").fetchall() if self.db.has_table(PHOTO_TABLE) else []
        m17_was_on = _wipe_voice_link(self.db)   # M-17 cannot outlive M-16: voices, "heard" entries, opt-in
        n = len(sample)
        _overwrite_unlink(self.data_dir / KEY_FILE)
        if photos:
            self.db.execute(f"DELETE FROM {PHOTO_TABLE}")
        self.db.execute("DELETE FROM fm_enrolled")
        if not keep_logs:
            self.db.execute("DELETE FROM fm_encounter")
            self.db.execute("DELETE FROM fm_unrecognized_count")
            _overwrite_unlink(self.data_dir / LOG_KEY_FILE)
        self.db.execute("DELETE FROM fm_state WHERE k='opt_in'")
        self._last_match.clear()
        self._compact()
        rows = self.db.execute("SELECT COUNT(*) FROM fm_enrolled").fetchone()[0]
        undecryptable = all(not self._can_open_with_any(t, b"tpl:" + pid.encode()) and not self._can_open_with_any(d, pid.encode())
                            for pid, d, t in sample)
        absent = not any(self._bytes_on_disk(t) for _, _, t in sample)
        key_gone = not (self.data_dir / KEY_FILE).exists()
        v_rows = voice_rows(self.db)
        v_undecryptable = all(not self._can_open_with_any(t, VOICE_AAD + pid.encode()) for pid, t in voices)
        v_absent = not any(self._bytes_on_disk(t) for _, t in voices)
        p_rows = photo_rows(self.db)
        p_undecryptable = all(not self._can_open_with_any(b, PHOTO_AAD + pid.encode()) for pid, b in photos)
        p_absent = not any(self._bytes_on_disk(b) for _, b in photos)
        ok = (rows == 0 and undecryptable and absent and key_gone and v_rows == 0 and v_undecryptable and v_absent
              and p_rows == 0 and p_undecryptable and p_absent)
        detail: dict = {"templates_destroyed": n, "verified": ok, "logs_kept": keep_logs}
        if voices:
            detail["voice_templates_destroyed"] = len(voices)
        if photos:
            detail["photos_destroyed"] = len(photos)
        log_change(self.db, self.chain, "disabled_wiped", detail)
        if voices or m17_was_on:
            log_voice_link_wiped(self.db, self.chain, len(voices), ok)
        self._fk = self._lk = None  # type: ignore[assignment]
        self.modes = set()
        what = _destroyed([f"{n} templates"] + ([f"{len(photos)} photos"] if photos else []))
        return WipeReport(n, ok, {"rows": rows, "undecryptable": undecryptable, "ciphertext_absent": absent, "key_file_absent": key_gone,
                                  "voice_rows": v_rows, "voice_undecryptable": v_undecryptable, "voice_ciphertext_absent": v_absent,
                                  "photo_rows": p_rows, "photo_undecryptable": p_undecryptable, "photo_ciphertext_absent": p_absent},
                          f"{what}, verified" if ok else "wipe NOT verified", photos_destroyed=len(photos))


def purge(db, now: float) -> int:
    """Retention (ADR-0039): expired encounters/counts, and templates past a set end date."""
    n = 0
    for t in ("fm_encounter", "fm_unrecognized_count"):
        if db.has_table(t):
            n += db.execute(f"DELETE FROM {t} WHERE expires_at < ?", (now,)).rowcount
    if db.has_table("fm_enrolled"):
        ids = [r[0] for r in db.execute("SELECT person_id FROM fm_enrolled WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))]
        for pid in ids:
            for t in ("fm_encounter", *VOICE_TABLES, PHOTO_TABLE):
                if db.has_table(t):
                    db.execute(f"DELETE FROM {t} WHERE person_id=?", (pid,))
        n += db.execute("DELETE FROM fm_enrolled WHERE expires_at IS NOT NULL AND expires_at < ?", (now,)).rowcount
    return n


def disabled_wipe_without_module(db, data_dir: str | Path) -> int:
    """Used when the flag is turned off while the module isn't loaded (e.g. CLI/config):
    same destruction, no backend needed. M-17 voice data and its opt-in go too (check with voice_rows()),
    and every face photo (check with photo_rows())."""
    d = Path(data_dir).expanduser()
    n = db.execute("SELECT COUNT(*) FROM fm_enrolled").fetchone()[0] if db.has_table("fm_enrolled") else 0
    _wipe_voice_link(db)
    _overwrite_unlink(d / KEY_FILE); _overwrite_unlink(d / LOG_KEY_FILE)
    for t in (PHOTO_TABLE, "fm_enrolled", "fm_encounter", "fm_unrecognized_count"):
        if db.has_table(t):
            db.execute(f"DELETE FROM {t}")
    if db.has_table("fm_state"):
        db.execute("DELETE FROM fm_state WHERE k='opt_in'")
    db.execute("VACUUM")
    if db.path != ":memory:":
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return n


def _wipe_voice_link(db) -> bool:
    """Delete every M-17 row and the M-17 opt-in. Returns True if M-17 was opted in."""
    was_on = db.has_table("vl_state") and db.execute("SELECT 1 FROM vl_state WHERE k='opt_in'").fetchone() is not None
    for t in VOICE_TABLES:
        if db.has_table(t):
            db.execute(f"DELETE FROM {t}")
    if db.has_table("vl_state"):
        db.execute("DELETE FROM vl_state WHERE k='opt_in'")
    return was_on


def voice_rows(db) -> int:
    """Number of M-17 voice templates stored (0 when the table does not exist)."""
    return db.execute("SELECT COUNT(*) FROM vl_voice").fetchone()[0] if db.has_table("vl_voice") else 0


def photo_rows(db) -> int:
    """Number of face photos stored (0 when the table does not exist)."""
    return db.execute(f"SELECT COUNT(*) FROM {PHOTO_TABLE}").fetchone()[0] if db.has_table(PHOTO_TABLE) else 0


def _destroyed(parts: list[str]) -> str:
    """["1 template", "1 photo"] -> "1 template and 1 photo destroyed"."""
    return (", ".join(parts[:-1]) + " and " + parts[-1] if len(parts) > 1 else parts[0]) + " destroyed"


def _merge_photo_reports(reps: list[WipeReport]) -> WipeReport:
    n = sum(r.photos_destroyed for r in reps)
    ok = all(r.verified for r in reps)
    checks = {k: all(r.checks.get(k, True) for r in reps)
              for k in ("photo_row_absent", "photo_undecryptable", "photo_ciphertext_absent")}
    msg = "no photo stored" if not n else f"{n} photo{'s' if n != 1 else ''} destroyed, " + ("verified" if ok else "NOT verified")
    return WipeReport(0, ok, checks, msg, photos_destroyed=n)


def log_voice_link_wiped(db, chain, n: int, verified: bool) -> None:
    """M-17's own entry in the shared tamper-evident chain when M-16's disable takes M-17 with it."""
    from ..voice_link.optin import log_change as voice_log_change
    voice_log_change(db, chain, "disabled_wiped", {"voice_templates_destroyed": n, "verified": verified, "by": "M-16 disable"})


__all__ = ["FaceMatchModule", "FaceMatchConfig", "FaceMatchLocked", "ExportBlocked", "ShotResult", "WipeReport",
           "OptInError", "purge"]


def consent_review_db(db) -> list[dict]:
    """Status helper without loading templates: enrolled people whose consent record is missing."""
    known = {r[0] for r in db.execute("SELECT consent_id FROM consent_record")}
    return [{"person_id": pid, "consent_id": cid, "status": "consent record missing"}
            for pid, cid in db.execute("SELECT person_id, consent_id FROM fm_enrolled ORDER BY enrolled_at")
            if cid and cid not in known]
