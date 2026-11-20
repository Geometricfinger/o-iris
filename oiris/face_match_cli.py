"""`oiris face-match ...`: opt-in, guided enrollment, review log, deletes, disable + wipe.
Only reached through an explicit `face-match` command; normal runs never import M-16.

  oiris face-match status
  oiris face-match enable --mode personal [--mode research] [--jurisdiction us-other]
  oiris face-match fetch-models
  oiris face-match consent --subject CODE [--days 30] [--id ID] [--scope SCOPES]   record consent (you type YES)
        SCOPES: comma-separated from face_enrollment (default), voice_enrollment (M-17 voice link), photo, guardian
  oiris face-match consent --withdraw --id ID   withdraw a consent record
  oiris face-match consent --withdraw --id ID --scope photo   withdraw only the photo scope: photos erased, verified
  oiris face-match enroll [--camera auto|N]     guided capture, as many people as you like
  oiris face-match list                         enrolled people, with "photo: yes/no" (never the image)
  oiris face-match review
  oiris face-match delete-entry --id ENTRY | delete-person --id PERSON   (also erases that person's voice template and photo)
  oiris face-match photo-delete --person ID     erase that person's photo, verified; the enrollment stays
  oiris face-match disable [--keep-logs]        wipes every face and voice template and photo, verified (voice link goes off too)

Photo (ADR-0057 decision 4): with the consent scope "photo", one face crop from the person's own best
front enrollment shot is kept, sealed with their key, only to show who was recognised. Never exported.

Cameras: OpenCV numbers cameras in its own order, which can differ from the list macOS shows. With an
iPhone nearby (Continuity Camera), camera 0 may be the phone rather than the Mac's FaceTime camera, and
the phone may chime while it is probed. `--camera auto` tries cameras 0..4 and uses the first one that
sees exactly one face.
"""
from __future__ import annotations
import getpass, json, os, re, secrets, sys, time, uuid
from pathlib import Path
from typing import Any, Callable

from .config import Config

AUTO_PROBE_INDICES = range(5)
AUTO_PROBE_FRAMES = 20   # about 0.7 s at 30 fps: covers the dark first frames of a just-opened Mac camera
WARMUP_FRAMES = 15       # frames read and dropped before a capture (measured on a Mac's built-in camera, 2026-11-15)
CONSENT_SCOPE = "face_enrollment"
# The only scopes `consent --scope` may record (ADR-0057 decision 6), with what the person agreed to.
CONSENT_SCOPES = {
    "face_enrollment": "to have their face enrolled",
    "voice_enrollment": "to have their voice enrolled (voice link)",
    "photo": "to have one face photo kept to show who was recognised",
    "guardian": "as the guardian of a minor, to the scopes above for the minor",
}
CONSENT_HINT = "record one with: oiris face-match consent --subject CODE"
_CODE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def _ctx(cfg: Config):
    from .security.keys import KeyStore
    from .store.db import DB
    from .modules.face_match.optin import DbChain
    d = Path(cfg.data_dir).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    ks = KeyStore(d / "keys.json")
    pw = os.environ.get("OIRIS_PASSPHRASE") or getpass.getpass("O Iris passphrase: ")
    ks.unlock(pw) if ks.exists() else ks.create(pw)
    db = DB(d / "asb.db", ks.subkey("db"))
    return d, ks, db, DbChain(db, ks.subkey("face-chain"))


def _save_flag(cfg_path: Path, cfg: Config, on: bool, modes: list[str] | None = None, jurisdiction: str | None = None) -> None:
    raw = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    raw.setdefault("flags", {})["M-16"] = on
    if not on:
        raw["flags"]["M-17"] = False   # the voice link cannot outlive face matching (ADR-0057)
    fm = raw.setdefault("face_match", {})
    if modes is not None:
        fm["modes"] = modes
    if jurisdiction:
        fm["jurisdiction"] = jurisdiction
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(raw, indent=1), encoding="utf-8")


def _ticks(needed: set, order: list[str]) -> str:
    return "  ".join(f"[{' ' if a in needed else 'x'}] {a}" for a in order)


def run(a, cfg: Config, cfg_path: Path, inp: Callable[[str], str] = input, out=print, backend: Any = None) -> int:
    from .modules.face_match import optin
    sub = a.sub or "status"
    if sub == "fetch-models":
        from .modules.face_match.backends.opencv_zoo import MODELS, fetch_models
        md = cfg.face_match.get("model_dir") or str(Path(cfg.data_dir).expanduser() / "models" / "face_match")
        out("Downloading face models (YuNet: MIT, SFace: Apache-2.0; see THIRD_PARTY.md) to " + md)
        for k, p in fetch_models(md).items():
            out(f"  {k}: {p.name} ({MODELS[k]['license']}) sha256 ok")
        return 0
    d, ks, db, chain = _ctx(cfg)
    try:
        if sub == "status":
            rec = optin.opt_in_record(db)
            from .modules.face_match.core import consent_review_db
            out(json.dumps({"flag_on": cfg.enabled("M-16"), "opted_in": rec is not None, "opt_in": rec,
                            "log_entries": chain.verify(), "consent_review": consent_review_db(db)}, indent=1))
            return 0
        if sub == "enable":
            modes = a.mode or ["personal"]
            jur = a.jurisdiction or "strictest"
            text = optin.warning_for(jur)
            out(text)
            typed = inp(f'\nType "{optin.CONFIRM_PHRASE}" to turn face matching on: ')
            rec = optin.record_opt_in(db, chain, typed, modes, device_id=uuid.uuid5(uuid.NAMESPACE_URL, str(d)).hex[:12],
                                      jurisdiction=jur, warning_shown=text)
            _save_flag(cfg_path, cfg, True, rec["modes"], rec["jurisdiction"])
            out(f"Face matching is on ({', '.join(rec['modes'])}). Logged. Turn it off any time: oiris face-match disable")
            return 0
        if sub == "disable":
            from .modules.face_match.core import disabled_wipe_without_module, log_voice_link_wiped, photo_rows, voice_rows
            voices, photos = voice_rows(db), photo_rows(db)
            m17_on = db.execute("SELECT 1 FROM vl_state WHERE k='opt_in'").fetchone() is not None
            n = disabled_wipe_without_module(db, d)   # M-17 cannot outlive M-16: voice data goes too; so do photos
            ok = (db.execute("SELECT COUNT(*) FROM fm_enrolled").fetchone()[0] == 0 and not (d / "face_match.key").exists()
                  and voice_rows(db) == 0 and photo_rows(db) == 0)
            detail = {"templates_destroyed": n, "verified": ok, "logs_kept": False}
            if voices:
                detail["voice_templates_destroyed"] = voices
            if photos:
                detail["photos_destroyed"] = photos
            optin.log_change(db, chain, "disabled_wiped", detail)
            if voices or m17_on:
                log_voice_link_wiped(db, chain, voices, ok)
            _save_flag(cfg_path, cfg, False)
            voice_note = f" {voices} voice templates destroyed; voice link is off." if voices or m17_on else ""
            photo_note = f" {photos} photos destroyed." if photos else ""
            out(f"Face matching is off. {n} templates destroyed, {'verified' if ok else 'NOT verified'}.{voice_note}{photo_note}")
            return 0 if ok else 1
        from .modules import face_match_gate
        from .registry import default_registry
        cfg = Config.load(cfg_path) if cfg_path.exists() else cfg
        if sub == "consent":
            return _consent(a, db, inp, out, lambda: face_match_gate.load(cfg, db, ks, default_registry(cfg.plugins),
                                                                          chain=chain, backend=backend))
        m = face_match_gate.load(cfg, db, ks, default_registry(cfg.plugins), chain=chain, backend=backend)
        if m is None:
            out("Face matching is off. Turn it on first: oiris face-match enable"); return 2
        if sub == "review":
            log = m.review_log()
            for p in log["people"]:
                badge = " [no consent recorded]" if p["no_consent_badge"] else ""
                out(f"{p['name']}{badge} (id {p['person_id'][:8]}) {p['hint']}")
                for line in p["summary"]:
                    out("  " + line)
                for e in p["entries"]:
                    out(f"    entry {e['entry_id']}  {e['place']}  confidence {e['confidence']:.2f}")
            for c in log["unrecognized_counts"]:
                out(f"unrecognized faces: {c['count']} at {c['place']} on {c['date']}")
            return 0
        if sub == "delete-entry":
            out("deleted" if m.delete_entry(a.id) else "no such entry"); return 0
        if sub == "delete-person":
            pid = next((r[0] for r in db.execute("SELECT person_id FROM fm_enrolled") if r[0].startswith(a.id or "?")), None)
            if pid is None:
                out("no such person"); return 1
            out(m.delete_person(pid).message); return 0
        if sub == "list":
            people = [r[0] for r in db.execute("SELECT person_id FROM fm_enrolled ORDER BY enrolled_at").fetchall()]
            for pid in people:
                out(f"{m.person_name(pid)} (id {pid[:8]})  photo: {'yes' if m.has_photo(pid) else 'no'}")
            if not people:
                out("no one is enrolled")
            return 0
        if sub == "photo-delete":
            prefix = getattr(a, "person", None) or "?"
            pid = next((r[0] for r in db.execute("SELECT person_id FROM fm_enrolled") if r[0].startswith(prefix)), None)
            if pid is None:
                out("no such person"); return 1
            rep = m.delete_photo(pid)
            out(f"{rep.message}; the enrollment is kept" if rep.photos_destroyed else "no photo stored for this person")
            return 0 if rep.verified else 1
        if sub == "enroll":
            return _enroll(m, a, inp, out)
        out(f"unknown face-match command {sub}"); return 2
    finally:
        db.close()


def _scopes(value: Any) -> tuple[list[str] | None, str]:
    """--scope: comma-separated subset of CONSENT_SCOPES (default face_enrollment only)."""
    if value is None:
        return [CONSENT_SCOPE], ""
    got = [s.strip() for s in str(value).split(",") if s.strip()]
    bad = [s for s in got if s not in CONSENT_SCOPES]
    if not got or bad:
        return None, f"--scope must be a comma-separated list of: {', '.join(CONSENT_SCOPES)}"
    return [s for s in CONSENT_SCOPES if s in got], ""


def _consent(a, db, inp, out, load_module: Callable[[], Any]) -> int:
    """Record (only after a typed YES) or withdraw a consent record in the local store. The scopes default
    to face_enrollment; --scope may add voice_enrollment, photo and guardian (ADR-0057)."""
    from .guard.consent import ConsentRecord, ConsentRegistry
    reg = ConsentRegistry(db)
    cid = getattr(a, "id", None)
    if getattr(a, "withdraw", False):
        rec = reg.get(cid or "")
        if rec is None:
            out("no such consent record"); return 1
        if rec.withdrawn:
            out(f"consent {rec.consent_id} was already withdrawn"); return 0
        if getattr(a, "scope", None) is not None:
            return _withdraw_photo_scope(a, rec, reg, out, load_module)
        reg.withdraw(rec.consent_id)
        out(f"Consent {rec.consent_id} withdrawn.")
        try:   # building the module erases templates enrolled under a withdrawn consent
            if load_module() is not None:
                out("Face templates enrolled under it have been erased.")
        except Exception as e:  # noqa: BLE001 - the withdrawal is stored; templates go on the next load
            out(f"Face templates enrolled under it are erased the next time face matching loads ({e}).")
        return 0
    subject = (getattr(a, "subject", None) or "").strip()
    if not _CODE.fullmatch(subject):
        out("give --subject CODE: a short code for the person (letters, digits, - _ .), not their name"); return 2
    days = getattr(a, "days", None)
    days = 30 if days is None else days
    if not 1 <= days <= 365:
        out("--days must be between 1 and 365"); return 2
    if cid is not None and not _CODE.fullmatch(cid):
        out("--id may use letters, digits, - _ . only"); return 2
    scopes, why = _scopes(getattr(a, "scope", None))
    if scopes is None:
        out(why); return 2
    cid = cid or f"consent-{secrets.token_hex(4)}"
    if reg.known(cid):
        out(f"consent {cid} already exists; nothing recorded"); return 2
    expires = time.time() + days * 86400
    out("This will record:")
    out(f"  consent ID:   {cid}")
    out(f"  subject code: {subject}")
    out(f"  scope:        {', '.join(scopes)}")
    out(f"  expires:      {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(expires))} ({days} days)")
    if scopes == [CONSENT_SCOPE]:
        out("Record it only if the person this code stands for has agreed to have their face enrolled.")
    else:
        out("Record it only if the person this code stands for has agreed:")
        for s in scopes:
            out(f"  - {s}: {CONSENT_SCOPES[s]}")
    if inp('Type "YES" to record it: ').strip() != "YES":
        out("Not recorded."); return 1
    reg.add(ConsentRecord(cid, subject, set(scopes), expires_utc=expires))
    out(f"Consent recorded. Enter this consent ID during enrollment: {cid}")
    return 0


def _withdraw_photo_scope(a, rec, reg, out, load_module: Callable[[], Any]) -> int:
    """`consent --withdraw --id ID --scope photo`: only the photo scope goes; the rest of the record stays.
    Photos of people enrolled under it are erased now (verified) when face matching loads, else on its next load."""
    from .modules.face_match.photo import PHOTO_SCOPE
    if [s.strip() for s in str(a.scope).split(",") if s.strip()] != [PHOTO_SCOPE]:
        out(f"only --scope {PHOTO_SCOPE} can be withdrawn on its own; to withdraw everything, leave out --scope"); return 2
    if PHOTO_SCOPE not in rec.scopes:
        out(f"consent {rec.consent_id} has no {PHOTO_SCOPE} scope; nothing to withdraw"); return 0
    try:
        m = load_module()
    except Exception as e:  # noqa: BLE001 - the scope is still withdrawn below; photos go on the next load
        m, why = None, str(e)
    else:
        why = "face matching is off"
    if m is not None:
        rep = m.withdraw_photo_scope(rec.consent_id)
        out(f"Photo scope withdrawn from consent {rec.consent_id}. {rep.message}.")
        return 0 if rep.verified else 1
    rec.scopes.discard(PHOTO_SCOPE)
    reg.add(rec)
    out(f"Photo scope withdrawn from consent {rec.consent_id}. Photos kept under it are erased the next time "
        f"face matching loads ({why}).")
    return 0


def _camera_choice(value: Any) -> int | str | None:
    if value is None or isinstance(value, int):
        return value
    v = str(value).strip().lower()
    return "auto" if v == "auto" else int(v)


def _enroll(m, a, inp, out) -> int:
    """Guided capture (spec 7.3). No limit on people: keep going until you say stop.
    `--camera auto` probes cameras 0..4 for one that sees a face: OpenCV's numbering can differ from
    macOS's camera list, and camera 0 may be an iPhone (Continuity Camera)."""
    from .guard.consent import ConsentError
    order = list(m.cfg.enroll_required_angles)
    cam = _camera_choice(a.camera)
    explicit = isinstance(cam, int)
    if cam == "auto":
        out("Looking for a camera that can see your face; sit in front of the camera you want to use.")
        out("(An iPhone nearby may chime while it is checked.)")
        cam = _auto_camera(m)
        if cam is None:
            out("No camera saw a face, so nothing was captured. Check the lighting and that no other app "
                "is using the camera, or choose one with --camera N.")
            return 1
        out(f"Using camera {cam} (a face was seen there).")
    grab = _camera(cam, inp) if isinstance(cam, int) else _file_reader(inp)
    sid = m.start_session()
    try:
        while True:
            name = inp("Name or label for this person (blank to stop): ").strip()
            if not name:
                break
            cid = inp("Consent record ID (blank = none recorded): ").strip() or None
            try:
                pid = m.add_person(sid, name, cid, guardian=inp("Enrolling a minor as their guardian? [y/N] ").lower() == "y",
                                   no_consent_override=cid is None)
            except ConsentError as e:
                out(f"Not enrolled: {e}. To record consent, {CONSENT_HINT}")
                continue
            except ValueError as e:
                out(f"Not enrolled: {e}.")
                continue
            while True:
                need = m.needed_angles(pid)
                out(_ticks(need, order))
                st = m.finish_person(pid) if not need and len(m._captures[pid].shots) >= m.cfg.enroll_min_shots else None
                if st and st.complete:
                    out(st.message); break
                angle = sorted(need, key=lambda x: order.index(x.value))[0].value if need else order[0]
                frame = grab(f"Look {angle.upper()} and hold still. ")
                if frame is None:
                    out("capture cancelled for this person; nothing saved"); break
                r = m.submit_shot(pid, frame, angle)
                out("  ok" if r.ok else f"  rejected: {r.reason}")
                if explicit and r.code == "faces_none":
                    out(f"  hint: camera {cam} may not be the one facing you; try --camera auto")
    finally:
        m.end_session(sid)
    return 0


def _auto_camera(m, indices=AUTO_PROBE_INDICES, frames: int = AUTO_PROBE_FRAMES) -> int | None:
    """First camera index where the module's backend sees exactly one face in one of a few frames."""
    import cv2
    timeout = getattr(getattr(m, "guard", None), "timeout_s", 2.0)
    for i in indices:
        cap = cv2.VideoCapture(i)
        try:
            if not cap.isOpened():
                continue
            for _ in range(frames):   # the first frames are often dark while exposure settles
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                try:
                    boxes = m.backend.detect(frame, timeout)
                except Exception:  # noqa: BLE001 - a frame the backend cannot read counts as no face
                    continue
                if len(boxes) == 1:
                    return i
        finally:
            cap.release()
    return None


def _file_reader(inp: Callable[[str], str] = input):
    def grab(prompt: str):
        import cv2
        p = inp(prompt + "Image path (blank to cancel): ").strip()
        return cv2.imread(p) if p else None
    return grab


def _camera(index: int, inp: Callable[[str], str] = input):
    def grab(prompt: str):
        import cv2
        if inp(prompt + "Press Enter to capture (q to cancel) ").strip().lower() == "q":
            return None
        cap = cv2.VideoCapture(index)
        try:
            ok, frame = False, None
            for _ in range(WARMUP_FRAMES + 1):   # a just-opened Mac camera gives dark frames while exposure settles
                ok, frame = cap.read()
            return frame if ok else None
        finally:
            cap.release()
    return grab


if __name__ == "__main__":  # pragma: no cover
    sys.exit(1)
