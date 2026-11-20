"""M-17 opt-in (ADR-0057 decisions 1 and 6; ADR-0058 adds the spoken-name and recording sentences): typed confirmation, plain-language warning, record in
vl_state. Every M-17 enable / disable / enroll / delete is appended to the same tamper-evident chain
M-16 uses (DbChain), with module "M-17"."""
from __future__ import annotations
import json, time

from ...guard.vocabulary import check_text
from ..face_match.optin import CONFIRM_PHRASE, DbChain
from ..face_match.optin import opt_in_record as face_opt_in_record

WARNING_VERSION = "v2"   # v2: ADR-0058 announcement sentences
WARNING_TEXT = """Voice link is about to be turned on.
- It stores a mathematical map of a person's voice (a "voiceprint", or voice template). Voiceprints are biometric data. Each one stays on this device, encrypted with that person's own key.
- Voice templates are made only for people already enrolled in face matching who have a consent record with scope voice_enrollment, and for a minor the guardian's recorded agreement (scope guardian).
- With ambient listening, the microphone hears everyone in the room while voice link runs, including people who never agreed and don't know it is on. macOS shows its microphone indicator while the microphone is open.
- Voices of people who are not enrolled are dropped at once. They are never stored or counted.
- When it recognises someone, this device plays a chime and says 'sounds like' and their name out loud; anyone near the speaker can hear it. You can change this to a chime only or turn it off.
- A recording or phone call of an enrolled person can sound like them: treat a match as a hint, not proof.
- No audio is ever saved. Sound is processed in short pieces in memory and then discarded.
- Biometric laws (for example Illinois, Texas, Washington, the EU/UK and others) restrict collecting voiceprints, sometimes with heavy penalties. Compliance is your responsibility. Whether to use this, and how, is your decision and your risk.
- You can turn it off any time. Turning it off permanently deletes every voice template."""  # noqa: E501 - user-facing notice
check_text(WARNING_TEXT)

__all__ = ["CONFIRM_PHRASE", "WARNING_TEXT", "WARNING_VERSION", "OptInError", "opt_in_record", "record_opt_in", "log_change"]


class OptInError(PermissionError):
    pass


def opt_in_record(db) -> dict | None:
    r = db.execute("SELECT v FROM vl_state WHERE k='opt_in'").fetchone() if db.has_table("vl_state") else None
    return json.loads(r[0]) if r else None


def log_change(db, chain, kind: str, detail: dict) -> None:
    """Counts and settings only: never names, audio or templates."""
    rec = {"module": "M-17", "kind": kind, "t_wall": round(time.time(), 3), **detail}
    if chain is not None:
        chain.append(rec)
    if not isinstance(chain, DbChain):
        db.audit(f"voice_link.{kind}", rec)


def record_opt_in(db, chain, typed: str, listening: str, device_id: str, warning_shown: str | None = None) -> dict:
    """Needs M-16 opted in with personal mode: voice templates exist only for face-enrolled people."""
    if typed.strip() != CONFIRM_PHRASE:
        raise OptInError(f'type "{CONFIRM_PHRASE}" to turn voice link on')
    if warning_shown is None or WARNING_TEXT not in warning_shown:
        raise OptInError("the full warning must be shown before opting in")
    face = face_opt_in_record(db)
    if face is None or "personal" not in face.get("modes", []):
        raise OptInError("voice link needs face matching (M-16) turned on in personal mode first: oiris face-match enable")
    rec = {"listening": listening, "warning_version": WARNING_VERSION, "device_id": device_id, "t_wall": round(time.time(), 3)}
    db.execute("DELETE FROM vl_state WHERE k='self'")   # a fresh opt-in never inherits an old self mark
    db.execute("INSERT OR REPLACE INTO vl_state(k, v) VALUES('opt_in', ?)", (json.dumps(rec),))
    log_change(db, chain, "opt_in", rec)
    return rec
