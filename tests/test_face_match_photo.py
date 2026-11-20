"""M-16 face photo (ADR-0057 decision 4): capture, sealed storage, access and verified erasure.
Synthetic frames only (generated arrays and the TEST-ONLY deterministic backend). No real face or photo."""
import json, sqlite3, threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from nacl.signing import SigningKey

from oiris.config import Config
from oiris.guard.consent import ConsentRecord, ConsentRegistry
from oiris.modules.face_match import optin
from oiris.modules.face_match.backends.test_deterministic import TestDeterministicBackend, synth_face_frame
from oiris.modules.face_match.core import FaceMatchConfig, FaceMatchModule
from oiris.modules.face_match.photo import (PHOTO_AAD, PHOTO_JPEG_QUALITY, PHOTO_MARGIN, PHOTO_SIZE_PX, PhotoCandidate,
                                            _resize, face_crop, shot_quality)
from oiris.modules.face_match.port import REQUIRED_ANGLES, Angle, FaceBox
from oiris.security.aead import Sealer
from oiris.security.hashchain import HashChain
from oiris.security.keys import KeyStore
from oiris.store.db import DB
from oiris.store.migrations import LATEST

cv2 = pytest.importorskip("cv2")   # JPEG encode/decode; CI installs opencv-python-headless

SIDES = [a for a in REQUIRED_ANGLES if a is not Angle.FRONT]
DULL, SHARP, MEDIUM = 0.75, 1.0, 0.9   # synthetic texture contrast: lower contrast = lower sharpness
NAMED_PHOTO_CONSTANTS = (0.25, 160, 85)   # margin, square size in px, JPEG quality (ADR-0057 decision 4 task)


@pytest.fixture
def env(tmp_path):
    ks = KeyStore(tmp_path / "keys.json"); ks.create("pw")
    db = DB(tmp_path / "asb.db", ks.subkey("db"))
    chain = HashChain(ks.subkey("face-chain"), SigningKey.generate())
    consents = ConsentRegistry()
    consents.add(ConsentRecord("c-ok", "s1", {"face_enrollment"}))
    consents.add(ConsentRecord("c-photo", "s2", {"face_enrollment", "photo"}))
    consents.add(ConsentRecord("c-photo2", "s3", {"face_enrollment", "photo"}))
    yield {"ks": ks, "db": db, "chain": chain, "consents": consents, "dir": tmp_path}
    db.close()


def make(env, consents=None):
    if optin.opt_in_record(env["db"]) is None:
        optin.record_opt_in(env["db"], env["chain"], "I UNDERSTAND", ["personal"], "dev1", "us-other",
                            warning_shown=optin.warning_for("us-other"))
    return FaceMatchModule(env["db"], env["ks"], env["dir"], TestDeterministicBackend(), env["chain"],
                           FaceMatchConfig(modes=["personal"], allow_test_backend=True),
                           env["consents"] if consents is None else consents, timeout_s=1.0)


def frame(seed, angle=Angle.FRONT, brightness=SHARP):
    return synth_face_frame(seed, angle, brightness=brightness, noise_seed=hash(angle.value) % 97)


def box_of(f):
    return TestDeterministicBackend().detect(f, 1.0)[0]


def enroll(m, sid, seed, consent="c-photo", name="Alex", front_brightness=SHARP):
    pid = m.add_person(sid, name, consent)
    assert m.submit_shot(pid, frame(seed, Angle.FRONT, front_brightness), Angle.FRONT).ok
    for a in SIDES:
        assert m.submit_shot(pid, frame(seed, a), a).ok
    st = m.finish_person(pid)
    assert st.complete, st.message
    return pid, st


def photo_row(db, pid):
    r = db.execute("SELECT photo FROM fm_photo WHERE person_id=?", (pid,)).fetchone()
    return None if r is None else r[0]


def on_disk(env, blob):
    return any(blob in f.read_bytes() for f in Path(env["dir"]).rglob("*") if f.is_file())


# ---- capture (RAM only) ---------------------------------------------------------------------
def test_best_front_shot_is_chosen_by_quality_and_side_shots_are_never_used(env):
    m = make(env)
    sid = m.start_session()
    pid = m.add_person(sid, "Alex", "c-photo")
    cand = m._captures[pid].photo
    assert isinstance(cand, PhotoCandidate) and cand.pixels is None
    dull, sharp, medium = (frame(31, Angle.FRONT, b) for b in (DULL, SHARP, MEDIUM))
    assert shot_quality(dull, box_of(dull)) < shot_quality(medium, box_of(medium)) < shot_quality(sharp, box_of(sharp))
    assert m.submit_shot(pid, dull, Angle.FRONT).ok
    first = cand.pixels
    assert np.array_equal(first, face_crop(dull, box_of(dull)))
    for a in SIDES:   # side shots are sharper than the dull FRONT one but are never candidates
        side = frame(31, a, SHARP)
        assert shot_quality(side, box_of(side)) > cand.score
        assert m.submit_shot(pid, side, a).ok
        assert cand.pixels is first
    assert m.submit_shot(pid, sharp, Angle.FRONT).ok   # sharper FRONT replaces; the old candidate is zeroed
    assert np.array_equal(cand.pixels, face_crop(sharp, box_of(sharp))) and not first.any()
    kept = cand.pixels
    assert m.submit_shot(pid, medium, Angle.FRONT).ok   # less sharp: not replaced
    assert cand.pixels is kept and kept.any()
    rejected = synth_face_frame(31, Angle.FRONT, blur=True)   # a rejected shot is never a candidate
    assert not m.submit_shot(pid, rejected, Angle.FRONT).ok and cand.pixels is kept
    assert m.finish_person(pid).complete
    jpeg = m.photo(pid)
    got = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_UNCHANGED).astype(float)
    want_sharp, want_dull = (face_crop(f, box_of(f)).astype(float) for f in (sharp, dull))
    assert np.abs(got - want_sharp).mean() < np.abs(got - want_dull).mean()


def test_crop_margin_clip_and_size_on_a_colour_array():
    assert (PHOTO_MARGIN, PHOTO_SIZE_PX, PHOTO_JPEG_QUALITY) == NAMED_PHOTO_CONSTANTS
    rng = np.random.default_rng(5)
    img = rng.integers(0, 255, size=(200, 240, 3), dtype=np.uint8)   # generated array, not a photo
    b = FaceBox(80, 60, 40, 40, 1.0, Angle.FRONT)
    c = face_crop(img, b)
    assert c.shape == (PHOTO_SIZE_PX, PHOTO_SIZE_PX, 3) and c.dtype == np.uint8
    assert np.array_equal(c, _resize(img[50:110, 70:130], PHOTO_SIZE_PX))   # box + 25 % each side
    edge = face_crop(img, FaceBox(0, 0, 40, 40, 1.0, Angle.FRONT))   # clipped to the frame
    assert edge.shape == (PHOTO_SIZE_PX, PHOTO_SIZE_PX, 3)
    assert np.array_equal(edge, _resize(img[0:50, 0:50], PHOTO_SIZE_PX))
    rgba = np.dstack([img, np.full(img.shape[:2], 7, np.uint8)])
    assert face_crop(rgba, b).shape[2] == 3   # alpha dropped


# ---- store at finish_person -----------------------------------------------------------------
def test_no_photo_without_the_photo_scope(env):
    m = make(env)
    sid = m.start_session()
    pid = m.add_person(sid, "Sam", "c-ok")
    assert m._captures[pid].photo is None   # without the scope no crop is kept even in RAM
    m.end_session(sid)
    sid = m.start_session()
    pid, st = enroll(m, sid, 41, consent="c-ok", name="Sam")
    assert "No photo kept" in st.message and "'photo' scope" in st.message
    assert photo_row(env["db"], pid) is None and m.photo(pid) is None and not m.has_photo(pid)


def test_with_the_scope_one_sealed_photo_that_decodes_to_the_named_size(env):
    m = make(env)
    pid, st = enroll(m, m.start_session(), 42)
    assert "One face photo kept" in st.message
    assert env["db"].execute("SELECT COUNT(*) FROM fm_photo").fetchone()[0] == 1 and m.has_photo(pid)
    jpeg = m.photo(pid)
    assert jpeg[:2] == b"\xff\xd8" and b"Exif" not in jpeg
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_UNCHANGED)
    assert img.shape[:2] == (PHOTO_SIZE_PX, PHOTO_SIZE_PX)
    blob = photo_row(env["db"], pid)
    assert jpeg not in blob and jpeg[100:164] not in blob   # sealed, not in clear
    env["db"].execute("PRAGMA wal_checkpoint(TRUNCATE)")
    assert not on_disk(env, jpeg[100:164])
    assert env["chain"].entries[-1][0]["kind"] == "enrolled" and env["chain"].entries[-1][0]["photo"] is True


def test_aad_binds_the_photo_to_its_person(env):
    m = make(env)
    sid = m.start_session()
    a, _ = enroll(m, sid, 43, name="A")
    b, _ = enroll(m, sid, 44, consent="c-photo2", name="B")
    ba, bb = photo_row(env["db"], a), photo_row(env["db"], b)
    env["db"].execute("UPDATE fm_photo SET photo=? WHERE person_id=?", (bb, a))
    env["db"].execute("UPDATE fm_photo SET photo=? WHERE person_id=?", (ba, b))
    for pid in (a, b):
        with pytest.raises(Exception):
            m.photo(pid)
    with pytest.raises(Exception):   # the right key with the wrong person in the AAD fails too
        m.person_sealer(a).open(ba, PHOTO_AAD + b.encode())


def test_reenrollment_replaces_the_photo_and_the_old_one_cannot_be_opened(env):
    m = make(env)
    pid, _ = enroll(m, m.start_session(), 45, front_brightness=DULL)
    old, old_jpeg = photo_row(env["db"], pid), m.photo(pid)
    sid = m.reenroll(pid)
    assert m.submit_shot(pid, frame(45, Angle.FRONT, SHARP), Angle.FRONT).ok
    for a in SIDES:
        assert m.submit_shot(pid, frame(45, a), a).ok
    assert "One face photo kept" in m.finish_person(pid).message
    m.end_session(sid)
    assert env["db"].execute("SELECT COUNT(*) FROM fm_photo").fetchone()[0] == 1
    new = photo_row(env["db"], pid)
    assert new != old and m.photo(pid) != old_jpeg
    assert not m._can_open_with_any(old, PHOTO_AAD + pid.encode())


def test_reenrollment_without_the_scope_drops_the_old_photo(env):
    m = make(env)
    pid, _ = enroll(m, m.start_session(), 46)
    old = photo_row(env["db"], pid)
    m.withdraw_photo_scope("c-photo")
    sid = m.reenroll(pid)
    assert m._captures[pid].photo is None
    for a in REQUIRED_ANGLES:
        assert m.submit_shot(pid, frame(46, a), a).ok
    assert "No photo kept" in m.finish_person(pid).message
    m.end_session(sid)
    assert photo_row(env["db"], pid) is None and not m._can_open_with_any(old, PHOTO_AAD + pid.encode())


def test_rekey_for_the_voice_link_keeps_the_photo(env):
    m = make(env)
    pid, _ = enroll(m, m.start_session(), 47)
    before, blob = m.photo(pid), photo_row(env["db"], pid)
    m.rekey_person(pid)
    assert m.photo(pid) == before and photo_row(env["db"], pid) != blob
    assert not m._can_open_with_any(blob, PHOTO_AAD + pid.encode())


# ---- erasure ----------------------------------------------------------------------------------
def test_delete_person_erases_the_photo_verified(env):
    m = make(env)
    pid, _ = enroll(m, m.start_session(), 48)
    blob = photo_row(env["db"], pid)
    rep = m.delete_person(pid)
    assert rep.verified and rep.photos_destroyed == 1 and rep.message == "1 template and 1 photo destroyed, verified"
    assert rep.checks["photo_row_absent"] and rep.checks["photo_undecryptable"] and rep.checks["photo_ciphertext_absent"]
    assert photo_row(env["db"], pid) is None and not on_disk(env, blob) and m.photo(pid) is None
    assert env["chain"].entries[-1][0]["photos_destroyed"] == 1


def test_delete_person_without_a_photo_keeps_its_old_message(env):
    m = make(env)
    pid, _ = enroll(m, m.start_session(), 49, consent="c-ok")
    rep = m.delete_person(pid)
    assert rep.verified and rep.photos_destroyed == 0 and rep.message == "1 template destroyed, verified"


def test_photo_delete_keeps_the_enrollment(env):
    m = make(env)
    pid, _ = enroll(m, m.start_session(), 50)
    blob = photo_row(env["db"], pid)
    rep = m.delete_photo(pid)
    assert rep.verified and rep.photos_destroyed == 1 and rep.templates_destroyed == 0
    assert rep.message == "1 photo destroyed, verified" and all(rep.checks.values())
    assert m.photo(pid) is None and not on_disk(env, blob) and not m._can_open_with_any(blob, PHOTO_AAD + pid.encode())
    assert m.person_name(pid) == "Alex" and m.process_frame(frame(50))["matched"] == 1   # still enrolled and matching
    again = m.delete_photo(pid)
    assert again.verified and again.photos_destroyed == 0 and again.message == "no photo stored"
    with pytest.raises(KeyError):
        m.delete_photo("nobody")


def test_withdrawing_the_photo_scope_erases_photos_and_keeps_enrollments(env):
    m = make(env)
    sid = m.start_session()
    a, _ = enroll(m, sid, 51, name="A")
    b, _ = enroll(m, sid, 52, name="B")
    c, _ = enroll(m, sid, 53, consent="c-photo2", name="C")
    blobs = [photo_row(env["db"], p) for p in (a, b)]
    rep = m.withdraw_photo_scope("c-photo")
    assert rep.verified and rep.photos_destroyed == 2 and rep.message == "2 photos destroyed, verified"
    assert m.photo(a) is None and m.photo(b) is None and m.photo(c) is not None
    assert not any(on_disk(env, x) for x in blobs)
    rec = env["consents"].get("c-photo")
    assert "photo" not in rec.scopes and "face_enrollment" in rec.scopes and not rec.withdrawn
    assert env["db"].execute("SELECT COUNT(*) FROM fm_enrolled").fetchone()[0] == 3


def test_whole_consent_withdrawal_erases_person_and_photo(env):
    m = make(env)
    pid, _ = enroll(m, m.start_session(), 54)
    blob = photo_row(env["db"], pid)
    env["consents"].withdraw("c-photo")
    assert photo_row(env["db"], pid) is None and not on_disk(env, blob)
    assert m.photo(pid) is None


def test_photo_without_current_consent_is_erased_on_load_and_on_access(env):
    m = make(env)
    pid, _ = enroll(m, m.start_session(), 55)
    blob = photo_row(env["db"], pid)
    other = ConsentRegistry()   # the photo scope is gone; face_enrollment stays
    other.add(ConsentRecord("c-photo", "s2", {"face_enrollment"}))
    m2 = make(env, consents=other)
    assert photo_row(env["db"], pid) is None and not m2._can_open_with_any(blob, PHOTO_AAD + pid.encode())
    assert m2.person_name(pid) == "Alex"
    m3 = make(env)
    pid2, _ = enroll(m3, m3.start_session(), 56, consent="c-photo2", name="B")
    env["consents"].get("c-photo2").scopes.discard("photo")   # changed behind the module's back
    assert m3.photo(pid2) is None and photo_row(env["db"], pid2) is None


def test_disable_erases_every_photo_verified_and_photo_is_none_while_off(env):
    m = make(env)
    sid = m.start_session()
    pids = [enroll(m, sid, 60, name="A")[0], enroll(m, sid, 61, consent="c-photo2", name="B")[0],
            enroll(m, sid, 62, consent="c-ok", name="C")[0]]
    blobs = [photo_row(env["db"], p) for p in pids[:2]]
    rep = m.disable()
    assert rep.verified and rep.photos_destroyed == 2 and rep.templates_destroyed == 3
    assert rep.message == "3 templates and 2 photos destroyed, verified"
    assert rep.checks["photo_rows"] == 0 and rep.checks["photo_undecryptable"] and rep.checks["photo_ciphertext_absent"]
    assert not any(on_disk(env, b) for b in blobs)
    assert all(m.photo(p) is None for p in pids)


def test_photo_is_none_for_people_not_enrolled(env):
    m = make(env)
    pid, _ = enroll(m, m.start_session(), 63)
    assert m.photo("not-enrolled") is None and m.photo("") is None
    sid = m.start_session()
    half = m.add_person(sid, "Half", "c-photo2")
    assert m.submit_shot(half, frame(64), Angle.FRONT).ok
    assert m.photo(half) is None   # an open capture is not an enrolled person
    m.end_session(sid)
    assert m.photo(pid) is not None


def test_photo_is_none_once_the_opt_in_is_gone(env):
    m = make(env)
    pid, _ = enroll(m, m.start_session(), 65)
    env["db"].execute("DELETE FROM fm_state WHERE k='opt_in'")
    assert m.photo(pid) is None


# ---- RAM buffers ----------------------------------------------------------------------------
def test_ram_buffers_zeroed_after_finish(env):
    m = make(env)
    sid = m.start_session()
    pid = m.add_person(sid, "Alex", "c-photo")
    cand = m._captures[pid].photo
    for a in REQUIRED_ANGLES:
        assert m.submit_shot(pid, frame(70, a), a).ok
    held = cand.pixels
    assert held is not None and held.any()
    assert m.finish_person(pid).complete
    assert not held.any() and cand.pixels is None and m.capture_buffer_size() == 0


@pytest.mark.parametrize("how", ["end_session", "next_person", "disable"])
def test_ram_buffers_zeroed_after_an_abandoned_capture(env, how):
    m = make(env)
    sid = m.start_session()
    pid = m.add_person(sid, "Alex", "c-photo")
    cand = m._captures[pid].photo
    assert m.submit_shot(pid, frame(71), Angle.FRONT).ok
    held = cand.pixels
    assert held.any()
    if how == "end_session":
        m.end_session(sid)
    elif how == "next_person":
        m.add_person(sid, "Next", "c-ok")
    else:
        m.disable()
    assert not held.any() and cand.pixels is None and pid not in m._captures
    assert env["db"].execute("SELECT COUNT(*) FROM fm_photo").fetchone()[0] == 0


# ---- migration ---------------------------------------------------------------------------------
def test_migration_adds_fm_photo_to_an_existing_store(tmp_path, key):
    p = tmp_path / "old.db"
    d = DB.__new__(DB)
    d.path = str(p); d.conn = sqlite3.connect(str(p), isolation_level=None)
    d.lock = threading.RLock(); d.sealer = Sealer(key)
    assert d.migrate(6) == 6 and not d.has_table("fm_photo")
    d.execute("INSERT INTO fm_enrolled(person_id, template) VALUES('p1', x'00')")
    assert d.migrate() == LATEST == 7
    assert d.has_table("fm_photo") and d.execute("SELECT person_id FROM fm_enrolled").fetchall() == [("p1",)]
    cols = [r[1] for r in d.execute("PRAGMA table_info(fm_photo)").fetchall()]
    assert cols == ["person_id", "photo", "created_at"]
    d.conn.close()


# ---- CLI -------------------------------------------------------------------------------------------
class InjectedBackend(TestDeterministicBackend):
    """The synthetic-frame test backend, injected through run(backend=...); it only ever sees synthetic frames."""
    test_only = False


def cli_args(sub, **kw):
    return SimpleNamespace(sub=sub, mode=kw.get("mode"), jurisdiction=kw.get("jurisdiction"), camera=None, id=kw.get("id"),
                           subject=kw.get("subject"), days=None, withdraw=kw.get("withdraw", False), scope=kw.get("scope"),
                           person=kw.get("person"))


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("OIRIS_PASSPHRASE", "pw")
    cfgp = tmp_path / "config.json"
    cfgp.write_text(json.dumps({"data_dir": str(tmp_path)}))
    return tmp_path, cfgp


def _cli(store, sub, inp=lambda p: "YES", **kw):
    from oiris.face_match_cli import run
    d, cfgp = store
    out: list[str] = []
    rc = run(cli_args(sub, **kw), Config.load(cfgp), cfgp, inp=inp, out=out.append, backend=InjectedBackend())
    return rc, out


def _enroll_direct(store, people):
    """Enroll through the same durable store the CLI uses (synthetic frames)."""
    d, _ = store
    ks = KeyStore(d / "keys.json"); ks.unlock("pw")
    db = DB(d / "asb.db", ks.subkey("db"))
    try:
        m = FaceMatchModule(db, ks, d, InjectedBackend(), optin.DbChain(db, ks.subkey("face-chain")),
                            FaceMatchConfig(modes=["personal"]))
        sid = m.start_session()
        pids = []
        for name, seed, cid in people:
            pid = m.add_person(sid, name, cid)
            for a in REQUIRED_ANGLES:
                assert m.submit_shot(pid, frame(seed, a), a).ok
            assert m.finish_person(pid).complete
            pids.append(pid)
        return pids
    finally:
        db.close()


def test_cli_list_photo_delete_scope_withdrawal_and_disable(store):
    assert _cli(store, "enable", inp=lambda p: optin.CONFIRM_PHRASE, mode=["personal"], jurisdiction="us-other")[0] == 0
    assert _cli(store, "list")[1] == ["no one is enrolled"]
    assert _cli(store, "consent", subject="S-1", id="cp", scope="face_enrollment,photo")[0] == 0
    assert _cli(store, "consent", subject="S-2", id="cn")[0] == 0
    a, b, c = _enroll_direct(store, [("Ann", 80, "cp"), ("Ben", 81, "cn"), ("Cal", 82, "cp")])
    rc, out = _cli(store, "list")
    assert rc == 0 and out == [f"Ann (id {a[:8]})  photo: yes", f"Ben (id {b[:8]})  photo: no", f"Cal (id {c[:8]})  photo: yes"]
    rc, out = _cli(store, "photo-delete", person=a[:8])
    assert rc == 0 and out == ["1 photo destroyed, verified; the enrollment is kept"]
    assert _cli(store, "photo-delete", person=b[:8])[1] == ["no photo stored for this person"]
    assert _cli(store, "photo-delete", person="zz-none")[0] == 1
    assert "photo: no" in _cli(store, "list")[1][0]
    rc, out = _cli(store, "consent", withdraw=True, id="cp", scope="guardian")
    assert rc == 2 and "only --scope photo" in out[0]
    assert _cli(store, "consent", withdraw=True, id="cn", scope="photo")[1] == ["consent cn has no photo scope; nothing to withdraw"]
    rc, out = _cli(store, "consent", withdraw=True, id="cp", scope="photo")
    assert rc == 0 and out == ["Photo scope withdrawn from consent cp. 1 photo destroyed, verified."]
    rc, out = _cli(store, "list")
    assert out == [f"Ann (id {a[:8]})  photo: no", f"Ben (id {b[:8]})  photo: no", f"Cal (id {c[:8]})  photo: no"]
    d, _ = store
    ks = KeyStore(d / "keys.json"); ks.unlock("pw")
    db = DB(d / "asb.db", ks.subkey("db"))
    try:
        rec = ConsentRegistry(db).get("cp")
        assert rec.scopes == {"face_enrollment"} and not rec.withdrawn
    finally:
        db.close()


def test_cli_disable_reports_photos(store):
    assert _cli(store, "enable", inp=lambda p: optin.CONFIRM_PHRASE, mode=["personal"], jurisdiction="us-other")[0] == 0
    assert _cli(store, "consent", subject="S-1", id="cp", scope="face_enrollment,photo")[0] == 0
    _enroll_direct(store, [("Ann", 90, "cp"), ("Ben", 91, "cp")])
    rc, out = _cli(store, "disable")
    assert rc == 0 and out[-1] == "Face matching is off. 2 templates destroyed, verified. 2 photos destroyed."
    d, _ = store
    ks = KeyStore(d / "keys.json"); ks.unlock("pw")
    db = DB(d / "asb.db", ks.subkey("db"))
    try:
        assert db.execute("SELECT COUNT(*) FROM fm_photo").fetchone()[0] == 0
    finally:
        db.close()
