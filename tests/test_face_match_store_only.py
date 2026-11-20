"""Store-only M-16 backend: a host (O Iris.app) that holds enrollments for the M-17 voice link but does no
face matching. Opens without models or OpenCV, never sees a face, never loads another model's templates,
refuses enrollment plainly. Synthetic data only (the deterministic test backend enrolls first)."""
import json, os, subprocess, sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import test_voice_link as tvl
from oiris.config import Config
from oiris.modules import face_match_gate, voice_link_gate
from oiris.modules.face_match.backends.store_only import ENROLL_REFUSAL, StoreOnlyBackend
from oiris.modules.face_match.backends.test_deterministic import synth_face_frame
from oiris.modules.face_match.core import FaceMatchLocked, FaceMatchModule
from oiris.modules.face_match.port import Angle
from oiris.registry import Registry, default_registry
from oiris.schema import Health

ROOT = Path(__file__).resolve().parents[1]
env = tvl.env   # same store fixture as the voice link tests (KeyStore, DB, DbChain, durable consents)


def _cfg(env, **flags) -> Config:
    return Config(data_dir=str(env["dir"]), flags={**Config().flags, "M-16": True, **flags},
                  face_match={"modes": ["personal"], "adapter": "store_only"})


def _registry() -> Registry:
    r = Registry()
    face_match_gate.register(r)
    return r


def _store_only(env) -> FaceMatchModule:
    return face_match_gate.load(_cfg(env), env["db"], env["ks"], _registry(), chain=env["chain"], consents=env["consents"])


@pytest.fixture
def enrolled(env):
    """Alex enrolled with the deterministic test backend; then the same store reopened with store_only."""
    real = tvl.face_on(env)
    pid = tvl.enroll_face(real, 11, "Alex")
    dek = real.person_sealer(pid)
    return SimpleNamespace(env=env, pid=pid, real=real, dek=dek, face=_store_only(env))


def test_registered_in_the_gate_and_default_registry():
    assert face_match_gate.BACKENDS["store_only"] == ("oiris.modules.face_match.backends.store_only", "StoreOnlyBackend",
                                                      "Apache-2.0")
    p = default_registry({"entry_points": False}).get("face_match", "store_only")
    assert p.license == "Apache-2.0" and p.module_flag == "M-16"
    b = StoreOnlyBackend()
    assert (b.name, b.model_id, b.test_only, b.enrolls) == ("store_only", "store-only", False, False)
    assert b.model_id != "opencv-zoo-yunet2023mar-sface2021dec" and b.model_id != "test-deterministic"


def test_opens_only_after_the_m16_opt_in(env):
    cfg = _cfg(env)
    cfg.flags["M-16"] = False
    assert face_match_gate.load(cfg, env["db"], env["ks"], _registry()) is None   # flag off: nothing built
    with pytest.raises(FaceMatchLocked):                                           # flag on, no opt-in recorded
        _store_only(env)
    tvl.face_optin.record_opt_in(env["db"], env["chain"], "I UNDERSTAND", ["personal"], "dev1", "us-other",
                                 warning_shown=tvl.face_optin.warning_for("us-other"))
    m = _store_only(env)
    assert isinstance(m, FaceMatchModule) and m.backend.name == "store_only" and m.modes == {"personal"}
    assert m.backend.health() == (Health.OK, "store only: no face matching")


def test_opens_without_opencv_or_models(tmp_path):
    """A fresh interpreter where `import cv2` fails, and no model files anywhere: the module still opens."""
    code = f"""
import sys
sys.modules["cv2"] = None   # any import of OpenCV now raises ImportError
from oiris.config import Config
from oiris.modules import face_match_gate
from oiris.modules.face_match import optin
from oiris.registry import default_registry
from oiris.security.keys import KeyStore
from oiris.store.db import DB
d = {str(tmp_path)!r}
ks = KeyStore(d + "/keys.json"); ks.create("pw")
db = DB(d + "/asb.db", ks.subkey("db"))
chain = optin.DbChain(db, ks.subkey("face-chain"))
optin.record_opt_in(db, chain, "I UNDERSTAND", ["personal"], "dev1", "us-other", warning_shown=optin.warning_for("us-other"))
cfg = Config(data_dir=d, flags={{**Config().flags, "M-16": True}}, face_match={{"adapter": "store_only"}})
m = face_match_gate.load(cfg, db, ks, default_registry({{"entry_points": False}}), chain=chain)
assert m.backend.name == "store_only" and m.process_frame([[0]])["faces"] == 0
import os
assert not os.path.exists(d + "/models"), "no model directory is needed or made"
print("ok")
"""
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_person_keys_and_name_work_for_an_enrolled_person(enrolled):
    face, pid = enrolled.face, enrolled.pid
    assert face.person_name(pid) == "Alex"
    blob = enrolled.dek.seal(b"voice bytes", b"voice:" + pid.encode())
    assert face.person_sealer(pid).open(blob, b"voice:" + pid.encode()) == b"voice bytes"   # the same DEK
    with pytest.raises(KeyError):
        face.person_sealer("nobody")
    with pytest.raises(KeyError):
        face.person_name("nobody")


def test_detect_finds_no_faces_so_embed_and_compare_never_run(enrolled, monkeypatch):
    face = enrolled.face
    calls = []
    monkeypatch.setattr(face.backend, "embed", lambda *a: calls.append("embed"))
    monkeypatch.setattr(face.backend, "compare", lambda *a: calls.append("compare"))
    frame = synth_face_frame(11, Angle.FRONT)   # Alex's synthetic face, in front of the camera
    assert face.backend.detect(frame, 1.0) == []
    r = face.process_frame(frame, place="Kitchen")
    assert (r["faces"], r["matched"], r["unrecognized"]) == (0, 0, 0) and calls == []
    assert face.db.execute("SELECT COUNT(*) FROM fm_encounter").fetchone()[0] == 0
    assert face.last_match_at(enrolled.pid) is None
    with pytest.raises(RuntimeError):
        StoreOnlyBackend().embed(frame, None)
    assert StoreOnlyBackend().compare(None, []) is None


def test_templates_of_another_model_are_not_loaded(enrolled):
    face, db = enrolled.face, enrolled.env["db"]
    rows = db.execute("SELECT model_id FROM fm_enrolled").fetchall()
    assert rows == [("test-deterministic",)]          # the enrollment is kept...
    assert face._load_templates() == []               # ...but never matched with the store-only model
    assert enrolled.real._load_templates() != []      # the model that made it still loads it


def test_enrollment_is_refused_plainly(enrolled):
    face = enrolled.face
    sid = face.start_session()
    for call in (lambda: face.add_person(sid, "Blake", "c-voice"),
                 lambda: face.submit_shot(enrolled.pid, synth_face_frame(12), Angle.FRONT),
                 lambda: face.reenroll(enrolled.pid)):
        with pytest.raises(FaceMatchLocked) as e:
            call()
        assert isinstance(e.value, ValueError) and str(e.value) == ENROLL_REFUSAL
    face.end_session(sid)
    assert "real face backend" in ENROLL_REFUSAL and "store_only" in ENROLL_REFUSAL and "opencv_zoo" in ENROLL_REFUSAL
    assert enrolled.env["db"].execute("SELECT COUNT(*) FROM fm_enrolled").fetchone()[0] == 1


def test_cli_enroll_with_store_only_says_so(tmp_path, monkeypatch):
    """`oiris face-match enroll` with an app-style config (adapter store_only): a plain line, no traceback."""
    from oiris import face_match_cli
    from oiris.modules.face_match.optin import DbChain, record_opt_in, warning_for
    from oiris.security.keys import KeyStore
    from oiris.store.db import DB
    ks = KeyStore(tmp_path / "keys.json"); ks.create("pw")
    db = DB(tmp_path / "asb.db", ks.subkey("db"))
    record_opt_in(db, DbChain(db, ks.subkey("face-chain")), "I UNDERSTAND", ["personal"], "dev1", "us-other",
                  warning_shown=warning_for("us-other"))
    db.close()
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"data_dir": str(tmp_path), "flags": {"M-16": True},
                                    "face_match": {"modes": ["personal"], "adapter": "store_only"}}))
    monkeypatch.setenv("OIRIS_PASSPHRASE", "pw")
    answers = iter(["Blake", "", "n", ""])   # name, no consent ID, not a minor, then stop
    lines: list[str] = []
    a = SimpleNamespace(sub="enroll", camera=None)
    rc = face_match_cli.run(a, Config.load(cfg_path), cfg_path, inp=lambda _p: next(answers), out=lines.append)
    assert rc == 0 and f"Not enrolled: {ENROLL_REFUSAL}." in lines, lines


def test_voice_link_gate_loads_and_matches_with_store_only(enrolled):
    """The app's case: M-16 store-only + M-17. Voice enrollment and matching use the person's keys only."""
    env = enrolled.env
    tvl.voice_opt_in(env)
    cfg = _cfg(env, **{"M-17": True})
    cfg.voice_link = {"allow_test_backend": True, "enroll_windows": 3}   # tests only: a config file may not set it
    vm = voice_link_gate.load(cfg, env["db"], env["ks"], tvl.registry(), enrolled.face, chain=env["chain"],
                              speaker=tvl.slots())
    assert vm is not None and vm.health() == ("listening", "")
    tvl.enroll_voice(vm, enrolled.pid, "A")
    r = vm.process_audio(tvl.synth_voice("A", seed=50), tvl.SR, place="Kitchen")
    assert r["heard"]["person_id"] == enrolled.pid and r["heard"]["name"] == "Alex"
    assert r["heard"]["face_agrees"] is False   # no face matching on this host
    # the voice template is sealed with the person's DEK, as with a real face backend
    blob = env["db"].execute("SELECT template FROM vl_voice WHERE person_id=?", (enrolled.pid,)).fetchone()[0]
    vec = np.frombuffer(enrolled.face.person_sealer(enrolled.pid).open(blob, b"voice:" + enrolled.pid.encode()), dtype=np.float32)
    assert vec.size == tvl.FakeSpeaker.dim


def test_no_cv2_import_in_the_backend_source():
    src = (ROOT / "oiris" / "modules" / "face_match" / "backends" / "store_only.py").read_text()
    assert "cv2" not in src and "onnx" not in src and "urllib" not in src
    assert os.path.basename(face_match_gate.STORE_ONLY_PATH.replace(".", "/")) == "store_only"
