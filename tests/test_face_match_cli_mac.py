"""`oiris face-match` CLI fixes from a live Mac test: camera auto-pick (OpenCV's camera numbering differs
from macOS's list; camera 0 was an iPhone via Continuity Camera), the consent command, and enroll not
crashing on consent/label errors. Fake cv2 and synthetic frames only: no real faces, no real camera."""
import json, sys, time, types
from types import SimpleNamespace

import numpy as np
import pytest
from nacl.signing import SigningKey

from oiris.cli import main
from oiris.config import Config
from oiris.face_match_cli import _auto_camera, _enroll, run
from oiris.guard.consent import ConsentRegistry
from oiris.modules.face_match import optin
from oiris.modules.face_match.backends.test_deterministic import TestDeterministicBackend, synth_face_frame
from oiris.modules.face_match.core import FaceMatchConfig, FaceMatchModule
from oiris.modules.face_match.port import Angle, FaceBox
from oiris.security.hashchain import HashChain
from oiris.security.keys import KeyStore
from oiris.store.db import DB

DARK = np.zeros((160, 160), dtype=np.float32)   # what the phone at index 0 gave: no face


class FakeCv2(types.ModuleType):
    """cv2 stand-in: `cams` maps index -> callable returning the next frame; other indices do not open."""

    def __init__(self, cams):
        super().__init__("cv2")
        self.cams, self.opened = cams, []
        fake = self

        class VideoCapture:
            def __init__(self, i):
                self.i = i
                fake.opened.append(i)

            def isOpened(self):
                return self.i in fake.cams

            def read(self):
                return (True, fake.cams[self.i]()) if self.i in fake.cams else (False, None)

            def release(self):
                pass

        self.VideoCapture = VideoCapture


class Prompter:
    """Scripted answers; remembers the angle last asked for so the fake camera can show that pose."""

    def __init__(self, answers):
        self.answers, self.prompts, self.angle = list(answers), [], Angle.FRONT

    def __call__(self, prompt):
        self.prompts.append(prompt)
        for a in Angle:
            if f"Look {a.value.upper()} " in prompt:
                self.angle = a
        return self.answers.pop(0) if self.answers else ""


@pytest.fixture
def env(tmp_path):
    ks = KeyStore(tmp_path / "keys.json"); ks.create("pw")
    db = DB(tmp_path / "asb.db", ks.subkey("db"))
    chain = HashChain(ks.subkey("face-chain"), SigningKey.generate())
    yield {"ks": ks, "db": db, "chain": chain, "dir": tmp_path}
    db.close()


def module(env, jurisdiction="us-other", consents=None):
    optin.record_opt_in(env["db"], env["chain"], "I UNDERSTAND", ["personal"], "dev1", jurisdiction,
                        warning_shown=optin.warning_for(jurisdiction))
    return FaceMatchModule(env["db"], env["ks"], env["dir"], TestDeterministicBackend(), env["chain"],
                           FaceMatchConfig(modes=["personal"], allow_test_backend=True), consents, timeout_s=1.0)


def count(db):
    return db.execute("SELECT COUNT(*) FROM fm_enrolled").fetchone()[0]


# ---- camera auto-pick -----------------------------------------------------------------------------
def test_auto_pick_uses_first_camera_that_sees_one_face(monkeypatch):
    face = object()
    cv2 = FakeCv2({0: lambda: "phone-frame", 1: lambda: face})
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    seen = []

    class Backend:
        def detect(self, frame, timeout_s):
            seen.append(frame)
            return [FaceBox(0, 0, 10, 10, 0.9)] if frame is face else []

    assert _auto_camera(SimpleNamespace(backend=Backend(), guard=SimpleNamespace(timeout_s=1.0))) == 1
    assert cv2.opened == [0, 1] and "phone-frame" in seen   # 0 was probed (and read) before 1 was chosen


def test_auto_pick_needs_exactly_one_face(monkeypatch):
    cv2 = FakeCv2({0: lambda: "two", 1: lambda: "one"})
    monkeypatch.setitem(sys.modules, "cv2", cv2)

    class Backend:
        def detect(self, frame, timeout_s):
            if frame == "boom":
                raise ValueError("unreadable")
            return [FaceBox(0, 0, 10, 10, 0.9)] * (2 if frame == "two" else 1)

    assert _auto_camera(SimpleNamespace(backend=Backend())) == 1
    cv2.cams = {0: lambda: "boom"}   # a backend error on a frame counts as no face, not a crash
    assert _auto_camera(SimpleNamespace(backend=Backend())) is None


def test_enroll_auto_picks_camera_1_and_enrolls(env, monkeypatch):
    m = module(env)
    inp = Prompter(["Alex", "", "n"])
    cv2 = FakeCv2({0: lambda: DARK, 1: lambda: synth_face_frame(21, inp.angle, noise_seed=len(inp.prompts))})
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    out: list[str] = []
    assert _enroll(m, SimpleNamespace(camera="auto"), inp, out.append) == 0
    assert "Using camera 1 (a face was seen there)." in out
    assert any("iPhone" in line for line in out)
    assert count(env["db"]) == 1
    assert sum("Press Enter to capture" in p for p in inp.prompts) >= len(m.cfg.enroll_required_angles)   # per-angle prompt kept
    assert not any("hint:" in line for line in out)


def test_enroll_auto_no_camera_sees_a_face_captures_nothing(env, monkeypatch):
    m = module(env)
    monkeypatch.setitem(sys.modules, "cv2", FakeCv2({0: lambda: DARK, 1: lambda: DARK}))
    inp = Prompter([])
    out: list[str] = []
    assert _enroll(m, SimpleNamespace(camera="auto"), inp, out.append) == 1
    assert any(line.startswith("No camera saw a face") for line in out)
    assert inp.prompts == [] and count(env["db"]) == 0 and m.capture_buffer_size() == 0


def test_enroll_explicit_index_hints_auto_when_no_face(env, monkeypatch):
    m = module(env)
    cv2 = FakeCv2({0: lambda: DARK})
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    inp = Prompter(["Alex", "", "n", "", "q"])
    out: list[str] = []
    assert _enroll(m, SimpleNamespace(camera=0), inp, out.append) == 0
    assert cv2.opened == [0]   # an explicit index is used as given, never probed
    assert any("rejected" in line for line in out)
    assert "  hint: camera 0 may not be the one facing you; try --camera auto" in out
    assert "capture cancelled for this person; nothing saved" in out and count(env["db"]) == 0


def test_explicit_index_with_a_face_enrolls_without_hint(env, monkeypatch):
    m = module(env)
    inp = Prompter(["Alex", "", "n"])
    monkeypatch.setitem(sys.modules, "cv2", FakeCv2({1: lambda: synth_face_frame(22, inp.angle, noise_seed=len(inp.prompts))}))
    out: list[str] = []
    assert _enroll(m, SimpleNamespace(camera="1"), inp, out.append) == 0
    assert count(env["db"]) == 1 and not any("hint:" in line for line in out)


def test_cli_camera_argument_accepts_auto_or_index(monkeypatch, tmp_path):
    got = {}
    monkeypatch.setattr("oiris.face_match_cli.run", lambda a, cfg, p: got.setdefault("a", a) and 0)
    for v, want in (("auto", "auto"), ("AUTO", "auto"), ("1", 1)):
        got.clear()
        main(["face-match", "enroll", "--camera", v, "--config", str(tmp_path / "none.json")])
        assert got["a"].camera == want
    got.clear()
    main(["face-match", "consent", "--subject", "S-01", "--days", "7", "--config", str(tmp_path / "none.json")])
    assert (got["a"].sub, got["a"].subject, got["a"].days, got["a"].withdraw) == ("consent", "S-01", 7, False)
    with pytest.raises(SystemExit):
        main(["face-match", "enroll", "--camera", "facetime"])


# ---- enroll never crashes on consent or label errors -----------------------------------------------
def test_strict_enroll_without_consent_explains_and_continues(env):
    m = module(env, jurisdiction="strictest", consents=ConsentRegistry())
    inp = Prompter(["Alex", "", "n", "Sam", "c-unknown", "n", "the robot", "", "n", ""])
    out: list[str] = []
    assert _enroll(m, SimpleNamespace(camera=None), inp, out.append) == 0   # no exception
    fails = [line for line in out if line.startswith("Not enrolled:")]
    assert len(fails) == 3
    assert "requires a consent record" in fails[0] and "oiris face-match consent --subject CODE" in fails[0]
    assert "consent ID" in fails[1] and "oiris face-match consent --subject CODE" in fails[1]
    assert "banned wording" in fails[2]
    assert sum("Name or label" in p for p in inp.prompts) == 4   # asked for the next person every time
    assert count(env["db"]) == 0


# ---- consent command --------------------------------------------------------------------------------
@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("OIRIS_PASSPHRASE", "pw")
    cfgp = tmp_path / "config.json"
    cfgp.write_text(json.dumps({"data_dir": str(tmp_path)}))
    return tmp_path, cfgp


class InjectedBackend(TestDeterministicBackend):
    """The deterministic synthetic-frame backend, injected through run(backend=...) because a config file
    may not set allow_test_backend; it still only ever sees synthetic frames here."""
    test_only = False


def args(sub, **kw):
    return SimpleNamespace(sub=sub, mode=kw.get("mode"), jurisdiction=kw.get("jurisdiction"), camera=kw.get("camera"),
                           id=kw.get("id"), subject=kw.get("subject"), days=kw.get("days"), withdraw=kw.get("withdraw", False))


def records(tmp_path):
    ks = KeyStore(tmp_path / "keys.json"); ks.unlock("pw")
    db = DB(tmp_path / "asb.db", ks.subkey("db"))
    try:
        return dict(ConsentRegistry(db)._r)
    finally:
        db.close()


def test_consent_records_only_after_typed_yes(store):
    d, cfgp = store
    for typed in ("", "yes", "Y", "YES please", "no"):
        out: list[str] = []
        assert run(args("consent", subject="S-01"), Config.load(cfgp), cfgp, inp=lambda p, t=typed: t, out=out.append) == 1
        assert out[-1] == "Not recorded."
        assert "  subject code: S-01" in out and "  scope:        face_enrollment" in out
        assert any(line.startswith("  expires:") and "(30 days)" in line for line in out)
    assert records(d) == {}
    out = []
    t0 = time.time()
    assert run(args("consent", subject="S-01"), Config.load(cfgp), cfgp, inp=lambda p: "YES", out=out.append) == 0
    (cid, rec), = records(d).items()
    assert cid.startswith("consent-") and len(cid) == len("consent-") + 8
    assert out[-1] == f"Consent recorded. Enter this consent ID during enrollment: {cid}"
    assert rec.subject_code == "S-01" and rec.scopes == {"face_enrollment"} and not rec.withdrawn
    assert t0 + 30 * 86400 - 5 < rec.expires_utc < time.time() + 30 * 86400 + 5


def test_consent_days_limits_and_given_id(store):
    d, cfgp = store
    for days in (0, -1, 366, 1000):
        out: list[str] = []
        assert run(args("consent", subject="S-02", days=days), Config.load(cfgp), cfgp,
                   inp=lambda p: pytest.fail("asked for YES with invalid --days"), out=out.append) == 2
        assert out == ["--days must be between 1 and 365"]
    for days in (1, 365):
        assert run(args("consent", subject="S-02", days=days, id=f"c-{days}"), Config.load(cfgp), cfgp,
                   inp=lambda p: "YES", out=[].append) == 0
    recs = records(d)
    assert set(recs) == {"c-1", "c-365"}
    assert recs["c-365"].expires_utc - recs["c-1"].expires_utc == pytest.approx(364 * 86400, abs=5)
    out = []   # an existing ID is never overwritten
    assert run(args("consent", subject="S-03", id="c-1"), Config.load(cfgp), cfgp, inp=lambda p: "YES", out=out.append) == 2
    assert records(d)["c-1"].subject_code == "S-02"


def test_consent_needs_a_subject_code(store):
    d, cfgp = store
    for subject in (None, "", "Jane Doe", "a/b"):
        out: list[str] = []
        assert run(args("consent", subject=subject), Config.load(cfgp), cfgp, inp=lambda p: "YES", out=out.append) == 2
        assert "not their name" in out[-1]
    assert records(d) == {}


def test_consent_withdraw(store):
    d, cfgp = store
    out: list[str] = []
    assert run(args("consent", withdraw=True, id="nope"), Config.load(cfgp), cfgp, out=out.append) == 1
    assert out == ["no such consent record"]
    run(args("consent", subject="S-04", id="c-w"), Config.load(cfgp), cfgp, inp=lambda p: "YES", out=[].append)
    out = []
    assert run(args("consent", withdraw=True, id="c-w"), Config.load(cfgp), cfgp, out=out.append) == 0
    assert out == ["Consent c-w withdrawn."] and records(d)["c-w"].withdrawn
    out = []
    assert run(args("consent", withdraw=True, id="c-w"), Config.load(cfgp), cfgp, out=out.append) == 0
    assert "already withdrawn" in out[0]


def test_consent_then_strict_enroll_then_withdraw_erases(store, monkeypatch):
    """Whole Mac flow: enable (strictest), record consent, enroll with --camera auto, withdraw."""
    d, cfgp = store
    be = InjectedBackend()
    assert run(args("enable", mode=["personal"], jurisdiction="strictest"), Config.load(cfgp), cfgp,
               inp=lambda p: optin.CONFIRM_PHRASE, out=[].append) == 0
    out: list[str] = []
    run(args("consent", subject="S-05"), Config.load(cfgp), cfgp, inp=lambda p: "YES", out=out.append)
    cid = out[-1].rsplit(" ", 1)[1]
    inp = Prompter(["Alex", cid, "n"])
    monkeypatch.setitem(sys.modules, "cv2", FakeCv2({0: lambda: DARK, 1: lambda: synth_face_frame(23, inp.angle, noise_seed=len(inp.prompts))}))
    out = []
    assert run(args("enroll", camera="auto"), Config.load(cfgp), cfgp, inp=inp, out=out.append, backend=be) == 0
    assert "Using camera 1 (a face was seen there)." in out and not any(line.startswith("Not enrolled") for line in out)
    ks = KeyStore(d / "keys.json"); ks.unlock("pw")
    db = DB(d / "asb.db", ks.subkey("db"))
    try:
        assert count(db) == 1
    finally:
        db.close()
    out = []
    assert run(args("consent", withdraw=True, id=cid), Config.load(cfgp), cfgp, out=out.append, backend=be) == 0
    assert out == [f"Consent {cid} withdrawn.", "Face templates enrolled under it have been erased."]
    db = DB(d / "asb.db", ks.subkey("db"))
    try:
        assert count(db) == 0
    finally:
        db.close()
