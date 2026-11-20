"""M-16 fixes from the 2026-11-15 real-webcam test: model download on python.org Python
(CA bundle fallback, plain certificate message) and up/down judged against the person's own
front view. Synthetic data only: no real faces, no network."""
import hashlib, io, ssl, sys, types, urllib.error, urllib.request

import numpy as np
import pytest
from nacl.signing import SigningKey

from oiris.config import validate_face_match
from oiris.guard.consent import ConsentRecord, ConsentRegistry
from oiris.modules.face_match import optin
from oiris.modules.face_match.backends import opencv_zoo as oz
from oiris.modules.face_match.backends.test_deterministic import TestDeterministicBackend, synth_face_frame
from oiris.modules.face_match.core import FaceMatchConfig, FaceMatchModule
from oiris.modules.face_match.port import REQUIRED_ANGLES, Angle, FaceBox
from oiris.modules.face_match.quality import REASONS
from oiris.security.hashchain import HashChain
from oiris.security.keys import KeyStore
from oiris.store.db import DB


# ---- certificate / CA bundle -----------------------------------------------------------------
def _fake(path):
    m = types.ModuleType("fake_certifi")
    m.where = lambda: path  # type: ignore[attr-defined]
    return m


def test_ca_prefers_certifi(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setitem(sys.modules, "certifi", _fake("/ca/certifi.pem"))
    monkeypatch.setitem(sys.modules, "pip._vendor.certifi", _fake("/ca/pip.pem"))
    assert oz.ca_file() == "/ca/certifi.pem"


def test_ca_falls_back_to_pip_copy_then_default(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setitem(sys.modules, "certifi", None)       # None in sys.modules -> ImportError
    monkeypatch.setitem(sys.modules, "pip._vendor.certifi", _fake("/ca/pip.pem"))
    assert oz.ca_file() == "/ca/pip.pem"
    monkeypatch.setitem(sys.modules, "pip._vendor.certifi", None)
    assert oz.ca_file() is None
    seen = {}
    monkeypatch.setattr(oz.ssl, "create_default_context", lambda cafile=None: seen.setdefault("cafile", cafile))
    oz.ssl_context()
    assert seen == {"cafile": None}


def test_user_ssl_cert_file_wins(monkeypatch):
    monkeypatch.setenv("SSL_CERT_FILE", "/user/bundle.pem")
    monkeypatch.setitem(sys.modules, "certifi", _fake("/ca/certifi.pem"))
    assert oz.ca_file() is None      # the default context reads SSL_CERT_FILE itself


def test_ssl_context_uses_selected_bundle(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setitem(sys.modules, "certifi", _fake("/ca/certifi.pem"))
    seen = {}
    monkeypatch.setattr(oz.ssl, "create_default_context", lambda cafile=None: seen.setdefault("cafile", cafile))
    oz.ssl_context()
    assert seen == {"cafile": "/ca/certifi.pem"}


def _verify_error():
    return urllib.error.URLError(ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate"))


@pytest.mark.parametrize("err", [_verify_error(), urllib.error.URLError("<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed>")])
def test_certificate_failure_is_plain_model_missing(tmp_path, err):
    def opener(url):
        raise err
    with pytest.raises(oz.ModelMissing) as ei:
        oz.fetch_models(tmp_path, opener=opener)
    msg = str(ei.value)
    assert "could not verify the download's certificate" in msg
    assert "Install Certificates.command" in msg and "/Applications/Python 3.x" in msg
    assert "pip install certifi" in msg and "SSL_CERT_FILE" in msg
    assert not list(tmp_path.glob("*.part")) and not list(tmp_path.glob("*.onnx"))


def test_other_url_errors_are_not_relabelled(tmp_path):
    def opener(url):
        raise urllib.error.URLError("no route to host")
    with pytest.raises(urllib.error.URLError) as ei:
        oz.fetch_models(tmp_path, opener=opener)
    assert not isinstance(ei.value, oz.ModelMissing)


def test_default_opener_passes_ssl_context(tmp_path, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(oz, "ssl_context", lambda: sentinel)
    seen = []

    timeouts = []

    def fake_urlopen(url, context=None, timeout=None):
        seen.append(context)
        timeouts.append(timeout)
        raise _verify_error()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(oz.ModelMissing, match="certificate"):
        oz.fetch_models(tmp_path)
    assert seen == [sentinel]
    assert all(t is None or t > 0 for t in timeouts)   # a network timeout, when passed, is positive


def test_checksum_still_enforced(tmp_path, monkeypatch):
    with pytest.raises(oz.ModelMissing, match="checksum mismatch"):
        oz.fetch_models(tmp_path, opener=lambda url: io.BytesIO(b"not the model"))
    assert not list(tmp_path.glob("*.onnx")) and not list(tmp_path.glob("*.part"))
    # with pinned hashes that do match, the same path accepts the files
    blobs = {"detector": b"synthetic detector bytes", "embedder": b"synthetic embedder bytes"}
    models = {k: dict(m, url=f"https://example.invalid/{k}", sha256=hashlib.sha256(blobs[k]).hexdigest()) for k, m in oz.MODELS.items()}
    monkeypatch.setattr(oz, "MODELS", models)
    out = oz.fetch_models(tmp_path, opener=lambda url: io.BytesIO(blobs[url.rsplit("/", 1)[1]]))
    assert oz.models_present(tmp_path) and {k: p.read_bytes() for k, p in out.items()} == blobs


# ---- pitch / yaw measure ----------------------------------------------------------------------
LEVEL = [[40, 50], [80, 50], [60, 72], [45, 90], [75, 90]]   # right eye, left eye, nose, right mouth, left mouth


def _row(lm, score=0.93):
    return np.array([20, 30, 80, 90, *np.asarray(lm, float).ravel(), score], np.float32)


def test_pitch_yaw_from_synthetic_yunet_row():
    row = _row(LEVEL)
    assert row.shape == (15,)
    b = oz.box_from_row(row)
    assert (b.x, b.y, b.w, b.h) == (20, 30, 80, 90) and b.score == pytest.approx(0.93)
    assert b.pitch == pytest.approx(22 / 40) and b.yaw == pytest.approx(0.0) and b.pose == Angle.FRONT
    turned = np.array(LEVEL, float) + [[0, 0], [0, 0], [10, 0], [0, 0], [0, 0]]
    b2 = oz.box_from_row(_row(turned))
    assert b2.yaw == pytest.approx(10 / 40) and b2.pose == Angle.RIGHT
    # one measure: pose_from_landmarks and the FaceBox fields agree
    assert oz.pitch_yaw(np.array(LEVEL, float)) == (pytest.approx(b.pitch), pytest.approx(b.yaw))
    assert oz.pose_from_landmarks(np.array(LEVEL, float)) == Angle.FRONT


# ---- relative up/down in enrollment ------------------------------------------------------------
class PitchBackend(TestDeterministicBackend):
    """Test backend that also reports a continuous pitch (set per shot by the test)."""
    pitch: float | None = None

    def detect(self, frame, timeout_s):
        boxes = super().detect(frame, timeout_s)
        return [FaceBox(b.x, b.y, b.w, b.h, b.score, b.pose, b.landmarks, pitch=self.pitch) for b in boxes]


@pytest.fixture
def env(tmp_path):
    ks = KeyStore(tmp_path / "keys.json"); ks.create("pw")
    db = DB(tmp_path / "asb.db", ks.subkey("db"))
    chain = HashChain(ks.subkey("face-chain"), SigningKey.generate())
    consents = ConsentRegistry()
    consents.add(ConsentRecord("c-ok", "s1", {"face_enrollment"}))
    yield {"ks": ks, "db": db, "chain": chain, "consents": consents, "dir": tmp_path}
    db.close()


def make(env, backend, **cfg):
    optin.record_opt_in(env["db"], env["chain"], "I UNDERSTAND", ["personal"], "dev1", "us-other",
                        warning_shown=optin.warning_for("us-other"))
    c = FaceMatchConfig(modes=["personal"], allow_test_backend=True, **cfg)
    return FaceMatchModule(env["db"], env["ks"], env["dir"], backend, env["chain"], c, env["consents"],
                           timeout_s=1.0, breaker_threshold=3, breaker_cooldown_s=60)


def shot(m, be, pid, want, pitch, pose=None, n=0):
    be.pitch = pitch
    return m.submit_shot(pid, synth_face_frame(7, pose or want, noise_seed=n), want)


def test_relative_up_down_against_front(env):
    be = PitchBackend()
    m = make(env, be)
    pid = m.add_person(m.start_session(), "Alex", "c-ok")
    assert shot(m, be, pid, Angle.FRONT, 0.55).ok
    # absolute buckets are ignored when pitch is known: the frame's own marker says UP/DOWN but pitch decides
    r = shot(m, be, pid, Angle.UP, 0.52, n=1)
    assert not r.ok and r.code == "angle" and r.reason == REASONS["angle"]
    assert shot(m, be, pid, Angle.UP, 0.49, pose=Angle.FRONT, n=2).ok
    r = shot(m, be, pid, Angle.DOWN, 0.60, n=3)
    assert not r.ok and r.code == "angle"
    r = shot(m, be, pid, Angle.DOWN, 0.64, pose=Angle.FRONT, n=4)
    assert r.ok and set(r.needed_angles) == {"left", "right"}


def test_reference_is_mean_of_front_shots_and_deltas_configurable(env):
    be = PitchBackend()
    m = make(env, be, pose_up_delta=0.10, pose_down_delta=0.10)
    pid = m.add_person(m.start_session(), "Alex", "c-ok")
    assert shot(m, be, pid, Angle.FRONT, 0.50).ok and shot(m, be, pid, Angle.FRONT, 0.60, n=1).ok   # ref 0.55
    assert not shot(m, be, pid, Angle.UP, 0.49, n=2).ok
    assert shot(m, be, pid, Angle.UP, 0.45, n=3).ok
    assert not shot(m, be, pid, Angle.DOWN, 0.64, n=4).ok
    assert shot(m, be, pid, Angle.DOWN, 0.65, n=5).ok


def test_up_down_before_front_asks_for_front_first(env):
    be = PitchBackend()
    m = make(env, be)
    pid = m.add_person(m.start_session(), "Alex", "c-ok")
    for want, p in ((Angle.UP, 0.30), (Angle.DOWN, 0.90)):
        r = shot(m, be, pid, want, p)
        assert not r.ok and r.code == "front_first" and r.reason == "capture the front view first"
    assert REASONS["front_first"] == "capture the front view first"
    assert m.capture_buffer_size() == 0
    # quality checks still run first: a blurry up shot reports blur, not front_first
    be.pitch = 0.30
    r = m.submit_shot(pid, synth_face_frame(7, Angle.UP, blur=True), Angle.UP)
    assert not r.ok and r.code == "blur"


def test_old_absolute_rule_when_no_pitch(env):
    be = PitchBackend()            # pitch stays None
    m = make(env, be)
    pid = m.add_person(m.start_session(), "Alex", "c-ok")
    assert shot(m, be, pid, Angle.UP, None).ok                       # no front needed without a pitch
    r = shot(m, be, pid, Angle.DOWN, None, pose=Angle.FRONT, n=1)
    assert not r.ok and r.code == "angle"
    assert shot(m, be, pid, Angle.DOWN, None, n=2).ok


def test_full_five_angle_enrollment_with_relative_rule(env):
    be = PitchBackend()
    m = make(env, be)
    pid = m.add_person(m.start_session(), "Alex", "c-ok")
    # level ~0.53, tilts only -0.07 / +0.11: the old fixed limits (<0.40, >0.72) could never pass these
    pitches = {Angle.FRONT: 0.53, Angle.LEFT: 0.53, Angle.RIGHT: 0.54, Angle.UP: 0.46, Angle.DOWN: 0.64}
    for i, a in enumerate(REQUIRED_ANGLES):
        assert shot(m, be, pid, a, pitches[a], pose=Angle.FRONT if a in (Angle.UP, Angle.DOWN) else a, n=i).ok
    st = m.finish_person(pid)
    assert st.complete, st.message
    assert m.capture_buffer_size() == 0
    assert m.db.execute("SELECT angles_covered FROM fm_enrolled").fetchone()[0] == "down,front,left,right,up"
    # matching is unchanged
    be.pitch = None
    assert m.process_frame(synth_face_frame(7, noise_seed=50), place="Home")["matched"] == 1


# ---- config ------------------------------------------------------------------------------------
def test_delta_config_validation():
    c = FaceMatchConfig()
    assert (c.pose_up_delta, c.pose_down_delta) == (0.06, 0.08)
    for k in ("pose_up_delta", "pose_down_delta"):
        for ok in (0.02, 0.15, 0.30):
            FaceMatchConfig(**{k: ok}).validate()
            assert not validate_face_match({k: ok})
        for bad in (0.0, 0.019, 0.31, -0.1, 1.0):
            with pytest.raises(ValueError, match=k):
                FaceMatchConfig(**{k: bad}).validate()
            assert validate_face_match({k: bad})
        with pytest.raises(ValueError):
            FaceMatchConfig.from_dict({k: "0.1"})
