"""M-17 speaker model (ADR-0057 decision 7): sherpa-onnx CAM++ behind the `speaker_embed` slot.
Fakes only: no network, no real model, no audio files, no sherpa-onnx install."""
from __future__ import annotations
import hashlib, io, ssl, subprocess, sys, types, urllib.error
from pathlib import Path

import numpy as np
import pytest

from oiris.modules.face_match.backends.opencv_zoo import CERT_HELP
from oiris.modules.voice_link.backends import sherpa_campp as sc
from oiris.registry import PluginMeta, default_registry
from oiris.slots import SpeakerEmbed, SlotManager

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = b"not a real model, just pinned test bytes " * 64
FAKE_DIM = 192
SR = 16000
MIN_SAMPLES = 1600   # fake extractor: is_ready needs at least this many samples


@pytest.fixture
def pinned(monkeypatch):
    """Pin MODEL to PAYLOAD for the duration of a test."""
    monkeypatch.setitem(sc.MODEL, "size", len(PAYLOAD))
    monkeypatch.setitem(sc.MODEL, "sha256", hashlib.sha256(PAYLOAD).hexdigest())
    return PAYLOAD


class Opener:
    def __init__(self, data: bytes = PAYLOAD, exc: BaseException | None = None):
        self.data, self.exc, self.urls = data, exc, []

    def __call__(self, url):
        self.urls.append(url)
        if self.exc is not None:
            raise self.exc
        return io.BytesIO(self.data)


def leftovers(d: Path) -> list[str]:
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


# ---- fetch_models ------------------------------------------------------------------------------

def test_good_download_leaves_only_the_model(tmp_path, pinned):
    op = Opener()
    p = sc.fetch_models(tmp_path, opener=op)
    assert p == tmp_path / sc.MODEL["file"] and p.read_bytes() == PAYLOAD
    assert op.urls == [sc.MODEL["url"]]            # only the pinned URL, nothing else
    assert leftovers(tmp_path) == [sc.MODEL["file"]]


def test_wrong_sha256_raises_and_leaves_no_file(tmp_path, monkeypatch, pinned):
    monkeypatch.setitem(sc.MODEL, "sha256", "0" * 64)
    with pytest.raises(sc.ModelMissing, match="checksum"):
        sc.fetch_models(tmp_path, opener=Opener())
    assert leftovers(tmp_path) == []


def test_oversize_stream_is_aborted(tmp_path, pinned):
    with pytest.raises(sc.ModelMissing, match="larger than the pinned"):
        sc.fetch_models(tmp_path, opener=Opener(PAYLOAD + b"x" * (2 * sc.CHUNK_BYTES)))
    assert leftovers(tmp_path) == []


def test_short_stream_is_refused(tmp_path, pinned):
    with pytest.raises(sc.ModelMissing, match="incomplete"):
        sc.fetch_models(tmp_path, opener=Opener(PAYLOAD[:-1]))
    assert leftovers(tmp_path) == []


def test_certificate_error_message_includes_cert_help(tmp_path, pinned):
    err = urllib.error.URLError(ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED]"))
    with pytest.raises(sc.ModelMissing) as e:
        sc.fetch_models(tmp_path, opener=Opener(exc=err))
    assert CERT_HELP in str(e.value)
    assert leftovers(tmp_path) == []


def test_other_network_error_is_plain_model_missing(tmp_path, pinned):
    with pytest.raises(sc.ModelMissing, match="download failed") as e:
        sc.fetch_models(tmp_path, opener=Opener(exc=urllib.error.URLError("no route")))
    assert CERT_HELP not in str(e.value)
    assert leftovers(tmp_path) == []


def test_present_file_makes_no_network_call(tmp_path, pinned):
    (tmp_path / sc.MODEL["file"]).write_bytes(PAYLOAD)
    op = Opener(exc=AssertionError("network used"))
    assert sc.fetch_models(tmp_path, opener=op) == tmp_path / sc.MODEL["file"]
    assert op.urls == []


def test_wrong_present_file_is_replaced(tmp_path, pinned):
    (tmp_path / sc.MODEL["file"]).write_bytes(b"stale")
    sc.fetch_models(tmp_path, opener=Opener())
    assert (tmp_path / sc.MODEL["file"]).read_bytes() == PAYLOAD
    assert leftovers(tmp_path) == [sc.MODEL["file"]]


def test_pinned_facts():
    assert sc.MODEL["size"] == 29596978
    assert sc.MODEL["sha256"] == "357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b"
    assert sc.MODEL["url"].endswith("/" + sc.MODEL["file"]) and sc.MODEL["url"].startswith("https://huggingface.co/")
    assert sc.SherpaCamppEmbedder.model_id == "3dspeaker_speech_campplus_sv_en_voxceleb_16k@357a834f"


# ---- embedder with a fake sherpa_onnx ------------------------------------------------------------

def fake_sherpa(seen: dict) -> types.ModuleType:
    m = types.ModuleType("sherpa_onnx")

    class SpeakerEmbeddingExtractorConfig:
        def __init__(self, model, num_threads, provider, debug):
            seen["config"] = dict(model=model, num_threads=num_threads, provider=provider, debug=debug)
            self.model = model

        def validate(self):
            return Path(self.model).is_file()

    class Stream:
        def __init__(self):
            self.n, self.done = 0, False

        def accept_waveform(self, sample_rate, waveform):
            seen["waveform"], seen["sr"] = waveform, sample_rate
            self.n += len(waveform)

        def input_finished(self):
            self.done = True

    class SpeakerEmbeddingExtractor:
        def __init__(self, config):
            self.dim = FAKE_DIM

        def create_stream(self):
            return Stream()

        def is_ready(self, s):
            return s.done and s.n >= MIN_SAMPLES

        def compute(self, s):
            return [0.5] * FAKE_DIM            # sherpa returns a Python list

    m.SpeakerEmbeddingExtractorConfig = SpeakerEmbeddingExtractorConfig
    m.SpeakerEmbeddingExtractor = SpeakerEmbeddingExtractor
    return m


@pytest.fixture
def sherpa(monkeypatch):
    seen: dict = {}
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake_sherpa(seen))
    return seen


@pytest.fixture
def model_dir(tmp_path, pinned):
    (tmp_path / sc.MODEL["file"]).write_bytes(PAYLOAD)
    return tmp_path


def test_missing_sherpa_gives_install_hint(monkeypatch, model_dir):
    monkeypatch.setitem(sys.modules, "sherpa_onnx", None)   # makes `import sherpa_onnx` fail
    with pytest.raises(ImportError, match=r'pip install "o-iris\[voice\]"'):
        sc.SherpaCamppEmbedder(str(model_dir))


def test_missing_model_raises(sherpa, tmp_path, pinned):
    with pytest.raises(sc.ModelMissing, match="voice-link fetch-models"):
        sc.SherpaCamppEmbedder(str(tmp_path))


def test_wrong_size_model_raises(sherpa, model_dir):
    (model_dir / sc.MODEL["file"]).write_bytes(PAYLOAD + b"!")
    with pytest.raises(sc.ModelMissing, match="voice-link fetch-models"):
        sc.SherpaCamppEmbedder(str(model_dir))


def test_wrong_hash_same_size_model_raises(sherpa, model_dir):
    (model_dir / sc.MODEL["file"]).write_bytes(b"X" + PAYLOAD[1:])
    with pytest.raises(sc.ModelMissing):
        sc.SherpaCamppEmbedder(str(model_dir))


def test_embed_returns_float32_dim_and_zeroes_its_copy(sherpa, model_dir):
    e = sc.SherpaCamppEmbedder(str(model_dir), num_threads=2)
    assert isinstance(e, SpeakerEmbed) and e.name == "sherpa_campp" and e.dim == FAKE_DIM
    assert sherpa["config"] == dict(model=str(model_dir / sc.MODEL["file"]), num_threads=2, provider="cpu", debug=False)
    audio = np.full(SR, 0.25, np.float32)
    v = e.embed(audio, SR)
    assert v.dtype == np.float32 and v.shape == (FAKE_DIM,)
    assert sherpa["sr"] == SR
    assert sherpa["waveform"] is not audio and not np.any(sherpa["waveform"])   # our copy, zeroed
    assert np.all(audio == np.float32(0.25))                                       # caller's audio untouched
    assert not any(isinstance(x, np.ndarray) for x in vars(e).values())            # nothing retained


def test_too_short_audio_raises(sherpa, model_dir):
    e = sc.SherpaCamppEmbedder(str(model_dir))
    with pytest.raises(ValueError, match="audio too short"):
        e.embed(np.full(MIN_SAMPLES - 1, 0.1, np.float32), SR)
    assert not np.any(sherpa["waveform"])


def test_stereo_audio_is_refused(sherpa, model_dir):
    e = sc.SherpaCamppEmbedder(str(model_dir))
    with pytest.raises(ValueError, match="mono"):
        e.embed(np.zeros((SR, 2), np.float32), SR)


# ---- registry and slot ---------------------------------------------------------------------------

def test_default_registry_has_sherpa_campp_without_warning():
    r = default_registry({"entry_points": False})
    p = r.get("speaker_embed", "sherpa_campp")
    assert p.license == "Apache-2.0" and p.port_version == "1.0"
    assert p.meta == PluginMeta(license="Apache-2.0", weights_license="CC-BY-4.0", data_tier="public_permissive",
                                commercial_ok=True, source=sc.MODEL["url"])
    assert not any("sherpa_campp" in w for w in r.warnings)


def test_slot_manager_resolves_fake_backed_embedder(sherpa, model_dir):
    sm = SlotManager(default_registry({"entry_points": False}), {"speaker_embed": "sherpa_campp"},
                     {"sherpa_campp": {"model_dir": str(model_dir), "num_threads": 1}})
    v = sm.speaker_embedding(np.full(SR, 0.1, np.float32), SR)
    assert sm.status["speaker_embed"] == "active"
    assert isinstance(sm.get("speaker_embed"), sc.SherpaCamppEmbedder)
    assert v is not None and v.dtype == np.float32 and v.shape == (FAKE_DIM,)


def test_slot_degrades_when_model_missing(sherpa, tmp_path, pinned):
    sm = SlotManager(default_registry({"entry_points": False}), {"speaker_embed": "sherpa_campp"},
                     {"sherpa_campp": {"model_dir": str(tmp_path)}})
    assert sm.speaker_embedding(np.zeros(SR, np.float32), SR) is None
    assert sm.status["speaker_embed"].startswith("failed:")


def test_building_default_registry_imports_neither_backend_nor_sherpa():
    code = ("import sys; from oiris.registry import default_registry; default_registry({'entry_points': False}); "
            "bad = [m for m in ('sherpa_onnx', 'oiris.modules.voice_link.backends.sherpa_campp') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
