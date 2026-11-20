"""`speaker_embed` slot plugin: sherpa-onnx (Apache-2.0) running the 3D-Speaker CAM++ English
model (ADR-0057 decision 7). The model is NOT in this repo; it is fetched only on explicit
request (`oiris voice-link fetch-models`), size- and SHA-256-pinned, with M-16's certificate
handling. Licences recorded in THIRD_PARTY.md. Needs the `voice` extra (pip install "o-iris[voice]").

Audio and embeddings stay in RAM: this module never stores or logs either."""
from __future__ import annotations
import functools, hashlib, hmac, os, ssl, tempfile, urllib.error, urllib.request
from pathlib import Path
from typing import Any, Callable

import numpy as np

MODEL: dict[str, Any] = {  # pinned; checked 2026-11-16
    "file": "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
    "url": "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
    "size": 29596978,
    "sha256": "357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b",
    "license": "Apache-2.0",          # 3D-Speaker code and model (ModelScope)
    "weights_license": "CC-BY-4.0",   # trained on VoxCeleb1/2: attribution required
    "training_data": "VoxCeleb1, VoxCeleb2 (CC BY 4.0)",
}
DEFAULT_MODEL_DIR = "~/.asb/models/voice_link"
CHUNK_BYTES = 1 << 20
FETCH_TIMEOUT_S = 60   # per network operation (connect, each read): a stalled download fails instead of hanging
MODEL_ID_HASH_CHARS = 8
INSTALL_HINT = 'sherpa-onnx is not installed; run: pip install "o-iris[voice]"'
FETCH_HINT = "run: oiris voice-link fetch-models"


class ModelMissing(RuntimeError):
    pass


def model_path(model_dir: str | Path = "") -> Path:
    return Path(model_dir or DEFAULT_MODEL_DIR).expanduser() / str(MODEL["file"])


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_ok(p: Path) -> bool:
    return hmac.compare_digest(_sha256(p), str(MODEL["sha256"]))


def model_ok(p: Path) -> bool:
    """True when `p` is the pinned file: right size (cheap check first), then right SHA-256."""
    return p.is_file() and p.stat().st_size == MODEL["size"] and _hash_ok(p)


def _cert_failure(e: BaseException) -> bool:
    reason = getattr(e, "reason", e)
    return isinstance(reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(e)


def _download(opener: Callable[[str], Any], url: str, tmp: Path, limit: int) -> None:
    got = 0
    with opener(url) as r, open(tmp, "wb") as f:
        while chunk := r.read(CHUNK_BYTES):
            got += len(chunk)
            if got > limit:
                raise ModelMissing(f"{MODEL['file']}: download is larger than the pinned {limit} bytes, aborted")
            f.write(chunk)


def fetch_models(model_dir: str | Path = "", opener: Callable[[str], Any] | None = None) -> Path:
    """Explicit, user-initiated download of the one pinned file. No network call when the file
    is already present and correct. `opener` is for tests; by default urllib with M-16's
    `ssl_context()` (SSL_CERT_FILE wins, then certifi, then pip's vendored certifi)."""
    p = model_path(model_dir)
    if model_ok(p):
        return p
    from oiris.modules.face_match.backends.opencv_zoo import CERT_HELP, ssl_context
    if opener is None:
        opener = functools.partial(urllib.request.urlopen, context=ssl_context(), timeout=FETCH_TIMEOUT_S)
    size = int(MODEL["size"])
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=p.parent, prefix=p.name + ".", suffix=".part")
    os.close(fd)
    tmp = Path(name)
    try:
        _download(opener, str(MODEL["url"]), tmp, size)
        if tmp.stat().st_size != size:
            raise ModelMissing(f"{p.name}: incomplete download ({tmp.stat().st_size} of {size} bytes)")
        if not _hash_ok(tmp):
            raise ModelMissing(f"{p.name}: checksum mismatch, refusing to use it")
        os.replace(tmp, p)
    except ModelMissing:
        tmp.unlink(missing_ok=True)
        raise
    except Exception as e:  # noqa: BLE001 - any failure: remove the partial file, report plainly
        tmp.unlink(missing_ok=True)
        if _cert_failure(e):
            raise ModelMissing(CERT_HELP) from e
        raise ModelMissing(f"{p.name}: download failed ({e})") from e
    return p


class SherpaCamppEmbedder:
    """oiris.slots.SpeakerEmbed backed by sherpa_onnx.SpeakerEmbeddingExtractor (CPU)."""
    name = "sherpa_campp"
    license = "Apache-2.0"
    model_id = Path(str(MODEL["file"])).stem + "@" + str(MODEL["sha256"])[:MODEL_ID_HASH_CHARS]

    def __init__(self, model_dir: str = "", num_threads: int = 1) -> None:
        try:
            import sherpa_onnx
        except ImportError as e:
            raise ImportError(INSTALL_HINT) from e
        p = model_path(model_dir)
        if not p.is_file():
            raise ModelMissing(f"speaker model {p.name} not found in {p.parent}; {FETCH_HINT}")
        if not model_ok(p):
            raise ModelMissing(f"speaker model {p} does not match the pinned size/SHA-256; {FETCH_HINT}")
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(p), num_threads=int(num_threads),
                                                             provider="cpu", debug=False)
        if not config.validate():
            raise ModelMissing(f"sherpa-onnx rejected the speaker model config for {p}; {FETCH_HINT}")
        self._x = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.dim = int(self._x.dim)

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """`audio` = mono float32 PCM in [-1, 1]. Returns float32[dim]. Our copy is zeroed after use."""
        buf = np.array(audio, dtype=np.float32, copy=True)
        try:
            if buf.ndim != 1:
                raise ValueError("audio must be mono (1-D)")
            if int(sample_rate) <= 0:
                raise ValueError("sample_rate must be positive")
            stream = self._x.create_stream()
            stream.accept_waveform(sample_rate=int(sample_rate), waveform=buf)
            stream.input_finished()
            if not self._x.is_ready(stream):
                raise ValueError("audio too short")
            v = np.asarray(self._x.compute(stream), dtype=np.float32).ravel()
        finally:
            buf.fill(0)
        if v.size != self.dim:
            raise ValueError(f"speaker model returned {v.size} values, expected {self.dim}")
        return v
