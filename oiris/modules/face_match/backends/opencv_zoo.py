"""Real backend: OpenCV `FaceDetectorYN` (YuNet, MIT) + `FaceRecognizerSF` (SFace, Apache-2.0),
models from github.com/opencv/opencv_zoo. Weights are NOT in this repo; they are fetched
only on explicit request (`oiris face-match fetch-models`) and checked against pinned
SHA-256 hashes. Licences recorded in THIRD_PARTY.md. Needs `opencv-python-headless`
(Apache-2.0, the existing `hw` extra)."""
from __future__ import annotations
import functools, hashlib, importlib, os, ssl, urllib.error, urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ....schema import Health
from ..port import Angle, Embedding, FaceBox, MatchResult, Template, cosine_best

ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
MODELS = {
    "detector": {"file": "face_detection_yunet_2023mar.onnx", "url": f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                 "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", "license": "MIT"},
    "embedder": {"file": "face_recognition_sface_2021dec.onnx", "url": f"{ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
                 "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79", "license": "Apache-2.0"},
}
FETCH_TIMEOUT_S = 60   # per network operation (connect, each read): a stalled download fails instead of hanging


class ModelMissing(RuntimeError):
    pass


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_paths(model_dir: str | Path) -> dict[str, Path]:
    d = Path(model_dir).expanduser()
    return {k: d / m["file"] for k, m in MODELS.items()}


def models_present(model_dir: str | Path) -> bool:
    return all(p.exists() and sha256(p) == MODELS[k]["sha256"] for k, p in model_paths(model_dir).items())


CERT_HELP = ("Python could not verify the download's certificate. To fix it, run "
             "\"Install Certificates.command\" in /Applications/Python 3.x, or `pip install certifi`, "
             "or set SSL_CERT_FILE to a CA bundle, then try again.")


def ca_file() -> str | None:
    """CA bundle for the download: certifi, else pip's vendored copy, else None (default context).
    python.org's macOS Python has no root certificates until "Install Certificates.command" runs.
    A user-set SSL_CERT_FILE wins: the default context honours it."""
    if os.environ.get("SSL_CERT_FILE"):
        return None
    for name in ("certifi", "pip._vendor.certifi"):
        try:
            return str(importlib.import_module(name).where())
        except ImportError:
            continue
    return None


def ssl_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=ca_file())


def _cert_failure(e: urllib.error.URLError) -> bool:
    return isinstance(e.reason, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(e)


def fetch_models(model_dir: str | Path, opener: Callable[[str], Any] | None = None) -> dict[str, Path]:
    """Explicit, user-initiated download. Rejects any file whose hash doesn't match.
    `opener` is for tests; by default urllib is used with `ssl_context()` and FETCH_TIMEOUT_S."""
    if opener is None:
        opener = functools.partial(urllib.request.urlopen, context=ssl_context(), timeout=FETCH_TIMEOUT_S)
    out = model_paths(model_dir)
    for k, p in out.items():
        if p.exists() and sha256(p) == MODELS[k]["sha256"]:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        try:
            with opener(MODELS[k]["url"]) as r, open(tmp, "wb") as f:
                f.write(r.read())
        except urllib.error.URLError as e:
            tmp.unlink(missing_ok=True)
            if _cert_failure(e):
                raise ModelMissing(CERT_HELP) from e
            raise
        if sha256(tmp) != MODELS[k]["sha256"]:
            tmp.unlink()
            raise ModelMissing(f"{p.name}: checksum mismatch, refusing to use it")
        tmp.replace(p)
    return out


def pitch_yaw(lm: np.ndarray) -> tuple[float, float]:
    """5 points (right eye, left eye, nose, right mouth, left mouth) -> (pitch, yaw).
    yaw: nose offset from the eye midpoint, normalised by eye distance (0 when level).
    pitch: nose height between the eye and mouth midlines, as a fraction (~0.55 when level)."""
    re_, le, nose, rm, lmo = (np.asarray(lm[i], dtype=np.float64) for i in range(5))
    eye_mid, mouth_mid = (re_ + le) / 2, (rm + lmo) / 2
    d = float(np.linalg.norm(le - re_)) or 1.0
    span = float(mouth_mid[1] - eye_mid[1]) or 1.0
    return float((nose[1] - eye_mid[1]) / span), float((nose[0] - eye_mid[0]) / d)


def pose_from_landmarks(lm: np.ndarray) -> Angle:
    """Coarse absolute pose bucket from `pitch_yaw`."""
    pitch, yaw = pitch_yaw(lm)
    if abs(yaw) >= 0.18 and abs(yaw) >= abs(pitch - 0.55):
        return Angle.RIGHT if yaw > 0 else Angle.LEFT
    if pitch < 0.40:
        return Angle.UP
    if pitch > 0.72:
        return Angle.DOWN
    return Angle.FRONT


def box_from_row(f: Any) -> FaceBox:
    """One YuNet row (x, y, w, h, 5 landmark x/y pairs, score) -> FaceBox with pose, pitch and yaw."""
    lm = np.asarray(f[4:14], dtype=np.float32).reshape(5, 2)
    pitch, yaw = pitch_yaw(lm)
    return FaceBox(int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[14]), pose_from_landmarks(lm),
                   np.asarray(f, dtype=np.float32), pitch=pitch, yaw=yaw)


class OpenCvZooBackend:
    name = "opencv_zoo"
    license = "Apache-2.0"           # code + weights: YuNet MIT, SFace Apache-2.0
    model_id = "opencv-zoo-yunet2023mar-sface2021dec"
    model_version = "1"
    test_only = False

    def __init__(self) -> None:
        self._det: Any = None
        self._rec: Any = None
        self._health = (Health.FAILED, "not opened")

    def open(self, cfg: dict) -> None:
        try:
            import cv2
        except ImportError as e:
            raise ModelMissing("opencv-python-headless not installed (pip install 'o-iris[hw]')") from e
        model_dir = cfg.get("model_dir") or "~/.asb/models/face_match"
        if not models_present(model_dir):
            raise ModelMissing("face models not downloaded; run `oiris face-match fetch-models`")
        p = model_paths(model_dir)
        self._det = cv2.FaceDetectorYN.create(str(p["detector"]), "", (320, 320), float(cfg.get("det_score", 0.8)), 0.3, 50)
        self._rec = cv2.FaceRecognizerSF.create(str(p["embedder"]), "")
        self._health = (Health.OK, "")

    def close(self) -> None:
        self._det = self._rec = None

    def health(self) -> tuple[Health, str]:
        return self._health

    def detect(self, frame: Any, timeout_s: float) -> list[FaceBox]:
        img = np.asarray(frame)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        h, w = img.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(img)
        out = []
        for f in faces if faces is not None else []:
            out.append(box_from_row(f))
        return out

    def embed(self, frame: Any, box: FaceBox) -> Embedding:
        img = np.asarray(frame)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        aligned = self._rec.alignCrop(img, box.landmarks)
        e = Embedding(np.asarray(self._rec.feature(aligned)).ravel())
        aligned.fill(0)
        return e

    def compare(self, e: Embedding, templates: Sequence[Template]) -> MatchResult | None:
        return cosine_best(e, templates)
