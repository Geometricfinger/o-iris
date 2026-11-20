"""One face photo per enrolled, photo-consenting person (ADR-0057 decision 4).

Capture (RAM only): while a person is being enrolled, the best accepted FRONT shot of that person's own
enrollment is kept as a colour face crop. Live frames are never used. A replaced candidate is zeroed, and
`PhotoCandidate.wipe()` zeroes everything when the capture ends, finished or not.

Storage, access and erasure live in core.py: the JPEG is sealed with the person's own data key (AAD
b"photo:" + person_id) in fm_photo, and every M-16 erasure path removes it with verified wipe.
Strangers' faces are never kept: only frames submitted to `submit_shot` for an open capture reach here."""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .port import FaceBox
from .quality import crop, sharpness, to_gray

PHOTO_SCOPE = "photo"            # consent scope the person must have agreed to (`face-match consent --scope ...,photo`)
PHOTO_AAD = b"photo:"            # + person_id: the photo is sealed with the person's DEK, bound to that person
PHOTO_TABLE = "fm_photo"
PHOTO_MARGIN = 0.25              # added on each side of the face box, as a fraction of its larger side
PHOTO_SIZE_PX = 160              # the stored photo is PHOTO_SIZE_PX x PHOTO_SIZE_PX
PHOTO_JPEG_QUALITY = 85          # cv2.imencode JPEG quality; OpenCV writes no EXIF or other metadata
# The rule for "best": among accepted FRONT shots, the highest sharpness (variance of the Laplacian of the
# grayscale face box), the same measure check_shot uses for its blur check. A tie keeps the earlier shot.
BEST_SHOT_RULE = "highest sharpness (check_shot's blur measure) among accepted FRONT shots; ties keep the earlier"


def shot_quality(frame, box: FaceBox) -> float:
    """check_shot's blur measure on the face box. The grayscale copy is zeroed after use."""
    g = to_gray(frame)
    try:
        return sharpness(crop(g, box))
    finally:
        g.fill(0)


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize to size x size (numpy only: capture must not need OpenCV). Intermediates are zeroed."""
    h, w = img.shape[:2]
    ys = np.clip((np.arange(size) + 0.5) * h / size - 0.5, 0, h - 1)
    xs = np.clip((np.arange(size) + 0.5) * w / size - 0.5, 0, w - 1)
    y0, x0 = ys.astype(int), xs.astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = (ys - y0).astype(np.float32), (xs - x0).astype(np.float32)
    if img.ndim == 3:
        wy, wx = wy[:, None, None], wx[None, :, None]
    else:
        wy, wx = wy[:, None], wx[None, :]
    a = img.astype(np.float32)
    rows0, rows1 = a[y0], a[y1]
    top = rows0[:, x0] * (1 - wx) + rows0[:, x1] * wx
    bot = rows1[:, x0] * (1 - wx) + rows1[:, x1] * wx
    out = np.clip(np.rint(top * (1 - wy) + bot * wy), 0, 255).astype(np.uint8)
    for t in (a, rows0, rows1, top, bot):
        t.fill(0)
    return out


def face_crop(frame, box: FaceBox, margin: float = PHOTO_MARGIN, size: int = PHOTO_SIZE_PX) -> np.ndarray:
    """The face box plus `margin` on each side (a square around the box centre), clipped to the frame,
    resized to size x size. Colour frames stay colour (BGR, alpha dropped); grayscale stays grayscale."""
    a = np.asarray(frame)
    if a.ndim == 3:
        a = a[..., :3]
    h, w = a.shape[:2]
    side = max(box.w, box.h) * (1 + 2 * margin)
    cx, cy = box.x + box.w / 2, box.y + box.h / 2
    x0, y0 = max(0, int(round(cx - side / 2))), max(0, int(round(cy - side / 2)))
    x1, y1 = min(w, int(round(cx + side / 2))), min(h, int(round(cy + side / 2)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("face box is outside the frame")
    return _resize(a[y0:y1, x0:x1], size)


@dataclass
class PhotoCandidate:
    """RAM-only best FRONT crop of one capture. `pixels` is None until a FRONT shot is offered."""
    score: float = float("-inf")
    pixels: np.ndarray | None = None

    def offer(self, frame, box: FaceBox) -> bool:
        """Keep this accepted FRONT shot's crop if it beats the current one (BEST_SHOT_RULE)."""
        q = shot_quality(frame, box)
        if self.pixels is not None and not q > self.score:
            return False
        new = face_crop(frame, box)
        self.wipe()
        self.score, self.pixels = q, new
        return True

    def wipe(self) -> None:
        if self.pixels is not None:
            self.pixels.fill(0)
        self.pixels, self.score = None, float("-inf")


def encode_jpeg(pixels: np.ndarray) -> bytes:
    """JPEG at PHOTO_JPEG_QUALITY with cv2.imencode (no EXIF/metadata). The encoder buffer is zeroed."""
    import cv2
    ok, buf = cv2.imencode(".jpg", pixels, [int(cv2.IMWRITE_JPEG_QUALITY), PHOTO_JPEG_QUALITY])
    if not ok:
        raise ValueError("JPEG encoding failed")
    try:
        return buf.tobytes()
    finally:
        buf.fill(0)


__all__ = ["PHOTO_SCOPE", "PHOTO_AAD", "PHOTO_TABLE", "PHOTO_MARGIN", "PHOTO_SIZE_PX", "PHOTO_JPEG_QUALITY",
           "BEST_SHOT_RULE", "PhotoCandidate", "face_crop", "shot_quality", "encode_jpeg"]
