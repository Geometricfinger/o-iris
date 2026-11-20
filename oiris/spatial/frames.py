"""Head frame, mount rotations, camera rays and direction words (ADR-0060). Pure NumPy.

Head frame: right-handed, origin midway between the eyes, x forward, y left, z up.
Azimuth az = atan2(y, x) in degrees, in (-180, 180]: 0 ahead, +90 left, -90 right, 180 behind.
Elevation el = atan2(z, hypot(x, y)) in degrees: +90 straight up.

A mount's yaw turns its forward axis left, pitch tilts it up and roll turns it clockwise as seen from
behind the mount (its top tilts to its right). Mount-to-head rotation R = Rz(yaw) @ Ry(-pitch) @ Rx(roll).
Bearings ignore the mount position: parallax is negligible beyond about a metre.

Camera pixels use pinhole intrinsics (fx, fy, cx, cy). The OpenCV optical frame is x right, y down,
z forward; the mount frame is forward = z_opt, left = -x_opt, up = -y_opt."""
from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np

# Direction-word sectors (degrees). The words are the phrase slot values in oiris/guard/phrases.py.
AHEAD_MAX_DEG = 45.0          # |az| <= this: "ahead"
SIDE_MAX_DEG = 135.0          # 45 < |az| <= this: "at left" / "at right"; beyond it: "behind"
VERTICAL_DEG = 45.0           # el > this: "above"; el < -this: "below" (overrides the horizontal word)
HYSTERESIS_DEG = 10.0         # a tracked bearing keeps its word until this far past a boundary

AHEAD, LEFT, RIGHT, BEHIND, ABOVE, BELOW = "ahead", "at left", "at right", "behind", "above", "below"
DIRECTION_WORDS = (AHEAD, LEFT, RIGHT, BEHIND, ABOVE, BELOW)

FULL_TURN_DEG = 360.0
HALF_TURN_DEG = 180.0
# Below this norm a vector has no direction (a zero vector from cancelling inputs).
MIN_NORM = 1e-12


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def mount_to_head(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Rotation taking a mount-frame vector into the head frame."""
    return rot_z(yaw_deg) @ rot_y(-pitch_deg) @ rot_x(roll_deg)


def normalize(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if not math.isfinite(n) or n < MIN_NORM:
        raise ValueError("vector has no direction")
    return a / n


def wrap_az(az_deg: float) -> float:
    """Wrap an azimuth into (-180, 180]."""
    a = math.fmod(az_deg, FULL_TURN_DEG)
    if a <= -HALF_TURN_DEG:
        a += FULL_TURN_DEG
    elif a > HALF_TURN_DEG:
        a -= FULL_TURN_DEG
    return a


def vec_to_az_el(v) -> tuple[float, float]:
    x, y, z = (float(c) for c in normalize(v))
    az = math.degrees(math.atan2(y, x))
    el = math.degrees(math.atan2(z, math.hypot(x, y)))
    return wrap_az(az), el


def az_el_to_vec(az_deg: float, el_deg: float) -> np.ndarray:
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def angle_between(a, b) -> float:
    """Angle in degrees between two direction vectors (atan2 form: accurate near 0 and 180)."""
    ua, ub = normalize(a), normalize(b)
    return math.degrees(math.atan2(float(np.linalg.norm(np.cross(ua, ub))), float(np.dot(ua, ub))))


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in pixels, with the image size (w, h) that bounds valid pixels."""
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        for k in ("fx", "fy", "cx", "cy"):
            if not math.isfinite(getattr(self, k)):
                raise ValueError(f"intrinsics {k} must be finite")
        if not (self.fx > 0 and self.fy > 0 and self.width > 0 and self.height > 0):
            raise ValueError("intrinsics need positive fx, fy and image size")


def fov_intrinsics(w: int, h: int, hfov_deg: float, vfov_deg: float) -> Intrinsics:
    """Intrinsics from fields of view: fx = (w/2)/tan(hfov/2), fy likewise, centre at (w/2, h/2)."""
    if not (0 < hfov_deg < HALF_TURN_DEG and 0 < vfov_deg < HALF_TURN_DEG):
        raise ValueError("a pinhole field of view must be in (0, 180) degrees")
    fx = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    fy = (h / 2) / math.tan(math.radians(vfov_deg) / 2)
    return Intrinsics(fx, fy, w / 2, h / 2, w, h)


def pixel_to_mount_ray(u: float, v: float, k: Intrinsics) -> np.ndarray:
    """Unit ray in the mount frame (forward, left, up) through pixel (u, v)."""
    xo, yo = (u - k.cx) / k.fx, (v - k.cy) / k.fy
    return normalize([1.0, -xo, -yo])


def mount_ray_to_pixel(ray, k: Intrinsics) -> tuple[float, float] | None:
    """Pixel (u, v) where a mount-frame ray lands, or None when it is behind the camera or off the image."""
    f, left, up = (float(c) for c in normalize(ray))
    if f <= 0:
        return None
    u, v = k.cx + k.fx * (-left / f), k.cy + k.fy * (-up / f)
    if not (0 <= u < k.width and 0 <= v < k.height):
        return None
    return u, v


def _horizontal_word(az: float) -> str:
    if abs(az) <= AHEAD_MAX_DEG:
        return AHEAD
    if AHEAD_MAX_DEG < az <= SIDE_MAX_DEG:
        return LEFT
    if -SIDE_MAX_DEG <= az < -AHEAD_MAX_DEG:
        return RIGHT
    return BEHIND


def direction_word(az_deg: float, el_deg: float = 0.0) -> str:
    """The phrase-slot word for a head-frame bearing. "above"/"below" override the horizontal word."""
    if el_deg > VERTICAL_DEG:
        return ABOVE
    if el_deg < -VERTICAL_DEG:
        return BELOW
    return _horizontal_word(wrap_az(az_deg))


def in_sector(word: str, az_deg: float, el_deg: float, margin_deg: float = 0.0) -> bool:
    """Whether a bearing lies in a word's sector widened by margin_deg past each boundary."""
    az, m = wrap_az(az_deg), margin_deg
    if word == ABOVE:
        return el_deg > VERTICAL_DEG - m
    if word == BELOW:
        return el_deg < -VERTICAL_DEG + m
    if abs(el_deg) > VERTICAL_DEG + m:
        return False
    if word == AHEAD:
        return abs(az) <= AHEAD_MAX_DEG + m
    if word == LEFT:
        return AHEAD_MAX_DEG - m < az <= SIDE_MAX_DEG + m
    if word == RIGHT:
        return -SIDE_MAX_DEG - m <= az < -AHEAD_MAX_DEG + m
    if word == BEHIND:
        return abs(az) > SIDE_MAX_DEG - m
    raise ValueError(f"unknown direction word {word!r}")


class DirectionTracker:
    """Direction word for one tracked bearing, with hysteresis: the word changes only once the bearing
    is HYSTERESIS_DEG past the boundary of the current word's sector."""

    def __init__(self, hysteresis_deg: float = HYSTERESIS_DEG) -> None:
        self.hysteresis_deg = hysteresis_deg
        self.word: str | None = None

    def update(self, az_deg: float, el_deg: float = 0.0) -> str:
        if self.word is None or not in_sector(self.word, az_deg, el_deg, self.hysteresis_deg):
            self.word = direction_word(az_deg, el_deg)
        return self.word


__all__ = ["rot_x", "rot_y", "rot_z", "mount_to_head", "normalize", "wrap_az", "vec_to_az_el", "az_el_to_vec",
           "angle_between", "Intrinsics", "fov_intrinsics", "pixel_to_mount_ray", "mount_ray_to_pixel",
           "direction_word", "in_sector", "DirectionTracker", "DIRECTION_WORDS", "AHEAD_MAX_DEG", "SIDE_MAX_DEG",
           "VERTICAL_DEG", "HYSTERESIS_DEG"]
