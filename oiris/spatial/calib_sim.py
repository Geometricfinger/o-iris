"""Calibration board, calibration ring and a pinhole renderer for proving mount calibration in simulation
(ADR-0064). Pure NumPy, deterministic: the same arguments (and seed) give the same image.

Board frame: origin at the board centre, x to the right and y up as seen from the printed side, z = x cross y
out of the printed face. The board is a grid of ArUco DICT_4X4_50 markers (oiris/features/aruco.py): each marker
is 6 x 6 cells (a black border round the 4 x 4 code bits), with ids first_id, first_id + 1, ... row by row from
the top left. A white gap separates markers and surrounds the grid; it is the quiet zone detection needs.

Poses are (R, t) with x_world = R @ x_local + t. A camera's local frame is the OpenCV optical frame (x right,
y down, z forward); its world is the head frame of ADR-0060 (x forward, y left, z up). Pixel centres sit at
integer coordinates (OpenCV): pixel (row i, column j) is the point (u = j, v = i).

The calibration ring: four boards at known head-frame poses around the wearer (ahead, left, right, behind), each
facing the wearer from RING_DISTANCE_M with its centre at eye height. The wearer stands on a marked spot facing
the ahead board. Each board uses its own ids, so a camera knows which board it sees."""
from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np

from ..features.aruco import DICT_4X4_50, code_bits
from .frames import Intrinsics, mount_to_head, rot_z

MARKER_CELLS = 6                       # border + 4 code bits + border
# Default printed board: 4 x 3 markers of 70 mm with 15 mm gaps, 355 x 270 mm, fits A3 landscape (420 x 297 mm)
# with room for the title line.
DEFAULT_ROWS, DEFAULT_COLS = 3, 4
DEFAULT_MARKER_M = 0.070
DEFAULT_GAP_M = 0.015
# Ring: the board centre distance from the marked spot, and its height above eye level.
RING_DISTANCE_M = 0.6
RING_HEIGHT_M = 0.0
RING_AZ_DEG = {"ahead": 0.0, "left": 90.0, "right": -90.0, "behind": 180.0}
RING_NAMES = ("ahead", "left", "right", "behind")
# Grey levels of the simulated scene.
INK, PAPER, WALL, BACK_OF_BOARD = 20.0, 235.0, 190.0, 120.0
SUPERSAMPLE = 3                        # samples per pixel side: edges are area-averaged, as a real sensor does
JITTER_SEED = 64                       # fixed sub-sample jitter, independent of the noise seed
BLUR_RADIUS_SIGMAS = 3.0               # Gaussian kernel half-width in sigmas
MIN_DEPTH_M = 1e-6                     # a board point must lie this far in front of the camera to be seen
UINT8_MAX = 255.0

# Optical-to-mount rotation: forward = z_opt, left = -x_opt, up = -y_opt (oiris/spatial/frames.py).
OPTICAL_TO_MOUNT = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
# A board facing the wearer from straight ahead: its x is the wearer's right (-y), its y is up (z), its z faces back (-x).
_FACING_AHEAD = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@dataclass(frozen=True)
class Pose:
    """x_world = R @ x_local + t."""
    R: np.ndarray
    t: np.ndarray

    def inverse(self) -> "Pose":
        return Pose(self.R.T, -self.R.T @ self.t)

    def compose(self, other: "Pose") -> "Pose":
        """self after other: x_world = self(other(x))."""
        return Pose(self.R @ other.R, self.R @ other.t + self.t)


def camera_pose_from_mount(yaw_deg: float, pitch_deg: float, roll_deg: float, position_m) -> Pose:
    """Head-frame pose of a camera's optical frame from its mount angles (ADR-0060) and position."""
    return Pose(mount_to_head(yaw_deg, pitch_deg, roll_deg) @ OPTICAL_TO_MOUNT, np.asarray(position_m, dtype=np.float64))


def look_at(eye, target, up=(0.0, 0.0, 1.0), roll_deg: float = 0.0) -> Pose:
    """Pose of a camera optical frame at eye looking at target: z towards the target, y down (against up),
    then turned roll_deg about its own z axis."""
    e, z = np.asarray(eye, dtype=np.float64), np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    z = z / np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, dtype=np.float64))
    x = x / np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z], axis=1)
    a = math.radians(roll_deg)
    c, s = math.cos(a), math.sin(a)
    return Pose(R @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), e)


@dataclass(frozen=True)
class Board:
    """A rows x cols grid of DICT_4X4_50 markers. marker_m is the printed marker side (border included)."""
    rows: int = DEFAULT_ROWS
    cols: int = DEFAULT_COLS
    marker_m: float = DEFAULT_MARKER_M
    gap_m: float = DEFAULT_GAP_M
    first_id: int = 0

    def __post_init__(self) -> None:
        if self.rows < 1 or self.cols < 1:
            raise ValueError("a board needs at least one row and one column")
        if not (0 <= self.first_id and self.first_id + self.rows * self.cols <= len(DICT_4X4_50)):
            raise ValueError(f"board ids must lie in 0..{len(DICT_4X4_50) - 1}")
        if not (self.marker_m > 0 and self.gap_m * MARKER_CELLS >= self.marker_m):
            raise ValueError("the gap between markers must be at least one cell (marker_m / 6), the quiet zone")

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(range(self.first_id, self.first_id + self.rows * self.cols))

    @property
    def pitch_m(self) -> float:
        return self.marker_m + self.gap_m

    @property
    def size_m(self) -> tuple[float, float]:
        """(width, height) including the outer gap."""
        return self.cols * self.pitch_m + self.gap_m, self.rows * self.pitch_m + self.gap_m

    def scaled_to(self, measured_marker_m: float) -> "Board":
        """The same board printed at another scale: the owner measures the marker side, the gap scales with it."""
        k = measured_marker_m / self.marker_m
        return Board(self.rows, self.cols, measured_marker_m, self.gap_m * k, self.first_id)

    def marker_corners(self, marker_id: int) -> np.ndarray:
        """(4, 2) board-frame corners of a marker in detection order: top left, top right, bottom right, bottom left."""
        i = marker_id - self.first_id
        if not 0 <= i < self.rows * self.cols:
            raise KeyError(marker_id)
        r, c = divmod(i, self.cols)
        w, h = self.size_m
        x0 = -w / 2 + self.gap_m + c * self.pitch_m
        y0 = h / 2 - self.gap_m - r * self.pitch_m
        s = self.marker_m
        return np.array([[x0, y0], [x0 + s, y0], [x0 + s, y0 - s], [x0, y0 - s]])

    def texture(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Grey level of board points (board frame, metres); WALL outside the board."""
        w, h = self.size_m
        out = np.full(x.shape, WALL)
        on = (np.abs(x) <= w / 2) & (np.abs(y) <= h / 2)
        out[on] = PAPER
        lx, ly = x + w / 2 - self.gap_m, h / 2 - self.gap_m - y          # from the first marker's top left, y down
        c, r = np.floor(lx / self.pitch_m), np.floor(ly / self.pitch_m)
        fx, fy = lx - c * self.pitch_m, ly - r * self.pitch_m
        inside = on & (c >= 0) & (c < self.cols) & (r >= 0) & (r < self.rows) & (fx < self.marker_m) & (fy < self.marker_m)
        cell = self.marker_m / MARKER_CELLS
        grids = self._grids()
        mi = (r[inside] * self.cols + c[inside]).astype(int)
        ci = np.minimum((fy[inside] / cell).astype(int), MARKER_CELLS - 1)
        cj = np.minimum((fx[inside] / cell).astype(int), MARKER_CELLS - 1)
        out[inside] = np.where(grids[mi, ci, cj] == 1, PAPER, INK)
        return out

    def _grids(self) -> np.ndarray:
        g = np.zeros((self.rows * self.cols, MARKER_CELLS, MARKER_CELLS), np.uint8)
        for k, mid in enumerate(self.ids):
            g[k, 1:5, 1:5] = code_bits(DICT_4X4_50[mid])
        return g


def ring_layout(distance_m: float = RING_DISTANCE_M, height_m: float = RING_HEIGHT_M,
                board: Board | None = None) -> dict[str, tuple[Pose, Board]]:
    """The calibration ring: name -> (head-frame board pose, board). Each board faces the wearer; ids do not overlap."""
    b = board or Board()
    n = b.rows * b.cols
    out = {}
    for k, name in enumerate(RING_NAMES):
        Rz = rot_z(RING_AZ_DEG[name])
        bk = Board(b.rows, b.cols, b.marker_m, b.gap_m, k * n)
        out[name] = (Pose(Rz @ _FACING_AHEAD, Rz @ np.array([distance_m, 0.0, height_m])), bk)
    return out


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(math.ceil(BLUR_RADIUS_SIGMAS * sigma)))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        p = np.pad(img, pad, mode="edge")
        n = img.shape[axis]
        img = sum(k[i] * (p[i:i + n] if axis == 0 else p[:, i:i + n]) for i in range(2 * r + 1))
    return img


def _clip_in_front(poly: np.ndarray) -> np.ndarray:
    """The part of a closed polygon (3, n), camera frame, with depth z >= MIN_DEPTH_M (Sutherland-Hodgman, one plane)."""
    out = []
    n = poly.shape[1]
    for i in range(n):
        a, b = poly[:, i], poly[:, (i + 1) % n]
        a_in, b_in = a[2] >= MIN_DEPTH_M, b[2] >= MIN_DEPTH_M
        if a_in:
            out.append(a)
        if a_in != b_in:
            f = (MIN_DEPTH_M - a[2]) / (b[2] - a[2])
            out.append(a + f * (b - a))
    return np.array(out).T if out else np.zeros((3, 0))


def render(intrinsics: Intrinsics, camera_pose: Pose, boards: list[tuple[Pose, Board]] | tuple[Pose, Board],
           image_size: tuple[int, int] | None = None, noise_sigma: float = 0.0, blur_sigma_px: float = 0.0,
           seed: int = 0) -> np.ndarray:
    """uint8 grayscale image (h, w) of one or more boards through a pinhole camera, by inverse-homography sampling.

    camera_pose and each board pose are in the same world frame. Each pixel averages SUPERSAMPLE^2 samples; then an
    optional Gaussian blur (sigma in pixels) and seeded Gaussian noise (sigma in grey levels) are applied."""
    w, h = image_size if image_size is not None else (intrinsics.width, intrinsics.height)
    K = np.array([[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]])
    s = SUPERSAMPLE
    offs = (np.arange(s) + 0.5) / s - 0.5
    us = (np.arange(w)[:, None] + offs[None, :]).ravel()
    vs = (np.arange(h)[:, None] + offs[None, :]).ravel()
    # Stratified jitter: each sample moves at random within its own sub-cell, so the area estimate has no grid bias
    # (a regular grid snaps every edge to a third of a pixel). A fixed seed keeps renders identical.
    jit = np.random.default_rng(JITTER_SEED)
    ju = jit.uniform(-0.5, 0.5, (vs.size, us.size)) / s
    jv = jit.uniform(-0.5, 0.5, (vs.size, us.size)) / s
    img = np.full((vs.size, us.size), WALL)
    depth = np.full((vs.size, us.size), np.inf)
    if isinstance(boards, tuple):
        boards = [boards]
    for board_pose, board in boards:
        rel = camera_pose.inverse().compose(board_pose)        # board -> camera
        bw, bh = board.size_m
        corners = rel.R[:, :2] @ (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [bw / 2, bh / 2]).T + rel.t[:, None]
        front_part = _clip_in_front(corners)
        if front_part.shape[1] == 0:
            continue                                            # wholly behind the camera
        p = K @ front_part                                      # only sample the visible part's bounding box
        u, v = p[0] / p[2], p[1] / p[2]
        c0, c1 = np.searchsorted(us, [u.min() - 1, u.max() + 1])
        r0, r1 = np.searchsorted(vs, [v.min() - 1, v.max() + 1])
        if r1 <= r0 or c1 <= c0:
            continue
        U, V = np.meshgrid(us[c0:c1], vs[r0:r1])
        U, V = U + ju[r0:r1, c0:c1], V + jv[r0:r1, c0:c1]
        rays = np.linalg.solve(K, np.stack([U.ravel(), V.ravel(), np.ones(U.size)]))   # optical-frame rays, z = 1
        n, d = rel.R[:, 2], float(rel.R[:, 2] @ rel.t)          # board plane: n . X = d
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = d / (n @ rays)                                # ray depth (z) where it meets the plane
        sub_depth = depth[r0:r1, c0:c1].ravel()
        hit = np.isfinite(lam) & (lam > MIN_DEPTH_M) & (lam < sub_depth)
        local = rel.R.T @ (rays[:, hit] * lam[hit] - rel.t[:, None])
        on = (np.abs(local[0]) <= bw / 2) & (np.abs(local[1]) <= bh / 2)
        idx = np.nonzero(hit)[0][on]
        sub_img = img[r0:r1, c0:c1].ravel()
        front = d < 0                                           # the camera is on the printed side (z out of the face)
        sub_img[idx] = board.texture(local[0, on], local[1, on]) if front else BACK_OF_BOARD
        sub_depth[idx] = lam[idx]
        img[r0:r1, c0:c1] = sub_img.reshape(r1 - r0, c1 - c0)
        depth[r0:r1, c0:c1] = sub_depth.reshape(r1 - r0, c1 - c0)
    out = img.reshape(h, s, w, s).mean(axis=(1, 3))
    if blur_sigma_px > 0:
        out = _blur(out, blur_sigma_px)
    if noise_sigma > 0:
        out = out + np.random.default_rng(seed).normal(0.0, noise_sigma, out.shape)
    return np.clip(np.round(out), 0, UINT8_MAX).astype(np.uint8)


__all__ = ["Board", "Pose", "ring_layout", "render", "camera_pose_from_mount", "look_at", "OPTICAL_TO_MOUNT", "RING_NAMES",
           "RING_AZ_DEG", "RING_DISTANCE_M", "RING_HEIGHT_M", "DEFAULT_MARKER_M", "DEFAULT_GAP_M", "MARKER_CELLS"]
