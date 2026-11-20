"""Camera intrinsics and mount-pose calibration from ArUco boards (ADR-0064). Pure NumPy.

Intrinsics: Zhang's method. Each view of a planar board (oiris/spatial/calib_sim.Board) gives a homography from
the detected, sub-pixel refined marker corners; the closed form solves fx, fy, cx, cy with zero skew, and
Levenberg-Marquardt refines them with every view's pose to minimise the reprojection error. No lens distortion
yet: the rig config has no keys for it (ADR-0064, the first fine-tuning item for wide-angle cameras).

Mount pose: the wearer stands on the marked spot of the calibration ring (calib_sim.ring_layout). In each frame a
camera sees one ring board; its planar pose (homography, then refined) composed with that board's known head-frame
pose gives the camera's optical frame in the head frame. The optical frame (x right, y down, z forward) turns into
the mount frame (x forward, y left, z up), and the rotation splits into yaw, pitch and roll by ADR-0060's
R = Rz(yaw) Ry(-pitch) Rx(roll). Frames are averaged (chordal mean of rotations, mean of positions) after dropping
outlier frames, and the result is a mount dict that oiris.config.validate_rig accepts.

Every result either carries numbers and their quality, or a refusal reason code; nothing below a named limit
is output. Pixel centres are at integer coordinates (OpenCV)."""
from __future__ import annotations
import math
from dataclasses import dataclass, field

import numpy as np

from ..config import IMAGING_KINDS, validate_rig
from ..features import aruco
from .calib_sim import MARKER_CELLS, OPTICAL_TO_MOUNT, Board, Pose
from .frames import Intrinsics

# --- quality limits (ADR-0064) ---------------------------------------------------------------------------------
MIN_INTRINSICS_VIEWS = 3              # Zhang's method needs three views of the plane (two fix only 4 of 5 numbers)
MIN_TILT_SPREAD_DEG = 10.0            # the board normals must differ by at least this much between some two views
MAX_INTRINSICS_RMS_PX = 1.0           # above this the pinhole model does not fit (distortion, blur, a bad view)
MIN_MARKERS_PER_VIEW = 4              # a view (or frame) needs at least this many refined markers (16 corners)
MIN_POSE_FRAMES = 3                   # mount pose: at least this many frames must survive outlier rejection
MIN_INLIER_FRACTION = 0.6             # ... and at least this share of the usable frames
MAX_POSE_RMS_PX = 1.0                 # a frame whose board pose reprojects worse than this is dropped
FRAME_MAX_ANGLE_DEV_DEG = 0.5         # a frame whose rotation is further than this from the consensus is an outlier
FRAME_MAX_POSITION_DEV_M = 0.01       # ... or whose position is further than this
MAX_POSE_ANGLE_SPREAD_DEG = 0.25      # the surviving frames' RMS angular deviation must stay below this
MAX_POSE_POSITION_SPREAD_M = 0.005    # ... and their RMS position deviation below this

# Refusal reason codes.
TOO_FEW_VIEWS = "too_few_views"
VIEWS_NEAR_PARALLEL = "views_near_parallel"
RMS_TOO_HIGH = "rms_too_high"
NO_RING_BOARD = "no_ring_board"
TOO_FEW_FRAMES = "too_few_frames"
FRAMES_DISAGREE = "frames_disagree"
INVALID_MOUNT = "invalid_mount"
REASONS = (TOO_FEW_VIEWS, VIEWS_NEAR_PARALLEL, RMS_TOO_HIGH, NO_RING_BOARD, TOO_FEW_FRAMES, FRAMES_DISAGREE, INVALID_MOUNT)

# --- corner refinement -----------------------------------------------------------------------------------------
REFINE_PASSES = 2
EDGE_T_RANGE = (0.15, 0.85)           # sample each marker side away from its corners
EDGE_SAMPLE_STEP_PX = 1.5             # one edge profile per this many pixels of side
MIN_EDGE_SAMPLES = 4
PROFILE_STEP_PX = 0.25
EDGE_WINDOW_CELLS = 0.6               # profile half-length: inside the one-cell black border, inside the quiet zone
EDGE_WINDOW_PX = (1.0, 6.0)           # ... clamped to this range
MIN_EDGE_CONTRAST = 30.0              # grey levels between ink and paper along a profile
PLATEAU_FRACTION = 0.2                # the outer fifth of each profile end gives the ink and paper levels
EDGE_OUTLIER_PX = 0.5                 # edge points further than this from the first line fit are dropped
MAX_CORNER_SHIFT_PX = 3.0             # refinement may move a detected corner at most this far
MIN_SIDE_LENGTH_PX = 1e-9
BILINEAR_EDGE_PX = 1e-6               # keep interpolation inside the last pixel column and row
MIN_LINE_SINE = 1e-6                  # two marker sides closer to parallel than this cannot meet in a corner

# --- solver ----------------------------------------------------------------------------------------------------
LM_MAX_ITERS = 100
LM_LAMBDA0 = 1e-3
LM_LAMBDA_UP, LM_LAMBDA_DOWN = 10.0, 10.0
LM_LAMBDA_MAX = 1e10
LM_STEP_TOL = 1e-10                   # stop when the relative step is below this
JAC_REL_STEP = 1e-6                   # numerical Jacobian step, relative to each parameter
SMALL_ANGLE_RAD = 1e-12
NEAR_HALF_TURN_COS = -0.9             # cos(angle) below this: take the rotation axis from the symmetric part
GIMBAL_COS_MIN = 1e-9                 # |pitch| this close to 90 degrees: roll folds into yaw
HALF_TURN_DEG = 180.0
FULL_TURN_DEG = 360.0
ANGLE_DECIMALS = 3                    # output rounding: 0.001 degree, 0.1 mm, 0.001 px
POSITION_DECIMALS = 4
PIXEL_DECIMALS = 3


# =============================================================================================================
# Observations
# =============================================================================================================

@dataclass(frozen=True)
class BoardView:
    """Matched board-frame points (n, 2) metres and image points (n, 2) pixels from one image of one board."""
    obj: np.ndarray
    img: np.ndarray
    ids: tuple[int, ...]


def _bilinear(g: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h, w = g.shape
    x = np.clip(pts[..., 0], 0, w - 1 - BILINEAR_EDGE_PX)
    y = np.clip(pts[..., 1], 0, h - 1 - BILINEAR_EDGE_PX)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    return ((1 - fy) * ((1 - fx) * g[y0, x0] + fx * g[y0, x0 + 1]) + fy * ((1 - fx) * g[y0 + 1, x0] + fx * g[y0 + 1, x0 + 1]))


def _fit_line(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total least squares: (point on the line, unit direction)."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c)
    return c, vt[0]


def _edge_line(g: np.ndarray, a: np.ndarray, b: np.ndarray, centre: np.ndarray, cell_px: float):
    """Sub-pixel line along the outer edge of a marker side a->b (ink inside, paper outside), or None."""
    d = b - a
    length = float(np.linalg.norm(d))
    if length < MIN_SIDE_LENGTH_PX:
        return None
    d = d / length
    n = np.array([d[1], -d[0]])
    if n @ ((a + b) / 2 - centre) < 0:
        n = -n                                                     # outward normal
    half = float(np.clip(EDGE_WINDOW_CELLS * cell_px, *EDGE_WINDOW_PX))
    ss = np.arange(-half, half + PROFILE_STEP_PX / 2, PROFILE_STEP_PX)
    ts = np.linspace(*EDGE_T_RANGE, max(MIN_EDGE_SAMPLES, int(length / EDGE_SAMPLE_STEP_PX)))
    base = a[None, :] + ts[:, None] * (b - a)[None, :]
    prof = _bilinear(g, base[:, None, :] + ss[None, :, None] * n[None, None, :])
    # Area estimate: a step from ink (lo) to paper (hi) at s = e leaves (s_end - e) of normalised paper in the window,
    # whatever the blur or pixel phase, so e = s_end - integral((p - lo) / (hi - lo)).
    m = max(1, int(round(PLATEAU_FRACTION * len(ss))))
    lo, hi = prof[:, :m].mean(axis=1), prof[:, -m:].mean(axis=1)
    pts = []
    for k in range(len(ts)):
        if hi[k] - lo[k] < MIN_EDGE_CONTRAST:
            continue
        frac = (prof[k] - lo[k]) / (hi[k] - lo[k])
        area = float(np.sum((frac[1:] + frac[:-1]) / 2) * PROFILE_STEP_PX)
        pts.append(base[k] + (ss[-1] - area) * n)
    if len(pts) < MIN_EDGE_SAMPLES:
        return None
    P = np.array(pts)
    c, u = _fit_line(P)
    resid = np.abs((P - c) @ np.array([-u[1], u[0]]))
    keep = resid <= EDGE_OUTLIER_PX
    if keep.sum() < MIN_EDGE_SAMPLES:
        return None
    return _fit_line(P[keep])


def _intersect(l1, l2) -> np.ndarray | None:
    (p, u), (q, v) = l1, l2
    den = u[0] * v[1] - u[1] * v[0]
    if abs(den) < MIN_LINE_SINE:
        return None
    s = ((q[0] - p[0]) * v[1] - (q[1] - p[1]) * v[0]) / den
    return p + s * u


def refine_corners(gray: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
    """Sub-pixel marker corners: fit a line to each outer edge (mid-level crossings of profiles across it) and
    intersect neighbouring lines. Corners keep the detection order. None when an edge cannot be located."""
    g = np.asarray(gray, dtype=np.float64)
    q = np.asarray(corners, dtype=np.float64)
    start = q.copy()
    for _ in range(REFINE_PASSES):
        centre = q.mean(axis=0)
        cell = float(np.mean([np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)])) / MARKER_CELLS
        lines = [_edge_line(g, q[i], q[(i + 1) % 4], centre, cell) for i in range(4)]
        if any(ln is None for ln in lines):
            return None
        new = [_intersect(lines[(i - 1) % 4], lines[i]) for i in range(4)]
        if any(c is None for c in new):
            return None
        q = np.array(new)
        if np.max(np.linalg.norm(q - start, axis=1)) > MAX_CORNER_SHIFT_PX:
            return None
    return q


def observe_board(gray: np.ndarray, board: Board) -> BoardView | None:
    """Refined corners of every marker of this board found in the image, or None when there are too few."""
    obj, img, ids = [], [], []
    for m in aruco.detect(gray):
        if m.id not in board.ids:
            continue
        q = refine_corners(gray, m.corners)
        if q is None:
            continue
        obj.append(board.marker_corners(m.id))
        img.append(q)
        ids.append(m.id)
    if len(ids) < MIN_MARKERS_PER_VIEW:
        return None
    return BoardView(np.concatenate(obj), np.concatenate(img), tuple(ids))


# =============================================================================================================
# Geometry helpers
# =============================================================================================================

def _normaliser(p: np.ndarray) -> np.ndarray:
    c = p.mean(axis=0)
    s = math.sqrt(2) / max(float(np.mean(np.linalg.norm(p - c, axis=1))), MIN_SIDE_LENGTH_PX)
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Normalised DLT (Hartley): 3x3 H mapping src (n, 2) to dst (n, 2), n >= 4, H[2, 2] = 1."""
    Ts, Td = _normaliser(src), _normaliser(dst)
    s = np.c_[src, np.ones(len(src))] @ Ts.T
    d = np.c_[dst, np.ones(len(dst))] @ Td.T
    A = np.zeros((2 * len(s), 9))
    A[0::2, 0:3] = -s
    A[0::2, 6:9] = d[:, 0:1] * s
    A[1::2, 3:6] = -s
    A[1::2, 6:9] = d[:, 1:2] * s
    _, _, vt = np.linalg.svd(A)
    H = np.linalg.inv(Td) @ vt[-1].reshape(3, 3) @ Ts
    return H / H[2, 2]


def rodrigues(r: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(r))
    if th < SMALL_ANGLE_RAD:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


def rotation_vector(R: np.ndarray) -> np.ndarray:
    """Inverse of rodrigues (axis times angle), accurate near 0 and 180 degrees."""
    c = float(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    s = float(np.linalg.norm(w))
    th = math.atan2(s, c)
    if c > NEAR_HALF_TURN_COS:
        return w * (th / s) if s > SMALL_ANGLE_RAD else w
    B = (R + np.eye(3)) / 2                                       # near a half turn: axis from the symmetric part
    k = B[:, int(np.argmax(np.diag(B)))]
    k = k / np.linalg.norm(k)
    if k @ w < 0:
        k = -k
    return k * th


def nearest_rotation(M: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(u @ vt)) or 1.0)])
    return u @ D @ vt


def rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    return math.degrees(float(np.linalg.norm(rotation_vector(R1.T @ R2))))


def _kmat(k: np.ndarray) -> np.ndarray:
    return np.array([[k[0], 0.0, k[2]], [0.0, k[1], k[3]], [0.0, 0.0, 1.0]])


def _project(k: np.ndarray, rvec: np.ndarray, t: np.ndarray, obj: np.ndarray) -> np.ndarray:
    X = rodrigues(rvec)[:, :2] @ obj.T + t[:, None]
    return np.stack([k[0] * X[0] / X[2] + k[2], k[1] * X[1] / X[2] + k[3]], axis=1)


def _lm(f, x0: np.ndarray) -> np.ndarray:
    """Levenberg-Marquardt on residual vector f(x), with a forward-difference Jacobian. Deterministic."""
    x = np.asarray(x0, dtype=np.float64).copy()
    r = f(x)
    cost = float(r @ r)
    lam = LM_LAMBDA0
    for _ in range(LM_MAX_ITERS):
        J = np.empty((r.size, x.size))
        for i in range(x.size):
            h = JAC_REL_STEP * max(1.0, abs(x[i]))
            xp = x.copy()
            xp[i] += h
            J[:, i] = (f(xp) - r) / h
        A, g = J.T @ J, J.T @ r
        while True:
            dx = np.linalg.solve(A + lam * np.diag(np.maximum(np.diag(A), SMALL_ANGLE_RAD)), -g)
            rn = f(x + dx)
            cn = float(rn @ rn)
            if cn < cost:
                x, r, cost = x + dx, rn, cn
                lam = lam / LM_LAMBDA_DOWN
                break
            lam *= LM_LAMBDA_UP
            if lam > LM_LAMBDA_MAX:
                return x
        if float(np.linalg.norm(dx)) <= LM_STEP_TOL * (1.0 + float(np.linalg.norm(x))):
            break
    return x


def planar_pose(H: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Board-to-camera (R, t) from a board->image homography and K; the board is in front of the camera."""
    M = np.linalg.solve(K, H)
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    return nearest_rotation(np.stack([r1, r2, np.cross(r1, r2)], axis=1)), t


def _rms(res: np.ndarray) -> float:
    return float(np.sqrt(np.mean(res.reshape(-1, 2) ** 2) * 2))


# =============================================================================================================
# Intrinsics (Zhang)
# =============================================================================================================

@dataclass
class IntrinsicsResult:
    ok: bool
    reason: str | None = None
    detail: str = ""
    intrinsics: Intrinsics | None = None
    rms_px: float | None = None
    views_used: int = 0
    tilt_spread_deg: float | None = None
    view_poses: list = field(default_factory=list)       # (R, t) board-to-camera for each view used

    def as_config(self) -> dict | None:
        """The rig config "intrinsics" block, or None when refused."""
        if self.intrinsics is None:
            return None
        k = self.intrinsics
        return {"fx": round(k.fx, PIXEL_DECIMALS), "fy": round(k.fy, PIXEL_DECIMALS), "cx": round(k.cx, PIXEL_DECIMALS),
                "cy": round(k.cy, PIXEL_DECIMALS)}


def _v(H: np.ndarray, i: int, j: int) -> np.ndarray:
    a, b = H[:, i], H[:, j]
    return np.array([a[0] * b[0], a[1] * b[1], a[0] * b[2] + a[2] * b[0], a[1] * b[2] + a[2] * b[1], a[2] * b[2]])


def zhang_closed_form(Hs: list[np.ndarray], image_size: tuple[int, int]) -> np.ndarray | None:
    """[fx, fy, cx, cy] (zero skew) from board->image homographies, or None when the views do not fix them."""
    w, h = image_size
    s = 2.0 / (w + h)
    N = np.array([[s, 0, -s * w / 2], [0, s, -s * h / 2], [0, 0, 1]])        # pixels -> about unit scale
    rows = []
    for H in Hs:
        Hn = N @ H
        Hn = Hn / np.linalg.norm(Hn)
        rows += [_v(Hn, 0, 1), _v(Hn, 0, 0) - _v(Hn, 1, 1)]
    _, _, vt = np.linalg.svd(np.array(rows))
    b = vt[-1]
    if b[0] < 0:
        b = -b
    B11, B22, B13, B23, B33 = b
    if B11 <= 0 or B22 <= 0:
        return None
    cx, cy = -B13 / B11, -B23 / B22
    lam = B33 - (B13 ** 2 / B11 + B23 ** 2 / B22)
    if lam <= 0:
        return None
    Kn = np.array([[math.sqrt(lam / B11), 0, cx], [0, math.sqrt(lam / B22), cy], [0, 0, 1]])
    K = np.linalg.solve(N, Kn)
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])


def _tilt_spread_deg(poses) -> float:
    normals = [R[:, 2] for R, _ in poses]
    best = 0.0
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            c = float(np.clip(normals[i] @ normals[j], -1.0, 1.0))
            best = max(best, math.degrees(math.acos(c)))
    return best


def intrinsics_from_views(views: list[BoardView], image_size: tuple[int, int]) -> IntrinsicsResult:
    """Zhang's method on matched board/image points, then LM refinement of the reprojection error."""
    views = [v for v in views if v is not None and len(v.ids) >= MIN_MARKERS_PER_VIEW]
    n = len(views)
    if n < MIN_INTRINSICS_VIEWS:
        return IntrinsicsResult(False, TOO_FEW_VIEWS, f"{n} usable views, need {MIN_INTRINSICS_VIEWS}", views_used=n)
    Hs = [homography(v.obj, v.img) for v in views]
    k0 = zhang_closed_form(Hs, image_size)
    if k0 is None:
        return IntrinsicsResult(False, VIEWS_NEAR_PARALLEL, "the views do not fix the focal length: tilt the board more",
                                views_used=n, tilt_spread_deg=0.0)
    poses = [planar_pose(H, _kmat(k0)) for H in Hs]
    spread = _tilt_spread_deg(poses)
    if spread < MIN_TILT_SPREAD_DEG:
        return IntrinsicsResult(False, VIEWS_NEAR_PARALLEL, f"board tilt spread {spread:.1f} deg, need {MIN_TILT_SPREAD_DEG:g}",
                                views_used=n, tilt_spread_deg=spread)

    def residual(x: np.ndarray) -> np.ndarray:
        k = x[:4]
        return np.concatenate([(_project(k, x[4 + 6 * i:7 + 6 * i], x[7 + 6 * i:10 + 6 * i], v.obj) - v.img).ravel()
                               for i, v in enumerate(views)])

    x0 = np.concatenate([k0] + [np.r_[rotation_vector(R), t] for R, t in poses])
    x = _lm(residual, x0)
    rms = _rms(residual(x))
    k = x[:4]
    w, h = image_size
    if not (np.all(np.isfinite(x)) and k[0] > 0 and k[1] > 0):
        return IntrinsicsResult(False, VIEWS_NEAR_PARALLEL, "refinement diverged", views_used=n, tilt_spread_deg=spread)
    final_poses = [(rodrigues(x[4 + 6 * i:7 + 6 * i]), x[7 + 6 * i:10 + 6 * i]) for i in range(n)]
    spread = _tilt_spread_deg(final_poses)
    intr = Intrinsics(float(k[0]), float(k[1]), float(k[2]), float(k[3]), int(w), int(h))
    if spread < MIN_TILT_SPREAD_DEG:
        return IntrinsicsResult(False, VIEWS_NEAR_PARALLEL, f"board tilt spread {spread:.1f} deg, need {MIN_TILT_SPREAD_DEG:g}",
                                views_used=n, tilt_spread_deg=spread)
    if rms > MAX_INTRINSICS_RMS_PX:
        return IntrinsicsResult(False, RMS_TOO_HIGH, f"reprojection RMS {rms:.3f} px, limit {MAX_INTRINSICS_RMS_PX:g}",
                                rms_px=rms, views_used=n, tilt_spread_deg=spread)
    return IntrinsicsResult(True, None, "", intr, rms, n, spread, final_poses)


def calibrate_intrinsics(images: list[np.ndarray], board: Board) -> IntrinsicsResult:
    """Intrinsics from grayscale images of one board held at different tilts (MIN_INTRINSICS_VIEWS or more)."""
    if not images:
        return IntrinsicsResult(False, TOO_FEW_VIEWS, "no images")
    h, w = np.asarray(images[0]).shape[:2]
    views = [observe_board(im, board) for im in images]
    return intrinsics_from_views([v for v in views if v is not None], (w, h))


# =============================================================================================================
# Mount pose
# =============================================================================================================

def refine_board_pose(view: BoardView, intr: Intrinsics) -> tuple[np.ndarray, np.ndarray, float]:
    """Board-to-camera (R, t) from the homography, refined by LM with the intrinsics fixed; and the RMS in pixels."""
    k = np.array([intr.fx, intr.fy, intr.cx, intr.cy])
    R0, t0 = planar_pose(homography(view.obj, view.img), _kmat(k))

    def residual(x: np.ndarray) -> np.ndarray:
        return (_project(k, x[:3], x[3:], view.obj) - view.img).ravel()

    x = _lm(residual, np.r_[rotation_vector(R0), t0])
    return rodrigues(x[:3]), x[3:], _rms(residual(x))


def mount_angles(R_mount: np.ndarray) -> tuple[float, float, float]:
    """(yaw, pitch, roll) degrees with R = Rz(yaw) Ry(-pitch) Rx(roll) (ADR-0060)."""
    sp = float(np.clip(R_mount[2, 0], -1.0, 1.0))
    pitch = math.degrees(math.asin(sp))
    if math.sqrt(max(0.0, 1 - sp * sp)) < GIMBAL_COS_MIN:
        return math.degrees(math.atan2(-R_mount[0, 1], R_mount[1, 1])), pitch, 0.0
    yaw = math.degrees(math.atan2(R_mount[1, 0], R_mount[0, 0]))
    roll = math.degrees(math.atan2(R_mount[2, 1], R_mount[2, 2]))
    return yaw, pitch, roll


@dataclass(frozen=True)
class FramePose:
    """One frame's camera pose in the head frame: mount rotation (mount-to-head), position, which board, RMS."""
    R: np.ndarray
    position: np.ndarray
    board: str
    markers: int
    rms_px: float


def frame_pose(gray: np.ndarray, intr: Intrinsics, ring: dict[str, tuple[Pose, Board]]) -> FramePose | str:
    """The camera's head-frame pose from the ring board it sees best in this frame, or a reason code."""
    best: tuple[str, BoardView] | None = None
    markers = aruco.detect(gray)
    for name, (_, board) in ring.items():
        found = [m for m in markers if m.id in board.ids]
        if len(found) < MIN_MARKERS_PER_VIEW or (best is not None and len(found) <= len(best[1].ids)):
            continue
        obj, img, ids = [], [], []
        for m in found:
            q = refine_corners(gray, m.corners)
            if q is not None:
                obj.append(board.marker_corners(m.id)); img.append(q); ids.append(m.id)
        if len(ids) >= MIN_MARKERS_PER_VIEW and (best is None or len(ids) > len(best[1].ids)):
            best = (name, BoardView(np.concatenate(obj), np.concatenate(img), tuple(ids)))
    if best is None:
        return NO_RING_BOARD
    name, view = best
    R_cb, t_cb, rms = refine_board_pose(view, intr)
    if rms > MAX_POSE_RMS_PX:
        return RMS_TOO_HIGH
    cam = ring[name][0].compose(Pose(R_cb, t_cb).inverse())       # optical frame in the head frame
    return FramePose(cam.R @ OPTICAL_TO_MOUNT.T, cam.t, name, len(view.ids), rms)


@dataclass
class MountResult:
    ok: bool
    reason: str | None = None
    detail: str = ""
    mount: dict | None = None
    quality: dict = field(default_factory=dict)


def _chordal_mean(Rs: list[np.ndarray]) -> np.ndarray:
    return nearest_rotation(np.sum(Rs, axis=0))


def _consensus(frames: list[FramePose]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Mean rotation and position of the frames that agree; start from the medoid so one outlier cannot drag it."""
    n = len(frames)
    cost = [sum(rotation_angle_deg(frames[i].R, frames[j].R) for j in range(n)) for i in range(n)]
    m = int(np.argmin(cost))
    R, p = frames[m].R, frames[m].position
    keep = list(range(n))
    for _ in range(2):
        keep = [i for i in range(n) if rotation_angle_deg(R, frames[i].R) <= FRAME_MAX_ANGLE_DEV_DEG
                and float(np.linalg.norm(frames[i].position - p)) <= FRAME_MAX_POSITION_DEV_M]
        if not keep:
            break
        R = _chordal_mean([frames[i].R for i in keep])
        p = np.mean([frames[i].position for i in keep], axis=0)
    return R, p, keep


def _r(x: float, nd: int) -> float:
    v = round(float(x), nd)
    return 0.0 if v == 0 else v                                   # no "-0.0" in the config


def calibrate_mount(mount_id: str, images: list[np.ndarray], intr: Intrinsics, ring: dict[str, tuple[Pose, Board]],
                    kind: str = "camera") -> MountResult:
    """A rig mount dict (ADR-0060) for one camera from frames taken while the wearer stands on the ring's mark."""
    if kind not in IMAGING_KINDS:
        return MountResult(False, INVALID_MOUNT, f"kind must be one of {list(IMAGING_KINDS)}")
    frames: list[FramePose] = []
    dropped: dict[str, int] = {}
    for im in images:
        fp = frame_pose(im, intr, ring)
        if isinstance(fp, str):
            dropped[fp] = dropped.get(fp, 0) + 1
        else:
            frames.append(fp)
    quality: dict = {"frames_given": len(images), "frames_dropped": dict(sorted(dropped.items()))}
    if len(frames) < MIN_POSE_FRAMES:
        reason = NO_RING_BOARD if not frames and dropped.get(NO_RING_BOARD) else TOO_FEW_FRAMES
        return MountResult(False, reason, f"{len(frames)} usable frames, need {MIN_POSE_FRAMES}", quality=quality)
    R, p, keep = _consensus(frames)
    quality["frames_used"] = len(keep)
    quality["frames_rejected_as_outliers"] = len(frames) - len(keep)
    if len(keep) < MIN_POSE_FRAMES or len(keep) < MIN_INLIER_FRACTION * len(frames):
        return MountResult(False, FRAMES_DISAGREE, f"only {len(keep)} of {len(frames)} frames agree", quality=quality)
    used = [frames[i] for i in keep]
    ang = math.sqrt(float(np.mean([rotation_angle_deg(R, f.R) ** 2 for f in used])))
    pos = math.sqrt(float(np.mean([float(np.sum((f.position - p) ** 2)) for f in used])))
    boards = sorted({f.board for f in used})
    quality.update({"board": boards[0] if len(boards) == 1 else boards, "markers_mean": round(float(np.mean([f.markers for f in used])), 1),
                    "rms_px": round(float(np.mean([f.rms_px for f in used])), PIXEL_DECIMALS),
                    "angle_spread_deg": round(ang, ANGLE_DECIMALS), "position_spread_m": round(pos, POSITION_DECIMALS)})
    if ang > MAX_POSE_ANGLE_SPREAD_DEG or pos > MAX_POSE_POSITION_SPREAD_M:
        return MountResult(False, FRAMES_DISAGREE, f"frames spread {ang:.3f} deg / {pos * 1000:.1f} mm", quality=quality)
    yaw, pitch, roll = mount_angles(R)
    if yaw <= -HALF_TURN_DEG:
        yaw += FULL_TURN_DEG
    mount = {"id": mount_id, "kind": kind, "position_m": [_r(x, POSITION_DECIMALS) for x in p],
             "yaw_deg": _r(yaw, ANGLE_DECIMALS), "pitch_deg": _r(pitch, ANGLE_DECIMALS), "roll_deg": _r(roll, ANGLE_DECIMALS),
             "image_size": [int(intr.width), int(intr.height)],
             "intrinsics": {"fx": _r(intr.fx, PIXEL_DECIMALS), "fy": _r(intr.fy, PIXEL_DECIMALS),
                            "cx": _r(intr.cx, PIXEL_DECIMALS), "cy": _r(intr.cy, PIXEL_DECIMALS)}}
    errs = validate_rig({"mounts": [mount]})
    if errs:
        return MountResult(False, INVALID_MOUNT, "; ".join(errs), quality=quality)
    return MountResult(True, None, "", mount, quality)


__all__ = ["BoardView", "IntrinsicsResult", "MountResult", "FramePose", "refine_corners", "observe_board", "homography",
           "zhang_closed_form", "intrinsics_from_views", "calibrate_intrinsics", "refine_board_pose", "planar_pose",
           "frame_pose", "mount_angles", "calibrate_mount", "rodrigues", "rotation_vector", "rotation_angle_deg", "REASONS",
           "MIN_INTRINSICS_VIEWS", "MIN_TILT_SPREAD_DEG", "MAX_INTRINSICS_RMS_PX", "MIN_MARKERS_PER_VIEW", "MIN_POSE_FRAMES",
           "MAX_POSE_RMS_PX", "FRAME_MAX_ANGLE_DEV_DEG", "FRAME_MAX_POSITION_DEV_M", "MAX_POSE_ANGLE_SPREAD_DEG",
           "MAX_POSE_POSITION_SPREAD_M"]
