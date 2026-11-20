"""Camera intrinsics and mount-pose calibration (ADR-0064), proved on images rendered by oiris/spatial/calib_sim.py.
Synthetic only: no camera, no photo. The accuracies measured on these seeds are noted beside each bound."""
from __future__ import annotations
import math

import numpy as np
import pytest

from oiris.config import validate_rig
from oiris.spatial import calib_sim as cs
from oiris.spatial import calibrate as cal
from oiris.spatial.detections import SpatialDetection
from oiris.spatial.frames import Intrinsics, az_el_to_vec, fov_intrinsics, mount_to_head, rot_x, rot_y, rot_z, vec_to_az_el, wrap_az
from oiris.spatial.rig import Mount, Rig

# Tolerances (the owner's request: about 1 %, 0.5 degree and 1 cm; looser under heavy noise).
INTRINSICS_REL_TOL = 0.01
ANGLE_TOL_DEG = 0.5
POSITION_TOL_M = 0.01
ROUNDTRIP_TOL_PX = 1.0
CORNER_RMS_TOL_PX, CORNER_MAX_TOL_PX = 0.1, 0.3
EXACT_DEG = 1e-9                              # geometry that is exact up to floating point
NOISY_ANGLE_TOL_DEG = 1.0
NOISY_POSITION_TOL_M = 0.02
NOISY_INTRINSICS_REL_TOL = 0.02
LIGHT_NOISE, LIGHT_BLUR = 1.0, 0.5            # grey levels, pixels
HEAVY_NOISE, HEAVY_BLUR = 8.0, 1.0
POSE_FRAMES = 3
BUMP_DEG = 3.0                                # a knocked frame: the camera turned this far
VIEW_DISTANCE_M = 0.45
POINT_RANGE_M = 2.0

SIM_CAMERA = fov_intrinsics(640, 480, 100.0, 75.0)          # the sim rig's camera (oiris/adapters/sim_rig.py)
OFFSET_CAMERA = Intrinsics(270.0, 268.0, 325.5, 236.0, 640, 480)
RING = cs.ring_layout()
RING_BOARDS = list(RING.values())
BOARD = cs.Board()
AT_ORIGIN = cs.Pose(np.eye(3), np.zeros(3))

# (id, yaw, pitch, roll, position): front, left, right, rear, a pitched and a rolled mount.
MOUNTS = [("front", 0.0, 0.0, 0.0, [0.05, 0.0, 0.03]), ("left", 90.0, 0.0, 0.0, [0.0, 0.08, 0.02]),
          ("right", -90.0, 0.0, 0.0, [0.0, -0.08, 0.02]), ("rear", 180.0, 0.0, 0.0, [-0.1, 0.0, 0.05]),
          ("pitched", 0.0, 15.0, 0.0, [0.05, 0.02, 0.04]), ("rolled", 10.0, 5.0, 20.0, [0.05, -0.02, 0.0])]
# Board tilts for intrinsics (degrees about the board's x and y axes) and the camera's roll.
TILTS = [(0, 0, 0), (25, 0, 10), (-25, 0, -15), (0, 25, 30), (0, -25, -5), (20, 20, 0)]
FLAT_TILTS = [(0, 0, 0), (1, 0, 20), (0, -1, -20), (-1, 1, 45), (1, -1, 90)]


def tilted_view(K: Intrinsics, ax: float, ay: float, roll: float, noise=0.0, blur=0.0, seed=0) -> np.ndarray:
    eye = rot_y(ay) @ rot_x(-ax) @ np.array([0.0, 0.0, VIEW_DISTANCE_M])
    cam = cs.look_at(eye, [0.0, 0.0, 0.0], up=(0.0, 1.0, 0.0), roll_deg=roll)
    return cs.render(K, cam, (AT_ORIGIN, BOARD), noise_sigma=noise, blur_sigma_px=blur, seed=seed)


def mount_frames(K: Intrinsics, yaw, pitch, roll, pos, n=POSE_FRAMES, noise=LIGHT_NOISE, blur=LIGHT_BLUR, seed0=0):
    cam = cs.camera_pose_from_mount(yaw, pitch, roll, pos)
    return [cs.render(K, cam, RING_BOARDS, noise_sigma=noise, blur_sigma_px=blur, seed=seed0 + s) for s in range(n)]


def true_mount(mid, yaw, pitch, roll, pos, K: Intrinsics) -> dict:
    return {"id": mid, "kind": "camera", "position_m": pos, "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll,
            "image_size": [K.width, K.height], "intrinsics": {"fx": K.fx, "fy": K.fy, "cx": K.cx, "cy": K.cy}}


def errors(m: dict, yaw, pitch, roll, pos) -> tuple[float, float, float, float]:
    return (abs(wrap_az(m["yaw_deg"] - yaw)), abs(m["pitch_deg"] - pitch), abs(wrap_az(m["roll_deg"] - roll)),
            float(np.linalg.norm(np.array(m["position_m"]) - pos)))


def rel_errors(k: Intrinsics, truth: Intrinsics) -> list[float]:
    return [abs(k.fx / truth.fx - 1), abs(k.fy / truth.fy - 1), abs(k.cx - truth.cx) / truth.width, abs(k.cy - truth.cy) / truth.height]


# --- the simulator itself ---------------------------------------------------------------------------------------

def test_render_is_deterministic_and_the_seed_changes_only_the_noise():
    a = tilted_view(OFFSET_CAMERA, 20, 10, 5, noise=3.0, blur=0.7, seed=4)
    assert np.array_equal(a, tilted_view(OFFSET_CAMERA, 20, 10, 5, noise=3.0, blur=0.7, seed=4))
    assert not np.array_equal(a, tilted_view(OFFSET_CAMERA, 20, 10, 5, noise=3.0, blur=0.7, seed=5))
    assert a.dtype == np.uint8 and a.shape == (480, 640)


def test_refined_corners_land_on_the_true_projections():
    """Sub-pixel corners against the exact projection of each board corner (measured RMS about 0.035 px)."""
    cam = cs.look_at([0.1, 0.05, VIEW_DISTANCE_M], [0.0, 0.0, 0.0], up=(0.0, 1.0, 0.0), roll_deg=12.0)
    view = cal.observe_board(cs.render(OFFSET_CAMERA, cam, (AT_ORIGIN, BOARD)), BOARD)
    assert view is not None and view.ids == BOARD.ids
    rel = cam.inverse().compose(AT_ORIGIN)
    X = rel.R[:, :2] @ view.obj.T + rel.t[:, None]
    k = OFFSET_CAMERA
    px = np.stack([k.fx * X[0] / X[2] + k.cx, k.fy * X[1] / X[2] + k.cy], axis=1)
    err = np.linalg.norm(view.img - px, axis=1)
    assert float(np.sqrt(np.mean(err ** 2))) < CORNER_RMS_TOL_PX and err.max() < CORNER_MAX_TOL_PX


def test_ring_boards_face_the_wearer_with_distinct_ids():
    ids = [set(b.ids) for _, b in RING_BOARDS]
    assert all(not (a & b) for i, a in enumerate(ids) for b in ids[i + 1:])
    for name, (pose, _) in RING.items():
        az, el = vec_to_az_el(pose.t)
        assert abs(wrap_az(az - cs.RING_AZ_DEG[name])) < EXACT_DEG and abs(el) < EXACT_DEG
        assert float(pose.R[:, 2] @ pose.t) < 0                     # the printed face points back at the mark


# --- intrinsics ---------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def offset_views():
    return [tilted_view(OFFSET_CAMERA, ax, ay, roll, noise=LIGHT_NOISE, blur=LIGHT_BLUR, seed=i)
            for i, (ax, ay, roll) in enumerate(TILTS)]


def test_intrinsics_recovered_within_one_percent(offset_views):
    """Six tilted views (measured: fx, fy within 0.02 %, cx, cy within 0.05 px, RMS about 0.05 px)."""
    r = cal.calibrate_intrinsics(offset_views, BOARD)
    assert r.ok, (r.reason, r.detail)
    assert r.views_used == len(TILTS) and r.rms_px is not None and r.rms_px < cal.MAX_INTRINSICS_RMS_PX
    assert r.tilt_spread_deg is not None and r.tilt_spread_deg >= cal.MIN_TILT_SPREAD_DEG
    assert max(rel_errors(r.intrinsics, OFFSET_CAMERA)) < INTRINSICS_REL_TOL
    assert set(r.as_config()) == {"fx", "fy", "cx", "cy"}


def test_intrinsics_refuse_too_few_views(offset_views):
    r = cal.calibrate_intrinsics(offset_views[:cal.MIN_INTRINSICS_VIEWS - 1], BOARD)
    assert not r.ok and r.reason == cal.TOO_FEW_VIEWS and r.intrinsics is None and r.as_config() is None


def test_intrinsics_refuse_near_parallel_views():
    """Five views that differ in distance and roll but all face the camera within a degree."""
    views = [tilted_view(OFFSET_CAMERA, ax, ay, roll, seed=i) for i, (ax, ay, roll) in enumerate(FLAT_TILTS)]
    r = cal.calibrate_intrinsics(views, BOARD)
    assert not r.ok and r.reason == cal.VIEWS_NEAR_PARALLEL and r.intrinsics is None


def test_intrinsics_refuse_high_rms():
    """Points that no pinhole fits (3 px of scatter) are refused, not averaged."""
    rng = np.random.default_rng(3)
    k = np.array([OFFSET_CAMERA.fx, OFFSET_CAMERA.fy, OFFSET_CAMERA.cx, OFFSET_CAMERA.cy])
    obj = np.concatenate([BOARD.marker_corners(i) for i in BOARD.ids])
    views = []
    for ax, ay, roll in TILTS:
        R = rot_y(ay) @ rot_x(ax) @ rot_z(roll)
        t = np.array([0.0, 0.0, VIEW_DISTANCE_M])
        img = cal._project(k, cal.rotation_vector(R), t, obj) + rng.normal(0.0, 3.0, (len(obj), 2))
        views.append(cal.BoardView(obj, img, BOARD.ids))
    r = cal.intrinsics_from_views(views, (OFFSET_CAMERA.width, OFFSET_CAMERA.height))
    assert not r.ok and r.reason == cal.RMS_TOO_HIGH and r.rms_px > cal.MAX_INTRINSICS_RMS_PX


def test_intrinsics_with_heavy_noise_stay_within_the_looser_bound():
    """Noise 8 grey levels and blur 1 px (measured: within about 0.1 %)."""
    views = [tilted_view(OFFSET_CAMERA, ax, ay, roll, noise=HEAVY_NOISE, blur=HEAVY_BLUR, seed=100 + i)
             for i, (ax, ay, roll) in enumerate(TILTS)]
    r = cal.calibrate_intrinsics(views, BOARD)
    assert r.ok, (r.reason, r.detail)
    assert max(rel_errors(r.intrinsics, OFFSET_CAMERA)) < NOISY_INTRINSICS_REL_TOL


# --- mount pose -----------------------------------------------------------------------------------------------------

def _calibrated(mid, yaw, pitch, roll, pos, K=SIM_CAMERA, **kw):
    r = cal.calibrate_mount(mid, mount_frames(K, yaw, pitch, roll, pos, **kw), K, RING)
    assert r.ok, (mid, r.reason, r.detail, r.quality)
    return r


@pytest.mark.parametrize("mid,yaw,pitch,roll,pos", MOUNTS, ids=[m[0] for m in MOUNTS])
def test_mount_pose_recovered_and_round_trips_through_the_rig(mid, yaw, pitch, roll, pos):
    """Angles within 0.5 degree and position within 1 cm (measured: within 0.05 degree and 0.5 mm), and a head-frame
    point lands on the same pixel through the calibrated mount as through the true one (oiris/spatial/rig.py)."""
    r = _calibrated(mid, yaw, pitch, roll, pos)
    m = r.mount
    assert validate_rig({"mounts": [m]}) == []
    ey, ep, er, epos = errors(m, yaw, pitch, roll, pos)
    assert max(ey, ep, er) < ANGLE_TOL_DEG and epos < POSITION_TOL_M, (ey, ep, er, epos)
    assert r.quality["frames_used"] == POSE_FRAMES and r.quality["rms_px"] < cal.MAX_POSE_RMS_PX
    got = Mount.from_dict(m)
    want = Mount.from_dict(true_mount(mid, yaw, pitch, roll, pos, SIM_CAMERA))
    rig = Rig.from_config({"mounts": [m]})
    hits = 0
    az0, el0 = vec_to_az_el(want.mount_vec_to_head([1.0, 0.0, 0.0]))     # where the true mount points
    for daz in (-30.0, -10.0, 0.0, 15.0, 35.0):
        for de in (-20.0, 0.0, 25.0):
            point = POINT_RANGE_M * az_el_to_vec(az0 + daz, el0 + de)      # a head-frame point 2 m away
            az, el = vec_to_az_el(point)
            a, b = want.head_to_pixel(az, el), got.head_to_pixel(az, el)
            if a is None or b is None:
                continue
            hits += 1
            assert math.dist(a, b) < ROUNDTRIP_TOL_PX, (daz, de, a, b)
            back = rig.to_head(SpatialDetection("glint", "optical", mid, 0, 0.5, pixel=a))
            assert abs(wrap_az(back[0] - az)) < ANGLE_TOL_DEG and abs(back[1] - el) < ANGLE_TOL_DEG
    assert hits >= 10


def test_mount_pose_with_calibrated_intrinsics_end_to_end():
    """Intrinsics from tilted views, then the mount pose with those intrinsics: the whole procedure."""
    views = [tilted_view(SIM_CAMERA, ax, ay, roll, noise=LIGHT_NOISE, blur=LIGHT_BLUR, seed=200 + i)
             for i, (ax, ay, roll) in enumerate(TILTS)]
    ki = cal.calibrate_intrinsics(views, BOARD)
    assert ki.ok and max(rel_errors(ki.intrinsics, SIM_CAMERA)) < INTRINSICS_REL_TOL
    mid, yaw, pitch, roll, pos = MOUNTS[4]
    r = cal.calibrate_mount(mid, mount_frames(SIM_CAMERA, yaw, pitch, roll, pos), ki.intrinsics, RING)
    assert r.ok, (r.reason, r.detail)
    ey, ep, er, epos = errors(r.mount, yaw, pitch, roll, pos)
    assert max(ey, ep, er) < ANGLE_TOL_DEG and epos < POSITION_TOL_M


@pytest.mark.parametrize("mid,yaw,pitch,roll,pos", [MOUNTS[1], MOUNTS[5]], ids=["left", "rolled"])
def test_mount_pose_with_heavy_noise_stays_within_the_looser_bound(mid, yaw, pitch, roll, pos):
    """Noise 8 grey levels, blur 1 px, four frames (measured: within 0.1 degree and 1 mm)."""
    r = _calibrated(mid, yaw, pitch, roll, pos, n=4, noise=HEAVY_NOISE, blur=HEAVY_BLUR, seed0=300)
    ey, ep, er, epos = errors(r.mount, yaw, pitch, roll, pos)
    assert max(ey, ep, er) < NOISY_ANGLE_TOL_DEG and epos < NOISY_POSITION_TOL_M, (ey, ep, er, epos)


def test_a_knocked_frame_is_rejected_as_an_outlier():
    mid, yaw, pitch, roll, pos = MOUNTS[0]
    good = mount_frames(SIM_CAMERA, yaw, pitch, roll, pos, n=4)
    knocked = mount_frames(SIM_CAMERA, yaw + BUMP_DEG, pitch, roll, pos, n=1, seed0=50)
    r = cal.calibrate_mount(mid, good[:2] + knocked + good[2:], SIM_CAMERA, RING)
    assert r.ok, (r.reason, r.detail)
    assert r.quality["frames_rejected_as_outliers"] == 1 and r.quality["frames_used"] == 4
    ey, ep, er, epos = errors(r.mount, yaw, pitch, roll, pos)
    assert max(ey, ep, er) < ANGLE_TOL_DEG and epos < POSITION_TOL_M


def test_frames_that_disagree_are_refused():
    mid, yaw, pitch, roll, pos = MOUNTS[0]
    frames = mount_frames(SIM_CAMERA, yaw, pitch, roll, pos, n=2) + mount_frames(SIM_CAMERA, yaw - BUMP_DEG, pitch, roll, pos, n=2)
    r = cal.calibrate_mount(mid, frames, SIM_CAMERA, RING)
    assert not r.ok and r.reason == cal.FRAMES_DISAGREE and r.mount is None


def test_too_few_frames_and_no_board_are_refused():
    mid, yaw, pitch, roll, pos = MOUNTS[0]
    r = cal.calibrate_mount(mid, mount_frames(SIM_CAMERA, yaw, pitch, roll, pos, n=cal.MIN_POSE_FRAMES - 1), SIM_CAMERA, RING)
    assert not r.ok and r.reason == cal.TOO_FEW_FRAMES
    sky = mount_frames(SIM_CAMERA, 0.0, 80.0, 0.0, pos, n=POSE_FRAMES)     # looking up: no ring board in view
    r = cal.calibrate_mount(mid, sky, SIM_CAMERA, RING)
    assert not r.ok and r.reason == cal.NO_RING_BOARD and r.quality["frames_dropped"] == {cal.NO_RING_BOARD: POSE_FRAMES}


def test_a_mount_the_rig_would_refuse_is_not_output():
    """A camera 1.6 m from the mark is no head-worn mount: validate_rig refuses it, so calibration refuses too."""
    far_ring = {k: (cs.Pose(p.R, p.t * 4.0), b) for k, (p, b) in RING.items()}
    cam = cs.camera_pose_from_mount(0.0, 0.0, 0.0, [0.0, 0.0, 0.0])
    near = [cs.render(SIM_CAMERA, cam, RING_BOARDS, noise_sigma=LIGHT_NOISE, seed=s) for s in range(POSE_FRAMES)]
    r = cal.calibrate_mount("front", near, SIM_CAMERA, far_ring)        # the ring is told it stands 2.4 m away
    assert not r.ok and r.reason == cal.INVALID_MOUNT and r.mount is None
    assert not cal.calibrate_mount("front", near, SIM_CAMERA, RING, kind="mic_array").ok


def test_calibration_is_repeatable():
    mid, yaw, pitch, roll, pos = MOUNTS[5]
    a = cal.calibrate_mount(mid, mount_frames(SIM_CAMERA, yaw, pitch, roll, pos), SIM_CAMERA, RING)
    b = cal.calibrate_mount(mid, mount_frames(SIM_CAMERA, yaw, pitch, roll, pos), SIM_CAMERA, RING)
    assert a.ok and a.mount == b.mount and a.quality == b.quality


def test_mount_angles_invert_the_adr_0060_rotation():
    for y, p, r in [(0, 0, 0), (90, 10, -5), (-90, -30, 40), (180, 5, 0), (-135, 60, -170), (45, -89, 10)]:
        got = cal.mount_angles(mount_to_head(y, p, r))
        assert np.allclose(mount_to_head(*got), mount_to_head(y, p, r), atol=1e-9)
    for th in (0.0, 1e-8, 0.5, 2.0, math.pi - 1e-6, math.pi):
        v = th * np.array([0.6, -0.48, 0.64])
        assert np.allclose(cal.rodrigues(cal.rotation_vector(cal.rodrigues(v))), cal.rodrigues(v), atol=1e-9)
