"""Spatial sensing (ADR-0060): frames, direction words, the rig ports, detections and merging. Synthetic only."""
from __future__ import annotations
import itertools, math

import numpy as np
import pytest

from oiris.config import DEFAULT_RIG, Config, ConfigError, validate_rig
from oiris.guard.phrases import SLOT_VALUES
from oiris.spatial import Rig, SpatialDetection, locate, merge
from oiris.spatial.detections import emitter_detections, glint_detections, hot_region_detections
from oiris.spatial.frames import (DIRECTION_WORDS, DirectionTracker, angle_between, az_el_to_vec, direction_word, fov_intrinsics,
                                  mount_ray_to_pixel, mount_to_head, pixel_to_mount_ray, vec_to_az_el, wrap_az)
from oiris.spatial.merge import EventTracker, Located, noisy_or

TOL = 1e-9
W, H = 640, 480


def cam(mid: str, yaw=0.0, pitch=0.0, roll=0.0, **kw) -> dict:
    return {"id": mid, "kind": "camera", "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll, "hfov_deg": 90.0,
            "vfov_deg": 70.0, "image_size": [W, H], **kw}


def det(mount="front", **kw) -> SpatialDetection:
    base = {"kind": "glint", "family": "optical", "mount_id": mount, "t_mono_ns": 0, "confidence": 0.5}
    return SpatialDetection(**{**base, **kw})


def loc(az, el=0.0, conf=0.5, fam="optical", mount="front", t_s=0.0, kind="glint") -> Located:
    return Located(az, el, kind, fam, mount, int(t_s * 1e9), conf)


# ---------- frames ----------

def test_words_are_the_phrase_slot_values():
    assert set(DIRECTION_WORDS) == SLOT_VALUES["direction"]


@pytest.mark.parametrize("az,el", [(a, e) for a in (-179.0, -135.0, -90.0, -45.0, 0.0, 30.0, 90.0, 135.0, 180.0)
                                   for e in (-89.0, -45.0, 0.0, 20.0, 89.0)])
def test_az_el_round_trip(az, el):
    a, e = vec_to_az_el(az_el_to_vec(az, el))
    assert abs(wrap_az(a - az)) < 1e-9 and abs(e - el) < 1e-9


def test_azimuth_range_is_half_open():
    assert vec_to_az_el([-1.0, -0.0, 0.0])[0] == 180.0
    assert vec_to_az_el([-1.0, 1e-300, 0.0])[0] == 180.0
    assert wrap_az(-180.0) == 180.0 and wrap_az(540.0) == 180.0 and wrap_az(-190.0) == 170.0
    assert vec_to_az_el([0, 1, 0]) == (90.0, 0.0) and vec_to_az_el([0, -1, 0]) == (-90.0, 0.0)
    assert vec_to_az_el([0, 0, 1])[1] == 90.0


@pytest.mark.parametrize("az,word", [(0.0, "ahead"), (45.0, "ahead"), (-45.0, "ahead"), (45.0001, "at left"), (135.0, "at left"),
                                     (135.0001, "behind"), (180.0, "behind"), (-180.0, "behind"), (-135.0, "at right"),
                                     (-45.0001, "at right"), (-135.0001, "behind"), (90.0, "at left"), (-90.0, "at right")])
def test_sector_boundaries(az, word):
    assert direction_word(az, 0.0) == word


def test_vertical_words_override():
    assert direction_word(180.0, 45.0) == "behind" and direction_word(180.0, 45.0001) == "above"
    assert direction_word(0.0, -45.0) == "ahead" and direction_word(0.0, -45.0001) == "below"


def test_hysteresis_holds_ten_degrees_past_a_boundary():
    t = DirectionTracker()
    assert t.update(40.0) == "ahead"
    assert t.update(50.0) == "ahead" and t.update(55.0) == "ahead"
    assert t.update(55.0001) == "at left"
    assert t.update(40.0) == "at left" and t.update(35.0001) == "at left"
    assert t.update(35.0) == "ahead"
    b = DirectionTracker()
    assert b.update(140.0) == "behind" and b.update(126.0) == "behind" and b.update(-126.0) == "behind"
    assert b.update(125.0) == "at left"
    v = DirectionTracker()
    assert v.update(0.0, 50.0) == "above" and v.update(0.0, 36.0) == "above" and v.update(0.0, 35.0) == "ahead"
    assert v.update(0.0, 54.0) == "ahead" and v.update(0.0, 55.0001) == "above"
    assert DirectionTracker().update(0.0, 50.0) == "above"   # a new track takes the plain word


def test_rotation_matrices_are_proper():
    for y, p, r in itertools.product((-170.0, 0.0, 33.0), (-80.0, 0.0, 10.0), (-120.0, 0.0, 45.0)):
        R = mount_to_head(y, p, r)
        assert np.allclose(R @ R.T, np.eye(3)) and abs(np.linalg.det(R) - 1) < TOL


def test_mount_orientation_conventions():
    assert np.allclose(mount_to_head(90, 0, 0) @ [1, 0, 0], [0, 1, 0])        # yaw + turns forward to the left
    az, el = vec_to_az_el(mount_to_head(0, 10, 0) @ [1, 0, 0])
    assert abs(az) < TOL and abs(el - 10) < TOL                               # the ADR check
    assert mount_to_head(0, 0, 30)[1, 2] < 0                                   # roll +: the top tilts to the right (-y)


def test_angle_between():
    assert angle_between([1, 0, 0], [1, 0, 0]) == 0.0
    assert abs(angle_between([1, 0, 0], [-1, 0, 0]) - 180) < TOL
    assert abs(angle_between(az_el_to_vec(0, 0), az_el_to_vec(10, 0)) - 10) < 1e-9
    with pytest.raises(ValueError):
        angle_between([0, 0, 0], [1, 0, 0])


def test_fov_intrinsics():
    k = fov_intrinsics(W, H, 90.0, 60.0)
    assert abs(k.fx - W / 2) < TOL and abs(k.fy - (H / 2) / math.tan(math.radians(30))) < TOL and (k.cx, k.cy) == (W / 2, H / 2)
    # the image's left edge centre is half the horizontal field of view to the left
    az, el = vec_to_az_el(pixel_to_mount_ray(0, H / 2, k))
    assert abs(az - 45) < 1e-9 and abs(el) < TOL
    az, el = vec_to_az_el(pixel_to_mount_ray(W / 2, 0, k))
    assert abs(el - 30) < 1e-9 and abs(az) < TOL
    for bad in (0.0, 180.0, -5.0):
        with pytest.raises(ValueError):
            fov_intrinsics(W, H, bad, 60.0)


def test_pixel_ray_round_trip_and_bounds():
    k = fov_intrinsics(W, H, 90.0, 60.0)
    for u, v in ((10.5, 20.25), (320, 240), (639, 479)):
        px = mount_ray_to_pixel(pixel_to_mount_ray(u, v, k), k)
        assert px is not None and abs(px[0] - u) < 1e-9 and abs(px[1] - v) < 1e-9
    assert mount_ray_to_pixel([-1, 0, 0], k) is None          # behind the camera
    assert mount_ray_to_pixel(az_el_to_vec(50, 0), k) is None  # outside a 90 degree field of view


# ---------- rig ----------

def rig(*mounts) -> Rig:
    return Rig.from_config({"mounts": list(mounts)})


def centre(mount_id: str) -> SpatialDetection:
    return det(mount_id, pixel=(W / 2, H / 2))


def test_mount_centres_point_the_right_way():
    r = rig(cam("front"), cam("rear", 180), cam("left", 90), cam("right", -90))
    cases = {"front": (0.0, "ahead"), "rear": (180.0, "behind"), "left": (90.0, "at left"), "right": (-90.0, "at right")}
    for mid, (az_want, word) in cases.items():
        az, el = r.to_head(centre(mid))
        assert abs(wrap_az(az - az_want)) < 1e-9 and abs(el) < 1e-9 and direction_word(az, el) == word


def test_pitched_mount_elevation():
    r = rig(cam("up", pitch=10.0), cam("down", yaw=90.0, pitch=-20.0))
    az, el = r.to_head(centre("up"))
    assert abs(az) < 1e-9 and abs(el - 10) < 1e-9
    az, el = r.to_head(centre("down"))
    assert abs(az - 90) < 1e-9 and abs(el + 20) < 1e-9


def test_roll_rotates_the_image_axes():
    k = fov_intrinsics(W, H, 90.0, 70.0)
    up_px = (W / 2, H / 2 - k.fy * math.tan(math.radians(20)))     # 20 degrees above centre in the image
    right_px = (W / 2 + k.fx * math.tan(math.radians(20)), H / 2)  # 20 degrees right of centre
    level = rig(cam("c"))
    az, el = level.to_head(det("c", pixel=up_px))
    assert abs(az) < 1e-9 and abs(el - 20) < 1e-9
    rolled = rig(cam("c", roll=90.0))   # clockwise from behind: the image top now points to the wearer's right
    az, el = rolled.to_head(det("c", pixel=up_px))
    assert abs(az + 20) < 1e-9 and abs(el) < 1e-9
    az, el = rolled.to_head(det("c", pixel=right_px))   # image right now points down
    assert abs(az) < 1e-9 and abs(el + 20) < 1e-9


def test_explicit_intrinsics_and_bearings():
    r = rig({"id": "c", "kind": "nir", "intrinsics": {"fx": 300.0, "fy": 300.0, "cx": 300.0, "cy": 200.0}, "image_size": [W, H]},
            {"id": "mics", "kind": "mic_array", "yaw_deg": 180.0, "mic_positions_m": [[0.05, 0, 0], [-0.05, 0, 0]]})
    az, el = r.to_head(det("c", pixel=(300.0, 200.0)))
    assert abs(az) < 1e-9 and abs(el) < 1e-9
    az, _ = r.to_head(det("c", pixel=(600.0, 200.0)))   # one focal length right of the centre: 45 degrees right
    assert abs(az + 45) < 1e-9
    az, el = r.to_head(det("mics", kind="sound", family="voice", mount_bearing=(30.0, 5.0)))
    assert abs(az - (-150.0)) < 1e-9 and abs(el - 5) < 1e-9
    az, el = r.to_head(det("anything", head_bearing=(-180.0, 0.0)))
    assert az == 180.0
    for bad in (det("nowhere", pixel=(1, 1)), det("mics", pixel=(1, 1)), det("c", pixel=(W + 1, 1))):
        with pytest.raises(ValueError):
            r.to_head(bad)


def test_default_rig_is_one_front_camera():
    assert validate_rig({}) == [] and Config().rig == {}
    r = Rig.from_config({})
    assert list(r.mounts) == ["front"] and r.mounts["front"].kind == "camera" and r.mounts["front"].intrinsics is not None
    assert DEFAULT_RIG["mounts"][0]["id"] == "front"
    assert Rig.from_config({"mounts": []}).mounts == {}   # no mounts: no coverage, not an error


@pytest.mark.parametrize("mounts,needle", [
    ([cam("a"), cam("a", 180)], "duplicate ids"),
    ([{"id": "", "kind": "camera"}], ".id must be"),
    ([{"id": "bot_cam", "kind": "camera"}], ".id must be"),
    ([{"id": "x", "kind": "lidar"}], ".kind must be"),
    ([cam("x", yaw=float("nan"))], "yaw_deg"),
    ([cam("x", pitch=95.0)], "pitch_deg"),
    ([cam("x", yaw=True)], "yaw_deg"),
    ([{**cam("x"), "hfov_deg": 180.0}], "hfov_deg"),
    ([{**cam("x"), "vfov_deg": 0.0}], "vfov_deg"),
    ([{**cam("x"), "image_size": [640]}], "image_size"),
    ([{**cam("x"), "image_size": [640.0, 480]}], "image_size"),
    ([{**cam("x"), "image_size": [0, 480]}], "image_size"),
    ([{**cam("x"), "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}}], "not both"),
    ([{"id": "x", "kind": "camera", "intrinsics": {"fx": -1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}}], "intrinsics"),
    ([{"id": "x", "kind": "camera", "position_m": [0, 0]}], "position_m"),
    ([{"id": "x", "kind": "camera", "position_m": [0, 0, 5.0]}], "position_m"),
    ([{"id": "m", "kind": "mic_array", "mic_positions_m": [[0, 0, 0]]}], "at least 2"),
    ([{"id": "m", "kind": "mic_array"}], "at least 2"),
    ([{"id": "m", "kind": "mic_array", "mic_positions_m": [[0, 0, 0], [0, 0, 0]]}], "same position"),
    ([{"id": "m", "kind": "imu", "hfov_deg": 60.0}], "only to imaging"),
    ([{"id": "m", "kind": "imu", "ir_illuminator": True}], "only to imaging"),
    ([{**cam("x"), "mic_positions_m": [[0, 0, 0], [1, 0, 0]]}], "only to mic_array"),
    ([{**cam("x"), "ir_illuminator": "yes"}], "true or false"),
    ([{**cam("x"), "zoom": 2}], "unknown keys"),
])
def test_rig_validation(mounts, needle):
    errs = validate_rig({"mounts": mounts})
    assert any(needle in e for e in errs), errs
    with pytest.raises(ConfigError):
        Rig.from_config({"mounts": mounts})
    with pytest.raises(ConfigError):
        Config(rig={"mounts": mounts}).validate()


def test_rig_section_shape():
    assert validate_rig([]) == ["rig must be an object"]  # type: ignore[arg-type]
    assert any("unknown keys" in e for e in validate_rig({"cameras": []}))
    assert any("list" in e for e in validate_rig({"mounts": {}}))
    assert any("list" in e for e in validate_rig({"mounts": [cam(f"c{i}") for i in range(17)]}))


@pytest.mark.parametrize("yaw,pitch", [(31.0, 0.0), (-31.0, 0.0), (180.0, 0.0), (90.0, 0.0), (0.0, 31.0), (0.0, -31.0)])
def test_eye_safety_refuses_off_axis_illuminators(yaw, pitch):
    m = {"id": "ir", "kind": "nir", "yaw_deg": yaw, "pitch_deg": pitch, "ir_illuminator": True}
    errs = validate_rig({"mounts": [m]})
    assert len(errs) == 1 and "7.3" in errs[0] and "ir_eye_safety_approved" in errs[0]
    with pytest.raises(ConfigError, match="7.3"):
        Config(rig={"mounts": [m]}).validate()
    approved = {**m, "ir_eye_safety_approved": True}
    assert validate_rig({"mounts": [approved]}) == []
    assert Rig.from_config({"mounts": [approved]}).mounts["ir"].ir_eye_safety_approved


def test_eye_safety_allows_forward_illuminators_and_unlit_mounts():
    for yaw, pitch, roll in ((0.0, 0.0, 0.0), (30.0, 0.0, 0.0), (-30.0, 30.0, 0.0), (0.0, 0.0, 90.0)):
        m = {"id": "ir", "kind": "nir", "yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll, "ir_illuminator": True}
        assert validate_rig({"mounts": [m]}) == []
    assert validate_rig({"mounts": [cam("rear", 180.0)]}) == []                          # no illuminator: no limit
    assert validate_rig({"mounts": [{**cam("rear", 180.0), "ir_illuminator": False}]}) == []


def test_config_load_reads_the_rig(tmp_path):
    p = tmp_path / "c.json"
    p.write_text('{"rig": {"mounts": [{"id": "rear", "kind": "camera", "yaw_deg": 180}]}}', encoding="utf-8")
    c = Config.load(p)
    assert Rig.from_config(c.rig).mounts["rear"].yaw_deg == 180.0


# ---------- detections ----------

def test_detection_validation_and_dict_round_trip():
    d = det(pixel=(1, 2), range_m=2.0, size=3)
    assert d.pixel == (1.0, 2.0) and SpatialDetection.from_dict(d.to_dict()) == d
    for bad in ({"kind": "person"}, {"confidence": 1.5}, {"confidence": float("nan")}, {"t_mono_ns": -1}, {"t_mono_ns": 1.0},
                {"mount_id": ""}, {"family": ""}, {}, {"pixel": (1, 2), "head_bearing": (0, 0)}, {"pixel": (1,)},
                {"pixel": (float("inf"), 0)}, {"pixel": (1, 2), "range_m": -1.0}):
        with pytest.raises(ValueError):
            det(**bad)
    with pytest.raises(ValueError):
        SpatialDetection.from_dict({**d.to_dict(), "label": "x"})


def test_feature_helpers_place_glints_emitters_and_heat():
    from oiris.features.optical import glint_candidates
    r = rig(cam("front"), {**cam("rear", 180), "kind": "thermal", "image_size": [32, 24]})
    off = np.zeros((H, W), np.uint8)
    on = off.copy()
    on[239:241, 319:321] = 250                      # a glint at the image centre
    gl = glint_detections(glint_candidates(on, off), r.mounts["front"], 5)
    assert len(gl) == 1 and gl[0].kind == "glint" and gl[0].family == "optical" and gl[0].confidence > 0.5
    az, el = r.to_head(gl[0])
    assert abs(az) < 0.2 and abs(el) < 0.2
    off2 = off.copy()
    off2[100:103, 600:603] = 255                    # a saturated spot right of centre and above
    em = emitter_detections(off2, r.mounts["front"], 5)
    assert len(em) == 1 and em[0].kind == "ir_emitter" and em[0].size == 9 and em[0].pixel == (601.0, 101.0)
    az, el = r.to_head(em[0])
    assert az < -30 and el > 10
    off3 = off.copy()
    off3[5, 5] = 255                                # one pixel: below nir_emitter_score's minimum
    assert emitter_detections(off3, r.mounts["front"], 5) == []
    f = np.full((24, 32), 21.0)
    f[10:14, 14:18] = 41.0                          # a heater-hot patch at the centre of the rear thermal image
    hot = hot_region_detections(f, r.mounts["rear"], 5)
    assert len(hot) == 1 and hot[0].family == "thermal" and hot[0].confidence == 1.0
    az, _ = r.to_head(hot[0])
    assert direction_word(az) == "behind"


# ---------- merge ----------

def test_two_sensors_ten_degrees_apart_merge():
    ev = merge([loc(0.0, conf=0.5, mount="front"), loc(10.0, conf=0.5, fam="thermal", mount="th", kind="thermal", t_s=0.4)])
    assert len(ev) == 1
    e = ev[0]
    assert abs(e.az - 5.0) < 1e-9 and e.direction == "ahead" and e.families == ("optical", "thermal")
    assert e.sensors == ("front", "th") and e.kinds == ("glint", "thermal") and e.evidence == 2 and e.count == 2
    assert abs(e.confidence - 0.75) < 1e-12 and e.t_mono_ns == int(0.4e9)


@pytest.mark.parametrize("a,b", [(loc(0.0), loc(20.0, mount="x")), (loc(0.0), loc(10.0, mount="x", t_s=1.0)),
                                 (loc(0.0), loc(15.0001, mount="x")), (loc(0.0), loc(5.0, mount="x", t_s=0.5000001))])
def test_far_apart_or_late_do_not_merge(a, b):
    assert len(merge([a, b])) == 2


def test_merge_limits_are_inclusive():
    assert len(merge([loc(0.0), loc(15.0 - 1e-9, mount="x", t_s=0.5)])) == 1


def test_weighted_bearing_and_noisy_or():
    e = merge([loc(0.0, conf=0.9), loc(12.0, conf=0.3, mount="x")])[0]
    v = 0.9 * az_el_to_vec(0, 0) + 0.3 * az_el_to_vec(12, 0)
    assert abs(e.az - vec_to_az_el(v)[0]) < 1e-9 and 0 < e.az < 6
    assert abs(e.confidence - (1 - 0.1 * 0.7)) < 1e-12
    z = merge([loc(0.0, conf=0.0), loc(10.0, conf=0.0, mount="x")])[0]   # all zero: plain mean
    assert abs(z.az - 5.0) < 1e-9 and z.confidence == 0.0
    assert noisy_or([]) == 0.0 and noisy_or([1.0, 0.2]) == 1.0
    assert merge([loc(1.0, fam="optical"), loc(2.0, fam="optical", mount="x")])[0].evidence == 1


def test_merge_across_the_rear_seam():
    e = merge([loc(175.0), loc(-175.0, mount="x")])
    assert len(e) == 1 and e[0].az == 180.0 and e[0].direction == "behind"


def test_merge_is_order_independent():
    rng = np.random.default_rng(7)
    dets = [loc(float(rng.uniform(-180, 180)), float(rng.uniform(-30, 30)), float(rng.uniform(0, 1)),
                fam=["optical", "thermal", "voice"][i % 3], mount=f"m{i % 4}", t_s=float(rng.uniform(0, 2))) for i in range(40)]
    want = merge(dets)
    for _ in range(10):
        perm = [dets[i] for i in rng.permutation(len(dets))]
        assert merge(perm) == want
    assert sum(e.count for e in want) == 40


def test_locate_reports_what_it_cannot_place():
    r = rig(cam("front"))
    good, bad = locate(r, [det(pixel=(W / 2, H / 2)), det("gone", pixel=(1, 1)), {"kind": "glint"}, det(head_bearing=(90, 0)).to_dict()])
    assert len(good) == 2 and len(bad) == 2 and {g.az for g in good} == {0.0, 90.0}


def test_event_tracker_keeps_a_word_with_hysteresis():
    tr = EventTracker()
    assert tr.update(merge([loc(40.0)]))[0].direction == "ahead"
    assert tr.update(merge([loc(52.0)]))[0].direction == "ahead"      # plain word would be "at left"
    assert tr.update(merge([loc(60.0)]))[0].direction == "at left"
    assert tr.update(merge([loc(-120.0)]))[0].direction == "at right"  # too far to continue: a new track
    assert EventTracker().update(merge([loc(52.0)]))[0].direction == "at left"
