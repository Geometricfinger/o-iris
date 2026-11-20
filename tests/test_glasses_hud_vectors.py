"""Glasses display list (oiris/ui/glasses.py): tests/vectors/glasses_hud.json is current (android/core GlassesHud.kt
must reproduce it), render() still writes the pre-display-list SVG byte for byte, and every op has the fixed schema."""
import base64, hashlib, json, re, subprocess, sys, zlib
from pathlib import Path

import numpy as np
import pytest

from oiris.ui import tokens as T
from oiris.ui.glasses import (COORD_DECIMALS, OP_KEYS, EnrollGuide, GlassesState, display_list, enroll_guide_display_list,
                              render, render_enroll_guide, thermal_image, to_svg)

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "tests/vectors/glasses_hud.json"
HEALTH = {"thermal": "ok", "ble": "degraded", "radar": "down", "x<y": "weird&"}
FRAME = np.random.default_rng(0).normal(30, 2, (24, 32))
PNG_PAYLOAD = re.compile(r'base64,([^"]*)')


def hud(level, text, moving, thermal=None):
    return GlassesState(level=level, alert_text=text, moving=moving, new_sighting=bool(text), thermal=None if thermal is None else FRAME,
                        thermal_colour=thermal == "c", health=dict(HEALTH), voice_tier="high" if moving else "low",
                        locked=level == "low", panic_armed=level == "medium")


# sha256 of the SVG the renderer wrote before the display list existed (captured from the previous source)
SAMPLES = [
    (lambda: render(hud(None, "", False)), "2bc40183ff9be39e8cfd2497e599fbd34ea2dad58536000c11c35a748b394407"),
    (lambda: render(hud("low", "Cold object near door", False)), "0e03896893b469bcb86c880928088dba18fc1b246c55e58813dca9a9f7f517b5"),
    (lambda: render(hud("medium", "Signatures A1 A3", True)), "d7fc8af5c7267869d918a4c20b0880c9202195fc871e1b4cc9261ba3ea003e7b"),
    (lambda: render(hud("confirmed", "a<b & c", False)), "542c67004ba66277cebba30c971e0342a3022ebbc77d42f32ee4157518d7e1eb"),
    (lambda: render(GlassesState(health={f"s{i}": "ok" for i in range(9)})),
     "06969fbf2da26b0caaed815fdb1b98b3da864a317a49ee38b78bd7067f31b184"),
    (lambda: render_enroll_guide(EnrollGuide(["left", "up"], True, "", False)),
     "5a94dd3a2c1c85160dd6cfa32eeb294549e2dbba17c592434be4b4079f233390"),
    (lambda: render_enroll_guide(EnrollGuide(["down"], False, "move closer, face too small", False)),
     "e4ccc8c8d6e195ca7581ca6733fae4d54b1d3c0ce39fb20b46b5756960f8472e"),
    (lambda: render_enroll_guide(EnrollGuide([], None, "", True)), "064ff246cd68ec96462b3f8b0c24a6972a581bbaa262b77e2fe9b77c7d80687c"),
]
# thermal states: the PNG bytes depend on the zlib build, so the payload is masked and its pixels checked apart
THERMAL_SAMPLES = [
    (hud("medium", "Cold object near door", False, "g"), "54585c5a479c8e1fb52ec0caa477eb550ee912063d5f8c8b9a71fb6495b061ca"),
    (hud("confirmed", "Signatures A1 A3", False, "c"), "c840dd2494c0bfe08fdbaa1ef0c05badb7e89ea3215ddc6665066f07559cc95a"),
]


def test_glasses_hud_vectors_are_current():
    r = subprocess.run([sys.executable, str(ROOT / "scripts/make_android_vectors.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("make, digest", SAMPLES)
def test_render_is_byte_identical_to_the_previous_renderer(make, digest):
    assert hashlib.sha256(make().encode()).hexdigest() == digest


@pytest.mark.parametrize("st, digest", THERMAL_SAMPLES)
def test_thermal_render_unchanged_and_pixels_intact(st, digest):
    svg = render(st)
    assert hashlib.sha256(PNG_PAYLOAD.sub("base64,", svg).encode()).hexdigest() == digest
    png = base64.b64decode(PNG_PAYLOAD.search(svg).group(1))
    idat = png[png.index(b"IDAT") + 4:png.index(b"IEND") - 8]
    img = thermal_image(st.thermal, st.thermal_colour)
    rows = zlib.decompress(idat)
    assert rows == b"".join(b"\x00" + img[y].tobytes() for y in range(img.shape[0]))


def _check_ops(ops):
    assert ops[0]["op"] == "canvas" and [o["id"] for o in ops if o["op"] == "zone"] == ["G1", "G2", "G3", "G4", "G5", "G6"]
    for op in ops:
        assert list(op) == list(OP_KEYS[op["op"]]), op
        for v in op.values():
            assert type(v) in (str, int, float, bool), op
            if type(v) is float:
                assert round(v, COORD_DECIMALS) == v, op
    assert json.loads(json.dumps(ops)) == ops


@pytest.mark.parametrize("st", [hud(lv, t, mv, th) for lv in (None, "low", "medium", "confirmed") for t in ("", "Heat source ahead")
                                for mv in (False, True) for th in (None, "g", "c")])
def test_display_list_ops_have_the_fixed_schema(st):
    ops = display_list(st)
    _check_ops(ops)
    images = [o for o in ops if o["op"] == "image"]
    assert len(images) == (st.thermal is not None) and all(o["source"] == "thermal" for o in images)   # no pixels in the list
    assert ("G2" in {o["id"] for o in ops if o["op"] == "text"}) == bool(st.alert_text and not st.moving)


@pytest.mark.parametrize("g", [EnrollGuide(["front", "left", "right", "up", "down"]), EnrollGuide(["up"], True),
                               EnrollGuide(["up"], False, "too dark"), EnrollGuide([], True, "", True)])
def test_enroll_guide_ops_have_the_fixed_schema_and_render_through_to_svg(g):
    ops = enroll_guide_display_list(g)
    _check_ops(ops)
    assert ops[0]["id"] == "enroll-guide" and to_svg(ops) == render_enroll_guide(g)
    assert any(o["id"] == "tick" for o in ops) == bool(g.last_ok)


def test_vectors_cover_levels_moving_errors_and_every_guide_step():
    v = json.loads(VECTORS.read_text())
    states = [c["state"] for c in v["hud"]]
    assert {(s["level"], s["moving"]) for s in states if s["level"] in T.LEVELS} >= {(lv, m) for lv in T.LEVELS for m in (False, True)}
    assert {c.get("error") for c in v["hud"]} >= {None, "LayoutError", "VocabularyViolation"}
    assert any(set(s["health"].values()) >= {"ok", "degraded", "down"} for s in states)
    assert any(s["alert_text"] == "" for s in states) and any(c["name"] == "one word over the limit" for c in v["hud"])
    assert {len(c["guide"]["needed"]) for c in v["guide"]} == {0, 1, 2, 3, 4, 5}
    assert any(c["name"] == "rejected, reason truncated" and "error" not in c for c in v["guide"])
    assert all(op["op"] != "image" or "pixels" not in op for c in v["hud"] for op in c.get("ops", []))
