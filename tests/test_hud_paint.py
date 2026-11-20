"""Reference painter (oiris/ui/paint.py): pure and deterministic, every call agrees with the SVG backend element for
element, every op kind and call kind is covered, and tests/vectors/hud_paint.json (android/core HudPainter.kt must
reproduce it) is current."""
import copy, importlib.util, json, math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from oiris.link import ops as LO
from oiris.ui import tokens as T
from oiris.ui.glasses import (HAIRLINE, OP_KEYS, EnrollGuide, GlassesState, display_list, enroll_guide_display_list,
                              to_svg)
from oiris.ui.paint import ANCHORS, CALL_KINDS, MARK_KINDS, WEIGHT_NORMAL, paint, parse_colour

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "tests/vectors/hud_paint.json"
SVG = "{http://www.w3.org/2000/svg}"
FRAME = np.zeros((2, 2))
SHAPE_KINDS = MARK_KINDS + ("dot", "tick")


def _gen():
    spec = importlib.util.spec_from_file_location("make_android_vectors", ROOT / "scripts/make_android_vectors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _state(s: dict) -> GlassesState:
    return GlassesState(**{**s, "thermal": FRAME if s["thermal"] else None})


def _vector_lists() -> list[list[dict]]:
    v = json.loads(VECTORS.read_text())
    gen = _gen()
    out = [display_list(_state(c["state"]), gen.gh_tokens(c["tokens_patch"])) for c in v["hud"]]
    out += [enroll_guide_display_list(EnrollGuide(list(g["guide"]["needed"]), g["guide"]["last_ok"], g["guide"]["last_reason"],
                                                  g["guide"]["complete"])) for g in v["guide"]]
    return out


LISTS = _vector_lists()


def test_hud_paint_vectors_are_current():
    gen = _gen()
    fresh = json.loads(json.dumps(gen.hud_paint()))
    assert gen.differences(json.loads(VECTORS.read_text()), fresh) == []


@pytest.mark.parametrize("ops", LISTS)
def test_paint_is_pure_and_deterministic(ops):
    before = copy.deepcopy(ops)
    first, second = paint(ops), paint(ops)
    assert first == second and ops == before
    assert all(type(x) in (str, float, tuple) for call in first for x in call[1:])
    assert all(type(p) is float for call in first if call[0] in ("polygon", "polyline") for p in call[1])


def _f(el, k, default=None) -> float:
    return float(el.get(k, default))


def _points(el) -> tuple:
    return tuple(float(n) for p in el.get("points").split() for n in p.split(","))


def _paint_of(el, k) -> str:
    v = el.get(k, "")
    return "" if v == "none" else v


@pytest.mark.parametrize("ops", LISTS)
def test_every_call_matches_the_svg_element(ops):
    """to_svg is the visual truth: element i of the SVG (after the background rect) is call i, attribute by attribute."""
    root = ET.fromstring(to_svg(ops, {"thermal": ""}))
    els = list(root)
    calls = paint(ops)
    assert len(els) == len(calls)
    bg, frame = els[0], calls[0]
    assert frame == ("frame", _f(root, "width"), _f(root, "height"), bg.get("fill"), _f(root, "opacity"), root.get("font-family"))
    for el, call in zip(els[1:], calls[1:]):
        tag = el.tag.removeprefix(SVG)
        if call[0] == "text":
            assert tag == "text" and call == ("text", _f(el, "x"), _f(el, "y"), _f(el, "font-size", root.get("font-size")),
                                              el.get("fill"), WEIGHT_NORMAL, el.get("text-anchor", "start"), el.text or "")
        elif call[0] == "circle":
            sw = _f(el, "stroke-width", HAIRLINE) if "stroke" in el.attrib else 0.0
            assert tag == "circle" and call == ("circle", _f(el, "cx"), _f(el, "cy"), _f(el, "r"), _paint_of(el, "fill"),
                                                el.get("stroke", ""), sw)
        elif call[0] == "polygon":
            assert tag == "polygon" and call == ("polygon", _points(el), el.get("fill"))
        elif call[0] == "polyline":
            assert tag == "polyline" and el.get("fill") == "none"
            assert call == ("polyline", _points(el), el.get("stroke"), _f(el, "stroke-width"))
        elif call[0] == "rect":
            assert tag == "rect" and call == ("rect", _f(el, "x"), _f(el, "y"), _f(el, "width"), _f(el, "height"),
                                              el.get("fill"), "", 0.0, 0.0)
        else:
            assert tag == "image" and call == ("image", _f(el, "x"), _f(el, "y"), _f(el, "width"), _f(el, "height"),
                                               "thermal", el.get("data-mode"))


def test_vectors_cover_every_op_kind_shape_kind_and_call_kind():
    kinds = {op["op"] for ops in LISTS for op in ops}
    assert kinds == set(OP_KEYS)
    assert {op["kind"] for ops in LISTS for op in ops if op["op"] == "shape"} == set(SHAPE_KINDS)
    v = json.loads(VECTORS.read_text())
    calls = [c for group in ("hud", "guide", "frames") for case in v[group] for c in case["calls"]]
    assert {c[0] for c in calls} == set(CALL_KINDS)
    assert {c[6] for c in calls if c[0] == "text"} == set(ANCHORS)
    names = {c["name"] for c in v["hud"]}
    assert "idle" in names and {f"level {lv}, moving {m}" for lv in T.LEVELS for m in (False, True)} <= names
    assert {c["state"]["cue_direction"] for c in v["hud"]} >= {"at left", "at right", "above", "below", "behind", "ahead"}
    assert any(c["calls"] == [] for c in v["frames"])


def test_hostile_frames_never_raise_and_skip_what_they_cannot_draw():
    canvas = display_list(GlassesState())[0]
    junk = [canvas, {"op": "text", "x": True}, {"op": ["list"]}, {"op": "shape", "kind": "hexagon", "x": 1, "y": 1, "size": 1,
            "fill": "", "stroke": "", "stroke_width": 0}, {"op": "cue", "x": math.nan, "y": 0, "w": 1, "h": 1, "colour": ""},
            {"op": "text", "x": 10 ** 400, "y": 0, "size": 1, "colour": "", "anchor": "", "text": ""}, None, 5, "text", {}]
    assert [c[0] for c in paint(junk)] == ["frame"]
    assert paint(junk[1:]) == [] and paint([]) == [] and paint([{**canvas, "w": "640"}]) == []
    reply = json.dumps({"ok": True, "version": 1, "ops": [canvas, {"op": "script", "src": "x"}]})
    assert paint(LO.parse_hud(reply)["ops"]) == [paint([canvas])[0]]


def test_colours():
    tok = T.load()
    for c in [tok["color"][k] for k in tok["color"] if k != "alert"] + list(tok["color"]["alert"].values()):
        assert parse_colour(c) == 0xFF000000 | int(c[1:], 16)
    assert parse_colour("#abc") == parse_colour("#AABBCC") == 0xFFAABBCC
    assert all(parse_colour(s) is None for s in ("", "none", "red", "#12345", "#fff\n", None, 0xFFFFFF))
