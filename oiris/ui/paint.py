"""Reference painter for the glasses display list (oiris/ui/glasses.py): each op becomes the draw call a plain canvas
executes, derived from how glasses.to_svg renders that op, so a device canvas draws what the simulator shows.
android/core HudPainter.kt makes the same calls in the same order with the same values (tests/vectors/hud_paint.json,
scripts/make_android_vectors.py) and the app's AndroidHudCanvas executes them on android.graphics.Canvas.

Draw calls (tuples; every number a float, colours the op's strings, "" = no paint):
  ("frame", w, h, background, opacity, font_family)            canvas: logical size, background, global opacity
  ("rect", x, y, w, h, fill, stroke, stroke_width, radius)      cue edge marker (no stroke, square corners)
  ("circle", cx, cy, r, fill, stroke, stroke_width)             level circle (r = size / 2, fill only); dot (r = size // 2)
  ("polygon", points, fill)                                     level triangle / diamond; points = (x0, y0, x1, y1, ...)
  ("polyline", points, stroke, stroke_width)                    the guide tick, unfilled
  ("text", x, y, size, colour, weight, anchor, text)            baseline at y; weight normal; anchor start|middle|end
  ("image", x, y, w, h, source, mode)                           the G3 placeholder: the canvas supplies the pixels
Zones draw nothing (an additive display has no frames). The first op must be a canvas or nothing is drawn; a later
canvas, an unknown op or shape kind and an op with a missing or mistyped field are skipped, as HudFrame.parse drops
them (the SVG backend would draw an unknown shape kind as a diamond; an untrusted frame gets nothing). Pure."""
from __future__ import annotations
import math
import re
from typing import Any

from .glasses import ANCHOR_START, TICK_POINTS, TICK_SIZE

WEIGHT_NORMAL = "normal"                  # SVG text has no font-weight: the default
ANCHORS = (ANCHOR_START, "middle", "end")  # SVG text-anchor values; anything else renders as start
NO_PAINT = ""
NO_STROKE_WIDTH = 0.0
SQUARE = 0.0                              # rect corner radius: the cue is a plain SVG rect
HALF = 2                                  # size -> radius
MARK_KINDS = ("circle", "triangle", "diamond")
CALL_KINDS = ("frame", "rect", "circle", "polygon", "polyline", "text", "image")
# Colours: "#rgb" or "#rrggbb" (the design-token form), opaque; anything else (named colours, "") is not painted.
_HEX = re.compile(r"#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})")
SHORT_HEX = 3
OPAQUE = 0xFF000000


def parse_colour(s: Any) -> int | None:
    """ARGB of a hex colour string (always opaque), or None: no paint."""
    m = _HEX.fullmatch(s) if isinstance(s, str) else None
    if m is None:
        return None
    h = m.group(1)
    if len(h) == SHORT_HEX:
        h = "".join(c * 2 for c in h)
    return OPAQUE | int(h, 16)


def _num(op: dict, k: str) -> float | None:
    v = op.get(k)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except OverflowError:  # python-brain: allow SILENT_EXCEPTION - an int beyond a double is not drawable: skip the op
        return None
    return f if math.isfinite(f) else None


def _nums(op: dict, *keys: str) -> list[float] | None:
    out = [_num(op, k) for k in keys]
    return None if any(v is None for v in out) else out   # type: ignore[return-value]


def _strs(op: dict, *keys: str) -> list[str] | None:
    out = [op.get(k) for k in keys]
    return out if all(isinstance(v, str) for v in out) else None   # type: ignore[return-value]


def _frame(op: dict) -> tuple | None:
    n, s = _nums(op, "w", "h", "opacity"), _strs(op, "colour", "font_family")
    if n is None or s is None:
        return None
    return ("frame", n[0], n[1], s[0], n[2], s[1])


def _text(op: dict) -> tuple | None:
    n, s = _nums(op, "x", "y", "size"), _strs(op, "colour", "anchor", "text")
    if n is None or s is None:
        return None
    anchor = s[1] if s[1] in ANCHORS else ANCHOR_START
    return ("text", n[0], n[1], n[2], s[0], WEIGHT_NORMAL, anchor, s[2])


def _shape(op: dict) -> tuple | None:
    n, s = _nums(op, "x", "y", "size", "stroke_width"), _strs(op, "kind", "fill", "stroke")
    if n is None or s is None:
        return None
    x, y, size, sw = n
    kind, fill, stroke = s
    if kind == "circle":       # glasses._shape: a filled circle, the op's stroke is not drawn
        return ("circle", x, y, size / HALF, fill, NO_PAINT, NO_STROKE_WIDTH)
    if kind == "triangle":
        return ("polygon", (x, y - size / HALF, x + size / HALF, y + size / HALF, x - size / HALF, y + size / HALF), fill)
    if kind == "diamond":
        return ("polygon", (x, y - size / HALF, x + size / HALF, y, x, y + size / HALF, x - size / HALF, y), fill)
    if kind == "dot":          # glasses._svg_shape: r = size // 2, fill "" = none
        return ("circle", x, y, size // HALF, fill, stroke, sw)
    if kind == "tick":         # TICK_POINTS scaled by size // TICK_SIZE, as the SVG polyline
        pts: list[float] = []
        for dx, dy in TICK_POINTS:
            pts += [x + dx * size // TICK_SIZE, y + dy * size // TICK_SIZE]
        return ("polyline", tuple(pts), stroke, sw)
    return None


def _cue(op: dict) -> tuple | None:
    n, s = _nums(op, "x", "y", "w", "h"), _strs(op, "colour")
    if n is None or s is None:
        return None
    return ("rect", n[0], n[1], n[2], n[3], s[0], NO_PAINT, NO_STROKE_WIDTH, SQUARE)


def _image(op: dict) -> tuple | None:
    n, s = _nums(op, "x", "y", "w", "h"), _strs(op, "source", "mode")
    if n is None or s is None:
        return None
    return ("image", n[0], n[1], n[2], n[3], s[0], s[1])


_DRAW = {"text": _text, "shape": _shape, "cue": _cue, "image": _image}


def paint(ops: list) -> list[tuple]:
    """The draw calls for a display list (glasses.display_list, or HudFrame ops after parse_hud), in op order."""
    if not ops or not isinstance(ops[0], dict) or ops[0].get("op") != "canvas":
        return []
    first = _frame(ops[0])
    if first is None:
        return []
    calls = [first]
    for op in ops[1:]:
        kind = op.get("op") if isinstance(op, dict) else None
        draw = _DRAW.get(kind) if isinstance(kind, str) else None
        call = draw(op) if draw else None
        if call is not None:
            calls.append(call)
    return calls
