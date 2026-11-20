"""Glasses display renderer (G1-G6, Module Map 7.1), simulated as SVG on the C7 canvas.

Additive display: black is transparent, so everything is sparse light-on-dark at low default
brightness. G1 level strip, G2 alert line (<=6 words, vocabulary-guarded), G3 thermal window
(grayscale default, colour overlay toggle, never the alert colours), G4 health dots, G5 voice
tier, G6 lock / panic-armed state. The centre of view stays clear except the thermal window.

The display is defined once as a display list of plain draw ops (display_list, enroll_guide_display_list);
the SVG here is one backend of it (to_svg) and android/core GlassesHud.kt reproduces the list exactly."""
from __future__ import annotations
import base64, html, io, zlib, struct
from dataclasses import dataclass, field

import numpy as np

from ..guard.vocabulary import check_text
from ..modules.motion import is_moving
from . import tokens as T


@dataclass
class GlassesState:
    level: str | None = None                 # low | medium | confirmed | None
    alert_text: str = ""
    new_sighting: bool = False
    thermal: np.ndarray | None = None         # degC frame
    thermal_colour: bool = False
    health: dict = field(default_factory=dict)   # sensor -> ok|degraded|down
    voice_tier: str = "low"
    locked: bool = False
    panic_armed: bool = False
    moving: bool = False                      # walking or fast (moving_from_motion): no G2 pop-ups
    cue_direction: str | None = None          # direction word of the current cue (oiris/spatial/cues.py): edge marker


class LayoutError(ValueError):
    pass


def zones(tok: dict) -> dict:
    W, H = tok["glasses"]["canvas"]
    return {"G1": (0, 0, W, 28), "G2": (0, 32, W, 36), "G3": tuple(tok["glasses"]["thermal_window"]),
            "G4": (8, H - 28, 200, 20), "G5": (W - 160, H - 28, 152, 20), "G6": (8, 72, 160, 24)}


def _centre(tok: dict) -> tuple:
    W, H = tok["glasses"]["canvas"]
    f = tok["glasses"]["centre_clear_fraction"]
    return (W * (1 - f) / 2, H * (1 - f) / 2, W * f, H * f)


def _overlap(a, b) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def _whole(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def check_layout(tok: dict) -> None:
    W, H = tok["glasses"]["canvas"]
    if not all(_whole(v) for v in (W, H, *tok["glasses"]["thermal_window"])):
        raise LayoutError("canvas and thermal window must be whole pixels")
    z, c = zones(tok), _centre(tok)
    for k, r in z.items():
        if r[0] < 0 or r[1] < 0 or r[0] + r[2] > W or r[1] + r[3] > H:
            raise LayoutError(f"{k} outside canvas")
        if k != "G3" and _overlap(r, c):
            raise LayoutError(f"{k} intrudes on the clear centre")
    ks = list(z)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            if _overlap(z[a], z[b]):
                raise LayoutError(f"{a} overlaps {b}")


def _png_gray(img: np.ndarray) -> bytes:
    h, w = img.shape[:2]
    ch = 3 if img.ndim == 3 else 1
    raw = b"".join(b"\x00" + img[y].astype(np.uint8).tobytes() for y in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2 if ch == 3 else 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def thermal_image(frame: np.ndarray, colour: bool) -> np.ndarray:
    f = np.asarray(frame, float)
    g = (255 * (f - f.min()) / (np.ptp(f) or 1)).astype(np.uint8)
    if not colour:
        return g
    # overlay ramp (black->purple->white); built from neither alert hue family
    r = np.clip(g * 1.0, 0, 255); gg = np.clip((g.astype(int) - 128) * 2, 0, 255); b = np.clip(255 - np.abs(g.astype(int) - 128) * 2, 0, 255)
    return np.stack([r, gg, b], axis=-1).astype(np.uint8)


# ---- Display list: the glasses display defined once ------------------------------------------------
# display_list() turns a state and the C7 tokens into plain draw ops; to_svg() is one backend (this
# simulator) and an Android Canvas will be another. android/core GlassesHud.kt builds the same list from
# the same state and tokens; tests/vectors/glasses_hud.json (scripts/make_android_vectors.py) holds them to it.
# Every op is a dict with the fixed keys of OP_KEYS[op["op"]] and only str / int / float / bool values.
# Numbers: integer pixels stay int; any other value is a float rounded to COORD_DECIMALS (so both languages
# agree exactly; Kotlin rounds the exact binary value half-even, as Python's round does).
#   canvas  id (HUD_ID | GUIDE_ID), w, h, colour (background), opacity, font_family, font_size (default text size)
#   zone    id (G1-G6), x, y, w, h: layout only; an additive display draws no zone frames
#   text    id, x, y (baseline), text (raw, unescaped), size, colour, anchor (start | middle | end)
#   shape   id, kind, x, y, size, fill, stroke ("" = none), stroke_width, tag
#           kind circle | triangle | diamond: the level mark centred on (x, y), size = full extent, filled
#           kind dot: a circle of diameter size centred on (x, y); tag = "sensor: health" (G4) or the angle (guide)
#           kind tick: polyline TICK_POINTS (offsets for TICK_SIZE, scaled by size / TICK_SIZE) from (x, y)
#   image   id, x, y, w, h, source (SOURCE_THERMAL), mode (grayscale | colour): a placeholder, no pixels
#   cue     id (CUE_ID), direction (the cue's direction word), edge (left | right | top | bottom), x, y, w, h, colour:
#           a filled edge marker (CUE_EDGES) saying where a cued signature lies; never in the clear centre
COORD_DECIMALS = 2
OP_KEYS = {
    "canvas": ("op", "id", "w", "h", "colour", "opacity", "font_family", "font_size"),
    "zone": ("op", "id", "x", "y", "w", "h"),
    "text": ("op", "id", "x", "y", "text", "size", "colour", "anchor"),
    "shape": ("op", "id", "kind", "x", "y", "size", "fill", "stroke", "stroke_width", "tag"),
    "image": ("op", "id", "x", "y", "w", "h", "source", "mode"),
    "cue": ("op", "id", "direction", "edge", "x", "y", "w", "h", "colour"),
}
HUD_ID, GUIDE_ID = "hud", "enroll-guide"
ANCHOR_START = "start"
SOURCE_THERMAL = "thermal"
ZONE_IDS = ("G1", "G2", "G3", "G4", "G5", "G6")
# HUD geometry (pixels on the C7 canvas), relative to the zone each sits in
MARK_X, MARK_SIZE = 16, 16                 # G1 level mark centre x and extent
G1_TEXT_X, G1_BASELINE_UP = 32, 7          # level word: x, and baseline above the zone bottom
G2_TEXT_X, G2_BASELINE_UP = 8, 10          # alert / guide line
CUE_FROM_RIGHT, CUE_BASELINE_DOWN = 150, 26   # "New sighting" cue: from the canvas right edge, below the G2 top
G5_BASELINE_UP, G6_BASELINE_UP = 4, 6
DOT_INSET, DOT_PITCH, DOT_SIZE = 8, 18, 10    # G4 health / guide progress dots
HAIRLINE, DOWN_STROKE = 1, 2               # stroke widths: default outline; a degraded or down sensor ring
TICK_FROM_RIGHT, TICK_DOWN, TICK_SIZE, TICK_STROKE = 40, 6, 24, 3
TICK_POINTS = ((0, 12), (8, 20), (24, 2))
CUE_TEXT = "New sighting"
VOICE_PREFIX = "Voice: "
G6_LOCKED, G6_PANIC = "Locked", "Panic armed"
GUIDE_COMPLETE = "Template complete"
# Direction cue (oiris/spatial/cues.py): a short bar on the edge the signature lies towards, its thickness and length
# taken from the C7 spacing scale (space_px[CUE_THICK_STEP], space_px[CUE_LENGTH_STEP]), in the body text colour
# (the alert colours stay reserved for the level). "behind" spans the whole bottom edge, "below" is a short bar at
# its centre, "above" a short bar at the top centre; "ahead" draws nothing (the wearer is already facing it).
CUE_ID = "cue"
CUE_THICK_STEP, CUE_LENGTH_STEP = 1, -1
CUE_EDGES = {"at left": "left", "at right": "right", "above": "top", "below": "bottom", "behind": "bottom"}
CUE_NONE = "ahead"
CUE_SPAN_WHOLE = "behind"


def _n(v):
    """A display-list number: int pixels unchanged, anything else rounded to COORD_DECIMALS."""
    if isinstance(v, bool):
        raise LayoutError("a coordinate cannot be a boolean")
    return v if isinstance(v, int) else round(float(v), COORD_DECIMALS)


def _canvas(tok: dict, ident: str) -> dict:
    W, H = tok["glasses"]["canvas"]
    return {"op": "canvas", "id": ident, "w": W, "h": H, "colour": tok["color"]["bg"],
            "opacity": _n(tok["glasses"]["default_brightness"]), "font_family": tok["type"]["font_family"],
            "font_size": _n(tok["type"]["glasses_body"])}


def _zone_ops(tok: dict) -> list[dict]:
    z = zones(tok)
    return [{"op": "zone", "id": k, "x": z[k][0], "y": z[k][1], "w": z[k][2], "h": z[k][3]} for k in ZONE_IDS]


def _text(ident: str, x, y, text: str, tok: dict, colour: str) -> dict:
    return {"op": "text", "id": ident, "x": _n(x), "y": _n(y), "text": text, "size": _n(tok["type"]["glasses_body"]),
            "colour": colour, "anchor": ANCHOR_START}


def _shape_op(ident: str, kind: str, x, y, size, fill: str, stroke: str = "", stroke_width: int = 0, tag: str = "") -> dict:
    return {"op": "shape", "id": ident, "kind": kind, "x": _n(x), "y": _n(y), "size": _n(size), "fill": fill,
            "stroke": stroke, "stroke_width": _n(stroke_width), "tag": tag}


def display_list(st: GlassesState, tok: dict | None = None) -> list[dict]:
    """The HUD for a state as draw ops (see OP_KEYS). Raises LayoutError / VocabularyViolation as render does."""
    tok = tok or T.load()
    check_layout(tok)
    W, H = tok["glasses"]["canvas"]
    z, c = zones(tok), tok["color"]
    ops = [_canvas(tok, HUD_ID)] + _zone_ops(tok)
    if st.level:
        if st.level not in T.LEVELS:
            raise LayoutError("unknown level")
        lv = tok["alert_levels"][st.level]
        x, y, w, h = z["G1"]
        ops.append(_shape_op("G1", lv["shape"], MARK_X, y + h / 2, MARK_SIZE, c["alert"][st.level]))
        ops.append(_text("G1", G1_TEXT_X, y + h - G1_BASELINE_UP, lv["word"], tok, c["alert"][st.level]))
    if st.alert_text and not st.moving:   # no pop-ups during movement; G1 strip still shows the level
        words = st.alert_text.split()
        if len(words) > tok["glasses"]["max_words_alert_line"]:
            raise LayoutError("alert line exceeds the word limit")
        check_text(st.alert_text)
        x, y, w, h = z["G2"]
        ops.append(_text("G2", G2_TEXT_X, y + h - G2_BASELINE_UP, st.alert_text, tok, c["text"]))
    if st.new_sighting:
        ops.append(_text("G2cue", W - CUE_FROM_RIGHT, z["G2"][1] + CUE_BASELINE_DOWN, CUE_TEXT, tok, c["text_muted"]))
    if st.thermal is not None:
        x, y, w, h = z["G3"]
        ops.append({"op": "image", "id": "G3", "x": x, "y": y, "w": w, "h": h, "source": SOURCE_THERMAL,
                    "mode": "colour" if st.thermal_colour else "grayscale"})
    x, y, w, h = z["G4"]
    for i, (name, hs) in enumerate(sorted(st.health.items())):
        col = {"ok": c["health_ok"], "degraded": c["health_degraded"]}.get(hs, c["health_down"])
        ops.append(_shape_op("G4", "dot", x + DOT_INSET + i * DOT_PITCH, y + h / 2, DOT_SIZE, col if hs == "ok" else "",
                             col, HAIRLINE if hs == "ok" else DOWN_STROKE, f"{name}: {hs}"))
    x, y, w, h = z["G5"]
    ops.append(_text("G5", x, y + h - G5_BASELINE_UP, VOICE_PREFIX + st.voice_tier, tok, c["text_muted"]))
    x, y, w, h = z["G6"]
    g6 = (G6_LOCKED if st.locked else "") + (" " + G6_PANIC if st.panic_armed else "")
    if g6:
        ops.append(_text("G6", x, y + h - G6_BASELINE_UP, g6.strip(), tok, c["text"]))
    # The direction cue shows while moving too: it is a steady edge marker, not a pop-up (Module Map 6.1 forbids
    # only pop-ups during movement), so the wearer still learns where; the G2 alert text stays hidden as above.
    if st.cue_direction and st.cue_direction != CUE_NONE:
        ops.append(_cue_op(st.cue_direction, tok))
    return ops


def _cue_op(direction: str, tok: dict) -> dict:
    if direction not in CUE_EDGES:
        raise LayoutError("unknown cue direction")
    W, H = tok["glasses"]["canvas"]
    space = tok["space_px"]
    t, n = space[CUE_THICK_STEP], space[CUE_LENGTH_STEP]
    if not (_whole(t) and _whole(n)):
        raise LayoutError("cue sizes must be whole pixels")
    edge = CUE_EDGES[direction]
    if direction == CUE_SPAN_WHOLE:
        x, y, w, h = 0, H - t, W, t
    elif edge in ("left", "right"):
        x, y, w, h = (0 if edge == "left" else W - t), (H - n) / 2, t, n
    else:
        x, y, w, h = (W - n) / 2, (0 if edge == "top" else H - t), n, t
    if _overlap((x, y, w, h), _centre(tok)):
        raise LayoutError("cue intrudes on the clear centre")
    return {"op": "cue", "id": CUE_ID, "direction": direction, "edge": edge, "x": _n(x), "y": _n(y), "w": _n(w),
            "h": _n(h), "colour": tok["color"]["text"]}


# ---- SVG backend ------------------------------------------------------------------------------------
def _shape(kind: str, x: float, y: float, s: float, col: str) -> str:
    if kind == "circle":
        return f'<circle cx="{x}" cy="{y}" r="{s / 2}" fill="{col}"/>'
    if kind == "triangle":
        return f'<polygon points="{x},{y - s / 2} {x + s / 2},{y + s / 2} {x - s / 2},{y + s / 2}" fill="{col}"/>'
    return f'<polygon points="{x},{y - s / 2} {x + s / 2},{y} {x},{y + s / 2} {x - s / 2},{y}" fill="{col}"/>'


def _svg_shape(op: dict) -> str:
    x, y, s, sw = op["x"], op["y"], op["size"], op["stroke_width"]
    if op["kind"] == "tick":
        pts = " ".join(f"{x + dx * s // TICK_SIZE},{y + dy * s // TICK_SIZE}" for dx, dy in TICK_POINTS)
        return f'<polyline id="{op["id"]}" points="{pts}" fill="none" stroke="{op["stroke"]}" stroke-width="{sw}"/>'
    if op["kind"] != "dot":
        return _shape(op["kind"], x, y, s, op["fill"])
    width = "" if sw == HAIRLINE else f' stroke-width="{sw}"'
    paint = f'r="{s // 2}" fill="{op["fill"] or "none"}" stroke="{op["stroke"]}"{width}'
    if op["id"] == "guide":
        return f'<circle class="guide" data-angle="{op["tag"]}" cx="{x}" cy="{y}" {paint}/>'
    return f'<circle class="{op["id"]}" cx="{x}" cy="{y}" {paint}><title>{html.escape(op["tag"])}</title></circle>'


def to_svg(ops: list[dict], images: dict[str, str] | None = None) -> str:
    """SVG backend of a display list. images: source tag -> base64 PNG for image ops (the list has no pixels)."""
    images = images or {}
    out: list[str] = []
    fs = None
    for op in ops:
        k = op["op"]
        if k == "canvas":
            W, H, fs = op["w"], op["h"], op["font_size"]
            mode = "" if op["id"] == HUD_ID else f' data-mode="{op["id"]}"'
            out += [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
                    f'font-family="{op["font_family"]}" font-size="{fs}" opacity="{op["opacity"]}"{mode}>',
                    f'<rect width="{W}" height="{H}" fill="{op["colour"]}"/>']
        elif k == "text":
            size = "" if op["size"] == fs else f' font-size="{op["size"]}"'
            anchor = "" if op["anchor"] == ANCHOR_START else f' text-anchor="{op["anchor"]}"'
            out.append(f'<text id="{op["id"]}" x="{op["x"]}" y="{op["y"]}"{size}{anchor} fill="{op["colour"]}">'
                       f'{html.escape(op["text"])}</text>')
        elif k == "shape":
            out.append(_svg_shape(op))
        elif k == "cue":
            out.append(f'<rect id="{op["id"]}" class="cue" data-direction="{html.escape(op["direction"])}" '
                       f'data-edge="{op["edge"]}" x="{op["x"]}" y="{op["y"]}" width="{op["w"]}" height="{op["h"]}" '
                       f'fill="{op["colour"]}"/>')
        elif k == "image":
            if op["source"] not in images:
                raise LayoutError(f"no pixels for image source {op['source']}")
            out.append(f'<image id="{op["id"]}" x="{op["x"]}" y="{op["y"]}" width="{op["w"]}" height="{op["h"]}" '
                       f'preserveAspectRatio="none" href="data:image/png;base64,{images[op["source"]]}" data-mode="{op["mode"]}"/>')
        elif k != "zone":
            raise LayoutError(f"unknown draw op {k}")
    out.append("</svg>")
    return "".join(out)


def render(st: GlassesState, tok: dict | None = None) -> str:
    ops = display_list(st, tok)
    images = {}
    if st.thermal is not None:
        images[SOURCE_THERMAL] = base64.b64encode(_png_gray(thermal_image(st.thermal, st.thermal_colour))).decode()
    return to_svg(ops, images)


def save(st: GlassesState, path: str) -> None:
    with io.open(path, "w", encoding="utf-8") as f:
        f.write(render(st))


ENGINE_LEVEL = {"fyi": "low", "attention": "medium", "confirmed": "confirmed"}


def moving_from_motion(motion: str | None) -> bool:
    """GlassesState.moving from the motion gate (oiris/modules/motion.py): walking or fast. Unknown (no IMU)
    counts as still."""
    return is_moving(str(motion or ""))


def state_from_result(res: dict, motion: str | None = None, cue_direction: str | None = None) -> GlassesState:
    """Map a C3 headless/engine result to the glasses state (signature IDs only; no identities). motion: the
    motion state (live snapshot()["output"]["motion"]); while moving, G2 pop-ups are hidden. cue_direction: the
    current cue's direction word (oiris/spatial/cues.py), shown as an edge marker even while moving."""
    lvl = ENGINE_LEVEL.get(str(res.get("level") or ""), None)
    sigs = list(res.get("signatures") or [])
    text = ("Signatures " + " ".join(sigs[:4])) if sigs else ""
    health = {k: (v["health"] if isinstance(v, dict) else str(v)) for k, v in (res.get("health") or {}).items()}
    return GlassesState(level=lvl, alert_text=text, health=health, moving=moving_from_motion(motion),
                        cue_direction=cue_direction)


# ---- M-16 enrollment mini-guide (FACE_MATCHING_MODULE_SPEC §7.3 step 3) ----------------------
# The glasses show only the next angle needed and a tick when a shot is accepted. No names, no
# face image, nothing in the clear centre, and never anything on a *match* (spec §6 "Never does").
GUIDE_ORDER = ("front", "left", "right", "up", "down")
GUIDE_WORDS = {"front": "Look straight ahead", "left": "Turn head left", "right": "Turn head right",
               "up": "Tilt head up", "down": "Tilt head down"}
GUIDE_ARROW = {"front": "", "left": "\u2190", "right": "\u2192", "up": "\u2191", "down": "\u2193"}


@dataclass
class EnrollGuide:
    needed: list[str]                 # angles still needed, in GUIDE_ORDER
    last_ok: bool | None = None        # result of the last submitted shot (None = no shot yet)
    last_reason: str = ""              # plain reason when the last shot was rejected
    complete: bool = False

    @property
    def next_angle(self) -> str | None:
        return self.needed[0] if self.needed else None


def guide_from_port(port, person_id: str, last=None) -> EnrollGuide:
    """Build the guide from an EnrollmentPort (local core or remote.py proxy) and the last
    ShotResult. Works with either Angle enums or plain strings."""
    have = {getattr(a, "value", a) for a in port.needed_angles(person_id)}
    needed = [a for a in GUIDE_ORDER if a in have]
    ok = None if last is None else bool(last.ok)
    reason = "" if last is None or last.ok else str(last.reason)
    return EnrollGuide(needed, ok, reason, complete=not needed)


def _short(reason: str, n: int) -> str:
    words = reason.replace(",", "").split()
    return " ".join(words[:n])


def enroll_guide_display_list(g: EnrollGuide, tok: dict | None = None) -> list[dict]:
    """The enrollment mini-guide as draw ops (see OP_KEYS), built like display_list."""
    tok = tok or T.load()
    check_layout(tok)
    W, H = tok["glasses"]["canvas"]
    z, c = zones(tok), tok["color"]
    maxw = tok["glasses"]["max_words_alert_line"]
    ops = [_canvas(tok, GUIDE_ID)] + _zone_ops(tok)
    if g.complete:
        line = GUIDE_COMPLETE
    elif g.last_ok is False and g.last_reason:
        line = _short(g.last_reason, maxw)
    else:
        a = g.next_angle or "front"
        line = (GUIDE_ARROW[a] + " " + GUIDE_WORDS[a]).strip()
    if len(line.split()) > maxw:
        raise LayoutError("guide line exceeds the word limit")
    check_text(line)
    x, y, w, h = z["G2"]
    ops.append(_text("G2", G2_TEXT_X, y + h - G2_BASELINE_UP, line, tok, c["text"]))
    if g.last_ok:   # tick for the accepted shot, top-right of G2, outside the clear centre
        ops.append(_shape_op("tick", "tick", W - TICK_FROM_RIGHT, y + TICK_DOWN, TICK_SIZE, "", c["health_ok"], TICK_STROKE))
    x, y, w, h = z["G4"]       # five progress dots: filled = captured
    for i, a in enumerate(GUIDE_ORDER):
        done = a not in g.needed
        ops.append(_shape_op("guide", "dot", x + DOT_INSET + i * DOT_PITCH, y + h / 2, DOT_SIZE, c["text"] if done else "",
                             c["text_muted"], HAIRLINE, a))
    return ops


def render_enroll_guide(g: EnrollGuide, tok: dict | None = None) -> str:
    return to_svg(enroll_guide_display_list(g, tok))
