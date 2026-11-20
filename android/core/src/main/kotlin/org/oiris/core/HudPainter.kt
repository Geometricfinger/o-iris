package org.oiris.core

/** What a glasses display list is drawn on: the primitives of oiris/ui/paint.py, one method per draw-call kind.
 *  Numbers are logical canvas pixels (the frame's w x h); colours are the ops' strings, "" = no paint (parse them
 *  with [HudPainter.colour]). The app implements it on android.graphics.Canvas (AndroidHudCanvas); [RecordingCanvas]
 *  records the calls for the JVM checks. */
interface HudCanvas {
    /** The canvas op: logical size, background colour, global opacity (the brightness cap), font family. Always first. */
    fun frame(w: Double, h: Double, background: String, opacity: Double, fontFamily: String)
    fun rect(x: Double, y: Double, w: Double, h: Double, fill: String, stroke: String, strokeWidth: Double, radius: Double)
    fun circle(cx: Double, cy: Double, r: Double, fill: String, stroke: String, strokeWidth: Double)
    /** A filled closed polygon; points = x0, y0, x1, y1, ... */
    fun polygon(points: List<Double>, fill: String)
    /** An open unfilled polyline. */
    fun polyline(points: List<Double>, stroke: String, strokeWidth: Double)
    /** Text with its baseline at y; anchor start | middle | end. */
    fun text(x: Double, y: Double, size: Double, colour: String, weight: String, anchor: String, text: String)
    /** A placeholder the canvas fills with its own pixels for `source` (thermal) in `mode` (grayscale | colour). */
    fun image(x: Double, y: Double, w: Double, h: Double, source: String, mode: String)
}

/** Executes a display list on a [HudCanvas], the twin of oiris/ui/paint.py: each op becomes the call glasses.to_svg
 *  implies for it, same order, same values (tests/vectors/hud_paint.json, from the Python code). Zones draw nothing.
 *  The first op must be a canvas or nothing is drawn; a later canvas, an unknown op or shape kind and an op with a
 *  missing or mistyped field are skipped, so an untrusted frame (HudFrame.parse) never throws here. Pure Kotlin. */
object HudPainter {
    const val WEIGHT_NORMAL = "normal"
    val ANCHORS = listOf(GlassesHud.ANCHOR_START, "middle", "end")
    const val NO_PAINT = ""
    const val NO_STROKE_WIDTH = 0.0
    const val SQUARE = 0.0
    private const val HALF = 2.0
    val MARK_KINDS = listOf("circle", "triangle", "diamond")
    val CALL_KINDS = listOf("frame", "rect", "circle", "polygon", "polyline", "text", "image")
    private val HEX = Regex("#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})")
    private const val SHORT_HEX = 3
    private const val OPAQUE = 0xFF000000.toInt()
    private const val ZERO = 0.0
    private const val ROUND_UP = 0.5

    /** paint.parse_colour: ARGB of "#rgb" / "#rrggbb" (always opaque), or null: no paint. */
    fun colour(s: String?): Int? {
        val m = s?.let { HEX.matchEntire(it) } ?: return null
        var h = m.groupValues[1]
        if (h.length == SHORT_HEX) h = h.map { "$it$it" }.joinToString("")
        return OPAQUE or h.toInt(16)
    }

    /** Python float `a // b` (CPython float_floor_div), which the SVG backend uses for dot radii and tick points. */
    fun pyFloorDiv(a: Double, b: Double): Double {
        var mod = a % b                        // Java's double % is C fmod
        var div = (a - mod) / b
        if (mod != ZERO) { if ((b < ZERO) != (mod < ZERO)) { mod += b; div -= 1.0 } }
        if (div == ZERO) return Math.copySign(ZERO, a / b)
        var floor = Math.floor(div)
        if (div - floor > ROUND_UP) floor += 1.0
        return floor
    }

    private fun num(op: Map<String, Any?>, k: String): Double? {
        val v = op[k]
        val d = when (v) { is Long -> v.toDouble(); is Int -> v.toDouble(); is Double -> v; is Float -> v.toDouble(); else -> return null }
        return if (d.isFinite()) d else null
    }

    private fun str(op: Map<String, Any?>, k: String): String? = op[k] as? String

    private fun frame(op: Map<String, Any?>, c: HudCanvas): Boolean {
        val w = num(op, "w"); val h = num(op, "h"); val o = num(op, "opacity")
        val bg = str(op, "colour"); val ff = str(op, "font_family")
        if (w == null || h == null || o == null || bg == null || ff == null) return false
        c.frame(w, h, bg, o, ff)
        return true
    }

    private fun text(op: Map<String, Any?>, c: HudCanvas) {
        val x = num(op, "x") ?: return; val y = num(op, "y") ?: return; val size = num(op, "size") ?: return
        val colour = str(op, "colour") ?: return; val anchor = str(op, "anchor") ?: return; val t = str(op, "text") ?: return
        c.text(x, y, size, colour, WEIGHT_NORMAL, if (anchor in ANCHORS) anchor else GlassesHud.ANCHOR_START, t)
    }

    private fun shape(op: Map<String, Any?>, c: HudCanvas) {
        val x = num(op, "x") ?: return; val y = num(op, "y") ?: return
        val s = num(op, "size") ?: return; val sw = num(op, "stroke_width") ?: return
        val kind = str(op, "kind") ?: return; val fill = str(op, "fill") ?: return; val stroke = str(op, "stroke") ?: return
        when (kind) {
            "circle" -> c.circle(x, y, s / HALF, fill, NO_PAINT, NO_STROKE_WIDTH)   // filled; the op's stroke is not drawn
            "triangle" -> c.polygon(listOf(x, y - s / HALF, x + s / HALF, y + s / HALF, x - s / HALF, y + s / HALF), fill)
            "diamond" -> c.polygon(listOf(x, y - s / HALF, x + s / HALF, y, x, y + s / HALF, x - s / HALF, y), fill)
            "dot" -> c.circle(x, y, pyFloorDiv(s, HALF), fill, stroke, sw)
            "tick" -> {
                val size = GlassesHud.TICK_SIZE.toDouble()
                c.polyline(GlassesHud.TICK_POINTS.flatMap { (dx, dy) ->
                    listOf(x + pyFloorDiv(dx * s, size), y + pyFloorDiv(dy * s, size)) }, stroke, sw)
            }
        }
    }

    private fun cue(op: Map<String, Any?>, c: HudCanvas) {
        val x = num(op, "x") ?: return; val y = num(op, "y") ?: return; val w = num(op, "w") ?: return; val h = num(op, "h") ?: return
        val colour = str(op, "colour") ?: return
        c.rect(x, y, w, h, colour, NO_PAINT, NO_STROKE_WIDTH, SQUARE)
    }

    private fun image(op: Map<String, Any?>, c: HudCanvas) {
        val x = num(op, "x") ?: return; val y = num(op, "y") ?: return; val w = num(op, "w") ?: return; val h = num(op, "h") ?: return
        val source = str(op, "source") ?: return; val mode = str(op, "mode") ?: return
        c.image(x, y, w, h, source, mode)
    }

    /** Paints a validated display list: HudFrame.ops, or GlassesHud ops through [paintHud]. Never throws on bad ops. */
    fun paint(ops: List<Map<String, Any?>>, canvas: HudCanvas) {
        val first = ops.firstOrNull() ?: return
        if (first["op"] != "canvas" || !frame(first, canvas)) return
        for (op in ops.subList(1, ops.size)) when (op["op"]) {
            "text" -> text(op, canvas)
            "shape" -> shape(op, canvas)
            "cue" -> cue(op, canvas)
            "image" -> image(op, canvas)
        }
    }

    /** Paints GlassesHud.displayList / guideDisplayList ops. */
    fun paintHud(ops: List<GlassesHud.Op>, canvas: HudCanvas) = paint(ops.map { it.toMap() }, canvas)
}

/** A [HudCanvas] that records each call as a list (kind first, then the arguments in order), as paint.py's tuples. */
class RecordingCanvas : HudCanvas {
    val calls = mutableListOf<List<Any>>()
    override fun frame(w: Double, h: Double, background: String, opacity: Double, fontFamily: String) {
        calls += listOf("frame", w, h, background, opacity, fontFamily)
    }
    override fun rect(x: Double, y: Double, w: Double, h: Double, fill: String, stroke: String, strokeWidth: Double, radius: Double) {
        calls += listOf("rect", x, y, w, h, fill, stroke, strokeWidth, radius)
    }
    override fun circle(cx: Double, cy: Double, r: Double, fill: String, stroke: String, strokeWidth: Double) {
        calls += listOf("circle", cx, cy, r, fill, stroke, strokeWidth)
    }
    override fun polygon(points: List<Double>, fill: String) { calls += listOf("polygon", points, fill) }
    override fun polyline(points: List<Double>, stroke: String, strokeWidth: Double) { calls += listOf("polyline", points, stroke, strokeWidth) }
    override fun text(x: Double, y: Double, size: Double, colour: String, weight: String, anchor: String, text: String) {
        calls += listOf("text", x, y, size, colour, weight, anchor, text)
    }
    override fun image(x: Double, y: Double, w: Double, h: Double, source: String, mode: String) {
        calls += listOf("image", x, y, w, h, source, mode)
    }
}
