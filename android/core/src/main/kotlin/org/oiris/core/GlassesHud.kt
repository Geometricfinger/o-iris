package org.oiris.core

import java.math.BigDecimal
import java.math.RoundingMode

/** The glasses display (zones G1-G6, Module Map 7.1 and 6.1) as a display list, ported from oiris/ui/glasses.py
 *  display_list / enroll_guide_display_list. The same state and the same C7 design-token JSON give exactly the
 *  same draw ops as Python (tests/vectors/glasses_hud.json, generated from the Python code): the zone layout and
 *  check_layout rules, the level mark shapes and colours, the word limit and vocabulary guard on the alert line,
 *  no G2 pop-up while moving, the health dots, the direction cue's edge marker (shown while moving too), and the
 *  enrollment mini-guide with its reason truncation.
 *  Pure Kotlin, no Android APIs: an Android Canvas backend draws these ops later (the SVG simulator is the other).
 *  Numbers: integer pixels are Long; anything else is a Double rounded to COORD_DECIMALS (half-even on the exact
 *  binary value, as Python's round), so an op equals its decoded vector, number types included. */
object GlassesHud {
    const val COORD_DECIMALS = 2
    val OP_KEYS: Map<String, List<String>> = mapOf(
        "canvas" to listOf("op", "id", "w", "h", "colour", "opacity", "font_family", "font_size"),
        "zone" to listOf("op", "id", "x", "y", "w", "h"),
        "text" to listOf("op", "id", "x", "y", "text", "size", "colour", "anchor"),
        "shape" to listOf("op", "id", "kind", "x", "y", "size", "fill", "stroke", "stroke_width", "tag"),
        "image" to listOf("op", "id", "x", "y", "w", "h", "source", "mode"),
        "cue" to listOf("op", "id", "direction", "edge", "x", "y", "w", "h", "colour"),
    )
    const val HUD_ID = "hud"
    const val GUIDE_ID = "enroll-guide"
    const val ANCHOR_START = "start"
    const val SOURCE_THERMAL = "thermal"
    val LEVELS = listOf("low", "medium", "confirmed")
    val ZONE_IDS = listOf("G1", "G2", "G3", "G4", "G5", "G6")

    // Zones (glasses.zones): fixed pixel boxes on the canvas; G3 is the token thermal window
    private const val G1_H = 28L
    private const val G2_Y = 32L; private const val G2_H = 36L
    private const val BOTTOM_UP = 28L                   // G4 and G5 top, above the canvas bottom
    private const val G4_X = 8L; private const val G4_W = 200L; private const val G4_H = 20L
    private const val G5_FROM_RIGHT = 160L; private const val G5_W = 152L; private const val G5_H = 20L
    private const val G6_X = 8L; private const val G6_Y = 72L; private const val G6_W = 160L; private const val G6_H = 24L

    // HUD geometry, relative to the zone each sits in (glasses.py constants of the same names)
    const val MARK_X = 16L; const val MARK_SIZE = 16L
    const val G1_TEXT_X = 32L; const val G1_BASELINE_UP = 7L
    const val G2_TEXT_X = 8L; const val G2_BASELINE_UP = 10L
    const val CUE_FROM_RIGHT = 150L; const val CUE_BASELINE_DOWN = 26L
    const val G5_BASELINE_UP = 4L; const val G6_BASELINE_UP = 6L
    const val DOT_INSET = 8L; const val DOT_PITCH = 18L; const val DOT_SIZE = 10L
    const val HAIRLINE = 1L; const val DOWN_STROKE = 2L
    const val TICK_FROM_RIGHT = 40L; const val TICK_DOWN = 6L; const val TICK_SIZE = 24L; const val TICK_STROKE = 3L
    val TICK_POINTS = listOf(0L to 12L, 8L to 20L, 24L to 2L)
    const val CUE_TEXT = "New sighting"
    const val VOICE_PREFIX = "Voice: "
    const val G6_LOCKED = "Locked"; const val G6_PANIC = "Panic armed"
    const val GUIDE_COMPLETE = "Template complete"
    const val GUIDE_DEFAULT_ANGLE = "front"

    // Direction cue (glasses.py CUE_*): an edge bar, thickness and length from the C7 spacing scale, body text colour.
    // "behind" spans the whole bottom edge, "below" is a short bar at its centre; "ahead" draws nothing.
    const val CUE_ID = "cue"
    const val CUE_THICK_STEP = 1                         // space_px index; the length is the last (largest) step
    val CUE_EDGES = mapOf("at left" to "left", "at right" to "right", "above" to "top", "below" to "bottom", "behind" to "bottom")
    const val CUE_NONE = "ahead"
    const val CUE_SPAN_WHOLE = "behind"

    // Enrollment mini-guide (FACE_MATCHING_MODULE_SPEC 7.3 step 3): next angle and a tick only, no names.
    val GUIDE_ORDER = listOf("front", "left", "right", "up", "down")
    val GUIDE_WORDS = mapOf("front" to "Look straight ahead", "left" to "Turn head left", "right" to "Turn head right",
        "up" to "Tilt head up", "down" to "Tilt head down")
    val GUIDE_ARROW = mapOf("front" to "", "left" to "←", "right" to "→", "up" to "↑", "down" to "↓")
    private val MOVING = setOf("walking", "fast")

    class LayoutError(message: String) : IllegalArgumentException(message)
    class VocabularyViolation(message: String) : IllegalArgumentException(message)

    /** glasses.GlassesState; `thermal` = a thermal frame is present (the list carries no pixels). */
    data class State(val level: String? = null, val alertText: String = "", val newSighting: Boolean = false,
                     val thermal: Boolean = false, val thermalColour: Boolean = false, val health: Map<String, String> = emptyMap(),
                     val voiceTier: String = "low", val locked: Boolean = false, val panicArmed: Boolean = false,
                     val moving: Boolean = false, val cueDirection: String? = null)

    /** glasses.EnrollGuide: angles still needed (GUIDE_ORDER), the last shot's result (null = none yet). */
    data class Guide(val needed: List<String>, val lastOk: Boolean? = null, val lastReason: String = "", val complete: Boolean = false) {
        val nextAngle: String? get() = needed.firstOrNull()
    }

    sealed class Op {
        abstract val id: String
        abstract fun toMap(): Map<String, Any?>
    }
    data class Canvas(override val id: String, val w: Long, val h: Long, val colour: String, val opacity: Number,
                      val fontFamily: String, val fontSize: Number) : Op() {
        override fun toMap() = mapOf("op" to "canvas", "id" to id, "w" to w, "h" to h, "colour" to colour, "opacity" to opacity,
            "font_family" to fontFamily, "font_size" to fontSize)
    }
    data class Zone(override val id: String, val x: Long, val y: Long, val w: Long, val h: Long) : Op() {
        override fun toMap() = mapOf("op" to "zone", "id" to id, "x" to x, "y" to y, "w" to w, "h" to h)
    }
    data class Text(override val id: String, val x: Number, val y: Number, val text: String, val size: Number,
                    val colour: String, val anchor: String) : Op() {
        override fun toMap() = mapOf("op" to "text", "id" to id, "x" to x, "y" to y, "text" to text, "size" to size,
            "colour" to colour, "anchor" to anchor)
    }
    /** kind circle / triangle / diamond (level mark), dot (diameter `size`) or tick (TICK_POINTS); "" = no fill / stroke. */
    data class Shape(override val id: String, val kind: String, val x: Number, val y: Number, val size: Number, val fill: String,
                     val stroke: String, val strokeWidth: Number, val tag: String) : Op() {
        override fun toMap() = mapOf("op" to "shape", "id" to id, "kind" to kind, "x" to x, "y" to y, "size" to size,
            "fill" to fill, "stroke" to stroke, "stroke_width" to strokeWidth, "tag" to tag)
    }
    data class Image(override val id: String, val x: Long, val y: Long, val w: Long, val h: Long, val source: String,
                     val mode: String) : Op() {
        override fun toMap() = mapOf("op" to "image", "id" to id, "x" to x, "y" to y, "w" to w, "h" to h, "source" to source,
            "mode" to mode)
    }
    /** The direction cue: a filled edge marker; `direction` is the cue's word, `edge` left / right / top / bottom. */
    data class Cue(override val id: String, val direction: String, val edge: String, val x: Number, val y: Number,
                   val w: Number, val h: Number, val colour: String) : Op() {
        override fun toMap() = mapOf("op" to "cue", "id" to id, "direction" to direction, "edge" to edge, "x" to x, "y" to y,
            "w" to w, "h" to h, "colour" to colour)
    }

    /** The C7 tokens the display reads, from the design-token JSON (oiris/ui/design_tokens.json, decoded by Json.kt). */
    @Suppress("UNCHECKED_CAST")
    class Tokens(t: Map<String, Any?>) {
        private val glasses = t["glasses"] as Map<String, Any?>
        private val type = t["type"] as Map<String, Any?>
        val colour = t["color"] as Map<String, Any?>
        val alertColour = colour["alert"] as Map<String, String>
        val levels = t["alert_levels"] as Map<String, Map<String, Any?>>
        val canvas = glasses["canvas"] as List<Number>
        val thermalWindow = glasses["thermal_window"] as List<Number>
        val brightness = glasses["default_brightness"] as Number
        val centreClear = (glasses["centre_clear_fraction"] as Number).toDouble()
        val maxWords = glasses["max_words_alert_line"] as Number
        val fontFamily = type["font_family"] as String
        val fontSize = type["glasses_body"] as Number
        val space = t["space_px"] as List<Number>
        fun c(k: String) = colour[k] as String

        companion object { fun fromJson(text: String) = Tokens(Json.decode(text) as Map<String, Any?>) }
    }

    /** A display-list number: integer pixels unchanged, a Double rounded to COORD_DECIMALS (glasses._n). */
    fun n(v: Number): Number = if (v is Double || v is Float) roundCoord(v.toDouble()) else v.toLong()
    fun roundCoord(v: Double): Double = BigDecimal(v).setScale(COORD_DECIMALS, RoundingMode.HALF_EVEN).toDouble()

    /** glasses.moving_from_motion: walking or fast; unknown (no IMU) and anything else counts as still. */
    fun movingFromMotion(motion: String?): Boolean = (motion ?: "") in MOVING

    private fun W(t: Tokens) = t.canvas[0].toLong()
    private fun H(t: Tokens) = t.canvas[1].toLong()

    /** glasses.zones, in G1..G6 order (x, y, w, h). */
    fun zones(t: Tokens): LinkedHashMap<String, LongArray> {
        val w = W(t); val h = H(t); val tw = t.thermalWindow.map { it.toLong() }
        return linkedMapOf("G1" to longArrayOf(0, 0, w, G1_H), "G2" to longArrayOf(0, G2_Y, w, G2_H),
            "G3" to longArrayOf(tw[0], tw[1], tw[2], tw[3]), "G4" to longArrayOf(G4_X, h - BOTTOM_UP, G4_W, G4_H),
            "G5" to longArrayOf(w - G5_FROM_RIGHT, h - BOTTOM_UP, G5_W, G5_H), "G6" to longArrayOf(G6_X, G6_Y, G6_W, G6_H))
    }

    private fun centre(t: Tokens): DoubleArray {
        val w = W(t).toDouble(); val h = H(t).toDouble(); val f = t.centreClear
        return doubleArrayOf(w * (1 - f) / 2, h * (1 - f) / 2, w * f, h * f)
    }

    private fun overlap(a: DoubleArray, b: DoubleArray) =
        a[0] < b[0] + b[2] && b[0] < a[0] + a[2] && a[1] < b[1] + b[3] && b[1] < a[1] + a[3]

    private fun d(r: LongArray) = DoubleArray(4) { r[it].toDouble() }

    /** glasses.check_layout: whole pixels, every zone inside the canvas, none but G3 in the clear centre, no overlaps. */
    fun checkLayout(t: Tokens) {
        if (!(t.canvas + t.thermalWindow).all { it is Long || it is Int }) throw LayoutError("canvas and thermal window must be whole pixels")
        val z = zones(t); val c = centre(t); val w = W(t); val h = H(t)
        for ((k, r) in z) {
            if (r[0] < 0 || r[1] < 0 || r[0] + r[2] > w || r[1] + r[3] > h) throw LayoutError("$k outside canvas")
            if (k != "G3" && overlap(d(r), c)) throw LayoutError("$k intrudes on the clear centre")
        }
        val ks = z.keys.toList()
        for (i in ks.indices) for (j in i + 1 until ks.size)
            if (overlap(d(z[ks[i]]!!), d(z[ks[j]]!!))) throw LayoutError("${ks[i]} overlaps ${ks[j]}")
    }

    /** Python str.isspace (Unicode White_Space plus the C0 separators); str.split() splits on these. */
    private fun isPySpace(c: Char) = Character.isWhitespace(c) || Character.isSpaceChar(c) || c == '\u0085'
    fun words(s: String): List<String> = s.pySplit { isPySpace(it) }
    private fun String.pySplit(sep: (Char) -> Boolean): List<String> {
        val out = ArrayList<String>(); val cur = StringBuilder()
        for (ch in this) if (sep(ch)) { if (cur.isNotEmpty()) { out.add(cur.toString()); cur.clear() } } else cur.append(ch)
        if (cur.isNotEmpty()) out.add(cur.toString())
        return out
    }

    /** Python string order (code points), not UTF-16 units. */
    private val CODE_POINT_ORDER = Comparator<String> { a, b ->
        val x = a.codePoints().toArray(); val y = b.codePoints().toArray()
        var r = x.size.compareTo(y.size)
        for (i in 0 until minOf(x.size, y.size)) if (x[i] != y[i]) { r = x[i].compareTo(y[i]); break }
        r
    }

    /** guard.vocabulary.check_text, with VoiceLink.bannedRegex built from schema/vocabulary.json. */
    private fun checkText(text: String, banned: Regex) {
        val v = banned.findAll(text).map { it.value }.toList()
        if (v.isNotEmpty()) throw VocabularyViolation("banned wording $v in output")
    }

    private fun canvasOp(t: Tokens, id: String) = Canvas(id, W(t), H(t), t.c("bg"), n(t.brightness), t.fontFamily, n(t.fontSize))
    private fun zoneOps(t: Tokens) = zones(t).map { (k, r) -> Zone(k, r[0], r[1], r[2], r[3]) }
    private fun text(id: String, x: Number, y: Number, s: String, t: Tokens, colour: String) =
        Text(id, n(x), n(y), s, n(t.fontSize), colour, ANCHOR_START)
    private fun shape(id: String, kind: String, x: Number, y: Number, size: Number, fill: String, stroke: String = "",
                      strokeWidth: Long = 0, tag: String = "") = Shape(id, kind, n(x), n(y), n(size), fill, stroke, n(strokeWidth), tag)

    /** glasses.display_list: the HUD for a state. Throws LayoutError / VocabularyViolation where Python raises. */
    fun displayList(st: State, t: Tokens, banned: Regex): List<Op> {
        checkLayout(t)
        val w = W(t); val z = zones(t)
        val ops = mutableListOf<Op>(canvasOp(t, HUD_ID)); ops += zoneOps(t)
        if (!st.level.isNullOrEmpty()) {
            if (st.level !in LEVELS) throw LayoutError("unknown level")
            val lv = t.levels[st.level]!!; val col = t.alertColour[st.level]!!
            val g1 = z["G1"]!!
            ops += shape("G1", lv["shape"] as String, MARK_X, g1[1] + g1[3] / 2.0, MARK_SIZE, col)
            ops += text("G1", G1_TEXT_X, g1[1] + g1[3] - G1_BASELINE_UP, lv["word"] as String, t, col)
        }
        if (st.alertText.isNotEmpty() && !st.moving) {   // no pop-ups during movement; G1 strip still shows the level
            if (words(st.alertText).size > t.maxWords.toDouble()) throw LayoutError("alert line exceeds the word limit")
            checkText(st.alertText, banned)
            val g2 = z["G2"]!!
            ops += text("G2", G2_TEXT_X, g2[1] + g2[3] - G2_BASELINE_UP, st.alertText, t, t.c("text"))
        }
        if (st.newSighting) ops += text("G2cue", w - CUE_FROM_RIGHT, z["G2"]!![1] + CUE_BASELINE_DOWN, CUE_TEXT, t, t.c("text_muted"))
        if (st.thermal) {
            val g3 = z["G3"]!!
            ops += Image("G3", g3[0], g3[1], g3[2], g3[3], SOURCE_THERMAL, if (st.thermalColour) "colour" else "grayscale")
        }
        val g4 = z["G4"]!!
        for ((i, e) in st.health.entries.sortedWith(compareBy(CODE_POINT_ORDER) { it.key }).withIndex()) {
            val (name, hs) = e
            val col = when (hs) { "ok" -> t.c("health_ok"); "degraded" -> t.c("health_degraded"); else -> t.c("health_down") }
            ops += shape("G4", "dot", g4[0] + DOT_INSET + i * DOT_PITCH, g4[1] + g4[3] / 2.0, DOT_SIZE, if (hs == "ok") col else "",
                col, if (hs == "ok") HAIRLINE else DOWN_STROKE, "$name: $hs")
        }
        val g5 = z["G5"]!!
        ops += text("G5", g5[0], g5[1] + g5[3] - G5_BASELINE_UP, VOICE_PREFIX + st.voiceTier, t, t.c("text_muted"))
        val g6 = z["G6"]!!
        val g6Text = ((if (st.locked) G6_LOCKED else "") + (if (st.panicArmed) " $G6_PANIC" else "")).trim { isPySpace(it) }
        if (g6Text.isNotEmpty()) ops += text("G6", g6[0], g6[1] + g6[3] - G6_BASELINE_UP, g6Text, t, t.c("text"))
        // The cue shows while moving too: a steady edge marker, not a pop-up (Module Map 6.1 forbids only pop-ups).
        val cue = st.cueDirection
        if (!cue.isNullOrEmpty() && cue != CUE_NONE) ops += cueOp(cue, t)
        return ops
    }

    /** glasses._cue_op: the edge marker for a direction word; LayoutError for an unknown word or bad sizes. */
    private fun cueOp(direction: String, t: Tokens): Cue {
        val edge = CUE_EDGES[direction] ?: throw LayoutError("unknown cue direction")
        val w = W(t); val h = H(t)
        val th = t.space[CUE_THICK_STEP]; val len = t.space.last()
        if (!(th is Long || th is Int) || !(len is Long || len is Int)) throw LayoutError("cue sizes must be whole pixels")
        val k = th.toLong(); val ln = len.toLong()
        val r: List<Number> = when {
            direction == CUE_SPAN_WHOLE -> listOf(0L, h - k, w, k)
            edge == "left" || edge == "right" -> listOf(if (edge == "left") 0L else w - k, (h - ln) / 2.0, k, ln)
            else -> listOf((w - ln) / 2.0, if (edge == "top") 0L else h - k, ln, k)
        }
        if (overlap(DoubleArray(4) { r[it].toDouble() }, centre(t))) throw LayoutError("cue intrudes on the clear centre")
        return Cue(CUE_ID, direction, edge, n(r[0]), n(r[1]), n(r[2]), n(r[3]), t.c("text"))
    }

    /** glasses._short: the reason without commas, cut to its first n words. */
    fun short(reason: String, n: Int): String = words(reason.replace(",", "")).take(n).joinToString(" ")

    /** glasses.enroll_guide_display_list: the line (complete / rejected reason / next angle), tick, progress dots. */
    fun guideDisplayList(g: Guide, t: Tokens, banned: Regex): List<Op> {
        checkLayout(t)
        val w = W(t); val z = zones(t)
        val ops = mutableListOf<Op>(canvasOp(t, GUIDE_ID)); ops += zoneOps(t)
        val line = when {
            g.complete -> GUIDE_COMPLETE
            g.lastOk == false && g.lastReason.isNotEmpty() -> short(g.lastReason, t.maxWords.toInt())
            else -> {
                val a = g.nextAngle?.takeIf { it.isNotEmpty() } ?: GUIDE_DEFAULT_ANGLE
                require(a in GUIDE_WORDS) { "unknown angle $a" }
                (GUIDE_ARROW[a] + " " + GUIDE_WORDS[a]).trim { isPySpace(it) }
            }
        }
        if (words(line).size > t.maxWords.toDouble()) throw LayoutError("guide line exceeds the word limit")
        checkText(line, banned)
        val g2 = z["G2"]!!
        ops += text("G2", G2_TEXT_X, g2[1] + g2[3] - G2_BASELINE_UP, line, t, t.c("text"))
        if (g.lastOk == true)   // tick for the accepted shot, top-right of G2, outside the clear centre
            ops += shape("tick", "tick", w - TICK_FROM_RIGHT, g2[1] + TICK_DOWN, TICK_SIZE, "", t.c("health_ok"), TICK_STROKE)
        val g4 = z["G4"]!!       // five progress dots: filled = captured
        for ((i, a) in GUIDE_ORDER.withIndex()) {
            val done = a !in g.needed
            ops += shape("guide", "dot", g4[0] + DOT_INSET + i * DOT_PITCH, g4[1] + g4[3] / 2.0, DOT_SIZE,
                if (done) t.c("text") else "", t.c("text_muted"), HAIRLINE, a)
        }
        return ops
    }
}
