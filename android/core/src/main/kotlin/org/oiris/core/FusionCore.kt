package org.oiris.core

import java.math.BigDecimal
import java.math.RoundingMode

/** Fusion v2, the twin of oiris/fusion/stack.py (+ the C1 Score rules of oiris/schema.py). Scores -> six families
 *  (in a family, the one score with the largest value * confidence) -> L2 logistic regression with a missing-family
 *  mask. Evidence rule: ATTENTION needs >= minFamilies active families, one family is FYI, more than maxUnhealthy
 *  unhealthy families suppress the index. Outputs anomaly signatures only; no field names a class of person.
 *
 *  Bad input, as Python does it (never thrown here):
 *  - value / confidence: clamped to [0, 1] like schema.clamp01. NaN, a non-number or an unparsable string is 0,
 *    +inf is 1, a numeric string is parsed like Python float(). Out of range is clamped, not rejected.
 *  - health: only "ok", "degraded", "failed". Anything else makes Python's Score() raise, so that score is
 *    rejected: [score] returns null and it never reaches [fuse].
 *  - a module not in [SCORE_FAMILY] (or not a string) is ignored by fuse.
 *  - weights: [FusionModel.fromJson] is null where Python's from_json raises; [loadWeights] then falls back.
 *  Checked against tests/vectors/fusion.json (scripts/make_android_vectors.py). */
object FusionCore {
    val FAMILIES = listOf("optical", "thermal", "rf", "vital", "eye_liveness", "voice")
    val SCORE_FAMILY = mapOf(
        "S_glint" to "optical", "S_nir_emit" to "optical", "S_motion" to "optical", "S_newobj" to "optical",
        "S_thermal" to "thermal",
        "S_rf" to "rf", "S_rf_drift" to "rf", "S_wifi" to "rf",
        "S_radar" to "vital", "S_live" to "vital",
        "S_eye" to "eye_liveness",
        "S_voice" to "voice",
    )
    val SCORE_SIGNATURE = mapOf(
        "S_glint" to "A1", "S_nir_emit" to "A2", "S_thermal" to "A3", "S_rf" to "A4", "S_radar" to "A5",
        "S_motion" to "A6", "S_eye" to "A7", "S_voice" to "A8", "S_rf_drift" to "A9", "S_newobj" to "A10",
        "S_live" to "A5", "S_wifi" to "A4",
    )
    const val FAMILY_THRESHOLD = 0.5
    const val MIN_FAMILIES = 2
    const val MAX_UNHEALTHY = 1
    const val INDEX_DECIMALS = 6
    /** Families whose joint activity adds the A11 signature (>= A11_MIN of them). */
    val A11_FAMILIES = setOf("vital", "eye_liveness", "voice", "thermal")
    const val A11_MIN = 2
    const val A11 = "A11"
    val DEFAULT_W = doubleArrayOf(1.2, 1.0, 0.8, 1.0, 1.0, 0.6)
    val DEFAULT_W_MASK = DoubleArray(6)
    const val DEFAULT_B = -2.5
    const val DEFAULT_VERSION = "prior-0"
    /** Missing-family mask values (m in the logit), and the confidence that never counts. */
    const val MISSING = 1.0
    const val PRESENT = 0.0
    const val NO_CONFIDENCE = 0.0

    enum class Health(val wire: String) { OK("ok"), DEGRADED("degraded"), FAILED("failed");
        companion object { fun of(x: Any?): Health? = values().firstOrNull { it.wire == x } } }
    enum class Level(val wire: String) { FYI("fyi"), ATTENTION("attention"), CONFIRMED("confirmed") }

    /** A C1 score after the Score() rules: value and confidence already clamped. module null = not a string. */
    data class Score(val module: String?, val value: Double, val confidence: Double, val health: Health = Health.OK)

    /** schema.clamp01. */
    fun clamp01(x: Any?): Double {
        val d = Py.float(x) ?: return 0.0
        return if (d.isNaN()) 0.0 else if (d < 0) 0.0 else if (d > 1) 1.0 else d
    }

    /** Score(module, value, confidence, health), or null where Python raises (health not a known name). */
    fun score(module: Any?, value: Any?, confidence: Any?, health: Any? = "ok"): Score? {
        val h = (health as? Health) ?: Health.of(health) ?: return null
        return Score(module as? String, clamp01(value), clamp01(confidence), h)
    }

    class FusionModel(val w: DoubleArray = DEFAULT_W.copyOf(), val wMask: DoubleArray = DEFAULT_W_MASK.copyOf(),
                      val b: Double = DEFAULT_B, val version: String = DEFAULT_VERSION) {
        /** b + sum(w * x * (1 - m)) + sum(w_mask * m), summed left to right like NumPy over six values. */
        fun logit(x: DoubleArray, m: DoubleArray): Double {
            var s1 = 0.0; var s2 = 0.0
            for (i in 0 until 6) s1 += w[i] * x[i] * (1 - m[i])
            for (i in 0 until 6) s2 += wMask[i] * m[i]
            return b + s1 + s2
        }

        fun predict(x: DoubleArray, m: DoubleArray): Double = 1 / (1 + Math.exp(-logit(x, m)))

        fun toJson(): String = Json.encode(mapOf("w" to w.toList(), "w_mask" to wMask.toList(), "b" to b, "version" to version))

        companion object {
            /** FusionModel.from_json: null where Python raises (bad JSON, missing key, shape != 6, non-finite). */
            fun fromJson(s: String?): FusionModel? {
                val d = try { Json.decode(s ?: return null) } catch (e: RuntimeException) { return null } as? Map<*, *> ?: return null
                if (!d.containsKey("w") || !d.containsKey("w_mask") || !d.containsKey("b")) return null
                val w = vec6(d["w"]) ?: return null
                val wm = vec6(d["w_mask"]) ?: return null
                val b = Py.float(d["b"]) ?: return null
                if (w.any { !it.isFinite() } || wm.any { !it.isFinite() } || !b.isFinite()) return null
                return FusionModel(w, wm, b, if (d.containsKey("version")) Py.str(d["version"]) else "?")
            }

            private fun vec6(v: Any?): DoubleArray? {
                val l = v as? List<*> ?: return null
                if (l.size != 6) return null
                return DoubleArray(6) { Py.float(l[it]) ?: return null }
            }
        }
    }

    /** load_weights: model-load failures never crash; fall back to the given model (or the prior). */
    fun loadWeights(json: String?, fallback: FusionModel? = null): FusionModel =
        FusionModel.fromJson(json) ?: fallback ?: FusionModel()

    class FamilyValues(val x: DoubleArray, val mask: DoubleArray, val contrib: Map<String, List<String>>, val unhealthy: List<String>)

    fun familyValues(scores: List<Score>): FamilyValues {
        val vals = DoubleArray(6); val conf = DoubleArray(6); val mask = DoubleArray(6) { MISSING }
        val contrib = FAMILIES.associateWith { mutableListOf<String>() }
        val unhealthy = mutableSetOf<String>()
        for (s in scores) {
            val module = s.module ?: continue
            val fam = SCORE_FAMILY[module] ?: continue
            val i = FAMILIES.indexOf(fam)
            if (s.health == Health.FAILED) { unhealthy += fam; continue }
            val weighted = s.value * s.confidence
            mask[i] = PRESENT
            if (conf[i] == NO_CONFIDENCE || weighted > vals[i] * conf[i]) { vals[i] = s.value; conf[i] = s.confidence }
            if (s.value >= FAMILY_THRESHOLD && s.confidence > NO_CONFIDENCE) contrib.getValue(fam) += module
        }
        // a family with at least one healthy score is not unhealthy
        return FamilyValues(vals, mask, contrib, unhealthy.filter { mask[FAMILIES.indexOf(it)] == MISSING }.sorted())
    }

    /** advisory: Kotlin only, not in Python. A fallback node (the phone alone: optical + RF) is advisory; the
     *  caller says so, as the app's Fusion.kt does. Every other field is fuse()'s output dict. */
    data class Result(val anomalyIndex: Double, val level: Level?, val familiesActive: List<String>,
                      val familiesMissing: List<String>, val familiesUnhealthy: List<String>, val suppressed: Boolean,
                      val signatures: List<String>, val modelVersion: String, val advisory: Boolean) {
        /** The same keys and values as Python's fuse() dict (advisory left out). */
        fun toMap(): Map<String, Any?> = mapOf("anomaly_index" to anomalyIndex, "level" to level?.wire,
            "families_active" to familiesActive, "families_missing" to familiesMissing,
            "families_unhealthy" to familiesUnhealthy, "suppressed" to suppressed, "signatures" to signatures,
            "model_version" to modelVersion)
    }

    fun fuse(scores: List<Score>, model: FusionModel? = null, minFamilies: Int = MIN_FAMILIES,
             maxUnhealthy: Int = MAX_UNHEALTHY, fallbackNode: Boolean = false): Result {
        val mdl = model ?: FusionModel()
        val fv = familyValues(scores)
        val active = FAMILIES.filter { fv.contrib.getValue(it).isNotEmpty() }
        val suppressed = fv.unhealthy.size > maxUnhealthy
        val idx = if (suppressed) 0.0 else mdl.predict(fv.x, fv.mask)
        val level = when {
            suppressed -> null
            active.size >= minFamilies -> Level.ATTENTION
            active.size == 1 -> Level.FYI
            else -> null
        }
        val sigs = scores.filter { s -> s.module != null && s.module in SCORE_SIGNATURE && s.value >= FAMILY_THRESHOLD && s.health != Health.FAILED }
            .map { SCORE_SIGNATURE.getValue(it.module!!) }.toSet().sortedBy { it.substring(1).toInt() }.toMutableList()
        if (active.count { it in A11_FAMILIES } >= A11_MIN) sigs += A11
        return Result(round6(idx), level, active, FAMILIES.filterIndexed { i, _ -> fv.mask[i] == MISSING }, fv.unhealthy,
            suppressed, sigs, mdl.version, fallbackNode)
    }

    /** Python round(x, 6): the exact binary value rounded half-even, as CPython does. */
    fun round6(x: Double): Double =
        if (!x.isFinite()) x else BigDecimal(x).setScale(INDEX_DECIMALS, RoundingMode.HALF_EVEN).toDouble()
}

/** The few Python value rules the twins need: float() of a JSON value, str()/repr() of one. */
object Py {
    private val FLOAT_RE = Regex("[+-]?(?:(?:\\d(?:_?\\d)*)(?:\\.(?:\\d(?:_?\\d)*)?)?|\\.\\d(?:_?\\d)*)(?:[eE][+-]?\\d(?:_?\\d)*)?")
    private val SPECIAL_RE = Regex("[+-]?(?:inf|infinity|nan)", RegexOption.IGNORE_CASE)

    /** Python float(x) for a JSON value; null where float() raises (None, list, dict, an unparsable string). */
    fun float(x: Any?): Double? = when (x) {
        is Boolean -> if (x) 1.0 else 0.0
        is Number -> x.toDouble()
        is String -> {
            val t = x.trim()
            when {
                FLOAT_RE.matches(t) -> t.replace("_", "").toDouble()
                SPECIAL_RE.matches(t) -> {
                    val neg = t.startsWith("-")
                    if (t.trimStart('+', '-').lowercase() == "nan") Double.NaN
                    else if (neg) Double.NEGATIVE_INFINITY else Double.POSITIVE_INFINITY
                }
                else -> null
            }
        }
        else -> null
    }

    /** True for a JSON int or float that is finite (Python _finite_number: bool is not a number). */
    fun finiteNumber(x: Any?): Boolean = x !is Boolean && x is Number && x.toDouble().isFinite()

    /** Python str() of a JSON value (scalars; lists and maps use repr). */
    fun str(x: Any?): String = if (x is String) x else repr(x)

    fun repr(x: Any?): String = when (x) {
        null -> "None"
        is Boolean -> if (x) "True" else "False"
        is Int, is Long -> x.toString()
        is Double -> floatRepr(x)
        is String -> if ('\'' in x && '"' !in x) "\"$x\"" else "'" + x.replace("\\", "\\\\").replace("'", "\\'") + "'"
        is List<*> -> x.joinToString(", ", "[", "]") { repr(it) }
        is Map<*, *> -> x.entries.joinToString(", ", "{", "}") { "${repr(it.key)}: ${repr(it.value)}" }
        else -> x.toString()
    }

    private const val PLAIN_MIN = 1e-4
    private const val PLAIN_MAX = 1e16
    private const val ZERO = 0.0

    /** Python repr(float): plain between 1e-4 and 1e16, else d.ddde+XX. */
    fun floatRepr(d: Double): String {
        if (d.isNaN()) return "nan"
        if (d.isInfinite()) return if (d > 0) "inf" else "-inf"
        val a = Math.abs(d)
        if (a == ZERO || (a >= PLAIN_MIN && a < PLAIN_MAX)) {
            val s = BigDecimal(d.toString()).toPlainString()
            return if ('.' in s) s.trimEnd('0').let { if (it.endsWith('.')) it + "0" else it } else "$s.0"
        }
        val (m, e) = d.toString().split('E')
        val mant = m.trimEnd('0').trimEnd('.')
        val exp = e.toInt()
        return mant + "e" + (if (exp < 0) "-" else "+") + "%02d".format(Math.abs(exp))
    }
}
