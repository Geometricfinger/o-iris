package org.oiris.core

/** Power governor (ADR-0061), the twin of oiris/power/governor.py and duty.py: battery and heat readings -> a level
 *  and that level's duty plan. Pure and deterministic; time comes from each reading, an explicit `now`, or the
 *  injected clock.
 *
 *  Levels, least to most restrictive: normal, saver, hot, critical. Battery axis -> saver or critical (charging
 *  cancels it); heat axis -> hot or critical from skin, device or the Android thermal status. The more
 *  restrictive axis wins. Hysteresis on every threshold; tightening is immediate; relaxing goes one level at a time
 *  after minDwellS at the level. A stale reading (older than staleAfterS, or no finite t) relaxes nothing. An
 *  unknown value (null, out of plausible range) keeps that signal's last state.
 *
 *  Config validation returns Python's error strings; FrameGate / ListenDuty refuse bad arguments as Python does
 *  (IllegalArgumentException where Python raises ValueError). Checked against tests/vectors/power_governor.json. */
object Power {
    const val NORMAL = "normal"
    const val SAVER = "saver"
    const val HOT = "hot"
    const val CRITICAL = "critical"
    val LEVELS = listOf(NORMAL, SAVER, HOT, CRITICAL)
    val RANK = LEVELS.withIndex().associate { it.value to it.index }
    val REASONS = listOf("battery_low", "battery_critical", "charging", "skin_hot", "skin_critical", "device_hot",
        "device_critical", "thermal_status_hot", "thermal_status_critical", "stale")

    // Android PowerManager.getCurrentThermalStatus (API 29+).
    const val THERMAL_STATUS_NONE = 0
    const val THERMAL_STATUS_LIGHT = 1
    const val THERMAL_STATUS_MODERATE = 2
    const val THERMAL_STATUS_SEVERE = 3
    const val THERMAL_STATUS_CRITICAL = 4
    const val THERMAL_STATUS_EMERGENCY = 5
    const val THERMAL_STATUS_SHUTDOWN = 6

    val BATTERY_PCT_RANGE = 0.0 to 100.0
    val TEMP_PLAUSIBLE_C = -40.0 to 150.0   // outside: a sensor fault, treated as unknown

    /** Named, provisional defaults (config "power"), in Python's key order. */
    val POWER_DEFAULTS: Map<String, Double> = linkedMapOf(
        "battery_saver_enter_pct" to 20.0, "battery_saver_exit_pct" to 25.0,
        "battery_critical_enter_pct" to 7.0, "battery_critical_exit_pct" to 12.0,
        "skin_hot_enter_c" to 41.0, "skin_hot_exit_c" to 39.5,
        "skin_critical_enter_c" to 43.0, "skin_critical_exit_c" to 41.5,
        "device_hot_enter_c" to 75.0, "device_hot_exit_c" to 70.0,
        "device_critical_enter_c" to 85.0, "device_critical_exit_c" to 80.0,
        "min_dwell_s" to 30.0,
        "stale_after_s" to 60.0,
    )
    val POWER_KEYS = POWER_DEFAULTS.keys + "plans"
    val PCT_KEYS = listOf("battery_saver_enter_pct", "battery_saver_exit_pct", "battery_critical_enter_pct", "battery_critical_exit_pct")
    val SKIN_RANGE_C = 30.0 to 60.0
    val DEVICE_RANGE_C = 30.0 to 130.0
    val DWELL_RANGE_S = 0.0 to 3600.0
    val STALE_RANGE_S = 1.0 to 3600.0
    /** In Python's dict order: the order of the error strings. */
    val POWER_RANGES: Map<String, Pair<Double, Double>> = LinkedHashMap<String, Pair<Double, Double>>().apply {
        PCT_KEYS.forEach { put(it, BATTERY_PCT_RANGE) }
        POWER_DEFAULTS.keys.filter { it.startsWith("skin_") }.forEach { put(it, SKIN_RANGE_C) }
        POWER_DEFAULTS.keys.filter { it.startsWith("device_") }.forEach { put(it, DEVICE_RANGE_C) }
        put("min_dwell_s", DWELL_RANGE_S); put("stale_after_s", STALE_RANGE_S)
    }
    val HEAT_SOURCES = listOf("skin", "device")
    private val SIGNALS = listOf("battery", "skin", "device", "thermal_status")

    // ---- duty plans (duty.py) ----
    val PLAN_KEYS = listOf("camera_max_fps", "frame_stride", "listen_duty", "listen_period_s", "radio_scan_interval_s",
        "ir_illuminator", "voice_match", "face_match", "display_brightness_cap")
    /** key -> (low, high) and how Python prints that range in an error. frame_stride is a whole number. */
    val PLAN_RANGES: Map<String, Pair<Double, Double>> = linkedMapOf(
        "camera_max_fps" to (0.0 to 60.0),
        "frame_stride" to (1.0 to 100.0),
        "listen_duty" to (0.0 to 1.0),
        "listen_period_s" to (1.0 to 600.0),
        "radio_scan_interval_s" to (1.0 to 600.0),   // never 0: radio scanning is the guard's minimum
        "display_brightness_cap" to (0.0 to 1.0),
    )
    val PLAN_INTS = setOf("frame_stride")
    val PLAN_BOOLS = listOf("ir_illuminator", "voice_match", "face_match")
    val LOWER_IS_TIGHTER = listOf("camera_max_fps", "listen_duty", "display_brightness_cap")
    val HIGHER_IS_TIGHTER = listOf("frame_stride", "radio_scan_interval_s")
    const val CAMERA_OFF_FPS = 0.0
    const val DUTY_OFF = 0.0
    const val DUTY_FULL = 1.0
    val CRITICAL_OFF = listOf("ir_illuminator", "voice_match", "face_match")

    val DEFAULT_PLANS: Map<String, Map<String, Any>> = linkedMapOf(
        NORMAL to linkedMapOf("camera_max_fps" to 30.0, "frame_stride" to 1L, "listen_duty" to 1.0, "listen_period_s" to 10.0,
            "radio_scan_interval_s" to 15.0, "ir_illuminator" to true, "voice_match" to true, "face_match" to true,
            "display_brightness_cap" to 1.0),
        SAVER to linkedMapOf("camera_max_fps" to 15.0, "frame_stride" to 2L, "listen_duty" to 0.5, "listen_period_s" to 10.0,
            "radio_scan_interval_s" to 30.0, "ir_illuminator" to true, "voice_match" to true, "face_match" to true,
            "display_brightness_cap" to 0.6),
        HOT to linkedMapOf("camera_max_fps" to 10.0, "frame_stride" to 3L, "listen_duty" to 0.25, "listen_period_s" to 10.0,
            "radio_scan_interval_s" to 30.0, "ir_illuminator" to false, "voice_match" to true, "face_match" to false,
            "display_brightness_cap" to 0.5),
        CRITICAL to linkedMapOf("camera_max_fps" to 0.0, "frame_stride" to 3L, "listen_duty" to 0.0, "listen_period_s" to 10.0,
            "radio_scan_interval_s" to 60.0, "ir_illuminator" to false, "voice_match" to false, "face_match" to false,
            "display_brightness_cap" to 0.3),
    )

    private fun rangeText(k: String, r: Pair<Double, Double>): String =
        if (k in PLAN_INTS) "[${r.first.toLong()}, ${r.second.toLong()}]" else "[${Py.floatRepr(r.first)}, ${Py.floatRepr(r.second)}]"

    private fun num(x: Any?): Double = (x as Number).toDouble()

    /** The config's power.plans laid over the defaults, per level and per field (merged_plans). */
    fun mergedPlans(d: Any?): Map<String, Map<String, Any?>> {
        val src = d as? Map<*, *> ?: emptyMap<String, Any?>()
        return LEVELS.associateWith { lv ->
            val out = LinkedHashMap<String, Any?>(DEFAULT_PLANS.getValue(lv))
            (src[lv] as? Map<*, *>)?.forEach { (k, v) -> out[k as String] = v }
            out
        }
    }

    private fun validatePlan(level: String, p: Map<String, Any?>): List<String> {
        val where = "power.plans.$level"
        val errs = mutableListOf<String>()
        val unknown = p.keys - PLAN_KEYS.toSet()
        if (unknown.isNotEmpty()) errs += "$where: unknown keys ${Py.repr(unknown.sorted())}"
        for ((k, r) in PLAN_RANGES) {
            val x = p[k]
            if (!Py.finiteNumber(x) || (k in PLAN_INTS && Math.floor(num(x)) != num(x)) || num(x) < r.first || num(x) > r.second)
                errs += "$where.$k must be in ${rangeText(k, r)}"
        }
        for (k in PLAN_BOOLS) if (p[k] !is Boolean) errs += "$where.$k must be true or false"
        return errs
    }

    /** validate_plans: ranges per field, the order rule between levels, the critical minimum. Never throws. */
    fun validatePlans(d: Any?): List<String> {
        if (d !is Map<*, *>) return listOf("power.plans must be an object")
        val errs = mutableListOf<String>()
        val unknown = d.keys.map { it as String } - LEVELS.toSet()
        if (unknown.isNotEmpty()) errs += "power.plans: unknown levels ${Py.repr(unknown.sorted())}"
        for (lv in LEVELS) if (d.containsKey(lv) && d[lv] !is Map<*, *>) errs += "power.plans.$lv must be an object"
        if (errs.isNotEmpty()) return errs
        val plans = mergedPlans(d)
        for (lv in LEVELS) errs += validatePlan(lv, plans.getValue(lv))
        if (errs.isNotEmpty()) return errs
        for ((prev, cur) in LEVELS.zipWithNext()) {
            val a = plans.getValue(prev); val b = plans.getValue(cur)
            for (k in LOWER_IS_TIGHTER) if (num(b[k]) > num(a[k]))
                errs += "power.plans.$cur.$k must be <= $prev's (${Py.repr(a[k])}): each level at least as restrictive"
            for (k in HIGHER_IS_TIGHTER) if (num(b[k]) < num(a[k]))
                errs += "power.plans.$cur.$k must be >= $prev's (${Py.repr(a[k])}): each level at least as restrictive"
            for (k in PLAN_BOOLS) if (b[k] == true && a[k] != true) errs += "power.plans.$cur.$k is off in $prev, so it must stay off"
        }
        val crit = plans.getValue(CRITICAL)
        if (num(crit["camera_max_fps"]) != CAMERA_OFF_FPS) errs += "power.plans.critical.camera_max_fps must be 0 (camera processing off)"
        for (k in CRITICAL_OFF) if (crit[k] == true) errs += "power.plans.critical.$k must be false"
        return errs
    }

    /** DutyPlan per level from an already validated power.plans (or null for the defaults). */
    fun buildPlans(d: Any?): Map<String, DutyPlan> = mergedPlans(d).mapValues { (_, p) -> DutyPlan.of(p) }

    /** validate_power: thresholds, hysteresis direction, plans. Error strings, never throws. */
    fun validatePower(d: Any?): List<String> {
        if (d !is Map<*, *>) return listOf("power must be an object")
        val errs = mutableListOf<String>()
        val unknown = d.keys.map { it as String } - POWER_KEYS
        if (unknown.isNotEmpty()) errs += "power: unknown keys ${Py.repr(unknown.sorted())}"
        val v = LinkedHashMap<String, Any?>(POWER_DEFAULTS)
        for ((k, x) in d) if (k in POWER_DEFAULTS) v[k as String] = x
        for ((k, r) in POWER_RANGES) {
            val x = v[k]
            if (!Py.finiteNumber(x) || num(x) < r.first || num(x) > r.second)
                errs += "power.$k must be in [${Py.floatRepr(r.first)}, ${Py.floatRepr(r.second)}]"
        }
        if (errs.isEmpty()) {
            fun n(k: String) = num(v[k])
            for (tier in listOf("saver", "critical"))   // battery: low is bad, so exit sits above enter
                if (!(n("battery_${tier}_exit_pct") > n("battery_${tier}_enter_pct")))
                    errs += "power.battery_${tier}_exit_pct must be above battery_${tier}_enter_pct (hysteresis)"
            if (!(n("battery_critical_enter_pct") < n("battery_saver_enter_pct")))
                errs += "power.battery_critical_enter_pct must be below battery_saver_enter_pct"
            for (src in HEAT_SOURCES) {                  // heat: high is bad, so exit sits below enter
                for (tier in listOf("hot", "critical"))
                    if (!(n("${src}_${tier}_exit_c") < n("${src}_${tier}_enter_c")))
                        errs += "power.${src}_${tier}_exit_c must be below ${src}_${tier}_enter_c (hysteresis)"
                if (!(n("${src}_hot_enter_c") < n("${src}_critical_enter_c")))
                    errs += "power.${src}_hot_enter_c must be below ${src}_critical_enter_c"
            }
        }
        errs += validatePlans(if (d.containsKey("plans")) d["plans"] else emptyMap<String, Any?>())
        return errs
    }

    /** thermal_status_level: MODERATE is hot, SEVERE and above critical, NONE and LIGHT normal; anything else
     *  (null, a bool, a float, out of range) is unknown: null. */
    fun thermalStatusLevel(status: Any?): String? {
        val s = when (status) { is Int -> status.toLong(); is Long -> status; else -> return null }
        if (s < THERMAL_STATUS_NONE || s > THERMAL_STATUS_SHUTDOWN) return null
        return if (s >= THERMAL_STATUS_SEVERE) CRITICAL else if (s >= THERMAL_STATUS_MODERATE) HOT else NORMAL
    }

    /** Battery axis with hysteresis (low is bad). Unknown keeps prev. */
    fun batteryState(prev: String, pct: Double?, s: Map<String, Double>): String {
        if (pct == null) return prev
        if (pct <= s.getValue("battery_critical_enter_pct")) return CRITICAL
        if (prev == CRITICAL && pct < s.getValue("battery_critical_exit_pct")) return CRITICAL
        if (pct <= s.getValue("battery_saver_enter_pct")) return SAVER
        if ((prev == SAVER || prev == CRITICAL) && pct < s.getValue("battery_saver_exit_pct")) return SAVER
        return NORMAL
    }

    /** One temperature source with hysteresis (high is bad). Unknown keeps prev. */
    fun heatState(prev: String, tempC: Double?, src: String, s: Map<String, Double>): String {
        if (tempC == null) return prev
        if (tempC >= s.getValue("${src}_critical_enter_c")) return CRITICAL
        if (prev == CRITICAL && tempC > s.getValue("${src}_critical_exit_c")) return CRITICAL
        if (tempC >= s.getValue("${src}_hot_enter_c")) return HOT
        if ((prev == HOT || prev == CRITICAL) && tempC > s.getValue("${src}_hot_exit_c")) return HOT
        return NORMAL
    }

    internal fun tighter(a: String, b: String): String = if (RANK.getValue(a) >= RANK.getValue(b)) a else b

    internal fun inRange(x: Double?, r: Pair<Double, Double>): Double? =
        if (x != null && x.isFinite() && x >= r.first && x <= r.second) x else null

    internal val SIGNAL_NAMES get() = SIGNALS
}

data class DutyPlan(val cameraMaxFps: Double, val frameStride: Int, val listenDuty: Double, val listenPeriodS: Double,
                    val radioScanIntervalS: Double, val irIlluminator: Boolean, val voiceMatch: Boolean,
                    val faceMatch: Boolean, val displayBrightnessCap: Double) {
    fun toMap(): Map<String, Any> = linkedMapOf("camera_max_fps" to cameraMaxFps, "frame_stride" to frameStride,
        "listen_duty" to listenDuty, "listen_period_s" to listenPeriodS, "radio_scan_interval_s" to radioScanIntervalS,
        "ir_illuminator" to irIlluminator, "voice_match" to voiceMatch, "face_match" to faceMatch,
        "display_brightness_cap" to displayBrightnessCap)

    companion object {
        /** From one merged, validated plan map. */
        fun of(p: Map<String, Any?>): DutyPlan {
            fun d(k: String) = (p[k] as Number).toDouble()
            return DutyPlan(d("camera_max_fps"), d("frame_stride").toInt(), d("listen_duty"), d("listen_period_s"),
                d("radio_scan_interval_s"), p["ir_illuminator"] as Boolean, p["voice_match"] as Boolean,
                p["face_match"] as Boolean, d("display_brightness_cap"))
        }
    }
}

class PowerSettings(val thresholds: Map<String, Double> = Power.POWER_DEFAULTS,
                    val plans: Map<String, DutyPlan> = Power.buildPlans(null)) {
    companion object {
        /** From the config "power" section (null or {} = all defaults). Throws IllegalArgumentException with
         *  Python's joined errors where PowerSettings.from_config raises ConfigError. */
        fun fromConfig(d: Map<String, Any?>?): PowerSettings {
            val c = d ?: emptyMap()
            val errs = Power.validatePower(c)
            require(errs.isEmpty()) { errs.joinToString("; ") }
            val th = LinkedHashMap(Power.POWER_DEFAULTS)
            for (k in Power.POWER_DEFAULTS.keys) if (c.containsKey(k)) th[k] = (c[k] as Number).toDouble()
            return PowerSettings(th, Power.buildPlans(c["plans"]))
        }
    }
}

/** One reading. t: monotonic seconds when it was taken (null or non-finite: stale). Every other field may be null:
 *  unknown. thermalStatus is the raw Android int; an out-of-range one is unknown. */
data class PowerReading(val t: Double?, val batteryPct: Double? = null, val charging: Boolean? = null,
                        val skinC: Double? = null, val deviceC: Double? = null, val thermalStatus: Long? = null) {
    companion object {
        /** From a JSON object with Python's field names and type rules: a bool is not a number, a float is not a
         *  thermal status, anything else unexpected is unknown. */
        fun fromMap(m: Map<String, Any?>): PowerReading {
            fun n(k: String) = m[k].let { if (it is Number && it !is Boolean) it.toDouble() else null }
            return PowerReading(n("t"), n("battery_pct"), m["charging"] as? Boolean, n("skin_c"), n("device_c"),
                m["thermal_status"] as? Long)
        }
    }
}

data class Decision(val level: String, val plan: DutyPlan, val reasons: List<String>) {
    fun toMap(): Map<String, Any?> = mapOf("level" to level, "plan" to plan.toMap(), "reasons" to reasons)
}

/** update(reading, now) -> Decision. One instance per device. now defaults to the injected clock, then reading.t. */
class PowerGovernor(val settings: PowerSettings = PowerSettings(), private val clock: (() -> Double)? = null) {
    var level: String = Power.NORMAL
        private set
    var since: Double? = null       // caller's clock when the current level was entered
        private set
    var signals: Map<String, String> = Power.SIGNAL_NAMES.associateWith { Power.NORMAL }
        private set
    var charging = false
        private set

    fun decision(stale: Boolean = false): Decision {
        val sig = signals
        val r = mutableSetOf<String>()
        if (sig["battery"] == Power.SAVER) r += "battery_low" else if (sig["battery"] == Power.CRITICAL) r += "battery_critical"
        if (charging) r += "charging"
        for (name in listOf("skin", "device", "thermal_status")) {
            val s = sig.getValue(name)
            if (s == Power.HOT || s == Power.CRITICAL) r += "${name}_$s"
        }
        if (stale) r += "stale"
        return Decision(level, settings.plans.getValue(level), Power.REASONS.filter { it in r })
    }

    fun update(reading: PowerReading, now: Double? = null): Decision {
        val s = settings.thresholds
        val t = reading.t?.takeIf { it.isFinite() }
        val nowS = now ?: clock?.invoke() ?: t ?: since ?: 0.0
        val stale = t == null || nowS - t > s.getValue("stale_after_s")
        val ch = reading.charging
        val pct = Power.inRange(reading.batteryPct, Power.BATTERY_PCT_RANGE)
        val sig = signals
        var new = LinkedHashMap<String, String>()
        new["battery"] = if (ch == true) Power.NORMAL else Power.batteryState(sig.getValue("battery"), pct, s)
        new["skin"] = Power.heatState(sig.getValue("skin"), Power.inRange(reading.skinC, Power.TEMP_PLAUSIBLE_C), "skin", s)
        new["device"] = Power.heatState(sig.getValue("device"), Power.inRange(reading.deviceC, Power.TEMP_PLAUSIBLE_C), "device", s)
        new["thermal_status"] = Power.thermalStatusLevel(reading.thermalStatus) ?: sig.getValue("thermal_status")
        if (stale) {   // old data may tighten a signal, never relax it
            new = LinkedHashMap(new.mapValues { (k, v) -> Power.tighter(v, sig.getValue(k)) })
        } else if (ch != null) {
            charging = ch
        }
        signals = new
        var target = Power.NORMAL
        for (v in new.values) target = Power.tighter(target, v)
        val rank = Power.RANK.getValue(level); val tr = Power.RANK.getValue(target)
        val since0 = since
        if (tr > rank) {
            level = target; since = nowS
        } else if (tr < rank && !stale && since0 != null && nowS - since0 >= s.getValue("min_dwell_s")) {
            level = Power.LEVELS[rank - 1]; since = nowS
        }
        if (since == null) since = nowS
        return decision(stale)
    }

    /** The decisions for a reading list, in order (each at its own t, or the injected clock). */
    fun replay(readings: List<PowerReading>): List<Decision> = readings.map { update(it) }
}

/** Which camera frames are processed: a rate cap, then every stride-th admitted frame (duty.FrameGate). */
class FrameGate(maxFps: Double, val stride: Int = 1) {
    val maxFps: Double
    val minIntervalS: Double
    private var last: Double? = null
    private var n = 0

    init {
        require(maxFps.isFinite() && maxFps >= 0) { "max_fps must be a number >= 0" }
        require(stride >= 1) { "stride must be a whole number >= 1" }
        this.maxFps = maxFps
        minIntervalS = if (maxFps > Power.CAMERA_OFF_FPS) 1 / maxFps else Double.POSITIVE_INFINITY
    }

    fun admit(t: Double): Boolean {
        if (maxFps <= Power.CAMERA_OFF_FPS) return false
        last?.let { if (t < it) { last = null; n = 0 } }
        last?.let { if (t - it < minIntervalS - TOLERANCE_S) return false }
        last = t
        n++
        return (n - 1) % stride == 0
    }

    companion object { const val TOLERANCE_S = 1e-9 }
}

/** Listening on for the first duty * periodS of every periodS, aligned to t = 0 (duty.ListenDuty). */
class ListenDuty(val duty: Double, val periodS: Double) {
    init {
        require(duty.isFinite() && duty >= Power.DUTY_OFF && duty <= Power.DUTY_FULL) { "duty must be in [0, 1]" }
        require(periodS.isFinite() && periodS > 0) { "period_s must be a positive number" }
    }

    /** Python's % on floats is floored (the result takes the divisor's sign): Double.mod, not rem. */
    fun active(t: Double): Boolean {
        if (duty >= Power.DUTY_FULL) return true
        if (duty <= Power.DUTY_OFF) return false
        return t.mod(periodS) < duty * periodS
    }
}
