package org.oiris.core

import java.util.regex.Pattern
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** M-17 voice link rules ported from oiris/modules/voice_link (ADR-0057, ADR-0058, ADR-0059):
 *  the matching decision (core.decide), the voice template and cosine score (core), the spoken name
 *  (announce.speakable_name / phrase), the per-person repeat limit (announce.Announcer) and the energy
 *  VAD (vad.detect). Vectors: tests/vectors/voice_link.json, generated from the Python code.
 *  Pure Kotlin: no Android APIs, nothing stored. The speaker model, capture and playback are not here. */
object VoiceLink {
    // oiris/config.py VOICE_LINK_DEFAULTS
    const val MATCH_THRESHOLD = 0.5        // raw cosine; PROVISIONAL until the live check
    const val UNSURE_MARGIN = 0.05
    const val SPOOF_THRESHOLD = 0.5
    const val VAD_SNR_DB = 10.0
    const val VAD_FLOOR_DBFS = -55.0
    const val ANNOUNCE_REPEAT_S = 60.0

    // ---- matching (core.py) ----------------------------------------------------------------------
    enum class Outcome(val wire: String) { HEARD("heard"), UNSURE("unsure"), NONE("none"), SELF("self"), SPOOFED("spoofed") }
    data class Decision(val outcome: Outcome, val personId: String? = null)

    /** core.decide. `scores`: (cosine, personId) per enrolled voice, any order; ties rank the larger id
     *  first (Python tuple order). In order: best below matchThreshold => NONE; a second voice within
     *  unsureMargin => UNSURE; the best is selfId => SELF; spoof score at or above spoofThreshold =>
     *  SPOOFED; else HEARD. `spoofScore` is called only at the spoof step; null = no anti-spoof slot
     *  (or a failed one): the match goes ahead. */
    fun decide(scores: List<Pair<Double, String>>, matchThreshold: Double, unsureMargin: Double, selfId: String?,
               spoofScore: () -> Double?, spoofThreshold: Double): Decision {
        if (scores.isEmpty()) return Decision(Outcome.NONE)
        val ranked = scores.sortedWith { a, b ->
            when { a.first > b.first -> -1; a.first < b.first -> 1; else -> b.second.compareTo(a.second) }
        }
        val (best, pid) = ranked[0]
        if (best < matchThreshold) return Decision(Outcome.NONE)
        if (ranked.size > 1 && best - ranked[1].first < unsureMargin) return Decision(Outcome.UNSURE)
        if (pid == selfId) return Decision(Outcome.SELF)
        val s = spoofScore()
        if (s != null && s >= spoofThreshold) return Decision(Outcome.SPOOFED)
        return Decision(Outcome.HEARD, pid)
    }

    fun decide(scores: List<Pair<Double, String>>, matchThreshold: Double, unsureMargin: Double, selfId: String?,
               spoofScore: Double?, spoofThreshold: Double): Decision =
        decide(scores, matchThreshold, unsureMargin, selfId, { spoofScore }, spoofThreshold)

    /** face_match.port.unit: v / |v|, or v unchanged when the norm is 0 or not finite. */
    fun unit(v: FloatArray): FloatArray {
        var ss = 0.0
        for (x in v) ss += x.toDouble() * x
        val n = sqrt(ss).toFloat()
        return if (n > 0f && n.isFinite()) FloatArray(v.size) { v[it] / n } else v.copyOf()
    }

    /** The score the core logs: unit(embedding) . template (a stored template is already unit). */
    fun cosine(embedding: FloatArray, template: FloatArray): Double {
        require(embedding.size == template.size) { "dimension mismatch" }
        val q = unit(embedding)
        var d = 0.0
        for (i in q.indices) d += q[i].toDouble() * template[i]
        q.fill(0f)
        return d
    }

    /** The voice template: the unit mean of the unit embeddings of the accepted windows. */
    fun template(embeddings: List<FloatArray>): FloatArray {
        require(embeddings.isNotEmpty() && embeddings.all { it.size == embeddings[0].size }) { "embeddings of one dimension needed" }
        val sum = DoubleArray(embeddings[0].size)
        for (e in embeddings) { val u = unit(e); for (i in u.indices) sum[i] += u[i].toDouble(); u.fill(0f) }
        return unit(FloatArray(sum.size) { (sum[it] / embeddings.size).toFloat() })
    }

    // ---- the spoken name (announce.py) -------------------------------------------------------------
    const val PHRASE_PREFIX = "sounds like "
    const val NAME_MAX_WORDS = 4
    const val NAME_MAX_CHARS = 40          // code points, as Python's len()
    private val NAME_JOINERS = setOf(' '.code, '-'.code, '\''.code, '’'.code)
    // Han, Hiragana and Katakana are written together in one Japanese name: one script for this rule.
    private val SCRIPT_FAMILIES = mapOf("CJK" to "HAN", "HIRAGANA" to "HAN", "KATAKANA" to "HAN")
    private const val UNKNOWN_SCRIPT = "UNKNOWN"
    // Python's \b: between a word character (str.isalnum() or "_") and anything else.
    private const val WORD_CHAR = "[\\p{L}\\p{N}_]"

    /** oiris/guard/vocabulary.py's banned-word matcher, built from schema/vocabulary.json "banned_patterns". */
    fun bannedRegex(patterns: List<String>): Regex =
        Pattern.compile("(?<!$WORD_CHAR)(" + patterns.joinToString("|") + ")(?!$WORD_CHAR)",
            Pattern.CASE_INSENSITIVE or Pattern.UNICODE_CASE or Pattern.UNICODE_CHARACTER_CLASS).toRegex()

    private fun isLetter(cp: Int) = when (Character.getType(cp).toByte()) {
        Character.UPPERCASE_LETTER, Character.LOWERCASE_LETTER, Character.TITLECASE_LETTER,
        Character.MODIFIER_LETTER, Character.OTHER_LETTER -> true
        else -> false
    }

    private fun isMark(cp: Int) = when (Character.getType(cp).toByte()) {
        Character.NON_SPACING_MARK, Character.COMBINING_SPACING_MARK, Character.ENCLOSING_MARK -> true
        else -> false
    }

    /** The first word of the Unicode character name (as Python's unicodedata.name), folded by SCRIPT_FAMILIES. */
    private fun script(cp: Int): String {
        val first = (Character.getName(cp) ?: UNKNOWN_SCRIPT).split(" ")[0]
        return SCRIPT_FAMILIES[first] ?: first
    }

    /** announce.speakable_name: the name when it may be spoken, else null (the chime still plays).
     *  Guard-clean, at most NAME_MAX_WORDS words split on single spaces and NAME_MAX_CHARS characters,
     *  each word starting with a letter, letters of one script plus combining marks and NAME_JOINERS.
     *  Mixed scripts are refused so a look-alike letter cannot slip a word past the guard. */
    fun speakableName(name: String?, banned: Regex): String? {
        if (name.isNullOrEmpty() || name.codePointCount(0, name.length) > NAME_MAX_CHARS || banned.containsMatchIn(name)) return null
        val words = name.split(" ")
        if (words.size > NAME_MAX_WORDS || words.any { it.isEmpty() }) return null
        val scripts = HashSet<String>()
        for (w in words) {
            if (!isLetter(w.codePointAt(0))) return null
            var i = 0
            while (i < w.length) {
                val cp = w.codePointAt(i); i += Character.charCount(cp)
                if (isLetter(cp)) scripts.add(script(cp))
                else if (!(isMark(cp) || cp in NAME_JOINERS)) return null
            }
        }
        return if (scripts.size == 1) name else null
    }

    fun phrase(name: String): String = PHRASE_PREFIX + name

    /** announce.Announcer's repeat rule: the same person at most once per repeatS seconds; different
     *  people back to back. RAM only. The caller records an announcement only when it is queued. */
    class RepeatLimiter(val repeatS: Double = ANNOUNCE_REPEAT_S) {
        private val last = HashMap<String, Double>()

        fun allow(personId: String, nowS: Double): Boolean {
            val t = last[personId]
            if (t != null && nowS - t < repeatS) return false
            last[personId] = nowS
            return true
        }

        fun clear() = last.clear()
    }

    // ---- energy VAD (vad.py) ------------------------------------------------------------------------
    const val FRAME_S = 0.030
    const val HANGOVER_S = 0.200
    const val MIN_SEGMENT_S = 0.250
    const val FLOOR_PERCENTILE = 10.0
    const val POWER_EPS = 1e-12            // keeps log10 finite on digital silence (-120 dBFS)
    const val SILENCE_DBFS = -120.0
    private const val PERCENT = 100.0
    private const val HALF = 0.5

    /** segments: [start, end) sample indices; speechS: their total length in seconds. */
    data class VadResult(val segments: List<Pair<Int, Int>>, val speechS: Double)

    /** Frame length in samples (Python round: half to even). */
    fun frameLength(sampleRate: Int): Int = max(1, Math.rint(FRAME_S * sampleRate).toInt())

    /** Per-frame RMS level in dBFS; a trailing partial frame is ignored. */
    fun frameLevels(audio: FloatArray, sampleRate: Int): DoubleArray {
        val n = frameLength(sampleRate)
        return DoubleArray(audio.size / n) { f ->
            var p = 0.0
            for (i in f * n until (f + 1) * n) p += audio[i].toDouble() * audio[i]
            10.0 * log10(p / n + POWER_EPS)
        }
    }

    /** numpy.percentile, linear method (numpy's lerp included). */
    private fun percentile(v: DoubleArray, q: Double): Double {
        val s = v.sortedArray()
        val idx = q / PERCENT * (s.size - 1)
        val lo = floor(idx).toInt(); val hi = min(lo + 1, s.size - 1); val t = idx - lo
        val a = s[lo]; val b = s[hi]; val d = b - a
        return if (t >= HALF) b - d * (1 - t) else a + d * t
    }

    /** vad.detect: a frame is speech when its level is at least max(floor + snrDb, floorDbfs), floor =
     *  the FLOOR_PERCENTILE of the frame levels; speech frames hold the detector on for HANGOVER_S; a
     *  segment needs MIN_SEGMENT_S from its first to its last speech frame (the hangover does not count). */
    fun detect(audio: FloatArray, sampleRate: Int, snrDb: Double = VAD_SNR_DB, floorDbfs: Double = VAD_FLOOR_DBFS): VadResult {
        val levels = frameLevels(audio, sampleRate)
        if (levels.isEmpty()) return VadResult(emptyList(), 0.0)
        val n = frameLength(sampleRate)
        val thr = max(percentile(levels, FLOOR_PERCENTILE) + snrDb, floorDbfs)
        val on = BooleanArray(levels.size) { levels[it] >= thr }
        val hang = Math.rint(HANGOVER_S / FRAME_S).toInt()
        val active = on.copyOf()
        for (i in on.indices) if (on[i]) for (j in i + 1..min(i + hang, on.size - 1)) active[j] = true
        val minFrames = ceil(MIN_SEGMENT_S / FRAME_S).toInt()
        val segments = ArrayList<Pair<Int, Int>>()
        var i = 0
        while (i < active.size) {
            if (!active[i]) { i++; continue }
            val start = i
            while (i < active.size && active[i]) i++
            var speechEnd = start
            for (k in start until i) if (on[k]) speechEnd = k + 1
            if (speechEnd - start >= minFrames) segments.add(start * n to i * n)
        }
        return VadResult(segments, segments.sumOf { it.second - it.first }.toDouble() / sampleRate)
    }
}
