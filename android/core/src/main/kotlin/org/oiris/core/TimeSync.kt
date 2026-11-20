package org.oiris.core

/** Clock sync between devices, ported from oiris/link/timesync.py: the NTP-style offset and delay of one
 *  exchange with the pod (op "time_sync"), minimum-delay sample selection, Theil-Sen drift and the conversion
 *  of remote monotonic stamps onto this device's timeline before merging detections. Vectors:
 *  tests/vectors/time_sync.json, generated from the Python code. Pure Kotlin, no Android APIs.
 *
 *  Exchange (integer ns): this device sends at t0 (its clock), the pod receives at t1 and replies at t2 (pod
 *  clock), this device receives at t3. offset = remote - host = floorDiv((t1 - t0) + (t2 - t3), 2);
 *  delay = (t3 - t0) - (t2 - t1). Math.floorDiv rounds toward negative infinity like Python //, so odd negative
 *  sums agree (-2001 / 2 is -1001, where Long division gives -1000).
 *
 *  Overflow bound: every timestamp must lie in [0, MAX_TIMESTAMP_NS) = [0, 2^61) (about 73 years of uptime;
 *  rejectReason refuses the rest). Then each difference lies in (-2^61, 2^61) and each sum of two differences in
 *  (-2^62, 2^62), inside Long. Drift slopes divide Long differences converted to Double, exact (as in Python)
 *  while they stay below 2^53 ns, about 104 days. */
class TimeSync(val maxSamples: Int = MAX_SAMPLES, val maxDelayNs: Long = MAX_DELAY_NS, val staleNs: Long = STALE_NS) {
    data class Exchange(val t0: Long, val t1: Long, val t2: Long, val t3: Long) {
        val offsetNs: Long get() = offsetNs(t0, t1, t2, t3)
        val delayNs: Long get() = delayNs(t0, t1, t2, t3)
        /** Host time the offset refers to: the midpoint of send and receive (floor). */
        val hostMidNs: Long get() = t0 + Math.floorDiv(t3 - t0, 2L)
    }

    enum class Quality(val wire: String) { OK("ok"), UNSYNCED("unsynced"), STALE("stale") }

    class NotSynced : IllegalStateException("not synced")

    private val window = ArrayDeque<Exchange>()
    private var lastT3: Long? = null
    val samples: List<Exchange> get() = window.toList()

    init { require(maxSamples >= 1) { "maxSamples must be at least 1" } }

    /** Records one exchange: null when accepted, else the refusal reason (the sample is dropped). */
    fun add(t0: Long, t1: Long, t2: Long, t3: Long): String? {
        val why = rejectReason(t0, t1, t2, t3, maxDelayNs)
        if (why == null) {
            if (window.size == maxSamples) window.removeFirst()
            window.addLast(Exchange(t0, t1, t2, t3))
            lastT3 = lastT3?.let { maxOf(it, t3) } ?: t3
        }
        return why
    }

    /** Minimum delay first; ties go to the most recent (larger t3). */
    private val byDelay = compareBy<Exchange>({ it.delayNs }, { -it.t3 })

    fun best(): Exchange? = window.minWithOrNull(byDelay)

    /** Remote rate minus host rate as a slope (ns per ns): the median pairwise slope over the lowest-delay half,
     *  or null while there are too few samples or their host midpoints span too little time. */
    fun drift(): Double? {
        if (window.size < DRIFT_MIN_SAMPLES) return null
        val ranked = window.sortedWith(byDelay)
        val half = ranked.subList(0, (ranked.size + 1) / 2)
        val mids = half.map { it.hostMidNs }
        if (mids.max() - mids.min() < DRIFT_MIN_SPAN_NS) return null
        val slopes = ArrayList<Double>()
        for (i in half.indices) for (j in i + 1 until half.size) {
            val a = half[i]; val b = half[j]
            if (b.hostMidNs != a.hostMidNs)
                slopes.add((b.offsetNs - a.offsetNs).toDouble() / (b.hostMidNs - a.hostMidNs).toDouble())
        }
        if (slopes.isEmpty()) return null
        slopes.sort()
        val m = slopes.size / 2
        return if (slopes.size % 2 == 1) slopes[m] else (slopes[m - 1] + slopes[m]) / 2
    }

    fun driftPpm(): Double? = drift()?.let { it * PPM }

    /** remote - host at a host time: the best sample's offset carried along the drift (floor of the ns). */
    fun offsetAt(atHostNs: Long): Long {
        val b = best() ?: throw NotSynced()
        val d = drift() ?: return b.offsetNs
        return b.offsetNs + Math.floor(d * (atHostNs - b.hostMidNs).toDouble()).toLong()
    }

    /** A remote monotonic stamp on this device's timeline. `atHostNs`: roughly when (host clock) it was taken. */
    fun remoteToHost(tRemoteNs: Long, atHostNs: Long): Long = tRemoteNs - offsetAt(atHostNs)

    fun quality(nowHostNs: Long): Quality {
        val last = lastT3 ?: return Quality.UNSYNCED
        return if (nowHostNs - last > staleNs) Quality.STALE else Quality.OK
    }

    companion object {
        const val MAX_SAMPLES = 64
        const val MAX_DELAY_NS = 250_000_000L
        const val STALE_NS = 30_000_000_000L
        const val DRIFT_MIN_SAMPLES = 8
        const val DRIFT_MIN_SPAN_NS = 10_000_000_000L
        const val PPM = 1_000_000L
        const val MAX_TIMESTAMP_NS = 1L shl 61

        const val REJECT_RANGE = "timestamp out of range"
        const val REJECT_ORDER = "device replied before it received"
        const val REJECT_NEGATIVE = "negative delay"
        const val REJECT_SLOW = "delay above maximum"
        const val REJECT_REPLY = "bad time_sync reply"

        fun offsetNs(t0: Long, t1: Long, t2: Long, t3: Long): Long = Math.floorDiv((t1 - t0) + (t2 - t3), 2L)
        fun delayNs(t0: Long, t1: Long, t2: Long, t3: Long): Long = (t3 - t0) - (t2 - t1)

        /** Why an exchange is refused, or null when it is usable. */
        fun rejectReason(t0: Long, t1: Long, t2: Long, t3: Long, maxDelayNs: Long = MAX_DELAY_NS): String? {
            if (!listOf(t0, t1, t2, t3).all { it in 0 until MAX_TIMESTAMP_NS }) return REJECT_RANGE
            if (t2 < t1) return REJECT_ORDER
            val d = delayNs(t0, t1, t2, t3)
            if (d < 0) return REJECT_NEGATIVE
            if (d > maxDelayNs) return REJECT_SLOW
            return null
        }

        /** The request body of the pod op: {"op":"time_sync","t0":<ns>} (canonical JSON, keys sorted). */
        fun request(t0: Long): String = Json.encode(mapOf("op" to "time_sync", "t0" to t0))
    }

    /** Records the pod's reply to request(t0Sent), received at t3: null when accepted, else why not (REJECT_REPLY
     *  for a refusal, a malformed reply or one that echoes another request's t0). */
    fun addReply(reply: String, t0Sent: Long, t3: Long): String? {
        // Json.decode throws IllegalArgumentException (or an index error on truncated text): reported as REJECT_REPLY
        val r = try { Json.decode(reply) } catch (e: RuntimeException) { return REJECT_REPLY }
        if (r !is Map<*, *>) return REJECT_REPLY
        val t1 = r["t1"]; val t2 = r["t2"]
        if (r["ok"] != true || r["t0"] != t0Sent || t1 !is Long || t2 !is Long) return REJECT_REPLY
        return add(t0Sent, t1, t2, t3)
    }
}
