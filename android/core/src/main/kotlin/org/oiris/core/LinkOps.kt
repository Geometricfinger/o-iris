package org.oiris.core

import java.util.Base64

/** Link operations a pair of glasses needs from the owner's phone (ADR-0062), ported from oiris/link/ops.py and
 *  oiris/link/sideload.py: request builders (canonical JSON, byte-identical to Python) and reply parsers for unlock,
 *  add_unlock_slot, panic, wipe_challenge, wipe, hud and the model side-load ops. Vectors: tests/vectors/link_ops.json,
 *  generated from the Python code. Pure Kotlin, no Android APIs; the app screens and the vault holding the unlock
 *  secret and the Ed25519 wipe key come later.
 *
 *  Binary values are base64url without padding and decoded strictly (only the canonical encoding of the expected
 *  number of bytes). Parsers never throw: a reply that is not what the op returns parses as ok=false, BAD_REPLY.
 *  HudFrame.parse treats the display list as untrusted: unknown op kinds and malformed ops are dropped, the op count
 *  and every string are capped, coordinates are clamped into the canvas. */
object LinkOps {
    const val SECRET_BYTES = 32
    const val NONCE_BYTES = 32
    const val SIGNATURE_BYTES = 64
    const val WIPE_PREFIX = "ASB-WIPE:"
    val UNLOCK_SLOTS = listOf("phone")
    const val MAX_HUD_OPS = 64
    const val MAX_HUD_TEXT = 80
    const val MAX_CANVAS_PX = 4096L
    val NUMBER_KEYS = listOf("x", "y", "w", "h", "size", "stroke_width", "opacity", "font_size")
    const val MAX_MODEL_BYTES = 256L * 1024 * 1024
    const val CHUNK_MIN = 512
    const val CHUNK_MAX = 65536
    const val MAX_MISSING_RANGES = 64
    val NAME_RE = Regex("[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
    val SLOT_RE = Regex("[a-z][a-z0-9_]{0,31}")
    val SHA256_RE = Regex("[0-9a-f]{64}")
    val PIN_KEYS = listOf("name", "sha256", "size", "slot")

    const val NOT_AVAILABLE = "not available"
    const val BAD_REPLY = "bad reply"
    const val UNLOCK_REFUSED = "unlock refused"
    const val ADD_SLOT_REFUSED = "add_unlock_slot refused"
    const val WIPE_REFUSED = "wipe refused"

    // ---- base64url, strict ----
    private val B64URL = Regex("[A-Za-z0-9_-]*")
    fun b64url(b: ByteArray): String = Base64.getUrlEncoder().withoutPadding().encodeToString(b)
    fun b64urlLen(n: Int) = (4 * n + 2) / 3

    /** The bytes of the canonical unpadded encoding of exactly `n` bytes (or at most `maxBytes`), else null. */
    fun fromB64url(s: Any?, n: Int? = null, maxBytes: Int? = null): ByteArray? {
        if (s !is String || !B64URL.matches(s) || s.length % 4 == 1) return null
        if (n != null && s.length != b64urlLen(n)) return null
        if (maxBytes != null && s.length > b64urlLen(maxBytes)) return null
        val b = try { Base64.getUrlDecoder().decode(s) } catch (e: IllegalArgumentException) { return null }
        if (b64url(b) != s) { b.fill(0); return null }
        return b
    }

    fun chunkCount(size: Long, chunkSize: Int): Long = Math.floorDiv(size + chunkSize - 1, chunkSize.toLong())

    // ---- request builders ----
    fun unlock(slot: String, secret: ByteArray) = Json.encode(mapOf("op" to "unlock", "slot" to slot, "secret" to b64url(secret)))
    fun addUnlockSlot(slot: String, secret: ByteArray) =
        Json.encode(mapOf("op" to "add_unlock_slot", "slot" to slot, "secret" to b64url(secret)))
    fun panic() = Json.encode(mapOf("op" to "panic"))
    fun wipeChallenge() = Json.encode(mapOf("op" to "wipe_challenge"))
    /** The bytes the phone signs with its Ed25519 key. */
    fun wipeCommand(nonce: ByteArray): ByteArray = WIPE_PREFIX.toByteArray(Charsets.US_ASCII) + nonce
    fun wipe(command: ByteArray, signature: ByteArray) =
        Json.encode(mapOf("op" to "wipe", "command" to b64url(command), "signature" to b64url(signature)))
    fun hud(since: Long) = Json.encode(mapOf("op" to "hud", "since" to since))
    fun modelOffer(name: String, slot: String, sha256: String, size: Long, chunkSize: Int) = Json.encode(mapOf(
        "op" to "model_offer", "name" to name, "slot" to slot, "sha256" to sha256, "size" to size, "chunk_size" to chunkSize))
    fun modelChunk(sha256: String, index: Long, data: ByteArray) =
        Json.encode(mapOf("op" to "model_chunk", "sha256" to sha256, "index" to index, "data" to b64url(data)))
    fun modelCommit(sha256: String) = Json.encode(mapOf("op" to "model_commit", "sha256" to sha256))
    fun modelStatus() = Json.encode(mapOf("op" to "model_status"))

    // ---- replies ----
    /** A parsed reply: `ok`, the refusal `error`, and the op's fields (empty when refused or malformed). */
    data class Reply(val ok: Boolean, val error: String?, val fields: Map<String, Any?> = emptyMap()) {
        fun toMap(): Map<String, Any?> = if (ok) mapOf("ok" to true, "error" to null) + fields else mapOf("ok" to false, "error" to error)
    }

    private val bad = Reply(false, BAD_REPLY)

    /** The reply object, or null when it is not a JSON object with a boolean "ok" (and a string error when false). */
    @Suppress("UNCHECKED_CAST")
    fun load(reply: String): Map<String, Any?>? {
        val decoded = try { Json.decode(reply) } catch (e: Exception) { return null }
        val r = decoded as? Map<String, Any?> ?: return null
        val ok = r["ok"] as? Boolean ?: return null
        if (!ok && r["error"] !is String) return null
        return r
    }

    private inline fun parse(reply: String, body: (Map<String, Any?>) -> Reply?): Reply {
        val r = load(reply) ?: return bad
        if (r["ok"] != true) return Reply(false, r["error"] as String)
        return body(r) ?: bad
    }

    private fun count(v: Any?) = v is Long && v >= 0

    /** unlock, add_unlock_slot, panic. */
    fun parseAck(reply: String) = parse(reply) { Reply(true, null) }

    /** fields: nonce (hex). */
    fun parseWipeChallenge(reply: String) = parse(reply) { r ->
        fromB64url(r["nonce"], NONCE_BYTES)?.let { n -> Reply(true, null, mapOf("nonce" to n.joinToString("") { "%02x".format(it) })) }
    }

    fun parseWipe(reply: String) = parse(reply) { r -> if (r["wiped"] == true) Reply(true, null, mapOf("wiped" to true)) else null }

    /** fields: chunks, missing (half-open index ranges), missing_count. */
    fun parseOffer(reply: String) = parse(reply) { r ->
        val n = r["chunks"]; val k = r["missing_count"]; val missing = r["missing"]
        if (!count(n) || !count(k) || missing !is List<*> || missing.size > MAX_MISSING_RANGES) return@parse null
        for (m in missing)
            if (m !is List<*> || m.size != 2 || !count(m[0]) || !count(m[1]) || (m[0] as Long) >= (m[1] as Long) ||
                (m[1] as Long) > (n as Long)) return@parse null
        Reply(true, null, mapOf("chunks" to n, "missing" to missing, "missing_count" to k))
    }

    fun parseChunk(reply: String) = parse(reply) { r ->
        if (count(r["missing_count"])) Reply(true, null, mapOf("missing_count" to r["missing_count"])) else null
    }

    fun parseCommit(reply: String) = parse(reply) { r ->
        (r["installed"] as? String)?.let { Reply(true, null, mapOf("installed" to it)) }
    }

    fun validPin(name: Any?, slot: Any?, sha256: Any?, size: Any?): Boolean =
        name is String && NAME_RE.matches(name) && slot is String && SLOT_RE.matches(slot) && sha256 is String &&
            SHA256_RE.matches(sha256) && size is Long && size >= 1 && size <= MAX_MODEL_BYTES

    /** fields: models, each {name, slot, sha256, size}. */
    fun parseStatus(reply: String) = parse(reply) { r ->
        val models = r["models"] as? List<*> ?: return@parse null
        for (m in models)
            if (m !is Map<*, *> || m.keys.map { it as String }.sorted() != PIN_KEYS || !validPin(m["name"], m["slot"], m["sha256"], m["size"]))
                return@parse null
        Reply(true, null, mapOf("models" to models))
    }
}

/** One poll of the pod's HUD feed (op "hud"), cleaned: an untrusted display list made safe to draw. */
data class HudFrame(val ok: Boolean, val error: String?, val version: Long, val unchanged: Boolean,
                    val ops: List<Map<String, Any?>>, val dropped: Long) {
    fun toMap(): Map<String, Any?> = if (!ok) mapOf("ok" to false, "error" to error) else
        mapOf("ok" to true, "error" to null, "version" to version, "unchanged" to unchanged, "ops" to ops, "dropped" to dropped)

    companion object {
        private fun bad() = HudFrame(false, LinkOps.BAD_REPLY, -1, false, emptyList(), 0)

        private fun number(v: Any?) = (v is Long) || (v is Double && v.isFinite())

        /** Python str[:n]: the first n code points. */
        private fun cap(s: String): String =
            if (s.codePointCount(0, s.length) <= LinkOps.MAX_HUD_TEXT) s else s.substring(0, s.offsetByCodePoints(0, LinkOps.MAX_HUD_TEXT))

        /** Clamp keeping the number's type: a Long stays a Long (a fractional bound truncates toward zero, as Python
         *  int()), a Double stays a Double. Explicit comparisons, so an in-range value comes back unchanged. */
        private fun clamp(v: Any, lo: Double, hi: Double): Any = when (v) {
            is Long -> if (v.toDouble() < lo) lo.toLong() else if (v.toDouble() > hi) hi.toLong() else v
            else -> { val d = v as Double; if (d < lo) lo else if (d > hi) hi else d }
        }

        private fun d(v: Any?) = (v as Number).toDouble()

        private fun fields(op: Map<*, *>, keys: List<String>): LinkedHashMap<String, Any?>? {
            val out = LinkedHashMap<String, Any?>()
            for (k in keys) {
                val v = op[k]
                out[k] = when {
                    k in LinkOps.NUMBER_KEYS -> if (number(v)) v else return null
                    v is String -> cap(v)
                    else -> return null
                }
            }
            return out
        }

        private fun canvas(op: Any?): LinkedHashMap<String, Any?>? {
            if (op !is Map<*, *> || op["op"] != "canvas") return null
            val w = op["w"]; val h = op["h"]
            if (w !is Long || h !is Long || w < 1 || h < 1 || w > LinkOps.MAX_CANVAS_PX || h > LinkOps.MAX_CANVAS_PX) return null
            val out = fields(op, GlassesHud.OP_KEYS.getValue("canvas")) ?: return null
            out["opacity"] = clamp(out["opacity"]!!, 0.0, 1.0)
            out["font_size"] = clamp(out["font_size"]!!, 0.0, minOf(w, h).toDouble())
            return out
        }

        private fun drawOp(op: Any?, W: Long, H: Long): LinkedHashMap<String, Any?>? {
            if (op !is Map<*, *>) return null
            val kind = op["op"] as? String ?: return null
            val keys = GlassesHud.OP_KEYS[kind] ?: return null
            if (kind == "canvas") return null
            val out = fields(op, keys) ?: return null
            val side = minOf(W, H).toDouble()
            for (k in keys) when (k) {
                "x" -> out[k] = clamp(out[k]!!, 0.0, W.toDouble())
                "y" -> out[k] = clamp(out[k]!!, 0.0, H.toDouble())
                "size", "stroke_width" -> out[k] = clamp(out[k]!!, 0.0, side)
            }
            if ("w" in out) out["w"] = clamp(out["w"]!!, 0.0, W.toDouble() - d(out["x"]))
            if ("h" in out) out["h"] = clamp(out["h"]!!, 0.0, H.toDouble() - d(out["y"]))
            return out
        }

        /** The first op must be a valid canvas (else every op is dropped); only the first MAX_HUD_OPS input ops are
         *  looked at; ops that fail the checks and any further canvas are dropped and counted in `dropped`. */
        fun parse(reply: String): HudFrame {
            val r = LinkOps.load(reply) ?: return bad()
            if (r["ok"] != true) return HudFrame(false, r["error"] as String, -1, false, emptyList(), 0)
            val version = r["version"] as? Long ?: return bad()
            if (version < 0) return bad()
            if (r["unchanged"] == true) return HudFrame(true, null, version, true, emptyList(), 0)
            val raw = r["ops"] as? List<*> ?: return bad()
            val ops = ArrayList<Map<String, Any?>>()
            val c = raw.firstOrNull()?.let { canvas(it) }
            if (c != null) {
                ops.add(c)
                val W = c["w"] as Long; val H = c["h"] as Long
                for (op in raw.subList(1, minOf(raw.size, LinkOps.MAX_HUD_OPS))) drawOp(op, W, H)?.let { ops.add(it) }
            }
            return HudFrame(true, null, version, false, ops, (raw.size - ops.size).toLong())
        }
    }
}
