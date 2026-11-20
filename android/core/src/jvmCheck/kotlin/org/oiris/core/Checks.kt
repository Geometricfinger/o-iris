package org.oiris.core

import java.io.File

/** JVM checks for the pure-Kotlin core (run by Gradle `:core:check` -> `jvmChecks`, and in CI).
 *  Self-contained on purpose: no test framework dependency to licence. Exit code 1 on any failure. */
private var failures = 0
private var passed = 0
fun expect(name: String, cond: Boolean) { if (cond) passed++ else { failures++; System.err.println("FAIL: $name") } }
fun hex(b: ByteArray) = b.joinToString("") { "%02x".format(it) }
fun unhex(s: String) = ByteArray(s.length / 2) { s.substring(2 * it, 2 * it + 2).toInt(16).toByte() }
inline fun throws(block: () -> Unit) = try { block(); false } catch (e: Exception) { true }

@Suppress("UNCHECKED_CAST")
fun main(args: Array<String>) {
    val root = File(args.getOrElse(0) { "../.." })
    val v = Json.decode(File(root, "tests/vectors/noise_xx.json").readText()) as Map<String, Any?>
    // ---- Noise vectors (bytes produced by the Python pod side) ----
    val ist = unhex(v["init_static"] as String); val rst = unhex(v["resp_static"] as String)
    expect("static pub", hex(Prim.pub(ist)) == v["init_static_pub"])
    expect("fingerprint", SecureChannel.fingerprint(Prim.pub(rst)) == v["resp_fingerprint"])
    val i = NoiseXX(true, ist, ephemeral = unhex(v["init_ephemeral"] as String))
    val r = NoiseXX(false, rst, ephemeral = unhex(v["resp_ephemeral"] as String))
    val hs = v["handshake"] as List<String>
    val m1 = i.writeMessage(); expect("msg1", hex(m1) == hs[0]); r.readMessage(m1)
    val m2 = r.writeMessage(); expect("msg2", hex(m2) == hs[1]); i.readMessage(m2)
    val m3 = i.writeMessage(); expect("msg3", hex(m3) == hs[2]); r.readMessage(m3)
    expect("handshake hash", hex(i.handshakeHash) == v["handshake_hash"] && hex(r.handshakeHash) == v["handshake_hash"])
    var rekeyed = false
    for (t in v["transport"] as List<Map<String, Any?>>) {
        val plain = (t["plain"] as String).toByteArray()
        val (tx, rx) = if (t["dir"] == "i2r") i.send to r.recv else r.send to i.recv
        if (t["gen"] == 1L && !rekeyed) { i.send.rekey(); r.recv.rekey(); rekeyed = true }
        val c = tx.encrypt(ByteArray(0), plain)
        expect("transport ${t["plain"]}", hex(c) == t["cipher"])
        expect("decrypt ${t["plain"]}", rx.decrypt(ByteArray(0), c).contentEquals(plain))
    }
    // tamper is rejected
    val c = i.send.encrypt(ByteArray(0), "x".toByteArray()); c[0] = (c[0].toInt() xor 1).toByte()
    expect("tamper rejected", throws { r.recv.decrypt(ByteArray(0), c) })

    // ---- SecureChannel: pairing check, rekey generations, fragments ----
    val pod = SecureChannel(false, rst, SecureChannel.fingerprint(Prim.pub(ist)), rekeyMessages = 3)
    val phone = SecureChannel(true, ist, SecureChannel.fingerprint(Prim.pub(rst)), rekeyMessages = 3)
    pod.readHandshake(phone.writeHandshake()); phone.readHandshake(pod.writeHandshake()); pod.readHandshake(phone.writeHandshake())
    expect("ready", pod.ready && phone.ready)
    repeat(8) { k -> expect("rekey msg $k", pod.decrypt(phone.encrypt("m$k".toByteArray())).contentEquals("m$k".toByteArray())) }
    expect("rekeyed", phone.rekeys == 2)
    val big = ByteArray(150_000) { (it % 251).toByte() }
    val ra = Reassembler(pod); var got: ByteArray? = null
    val frames = Fragments.split(phone, big); frames.forEach { got = ra.feed(it) ?: got }
    expect("fragments", frames.size == 3 && got!!.contentEquals(big))
    val wrong = SecureChannel(true, ist, "00".repeat(16))
    val pod2 = SecureChannel(false, rst, SecureChannel.fingerprint(Prim.pub(ist)))
    pod2.readHandshake(wrong.writeHandshake())
    expect("wrong pairing refused", throws { wrong.readHandshake(pod2.writeHandshake()) })

    // ---- Enrollment protocol strings (shared with Python) ----
    val req = (Json.decode(File(root, "tests/vectors/enrollment_protocol.json").readText()) as Map<String, Any?>)["requests"] as Map<String, String>
    expect("p status", EnrollmentProtocol.status() == req["status"])
    expect("p warning", EnrollmentProtocol.warning("us-other") == req["warning"])
    expect("p start", EnrollmentProtocol.startSession() == req["start_session"])
    expect("p add", EnrollmentProtocol.addPerson("S", "Alex", "c-ok", false) == req["add_person"])
    expect("p add unicode/null", EnrollmentProtocol.addPerson("S", "Mom é", null, false) == req["add_person_no_consent"])
    expect("p shot", EnrollmentProtocol.shot("P", Angle.LEFT, 2, 1, byteArrayOf(1, 2, 3, 4, 5, 6)) == req["shot"])
    expect("p finish", EnrollmentProtocol.finish("P") == req["finish"])
    expect("p end", EnrollmentProtocol.endSession("S") == req["end_session"])

    // ---- EnrollmentFlow against a scripted pod ----
    var optedIn = false; val sent = mutableListOf<Map<String, Any?>>()
    val pod3: (String) -> String = { s ->
        val m = Json.decode(s) as Map<String, Any?>; sent += m
        when (m["op"]) {
            "status" -> Json.encode(mapOf("ok" to true, "opted_in" to optedIn, "modes" to if (optedIn) listOf("personal") else listOf<String>()))
            "warning" -> Json.encode(mapOf("ok" to true, "text" to "Face matching is about to be turned on."))
            "opt_in" -> { optedIn = m["typed"] == CONFIRM_PHRASE; Json.encode(mapOf("ok" to optedIn)) }
            "start_session" -> Json.encode(mapOf("ok" to true, "session" to "S1"))
            "add_person" -> Json.encode(mapOf("ok" to true, "person" to "P${sent.size}"))
            "shot" -> { val a = m["angle"]; val ok = (m["w"] as Long) >= 64
                Json.encode(mapOf("ok" to true, "accepted" to ok, "reason" to if (ok) "ok" else "move closer", "code" to a)) }
            "finish" -> Json.encode(mapOf("ok" to true, "complete" to true, "message" to "complete"))
            else -> Json.encode(mapOf("ok" to true))
        }
    }
    val f = EnrollmentFlow(pod3)
    expect("off by default -> locked", f.open() == EnrollmentFlow.Stage.LOCKED)
    expect("no capture while locked", throws { f.startPerson("Alex", "c") })
    f.showWarning()
    expect("wrong phrase keeps it off", !f.optIn("yes") && !optedIn && f.stage == EnrollmentFlow.Stage.WARNING)
    expect("typed opt-in", f.optIn("I UNDERSTAND") && f.stage == EnrollmentFlow.Stage.READY)
    for (n in 1..12) {   // no count limit: enroll many people in one session
        f.startPerson("Person $n", "c$n")
        expect("first prompt", f.lastMessage == Angle.FRONT.prompt || n > 1)
        val small = ByteArray(10 * 10 * 3)
        expect("rejected shot reason", !f.submit(Angle.FRONT, 10, 10, small) && f.lastMessage == "move closer")
        for (a in Angle.values()) { val px = ByteArray(64 * 64 * 3) { 7 }; expect("accept $a", f.submit(a, 64, 64, px)); expect("frame wiped", px.all { it == 0.toByte() }) }
        expect("checklist all ticked", f.checklist().all { it.second } && f.nextAngle() == null)
        expect("finish", f.finish() && f.lastMessage == "Template complete for Person $n")
    }
    expect("12 enrolled, one session", f.enrolledThisSession == 12 && sent.count { it["op"] == "start_session" } == 1)
    f.end(); expect("end session sent", sent.last()["op"] == "end_session")

    // ---- Alert visuals from the C7 tokens ----
    val av = AlertVisuals.fromTokens(File(root, "oiris/ui/design_tokens.json").readText())
    expect("three distinct shapes", av.values.map { it.shape }.toSet().size == 3)
    expect("words", av["medium"]!!.word == "Medium signal" && av.values.none { "robot" in it.word.lowercase() })
    expect("engine map", AlertVisuals.forEngine("attention") == "medium" && AlertVisuals.forEngine(null) == null)
    expect("bad tokens refused", throws { AlertVisuals.fromTokens("{\"alert_levels\":{\"low\":{}},\"color\":{\"alert\":{}}}") })
    expect("json roundtrip", Json.encode(Json.decode("{\"b\":[1,2.5,\"x\\n\"],\"a\":null}")) == "{\"a\":null,\"b\":[1,2.5,\"x\\n\"]}")


    // ---- BLE GATT framing of Noise frames (shared with oiris/link/ble.py) ----
    val bv = (Json.decode(File(root, "tests/vectors/ble_framing.json").readText()) as Map<String, Any?>)["cases"] as List<Map<String, Any?>>
    for (c in bv) {
        val frame = unhex(c["frame"] as String); val mtu = (c["mtu"] as Long).toInt(); val seq0 = (c["seq0"] as Long).toInt()
        val segs = BleFraming.segment(frame, mtu, seq0)
        expect("ble segments n=${frame.size} mtu=$mtu", segs.map { hex(it) } == c["segments"] as List<String>)
        expect("ble segment fits ATT mtu=$mtu", segs.all { it.size <= mtu - 3 })
        val d = Desegmenter(); var got: ByteArray? = null
        segs.forEach { s -> d.feed(s)?.let { got = it } }
        expect("ble reassembly n=${frame.size}", got != null && got!!.contentEquals(frame))
    }
    run {   // a lost segment drops the partial frame; the next frame still arrives
        val f1 = ByteArray(1000) { 1 }; val f2 = ByteArray(700) { 2 }
        val s1 = BleFraming.segment(f1, 185, 0); val s2 = BleFraming.segment(f2, 185, s1.size)
        val d = Desegmenter(); val out = mutableListOf<ByteArray>()
        (s1.filterIndexed { i, _ -> i != 2 } + s2).forEach { s -> d.feed(s)?.let { out.add(it) } }
        expect("ble gap drops partial frame", out.size == 1 && out[0].contentEquals(f2) && d.dropped >= 1)
    }
    run {   // Noise frames over BLE segments end to end (loopback)
        val podB = SecureChannel(false, rst, SecureChannel.fingerprint(Prim.pub(ist)))
        val phoneB = SecureChannel(true, ist, SecureChannel.fingerprint(Prim.pub(rst)))
        val toPod = Desegmenter(); val toPhone = Desegmenter(); var seqA = 0; var seqB = 0
        fun p2pod(f: ByteArray): ByteArray { var r: ByteArray? = null; BleFraming.segment(f, 23, seqA).also { seqA += it.size }.forEach { s -> toPod.feed(s)?.let { r = it } }; return r!! }
        fun pod2p(f: ByteArray): ByteArray { var r: ByteArray? = null; BleFraming.segment(f, 23, seqB).also { seqB += it.size }.forEach { s -> toPhone.feed(s)?.let { r = it } }; return r!! }
        podB.readHandshake(p2pod(phoneB.writeHandshake())); phoneB.readHandshake(pod2p(podB.writeHandshake())); podB.readHandshake(p2pod(phoneB.writeHandshake()))
        expect("noise over ble handshake", podB.ready && phoneB.ready)
        val msg = ByteArray(5000) { (it % 97).toByte() }
        expect("noise over ble message", podB.decrypt(p2pod(phoneB.encrypt(msg))).contentEquals(msg))
    }

    // ---- Thermal live view pixels (shared with oiris/ui/glasses.thermal_image) ----
    val tv = (Json.decode(File(root, "tests/vectors/thermal_render.json").readText()) as Map<String, Any?>)["cases"] as List<Map<String, Any?>>
    for (c in tv) {
        val cel = (c["celsius"] as List<Any?>).map { (it as Number).toDouble() }.toDoubleArray()
        val g = (c["gray"] as List<Any?>).map { (it as Number).toInt() }
        val rgb = (c["rgb"] as List<Any?>).map { (it as Number).toInt() }
        expect("thermal gray ${c["h"]}x${c["w"]}", ThermalRender.gray(cel).toList() == g)
        expect("thermal colour ${c["h"]}x${c["w"]}", ThermalRender.rgb(cel).toList() == rgb)
        val a = ThermalRender.argb(cel, false)
        expect("thermal argb gray ${c["h"]}x${c["w"]}", a.size == g.size && a.indices.all { (a[it] and 0xFF) == g[it] && (a[it] ushr 24) == 0xFF })
    }

    // ---- Front-camera eye features (shared with oiris/features/eye.py) ----
    val ev = Json.decode(File(root, "tests/vectors/eye_android.json").readText()) as Map<String, Any?>
    for (c in ev["blinks"] as List<Map<String, Any?>>) {
        val o = (c["openness"] as List<Any?>).map { (it as Number).toDouble() }.toDoubleArray()
        val exp = (c["blinks"] as List<List<Any?>>).map { (it[0] as Number).toInt() to (it[1] as Number).toInt() }
        expect("blinks fps=${c["fps"]}", EyeFeatures.blinks(o, (c["fps"] as Number).toDouble()) == exp)
    }
    for (c in ev["pupil"] as List<Map<String, Any?>>) {
        val roi = (c["roi"] as List<Any?>).map { (it as Number).toDouble() }.toDoubleArray()
        val (d, q) = EyeFeatures.pupilDiameter(roi, (c["w"] as Number).toInt(), (c["h"] as Number).toInt())
        expect("pupil diameter ${c["diameter"]}", kotlin.math.abs(d - (c["diameter"] as Number).toDouble()) < 1e-9)
        expect("pupil quality ${c["quality"]}", kotlin.math.abs(q - (c["quality"] as Number).toDouble()) < 1e-9)
    }

    // ---- Scan-rate limit, GPS policy, mic level, timeline ----
    var now = 0L; val lim = ScanRateLimiter(clock = { now })
    val allowed = (0 until 8).count { now = it * 1000L; lim.tryStart() }
    expect("<=5 scan starts per 30 s", allowed == 5 && lim.waitMs() == 30_000L - 7_000L)
    now = 30_000L; expect("scan allowed after window", lim.tryStart())
    var burst = 0; now = 100_000L; repeat(100) { if (lim.tryStart()) burst++ }
    expect("burst capped", burst == 5)
    expect("gps ok", GpsPolicy.classify(12.35, -45.68, 8f, 1_000).fix == "gps")
    expect("gps coarse", GpsPolicy.classify(12.35, -45.68, 60f, 1_000).fix == "gps_coarse")
    expect("gps stale -> no_fix", GpsPolicy.classify(12.35, -45.68, 8f, 600_000).let { it.fix == "no_fix" && it.lat == null })
    expect("gps inaccurate -> no_fix", GpsPolicy.classify(12.35, -45.68, 500f, 1_000).fix == "no_fix")
    expect("gps none -> no_fix", GpsPolicy.classify(null, null, null, null).fix == "no_fix")
    expect("mic silence", MicLevel.dbfs(ShortArray(256)) == -120.0)
    expect("mic full scale ~0 dBFS", kotlin.math.abs(MicLevel.dbfs(ShortArray(256) { if (it % 2 == 0) 32767 else -32768 })) < 0.01)
    val sw = listOf(Sweep("a", "home", 10.0, 20.0, "0", "clear"), Sweep("b", "home", 30.0, 40.0, "2", "signatures"), Sweep("c", "shop", 50.0, 60.0, "1", "signatures"))
    expect("timeline newest first", Timeline.ordered(sw).map { it.id } == listOf("c", "b", "a"))
    expect("timeline new since review", Timeline.newSince(sw, 35.0).map { it.id } == listOf("c", "b"))

    voiceLinkChecks(root)
    glassesHudChecks(root)
    timeSyncChecks(root)
    linkOpsChecks(root)
    voiceEnrollmentChecks(root)
    fusionChecks(root)
    powerChecks(root)
    hudPaintChecks(root)

    println("core checks: $passed passed, $failures failed")
    if (failures > 0) kotlin.system.exitProcess(1)
}

private const val VL_TOL = 1e-6   // cosine, template components and speech seconds

private fun nums(v: Any?): List<Double> = (v as List<Any?>).map { (it as Number).toDouble() }
private fun floats(v: Any?): FloatArray = nums(v).map { it.toFloat() }.toFloatArray()

/** The VAD test audio of scripts/make_android_vectors.py vl_signal: the bed sines always, plus the tone
 *  sines times a piecewise-constant gain over [start, end) samples; float32. */
@Suppress("UNCHECKED_CAST")
fun vlSignal(rate: Int, samples: Int, bed: List<List<Any?>>, tones: List<List<Any?>>, envelope: List<List<Any?>>): FloatArray {
    val env = DoubleArray(samples)
    for (e in envelope) { val g = (e[2] as Number).toDouble(); for (i in (e[0] as Number).toInt() until kotlin.math.min(samples,(e[1] as Number).toInt())) env[i] = g }
    return FloatArray(samples) { i ->
        val t = i.toDouble() / rate
        var x = 0.0
        for (b in bed) x += (b[1] as Number).toDouble() * kotlin.math.sin(2 * kotlin.math.PI * (b[0] as Number).toDouble() * t)
        var s = 0.0
        for (k in tones) s += (k[1] as Number).toDouble() * kotlin.math.sin(2 * kotlin.math.PI * (k[0] as Number).toDouble() * t)
        (x + env[i] * s).toFloat()
    }
}

// ---- M-17 voice link rules (shared with oiris/modules/voice_link; vectors from the Python code) ----
@Suppress("UNCHECKED_CAST")
fun voiceLinkChecks(root: File) {
    val v = Json.decode(File(root, "tests/vectors/voice_link.json").readText()) as Map<String, Any?>
    val c = v["constants"] as Map<String, Any?>
    fun num(k: String) = (c[k] as Number).toDouble()
    expect("vl constants: decision", VoiceLink.MATCH_THRESHOLD == num("match_threshold") && VoiceLink.UNSURE_MARGIN == num("unsure_margin") &&
        VoiceLink.SPOOF_THRESHOLD == num("spoof_threshold"))
    expect("vl constants: names", VoiceLink.NAME_MAX_WORDS.toDouble() == num("name_max_words") &&
        VoiceLink.NAME_MAX_CHARS.toDouble() == num("name_max_chars") && VoiceLink.PHRASE_PREFIX == c["phrase_prefix"] &&
        VoiceLink.ANNOUNCE_REPEAT_S == num("announce_repeat_s"))
    expect("vl constants: vad", VoiceLink.FRAME_S == num("frame_s") && VoiceLink.HANGOVER_S == num("hangover_s") &&
        VoiceLink.MIN_SEGMENT_S == num("min_segment_s") && VoiceLink.FLOOR_PERCENTILE == num("floor_percentile") &&
        VoiceLink.POWER_EPS == num("power_eps") && VoiceLink.SILENCE_DBFS == num("silence_dbfs") &&
        VoiceLink.VAD_SNR_DB == num("vad_snr_db") && VoiceLink.VAD_FLOOR_DBFS == num("vad_floor_dbfs"))

    for ((k, cs) in (v["cosine"] as List<Map<String, Any?>>).withIndex()) {
        val got = VoiceLink.cosine(floats(cs["embedding"]), floats(cs["template"]))
        expect("vl cosine #$k", kotlin.math.abs(got - (cs["cosine"] as Number).toDouble()) < VL_TOL)
    }
    for ((k, t) in (v["template"] as List<Map<String, Any?>>).withIndex()) {
        val got = VoiceLink.template((t["embeddings"] as List<Any?>).map { floats(it) })
        val exp = nums(t["template"])
        expect("vl template #$k", got.size == exp.size && got.indices.all { kotlin.math.abs(got[it] - exp[it]) < VL_TOL })
    }

    for (d in v["decide"] as List<Map<String, Any?>>) {
        val scores = (d["scores"] as List<List<Any?>>).map { (it[0] as Number).toDouble() to it[1] as String }
        val spoof = (d["spoof_score"] as Number?)?.toDouble()
        var asked = false
        val r = VoiceLink.decide(scores, (d["match_threshold"] as Number).toDouble(), (d["unsure_margin"] as Number).toDouble(),
            d["self_id"] as String?, { asked = true; spoof }, (d["spoof_threshold"] as Number).toDouble())
        expect("vl decide ${d["name"]}", r.outcome.wire == d["outcome"] && r.personId == d["person_id"])
        val reachesSpoof = r.outcome == VoiceLink.Outcome.HEARD || r.outcome == VoiceLink.Outcome.SPOOFED
        expect("vl decide ${d["name"]}: spoof asked only at its step", asked == reachesSpoof)
    }

    val vocab = Json.decode(File(root, "schema/vocabulary.json").readText()) as Map<String, Any?>
    val banned = VoiceLink.bannedRegex(vocab["banned_patterns"] as List<String>)
    for (n in v["names"] as List<Map<String, Any?>>) {
        val got = VoiceLink.speakableName(n["name"] as String?, banned)
        expect("vl name ${n["name"]}", got == n["speakable"] && (got == null || VoiceLink.phrase(got) == n["phrase"]))
    }

    for ((k, seq) in (v["repeat"] as List<Map<String, Any?>>).withIndex()) {
        val lim = VoiceLink.RepeatLimiter((seq["repeat_s"] as Number).toDouble())
        val steps = seq["steps"] as List<Map<String, Any?>>
        val got = steps.map { lim.allow(it["person_id"] as String, (it["t"] as Number).toDouble()) }
        expect("vl repeat #$k", got == steps.map { it["allowed"] })
    }

    for (a in v["vad"] as List<Map<String, Any?>>) {
        val rate = (a["rate"] as Number).toInt()
        val x = vlSignal(rate, (a["samples"] as Number).toInt(), a["bed"] as List<List<Any?>>, a["tones"] as List<List<Any?>>,
            a["envelope"] as List<List<Any?>>)
        val r = VoiceLink.detect(x, rate, (a["snr_db"] as Number).toDouble(), (a["floor_dbfs"] as Number).toDouble())
        val exp = (a["segments"] as List<List<Any?>>).map { (it[0] as Number).toInt() to (it[1] as Number).toInt() }
        expect("vl vad ${a["name"]}: segments", r.segments == exp)
        expect("vl vad ${a["name"]}: speech seconds", kotlin.math.abs(r.speechS - (a["speech_s"] as Number).toDouble()) < VL_TOL)
    }
}

// ---- Glasses HUD display list (shared with oiris/ui/glasses.display_list; vectors from the Python code) ----
@Suppress("UNCHECKED_CAST")
fun glassesHudChecks(root: File) {
    val v = Json.decode(File(root, "tests/vectors/glasses_hud.json").readText()) as Map<String, Any?>
    val tokenText = File(root, "oiris/ui/design_tokens.json").readText()
    val vocab = Json.decode(File(root, "schema/vocabulary.json").readText()) as Map<String, Any?>
    val banned = VoiceLink.bannedRegex(vocab["banned_patterns"] as List<String>)
    val c = v["constants"] as Map<String, Any?>
    expect("hud constants: decimals", GlassesHud.COORD_DECIMALS.toLong() == c["coord_decimals"])
    expect("hud constants: op keys", GlassesHud.OP_KEYS == c["op_keys"])
    expect("hud constants: tick", GlassesHud.TICK_POINTS.map { listOf(it.first, it.second) } == c["tick_points"] &&
        GlassesHud.TICK_SIZE == c["tick_size"])
    expect("hud constants: guide order", GlassesHud.GUIDE_ORDER == c["guide_order"])
    expect("hud constants: cue edges", GlassesHud.CUE_EDGES == c["cue_edges"] && GlassesHud.CUE_NONE == c["cue_none"] &&
        GlassesHud.CUE_THICK_STEP.toLong() == c["cue_thick_step"])
    // every phrase direction word is either drawn on an edge or deliberately not drawn ("ahead")
    expect("hud cue: every direction word covered", (GlassesHud.CUE_EDGES.keys + GlassesHud.CUE_NONE) == (c["direction_words"] as List<String>).toSet())
    for (r in v["rounding"] as List<List<Any?>>)
        expect("hud rounding ${r[0]}", GlassesHud.roundCoord((r[0] as Number).toDouble()) == (r[1] as Number).toDouble())

    // the shipped tokens, with a vector's [path, value] overrides applied
    fun tokens(patch: List<List<Any?>>): GlassesHud.Tokens {
        val t = Json.decode(tokenText) as MutableMap<String, Any?>
        for (p in patch) {
            val path = p[0] as List<String>
            var d = t
            for (k in path.dropLast(1)) d = d[k] as MutableMap<String, Any?>
            d[path.last()] = p[1]
        }
        return GlassesHud.Tokens(t)
    }
    // ops exactly as Python (number types included) and in the fixed schema; or the same refusal
    fun same(name: String, case: Map<String, Any?>, build: (GlassesHud.Tokens) -> List<GlassesHud.Op>) {
        var got: List<Map<String, Any?>>? = null; var kind: String? = null; var msg: String? = null
        try { got = build(tokens(case["tokens_patch"] as List<List<Any?>>)).map { it.toMap() } }
        catch (e: Exception) { kind = e::class.simpleName; msg = e.message }
        val err = case["error"] as String?
        if (err != null) { expect("hud $name: $err", kind == err && (err != "LayoutError" || msg == case["message"])); return }
        val exp = case["ops"] as List<Map<String, Any?>>
        expect("hud $name: ops", got == exp)
        if (got != exp) System.err.println("  got ${got?.let { Json.encode(it) } ?: "$kind: $msg"}\n  exp ${Json.encode(exp)}")
        expect("hud $name: fixed schema", got != null && got.all { it.keys.toList() == GlassesHud.OP_KEYS[it["op"]] &&
            it.values.all { x -> x is String || x is Long || x is Double || x is Boolean } })
    }
    for (h in v["hud"] as List<Map<String, Any?>>) {
        val s = h["state"] as Map<String, Any?>
        val st = GlassesHud.State(level = s["level"] as String?, alertText = s["alert_text"] as String, newSighting = s["new_sighting"] as Boolean,
            thermal = s["thermal"] as Boolean, thermalColour = s["thermal_colour"] as Boolean, health = s["health"] as Map<String, String>,
            voiceTier = s["voice_tier"] as String, locked = s["locked"] as Boolean, panicArmed = s["panic_armed"] as Boolean,
            moving = s["moving"] as Boolean, cueDirection = s["cue_direction"] as String?)
        same(h["name"] as String, h) { GlassesHud.displayList(st, it, banned) }
    }
    for (g in v["guide"] as List<Map<String, Any?>>) {
        val s = g["guide"] as Map<String, Any?>
        val eg = GlassesHud.Guide(s["needed"] as List<String>, s["last_ok"] as Boolean?, s["last_reason"] as String, s["complete"] as Boolean)
        same("guide ${g["name"]}", g) { GlassesHud.guideDisplayList(eg, it, banned) }
    }
    for (m in v["motion"] as List<Map<String, Any?>>)
        expect("hud moving from motion ${m["motion"]}", GlassesHud.movingFromMotion(m["motion"] as String?) == m["moving"])
}

// ---- Clock sync between devices (shared with oiris/link/timesync.py; vectors from the Python code) ----
private const val TS_DRIFT_TOL = 1e-6   // drift ppm: the vectors' FLOAT_TOL

@Suppress("UNCHECKED_CAST")
fun timeSyncChecks(root: File) {
    val v = Json.decode(File(root, "tests/vectors/time_sync.json").readText()) as Map<String, Any?>
    val c = v["constants"] as Map<String, Any?>
    expect("ts constants", TimeSync.MAX_SAMPLES.toLong() == c["max_samples"] && TimeSync.MAX_DELAY_NS == c["max_delay_ns"] &&
        TimeSync.STALE_NS == c["stale_ns"] && TimeSync.DRIFT_MIN_SAMPLES.toLong() == c["drift_min_samples"] &&
        TimeSync.DRIFT_MIN_SPAN_NS == c["drift_min_span_ns"] && TimeSync.MAX_TIMESTAMP_NS == c["max_timestamp_ns"] &&
        TimeSync.PPM == c["ppm"])
    expect("ts reasons", listOf(TimeSync.REJECT_RANGE, TimeSync.REJECT_ORDER, TimeSync.REJECT_NEGATIVE, TimeSync.REJECT_SLOW,
        TimeSync.REJECT_REPLY) == c["reasons"])
    expect("ts quality names", TimeSync.Quality.values().map { it.wire } == c["quality"])
    fun four(x: Any?): List<Long> = (x as List<Any?>).map { it as Long }

    for (e in v["exchanges"] as List<Map<String, Any?>>) {
        val t0 = e["t0"] as Long; val t1 = e["t1"] as Long; val t2 = e["t2"] as Long; val t3 = e["t3"] as Long
        expect("ts ${e["name"]}: offset", TimeSync.offsetNs(t0, t1, t2, t3) == e["offset_ns"])
        expect("ts ${e["name"]}: delay", TimeSync.delayNs(t0, t1, t2, t3) == e["delay_ns"])
        expect("ts ${e["name"]}: reject", TimeSync.rejectReason(t0, t1, t2, t3) == e["reject"])
    }

    for (s in v["sessions"] as List<Map<String, Any?>>) {
        val n = s["name"]; val p = s["params"] as Map<String, Any?>
        val ts = TimeSync((p["max_samples"] as Long).toInt(), p["max_delay_ns"] as Long, p["stale_ns"] as Long)
        val accepted = (s["exchanges"] as List<Any?>).map { val x = four(it); ts.add(x[0], x[1], x[2], x[3]) == null }
        expect("ts $n: accepted", accepted == s["accepted"])
        val b = ts.best()
        expect("ts $n: best", (b?.let { listOf(it.t0, it.t1, it.t2, it.t3) }) == s["best"]?.let { four(it) })
        val d = ts.driftPpm(); val expD = (s["drift_ppm"] as Number?)?.toDouble()
        expect("ts $n: drift", if (expD == null) d == null else d != null && kotlin.math.abs(d - expD) <= TS_DRIFT_TOL)
        for (q in s["quality"] as List<Map<String, Any?>>)
            expect("ts $n: quality at ${q["now"]}", ts.quality(q["now"] as Long).wire == q["quality"])
        for (k in s["conversions"] as List<Map<String, Any?>>) {
            val got = try { ts.remoteToHost(k["t_remote"] as Long, k["at_host"] as Long) } catch (e: TimeSync.NotSynced) { null }
            expect("ts $n: remote ${k["t_remote"]} -> host", got == k["t_host"])
        }
    }

    for (r in v["requests"] as List<Map<String, Any?>>)
        expect("ts request ${r["t0"]}", TimeSync.request(r["t0"] as Long) == r["request"])
    for (r in v["replies"] as List<Map<String, Any?>>) {
        val ts = TimeSync()
        val why = ts.addReply(r["reply"] as String, r["t0"] as Long, r["t3"] as Long)
        val b = ts.best()
        expect("ts reply ${r["name"]}", why == r["reject"] && b?.offsetNs == r["offset_ns"] && b?.delayNs == r["delay_ns"])
    }

    // the floor rule itself, beside the vectors: Long division truncates toward zero, Python // does not
    expect("ts floorDiv, not truncation", TimeSync.offsetNs(1000, 0, 0, 1001) == -1001L && (-2001L) / 2 == -1000L)
    expect("ts maxSamples >= 1", throws { TimeSync(maxSamples = 0) })
}

// ---- Link operations for glasses (shared with oiris/link/ops.py and sideload.py; vectors from the Python code) ----
private val LO_OPS = setOf("unlock", "add_unlock_slot", "panic", "wipe_challenge", "wipe", "hud", "model_offer",
    "model_chunk", "model_commit", "model_status")

@Suppress("UNCHECKED_CAST")
fun linkOpsChecks(root: File) {
    val v = Json.decode(File(root, "tests/vectors/link_ops.json").readText()) as Map<String, Any?>
    val c = v["constants"] as Map<String, Any?>
    expect("lo constants", LinkOps.SECRET_BYTES.toLong() == c["secret_bytes"] && LinkOps.NONCE_BYTES.toLong() == c["nonce_bytes"] &&
        LinkOps.SIGNATURE_BYTES.toLong() == c["signature_bytes"] && LinkOps.WIPE_PREFIX == c["wipe_prefix"] &&
        LinkOps.UNLOCK_SLOTS == c["unlock_slots"] && LinkOps.MAX_HUD_OPS.toLong() == c["max_hud_ops"] &&
        LinkOps.MAX_HUD_TEXT.toLong() == c["max_hud_text"] && LinkOps.MAX_CANVAS_PX == c["max_canvas_px"] &&
        LinkOps.NUMBER_KEYS == c["number_keys"] && LinkOps.MAX_MODEL_BYTES == c["max_model_bytes"] &&
        LinkOps.CHUNK_MIN.toLong() == c["chunk_min"] && LinkOps.CHUNK_MAX.toLong() == c["chunk_max"] &&
        LinkOps.MAX_MISSING_RANGES.toLong() == c["max_missing_ranges"])
    expect("lo patterns", LinkOps.NAME_RE.pattern == c["name_re"] && LinkOps.SLOT_RE.pattern == c["slot_re"] &&
        LinkOps.SHA256_RE.pattern == c["sha256_re"])
    val e = c["errors"] as Map<String, Any?>
    expect("lo error texts", LinkOps.NOT_AVAILABLE == e["not_available"] && LinkOps.BAD_REPLY == e["bad_reply"] &&
        LinkOps.UNLOCK_REFUSED == e["unlock_refused"] && LinkOps.ADD_SLOT_REFUSED == e["add_slot_refused"] &&
        LinkOps.WIPE_REFUSED == e["wipe_refused"])

    fun bytes(x: Any?) = unhex(x as String)
    fun str(a: Map<String, Any?>, k: String) = a[k] as String
    fun build(op: String, a: Map<String, Any?>): String = when (op) {
        "unlock" -> LinkOps.unlock(str(a, "slot"), bytes(a["secret"]))
        "add_unlock_slot" -> LinkOps.addUnlockSlot(str(a, "slot"), bytes(a["secret"]))
        "panic" -> LinkOps.panic()
        "wipe_challenge" -> LinkOps.wipeChallenge()
        "wipe" -> LinkOps.wipe(bytes(a["command"]), bytes(a["signature"]))
        "hud" -> LinkOps.hud(a["since"] as Long)
        "model_offer" -> LinkOps.modelOffer(str(a, "name"), str(a, "slot"), str(a, "sha256"), a["size"] as Long, (a["chunk_size"] as Long).toInt())
        "model_chunk" -> LinkOps.modelChunk(str(a, "sha256"), a["index"] as Long, bytes(a["data"]))
        "model_commit" -> LinkOps.modelCommit(str(a, "sha256"))
        "model_status" -> LinkOps.modelStatus()
        else -> error("no builder for $op")
    }
    fun parsed(parser: String, reply: String): Map<String, Any?> = when (parser) {
        "unlock", "add_unlock_slot", "panic", "ack" -> LinkOps.parseAck(reply).toMap()
        "wipe_challenge" -> LinkOps.parseWipeChallenge(reply).toMap()
        "wipe" -> LinkOps.parseWipe(reply).toMap()
        "hud" -> HudFrame.parse(reply).toMap()
        "model_offer", "offer" -> LinkOps.parseOffer(reply).toMap()
        "model_chunk", "chunk" -> LinkOps.parseChunk(reply).toMap()
        "model_commit", "commit" -> LinkOps.parseCommit(reply).toMap()
        "model_status", "status" -> LinkOps.parseStatus(reply).toMap()
        else -> error("no parser $parser")
    }

    // one request/response pair (or more) per op: the request built here equals Python's byte for byte, and the
    // real pod reply parses to the same values
    val seen = HashSet<String>()
    for (k in v["ops"] as List<Map<String, Any?>>) {
        val op = k["op"] as String; seen += op
        expect("lo ${k["name"]}: request", build(op, k["args"] as Map<String, Any?>) == k["request"])
        expect("lo ${k["name"]}: parsed", parsed(op, k["reply"] as String) == k["parsed"])
    }
    expect("lo every op has a vector", seen == LO_OPS)
    for (r in v["replies"] as List<Map<String, Any?>>)
        expect("lo reply ${r["parser"]} ${r["name"]}", parsed(r["parser"] as String, r["reply"] as String) == r["parsed"])
    for (h in v["hud"] as List<Map<String, Any?>>)
        expect("lo hud ${h["name"]}", HudFrame.parse(h["reply"] as String).toMap() == h["parsed"])

    // the wipe command and its Ed25519 signature (deterministic), reproduced with Bouncy Castle
    val w = v["wipe"] as Map<String, Any?>
    val cmd = LinkOps.wipeCommand(bytes(w["nonce"]))
    expect("lo wipe command", hex(cmd) == w["command"])
    val priv = org.bouncycastle.crypto.params.Ed25519PrivateKeyParameters(bytes(w["seed"]), 0)
    expect("lo wipe verify key", hex(priv.generatePublicKey().encoded) == w["verify_key"])
    val signer = org.bouncycastle.crypto.signers.Ed25519Signer().apply { init(true, priv); update(cmd, 0, cmd.size) }
    expect("lo wipe signature", hex(signer.generateSignature()) == w["signature"])

    for (k in v["chunking"] as List<Map<String, Any?>>)
        expect("lo chunks ${k["size"]}/${k["chunk_size"]}", LinkOps.chunkCount(k["size"] as Long, (k["chunk_size"] as Long).toInt()) == k["chunks"])
    for (b in v["base64url"] as List<Map<String, Any?>>) {
        val raw = bytes(b["hex"])
        expect("lo b64url ${b["hex"]}", LinkOps.b64url(raw) == b["b64url"] &&
            LinkOps.fromB64url(b["b64url"], raw.size)?.contentEquals(raw) == true)
    }
    val s = LinkOps.b64url(ByteArray(32) { it.toByte() })
    expect("lo b64url strict", LinkOps.fromB64url("$s=", 32) == null && LinkOps.fromB64url(s.dropLast(1) + "B", 32) == null &&
        LinkOps.fromB64url("+" + s.drop(1), 32) == null && LinkOps.fromB64url(s, 31) == null && LinkOps.fromB64url(5L, 32) == null &&
        LinkOps.fromB64url("AAE", maxBytes = 1) == null && LinkOps.fromB64url("AAE", maxBytes = 2)?.contentEquals(byteArrayOf(0, 1)) == true)
}

// ---- Phone-guided voice enrollment (shared with oiris/modules/face_match/remote.py; vectors from the real pod) ----
fun sha256(b: ByteArray): String = hex(java.security.MessageDigest.getInstance("SHA-256").digest(b))

/** The integer-only synthetic PCM of scripts/make_android_vectors.py ve_pcm: a triangle wave gated on and off. */
fun vePcm(g: Map<String, Any?>): ShortArray {
    fun l(k: String) = g[k] as Long
    val period = l("period"); val amp = l("amp"); val on = l("on"); val off = l("off"); val lead = l("lead"); val half = period / 2
    return ShortArray(l("samples").toInt()) { i ->
        val t = i % period; val k = i - lead
        if (k >= 0 && k % (on + off) < on) (Math.floorDiv(2 * amp * minOf(t, period - t), half) - amp).toShort() else 0
    }
}

fun leBytes(pcm: ShortArray) = ByteArray(pcm.size * 2) { if (it % 2 == 0) pcm[it / 2].toByte() else (pcm[it / 2].toInt() shr 8).toByte() }

private const val VE_UNEXPECTED = "{\"ok\":false,\"error\":\"unexpected request\"}"
private const val VE_ONE_PROMPT = "{\"ok\":true,\"prompts\":[{\"id\":\"a\",\"text\":\"Say a.\",\"tone\":\"low\"}]}"
private const val VE_STARTED = "{\"ok\":true,\"windows\":3,\"max_windows\":10,\"target_speech_s\":20.0,\"window_s\":7.0}"
private val VE_JUNK = listOf("", "{", "[]", "null", "{\"ok\":true,\"prompts\":[null]}", "{\"ok\":true,\"windows\":true}", "\"x\"",
    "{\"ok\":1e999}", "{\"ok\":true,\"accepted\":true,\"remaining\":\"0\"}")

@Suppress("UNCHECKED_CAST")
fun voiceEnrollmentChecks(root: File) {
    val v = Json.decode(File(root, "tests/vectors/voice_enrollment.json").readText()) as Map<String, Any?>
    val c = v["constants"] as Map<String, Any?>
    expect("ve constants", VoiceEnrollment.OPS == c["ops"] && VoiceEnrollment.VOICE_OFF == c["voice_off"] &&
        LinkOps.BAD_REPLY == c["bad_reply"] && VoiceEnrollment.MAX_SAMPLE_S.toLong() == c["max_sample_s"] &&
        VoiceEnrollment.MAX_PCM_BYTES.toLong() == c["max_pcm_bytes"] && VoiceEnrollment.MAX_PCM_B64.toLong() == c["max_pcm_b64"] &&
        VoiceEnrollment.RATE_MIN.toLong() == c["rate_min"] && VoiceEnrollment.RATE_MAX.toLong() == c["rate_max"] &&
        VoiceEnrollment.MAX_PROMPTS.toLong() == c["max_prompts"] && VoiceEnrollment.MAX_REPLY_TEXT.toLong() == c["max_reply_text"] &&
        VoiceEnrollment.MAX_REFUSED.toLong() == c["max_refused"])
    expect("ve stages", VoiceEnrollmentFlow.Stage.values().map { it.wire } == c["stages"])
    val fullMax = (c["full_request_max"] as Long).toInt()

    for (p in v["pcm"] as List<Map<String, Any?>>)
        expect("ve pcm ${p["gen"]}", sha256(leBytes(vePcm(p["gen"] as Map<String, Any?>))) == p["sha256"])

    fun str(a: Map<String, Any?>, k: String) = a[k] as String
    fun pcmOf(a: Map<String, Any?>): ShortArray =
        (a["pcm"] as List<Any?>?)?.let { l -> ShortArray(l.size) { (l[it] as Long).toInt().toShort() } } ?: vePcm(a["gen"] as Map<String, Any?>)
    fun build(op: String, a: Map<String, Any?>): String = when (op) {
        "voice_prompts" -> EnrollmentProtocol.voicePrompts()
        "voice_start" -> EnrollmentProtocol.voiceStart(str(a, "person"), str(a, "consent"))
        "voice_sample" -> EnrollmentProtocol.voiceSample(str(a, "person"), str(a, "prompt_id"), (a["rate"] as Long).toInt(), pcmOf(a))
        "voice_finish" -> EnrollmentProtocol.voiceFinish(str(a, "person"))
        "voice_cancel" -> EnrollmentProtocol.voiceCancel(str(a, "person"))
        else -> error("no builder for $op")
    }
    fun parsed(op: String, reply: String): Map<String, Any?> = when (op) {
        "voice_prompts" -> VoiceEnrollment.parsePrompts(reply)
        "voice_start" -> VoiceEnrollment.parseStart(reply)
        "voice_sample" -> VoiceEnrollment.parseSample(reply)
        "voice_finish" -> VoiceEnrollment.parseFinish(reply)
        "voice_cancel", "ack" -> VoiceEnrollment.parseAck(reply)
        else -> error("no parser $op")
    }.toMap()
    /** The request matches the vector: in full, or by SHA-256 and length for the long ones. */
    fun sameRequest(req: String, k: Map<String, Any?>): Boolean =
        if (k["request"] != null) req == k["request"]
        else req.length > fullMax && req.length.toLong() == k["request_length"] && sha256(req.toByteArray(Charsets.US_ASCII)) == k["request_sha256"]

    for (k in v["requests"] as List<Map<String, Any?>>)
        expect("ve request ${k["name"]}", build(k["op"] as String, k["args"] as Map<String, Any?>) == k["request"])
    run {   // a sample request carries exactly the samples, little-endian; the builder leaves the caller's array alone
        val pcm = shortArrayOf(1, -2, 0x1234)
        val req = Json.decode(EnrollmentProtocol.voiceSample("P", "count", 16000, pcm)) as Map<String, Any?>
        expect("ve pcm16 little-endian", java.util.Base64.getDecoder().decode(req["pcm16"] as String).contentEquals(
            byteArrayOf(1, 0, -2, -1, 0x34, 0x12)) && pcm.contentEquals(shortArrayOf(1, -2, 0x1234)))
    }

    // the real pod's replies (temporary store, fake speaker): every op, accepted and refused, parsed alike
    val seen = HashSet<String>()
    for (k in v["ops"] as List<Map<String, Any?>>) {
        val op = k["op"] as String; seen += op
        val a = k["args"] as Map<String, Any?>?
        if (a != null) expect("ve ${k["name"]}: request", sameRequest(build(op, a), k))
        expect("ve ${k["name"]}: parsed", parsed(op, k["reply"] as String) == k["parsed"])
    }
    expect("ve every op has a vector", seen == VoiceEnrollment.OPS.toSet())
    val codes = (v["ops"] as List<Map<String, Any?>>).mapNotNull { (it["parsed"] as Map<String, Any?>)["code"] }.toSet()
    expect("ve refusal codes covered", codes.containsAll(listOf("ok", "prompt", "rate", "invalid", "loud", "speech", "enough",
        "windows", "speech_total", "done")))
    for (r in v["replies"] as List<Map<String, Any?>>)
        expect("ve reply ${r["parser"]} ${r["name"]}", parsed(r["parser"] as String, r["reply"] as String) == r["parsed"])
    for (junk in VE_JUNK) for (op in VoiceEnrollment.OPS) {
        val ack = op == "voice_cancel" && junk.startsWith("{\"ok\":true")   // any {"ok":true} is an ack, as LinkOps.parseAck
        expect("ve parser never throws: $op $junk", !throws { parsed(op, junk) } && parsed(op, junk)["ok"] == ack)
    }

    // scripted flows against the recorded pod replies: the same requests in the same order, the same end state
    for (fl in v["flows"] as List<Map<String, Any?>>) {
        val name = fl["name"]
        val steps = fl["steps"] as List<Map<String, Any?>>
        var i = 0; var inOrder = true
        val flow = VoiceEnrollmentFlow { req ->
            val s = steps.getOrNull(i++)
            if (s == null || !sameRequest(req, s)) { inOrder = false; VE_UNEXPECTED } else s["reply"] as String
        }
        flow.begin(fl["person"] as String, fl["consent"] as String)
        val rate = (fl["rate"] as Long).toInt()
        var zeroed = true; var held = 0
        for (g in fl["attempts"] as List<Map<String, Any?>>) {
            if (flow.stage != VoiceEnrollmentFlow.Stage.RECORDING) break
            val pcm = vePcm(g); held++
            flow.submit(pcm, rate)
            zeroed = zeroed && pcm.all { it == 0.toShort() }
        }
        val f = fl["final"] as Map<String, Any?>
        expect("ve flow $name: requests in order", inOrder && i == steps.size)
        expect("ve flow $name: every attempt used", held == (fl["attempts"] as List<*>).size)
        expect("ve flow $name: audio zeroed after each send", zeroed)
        expect("ve flow $name: end state", flow.stage.wire == f["stage"] && flow.accepted == f["accepted"] &&
            flow.refused.toLong() == f["refused"] && flow.shown.toLong() == f["shown"] && (flow.stage == VoiceEnrollmentFlow.Stage.SAVED) == f["saved"])
        expect("ve flow $name: a sentence only while recording", (flow.current == null) == (flow.stage != VoiceEnrollmentFlow.Stage.RECORDING))
    }
    run {   // a transport failure still zeroes the audio; submit outside RECORDING refuses and zeroes too
        val f = VoiceEnrollmentFlow { req ->
            when {
                "voice_sample" in req -> throw java.io.IOException("link down")
                "voice_prompts" in req -> VE_ONE_PROMPT
                else -> VE_STARTED
            }
        }
        expect("ve flow begin", f.begin("P", "c") == VoiceEnrollmentFlow.Stage.RECORDING && f.current?.id == "a" && f.windowS == 7.0)
        val pcm = ShortArray(64) { 9 }
        expect("ve flow transport failure surfaces", throws { f.submit(pcm, 16000) })
        expect("ve flow transport failure zeroes", pcm.all { it == 0.toShort() })
        val idle = VoiceEnrollmentFlow { "{\"ok\":true}" }
        val p2 = ShortArray(8) { 3 }
        expect("ve flow submit before begin refused and zeroed", throws { idle.submit(p2, 16000) } && p2.all { it == 0.toShort() })
    }
}

// ---- Fusion v2 and the power governor (twins of oiris/fusion/stack.py and oiris/power; vectors from the Python code) ----
private const val PARITY_TOL = 1e-6   // the vectors' FLOAT_TOL
private val NON_FINITE = setOf("nan", "inf", "-inf")
private const val FU_NEAR_TIE = 0.1234565            // Python: round(0.1234565, 6) == 0.123456
private const val FU_NEAR_TIE_ROUNDED = 0.123456

/** A vector value as the Python call received it: "nan" / "inf" / "-inf" are those floats. */
private fun nf(x: Any?): Any? = if (x is String && x in NON_FINITE) Py.float(x) else x
private fun close(a: Double, b: Any?): Boolean =
    (b as? Number)?.toDouble()?.let { a == it || Math.abs(a - it) <= PARITY_TOL } ?: false
private fun closeAll(a: DoubleArray, b: Any?): Boolean = (b as List<*>).let { l -> l.size == a.size && a.indices.all { close(a[it], l[it]) } }
/** A JSON number other than a bool, or a spelled non-finite float; else null (Python would refuse it). */
private fun pyNumber(x: Any?): Double? = nf(x).let { if (it is Number && it !is Boolean) it.toDouble() else null }
/** Python keeps a config's int thresholds as ints; Kotlin holds every threshold as a Double. */
private fun thresholdsMatch(t: Map<String, Double>, e: Any?): Boolean =
    (e as Map<*, *>).let { m -> m.keys == t.keys && t.all { (k, x) -> close(x, m[k]) } }
private fun planMatches(p: DutyPlan, e: Any?): Boolean {
    val m = e as Map<*, *>; val k = p.toMap()
    return k.keys == m.keys && k.all { (key, x) -> if (x is Boolean) x == m[key] else close((x as Number).toDouble(), m[key]) }
}

@Suppress("UNCHECKED_CAST")
fun fusionChecks(root: File) {
    val v = Json.decode(File(root, "tests/vectors/fusion.json").readText()) as Map<String, Any?>
    val c = v["constants"] as Map<String, Any?>
    expect("fu families", FusionCore.FAMILIES == c["families"])
    expect("fu family map (S_wifi is rf)", FusionCore.SCORE_FAMILY == c["score_family"] && FusionCore.SCORE_FAMILY["S_wifi"] == "rf")
    expect("fu signatures", FusionCore.SCORE_SIGNATURE == c["score_signature"])
    expect("fu threshold", FusionCore.FAMILY_THRESHOLD == c["family_threshold"])
    expect("fu min families / max unhealthy", FusionCore.MIN_FAMILIES.toLong() == c["min_families"] &&
        FusionCore.MAX_UNHEALTHY.toLong() == c["max_unhealthy"])
    expect("fu health and level names", FusionCore.Health.values().map { it.wire } == c["health"] &&
        FusionCore.Level.values().map { it.wire } == c["levels"])
    val dm = c["default_model"] as Map<String, Any?>
    val prior = FusionCore.FusionModel()
    expect("fu default model", closeAll(prior.w, dm["w"]) && closeAll(prior.wMask, dm["w_mask"]) && prior.b == dm["b"] &&
        prior.version == dm["version"])

    for (s in v["scores"] as List<Map<String, Any?>>) {
        val raw = s["raw"] as List<Any?>
        val e = s["score"] as List<Any?>?
        val got = FusionCore.score(raw[0], nf(raw[1]), nf(raw[2]), raw[3])
        expect("fu score $raw", if (e == null) got == null else got != null && got.module == e[0] as? String &&
            got.value == (e[1] as Number).toDouble() && got.confidence == (e[2] as Number).toDouble() && got.health.wire == e[3])
    }

    for (k in v["cases"] as List<Map<String, Any?>>) {
        val n = k["name"]
        val built = (k["scores"] as List<List<Any?>>).map { FusionCore.score(it[0], nf(it[1]), nf(it[2]), it[3]) }
        expect("fu $n: accepted", built.map { it != null } == k["accepted"])
        val scores = built.filterNotNull()
        val model = (k["weights"] as String?)?.let { FusionCore.loadWeights(it) }
        val p = k["params"] as Map<String, Any?>? ?: emptyMap()
        val fv = FusionCore.familyValues(scores)
        val ef = k["family"] as Map<String, Any?>
        expect("fu $n: family values", closeAll(fv.x, ef["x"]) && closeAll(fv.mask, ef["mask"]))
        expect("fu $n: contributors", fv.contrib == ef["contrib"] && fv.unhealthy == ef["unhealthy"])
        expect("fu $n: logit", close((model ?: FusionCore.FusionModel()).logit(fv.x, fv.mask), k["logit"]))
        val fallback = k["fallback_node"] == true
        val r = FusionCore.fuse(scores, model, (p["min_families"] as Long? ?: FusionCore.MIN_FAMILIES.toLong()).toInt(),
            (p["max_unhealthy"] as Long? ?: FusionCore.MAX_UNHEALTHY.toLong()).toInt(), fallback)
        val er = k["result"] as Map<String, Any?>
        val got = r.toMap()
        expect("fu $n: anomaly index", close(r.anomalyIndex, er["anomaly_index"]))
        expect("fu $n: output", got.keys == er.keys && got.filterKeys { it != "anomaly_index" } == er.filterKeys { it != "anomaly_index" })
        expect("fu $n: advisory only for a fallback node", r.advisory == fallback)
    }

    for (w in v["weights"] as List<Map<String, Any?>>) {
        val m = FusionCore.FusionModel.fromJson(w["json"] as String)
        val e = w["model"] as Map<String, Any?>?
        expect("fu weights ${w["name"]}", if (e == null) m == null && w["accepted"] == false else m != null &&
            closeAll(m.w, e["w"]) && closeAll(m.wMask, e["w_mask"]) && close(m.b, e["b"]) && m.version == e["version"])
    }
    val lastGood = FusionCore.FusionModel(version = "last-good")
    expect("fu load_weights falls back to last-good", FusionCore.loadWeights("{", lastGood) === lastGood &&
        FusionCore.loadWeights(null).version == FusionCore.DEFAULT_VERSION)
    // FU_NEAR_TIE is stored just below the decimal tie, so Python's round(x, 6) gives FU_NEAR_TIE_ROUNDED, not ...457
    expect("fu round6 on the exact binary value", FusionCore.round6(FU_NEAR_TIE) == FU_NEAR_TIE_ROUNDED &&
        FusionCore.round6(FusionCore.FAMILY_THRESHOLD) == FusionCore.FAMILY_THRESHOLD)
}

@Suppress("UNCHECKED_CAST")
fun powerChecks(root: File) {
    val v = Json.decode(File(root, "tests/vectors/power_governor.json").readText()) as Map<String, Any?>
    val c = v["constants"] as Map<String, Any?>
    expect("pg levels", Power.LEVELS == c["levels"])
    expect("pg reasons", Power.REASONS == c["reasons"])
    expect("pg default thresholds", Power.POWER_DEFAULTS == c["defaults"] &&
        Power.POWER_DEFAULTS.keys.toList() == (c["defaults"] as Map<*, *>).keys.toList())
    expect("pg threshold ranges", Power.POWER_RANGES.mapValues { listOf(it.value.first, it.value.second) } == c["ranges"] &&
        Power.POWER_RANGES.keys.toList() == (c["ranges"] as Map<*, *>).keys.toList())
    expect("pg plausible ranges", listOf(Power.BATTERY_PCT_RANGE.first, Power.BATTERY_PCT_RANGE.second) == c["battery_pct_range"] &&
        listOf(Power.TEMP_PLAUSIBLE_C.first, Power.TEMP_PLAUSIBLE_C.second) == c["temp_plausible_c"])
    expect("pg plan keys", Power.PLAN_KEYS == c["plan_keys"])
    expect("pg plan ranges", (c["plan_ranges"] as Map<String, List<Any?>>).let { e -> e.keys == Power.PLAN_RANGES.keys &&
        e.all { (k, r) -> close(Power.PLAN_RANGES.getValue(k).first, r[0]) && close(Power.PLAN_RANGES.getValue(k).second, r[1]) } })
    expect("pg default plans", Power.DEFAULT_PLANS == c["default_plans"])
    val defaults = Power.buildPlans(null)
    expect("pg built default plans", Power.LEVELS.all { planMatches(defaults.getValue(it), (c["default_plans"] as Map<*, *>)[it]) })
    expect("pg frame gate tolerance", FrameGate.TOLERANCE_S == c["frame_gate_tolerance_s"])
    val ts = c["thermal_status"] as Map<String, Any?>
    expect("pg thermal status ints", listOf(Power.THERMAL_STATUS_NONE, Power.THERMAL_STATUS_LIGHT, Power.THERMAL_STATUS_MODERATE,
        Power.THERMAL_STATUS_SEVERE, Power.THERMAL_STATUS_CRITICAL, Power.THERMAL_STATUS_EMERGENCY, Power.THERMAL_STATUS_SHUTDOWN)
        .map { it.toLong() } == listOf("none", "light", "moderate", "severe", "critical", "emergency", "shutdown").map { ts[it] })

    for (t in v["thermal_status"] as List<Map<String, Any?>>)
        expect("pg thermal status ${t["status"]}", Power.thermalStatusLevel(t["status"]) == t["level"])

    for (k in v["validation"] as List<Map<String, Any?>>) {
        val n = k["name"]
        val cfg = k["power"]
        expect("pg validate $n", Power.validatePower(cfg) == k["errors"])
        if (cfg is Map<*, *>) {
            val ok = (k["errors"] as List<*>).isEmpty()
            val s = try { PowerSettings.fromConfig(cfg as Map<String, Any?>) } catch (e: IllegalArgumentException) { null }
            expect("pg settings $n", if (!ok) s == null else s != null && thresholdsMatch(s.thresholds, k["thresholds"]) &&
                Power.LEVELS.all { planMatches(s.plans.getValue(it), (k["plans"] as Map<*, *>)[it]) })
        }
    }

    for (tl in v["timelines"] as List<Map<String, Any?>>) {
        val n = tl["name"]
        val settings = PowerSettings.fromConfig(tl["power"] as Map<String, Any?>?)
        if (tl["plans"] != null) expect("pg $n: plans and thresholds", thresholdsMatch(settings.thresholds, tl["thresholds"]) &&
            Power.LEVELS.all { planMatches(settings.plans.getValue(it), (tl["plans"] as Map<*, *>)[it]) })
        val gov = PowerGovernor(settings)
        var cur = 0.0
        val clocked = PowerGovernor(settings) { cur }   // the same timeline on an injected clock
        for ((i, st) in (tl["steps"] as List<Map<String, Any?>>).withIndex()) {
            val r = PowerReading.fromMap(st.filterKeys { it != "now" }.mapValues { nf(it.value) })
            val now = (st["now"] as Number?)?.toDouble()
            cur = now ?: r.t?.takeIf { it.isFinite() } ?: clocked.since ?: 0.0
            val d = gov.update(r, now)
            val dc = clocked.update(r)
            expect("pg $n: step $i", d.level == st["level"] && d.reasons == st["reasons"] && d.plan == settings.plans[d.level])
            expect("pg $n: step $i on the injected clock", dc == d)
        }
    }

    for (sim in v["sim"] as List<Map<String, Any?>>) {
        val readings = (sim["readings"] as List<List<Any?>>).map {
            PowerReading((it[0] as Number).toDouble(), (it[1] as Number).toDouble(), it[2] as Boolean,
                (it[3] as Number).toDouble(), (it[4] as Number).toDouble())
        }
        val ds = PowerGovernor().replay(readings)
        expect("pg sim ${sim["scene"]}: levels", ds.map { it.level } == sim["levels"])
        expect("pg sim ${sim["scene"]}: reasons", ds.map { it.reasons } == sim["reasons"])
    }

    for (g in v["frame_gate"] as List<Map<String, Any?>>) {
        val fps = pyNumber(g["max_fps"])
        val stride = (g["stride"] as? Long)?.toInt()
        val gate = if (fps == null || stride == null) null else try { FrameGate(fps, stride) } catch (e: IllegalArgumentException) { null }
        val name = "pg frame gate ${g["max_fps"]}/${g["stride"]}"
        expect("$name: refused", (gate == null) == (g["rejected"] == true))
        if (gate != null) expect("$name: admits", (g["t"] as List<Any?>).map { gate.admit(pyNumber(it)!!) } == g["admit"])
    }
    for (g in v["listen_duty"] as List<Map<String, Any?>>) {
        val duty = pyNumber(g["duty"])
        val period = pyNumber(g["period_s"])
        val ld = if (duty == null || period == null) null else try { ListenDuty(duty, period) } catch (e: IllegalArgumentException) { null }
        val name = "pg listen duty ${g["duty"]}/${g["period_s"]}"
        expect("$name: refused", (ld == null) == (g["rejected"] == true))
        if (ld != null) expect("$name: active", (g["t"] as List<Any?>).map { ld.active(pyNumber(it)!!) } == g["active"])
    }
}

// ---- Painting the glasses display list (twin of oiris/ui/paint.py; vectors from the Python code) ----
private const val ARGB_MASK = 0xFFFFFFFFL

@Suppress("UNCHECKED_CAST")
fun hudPaintChecks(root: File) {
    val v = Json.decode(File(root, "tests/vectors/hud_paint.json").readText()) as Map<String, Any?>
    val tokenText = File(root, "oiris/ui/design_tokens.json").readText()
    val vocab = Json.decode(File(root, "schema/vocabulary.json").readText()) as Map<String, Any?>
    val banned = VoiceLink.bannedRegex(vocab["banned_patterns"] as List<String>)
    val c = v["constants"] as Map<String, Any?>
    expect("hp constants", HudPainter.WEIGHT_NORMAL == c["weight_normal"] && HudPainter.ANCHORS == c["anchors"] &&
        HudPainter.NO_PAINT == c["no_paint"] && HudPainter.NO_STROKE_WIDTH == c["no_stroke_width"] &&
        HudPainter.SQUARE == c["square"] && HudPainter.MARK_KINDS == c["mark_kinds"] && HudPainter.CALL_KINDS == c["call_kinds"])

    fun tokens(patch: List<List<Any?>>): GlassesHud.Tokens {
        val t = Json.decode(tokenText) as MutableMap<String, Any?>
        for (p in patch) {
            val path = p[0] as List<String>
            var d = t
            for (k in path.dropLast(1)) d = d[k] as MutableMap<String, Any?>
            d[path.last()] = p[1]
        }
        return GlassesHud.Tokens(t)
    }
    val seen = HashSet<String>()
    fun same(name: String, exp: Any?, paint: (HudCanvas) -> Unit) {
        val rec = RecordingCanvas(); paint(rec)
        val again = RecordingCanvas(); paint(again)
        expect("hp $name: calls", rec.calls == exp)
        if (rec.calls != exp) System.err.println("  got ${Json.encode(rec.calls)}\n  exp ${Json.encode(exp)}")
        expect("hp $name: deterministic", rec.calls == again.calls)
        rec.calls.forEach { seen += it[0] as String }
    }
    for (h in v["hud"] as List<Map<String, Any?>>) {
        val s = h["state"] as Map<String, Any?>
        val st = GlassesHud.State(level = s["level"] as String?, alertText = s["alert_text"] as String, newSighting = s["new_sighting"] as Boolean,
            thermal = s["thermal"] as Boolean, thermalColour = s["thermal_colour"] as Boolean, health = s["health"] as Map<String, String>,
            voiceTier = s["voice_tier"] as String, locked = s["locked"] as Boolean, panicArmed = s["panic_armed"] as Boolean,
            moving = s["moving"] as Boolean, cueDirection = s["cue_direction"] as String?)
        val ops = GlassesHud.displayList(st, tokens(h["tokens_patch"] as List<List<Any?>>), banned)
        same("hud ${h["name"]}", h["calls"]) { HudPainter.paintHud(ops, it) }
    }
    for (g in v["guide"] as List<Map<String, Any?>>) {
        val s = g["guide"] as Map<String, Any?>
        val eg = GlassesHud.Guide(s["needed"] as List<String>, s["last_ok"] as Boolean?, s["last_reason"] as String, s["complete"] as Boolean)
        val ops = GlassesHud.guideDisplayList(eg, tokens(g["tokens_patch"] as List<List<Any?>>), banned)
        same("guide ${g["name"]}", g["calls"]) { HudPainter.paintHud(ops, it) }
    }
    for (f in v["frames"] as List<Map<String, Any?>>) {   // untrusted frames: HudFrame.parse, then paint
        val ops = HudFrame.parse(f["reply"] as String).ops
        same("frame ${f["name"]}", f["calls"]) { HudPainter.paint(ops, it) }
    }
    expect("hp every call kind painted", seen == HudPainter.CALL_KINDS.toSet())

    for (d in v["floordiv"] as List<List<Any?>>) {   // boxed equality: -0.0 and 0.0 differ, as in Python
        val got: Any = HudPainter.pyFloorDiv((d[0] as Number).toDouble(), (d[1] as Number).toDouble())
        expect("hp floordiv ${d[0]} // ${d[1]}", got == d[2])
    }
    for (k in v["colours"] as List<List<Any?>>)
        expect("hp colour '${k[0]}'", HudPainter.colour(k[0] as String)?.let { it.toLong() and ARGB_MASK } == k[1])

    // never throws, whatever the list holds
    val junk: List<Map<String, Any?>> = listOf(mapOf("op" to "canvas", "w" to 10L, "h" to 10L, "colour" to "#000", "opacity" to 1L,
        "font_family" to "x"), mapOf("op" to "text", "x" to true), mapOf("op" to "shape", "kind" to null), mapOf("op" to null),
        mapOf("op" to "cue", "x" to Double.NaN, "y" to 0L, "w" to 1L, "h" to 1L, "colour" to ""), emptyMap())
    val rec = RecordingCanvas()
    expect("hp junk ops skipped, never thrown", !throws { HudPainter.paint(junk, rec) } && rec.calls.map { it[0] } == listOf("frame"))
    val none = RecordingCanvas()
    expect("hp no canvas, nothing drawn", !throws { HudPainter.paint(junk.drop(1), none); HudPainter.paint(emptyList(), none) } &&
        none.calls.isEmpty())
}
