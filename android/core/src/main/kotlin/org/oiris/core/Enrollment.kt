package org.oiris.core

import java.util.Base64

/** M-16 guided enrollment on the phone (spec 7.3). The phone only captures and guides; the pod's
 *  EnrollmentPort (oiris/modules/face_match/remote.py) checks quality, builds and seals templates.
 *  Off by default: locked until the user reads the warning and types "I UNDERSTAND". No count limit. */
enum class Angle(val wire: String, val prompt: String) {
    FRONT("front", "Look straight at the camera"), LEFT("left", "Turn your head to the left"),
    RIGHT("right", "Turn your head to the right"), UP("up", "Tilt your head up"), DOWN("down", "Tilt your head down");
    companion object { fun of(w: String) = values().first { it.wire == w } }
}

const val CONFIRM_PHRASE = "I UNDERSTAND"

/** Request builders: canonical JSON strings carried inside SecureChannel frames. */
object EnrollmentProtocol {
    fun status() = Json.encode(mapOf("op" to "status"))
    fun warning(jurisdiction: String) = Json.encode(mapOf("op" to "warning", "jurisdiction" to jurisdiction))
    fun optIn(typed: String, modes: List<String>, warningShown: String, jurisdiction: String) =
        Json.encode(mapOf("op" to "opt_in", "typed" to typed, "modes" to modes, "warning_shown" to warningShown, "jurisdiction" to jurisdiction))
    fun startSession() = Json.encode(mapOf("op" to "start_session"))
    fun addPerson(session: String, label: String, consentId: String?, guardian: Boolean) =
        Json.encode(mapOf("op" to "add_person", "session" to session, "label" to label, "consent_id" to consentId, "guardian" to guardian))
    fun shot(person: String, angle: Angle, w: Int, h: Int, bgr: ByteArray) =
        Json.encode(mapOf("op" to "shot", "person" to person, "angle" to angle.wire, "w" to w, "h" to h,
            "fmt" to "bgr", "pixels_b64" to Base64.getEncoder().encodeToString(bgr)))
    fun finish(person: String) = Json.encode(mapOf("op" to "finish", "person" to person))
    fun endSession(session: String) = Json.encode(mapOf("op" to "end_session", "session" to session))

    // ---- phone-guided voice enrollment (M-17, ADR-0057): the pod runs every check and builds the template ----
    fun voicePrompts() = Json.encode(mapOf("op" to "voice_prompts"))
    fun voiceStart(person: String, consent: String) =
        Json.encode(mapOf("op" to "voice_start", "person" to person, "consent" to consent))
    /** `pcm`: mono 16-bit samples, sent as standard base64 of their little-endian bytes. The little-endian copy is
     *  zeroed here; the caller zeroes `pcm` (VoiceEnrollmentFlow.submit does). */
    fun voiceSample(person: String, promptId: String, rate: Int, pcm: ShortArray): String {
        val le = ByteArray(pcm.size * VoiceEnrollment.PCM_SAMPLE_BYTES)
        try {
            for (i in pcm.indices) { val s = pcm[i].toInt(); le[2 * i] = s.toByte(); le[2 * i + 1] = (s shr 8).toByte() }
            return Json.encode(mapOf("op" to "voice_sample", "person" to person, "prompt_id" to promptId, "rate" to rate,
                "pcm16" to Base64.getEncoder().encodeToString(le)))
        } finally { le.fill(0) }
    }
    fun voiceFinish(person: String) = Json.encode(mapOf("op" to "voice_finish", "person" to person))
    fun voiceCancel(person: String) = Json.encode(mapOf("op" to "voice_cancel", "person" to person))
}

/** Reply parsers for the voice ops, ported from oiris/modules/face_match/remote.py parse_voice_*; vectors
 *  tests/vectors/voice_enrollment.json. They never throw: a reply that is not what the op returns parses as
 *  ok=false, LinkOps.BAD_REPLY; a pod refusal passes its error through. Replies carry no audio. */
object VoiceEnrollment {
    val OPS = listOf("voice_prompts", "voice_start", "voice_sample", "voice_finish", "voice_cancel")
    const val VOICE_OFF = "voice link is off"
    const val MAX_SAMPLE_S = 12
    const val PCM_SAMPLE_BYTES = 2
    const val RATE_MIN = 8000
    const val RATE_MAX = 48000
    const val MAX_PCM_BYTES = MAX_SAMPLE_S * RATE_MAX * PCM_SAMPLE_BYTES
    const val MAX_PCM_B64 = 4 * ((MAX_PCM_BYTES + 2) / 3)
    const val MAX_PROMPTS = 64
    const val MAX_REPLY_TEXT = 400
    const val MAX_REFUSED = 5          // = oiris/voice_link_cli.MAX_REFUSED_WINDOWS
    val PROMPT_KEYS = setOf("id", "text", "tone")

    data class Prompt(val id: String, val text: String, val tone: String)

    private val bad = LinkOps.Reply(false, LinkOps.BAD_REPLY)

    private inline fun parse(reply: String, body: (Map<String, Any?>) -> LinkOps.Reply?): LinkOps.Reply {
        val r = LinkOps.load(reply) ?: return bad
        if (r["ok"] != true) return LinkOps.Reply(false, r["error"] as String)
        return try { body(r) ?: bad } catch (e: RuntimeException) { bad }
    }

    private fun count(v: Any?) = v is Long && v >= 0
    private fun seconds(v: Any?) = (v is Long && v >= 0) || (v is Double && v.isFinite() && v >= 0)
    private fun text(v: Any?) = v is String && v.codePointCount(0, v.length) <= MAX_REPLY_TEXT
    private fun d(v: Any?) = (v as Number).toDouble()

    /** fields: prompts, each {id, text, tone}, in enrollment order. */
    fun parsePrompts(reply: String) = parse(reply) { r ->
        val ps = r["prompts"] as? List<*> ?: return@parse null
        if (ps.isEmpty() || ps.size > MAX_PROMPTS) return@parse null
        for (p in ps) if (p !is Map<*, *> || p.keys != PROMPT_KEYS || !p.values.all { text(it) && (it as String).isNotEmpty() }) return@parse null
        LinkOps.Reply(true, null, mapOf("prompts" to ps.map { p -> (p as Map<*, *>).let { mapOf("id" to it["id"], "text" to it["text"], "tone" to it["tone"]) } }))
    }

    fun prompts(reply: LinkOps.Reply): List<Prompt> =
        (reply.fields["prompts"] as? List<*>)?.map { p -> (p as Map<*, *>).let { Prompt(it["id"] as String, it["text"] as String, it["tone"] as String) } } ?: emptyList()

    /** fields: windows, max_windows, target_speech_s, window_s. */
    fun parseStart(reply: String) = parse(reply) { r ->
        val w = r["windows"]; val mx = r["max_windows"]; val t = r["target_speech_s"]; val ws = r["window_s"]
        if (!count(w) || !count(mx) || (w as Long) <= 0 || w > (mx as Long) || !seconds(t) || !seconds(ws) || d(ws) <= 0) return@parse null
        LinkOps.Reply(true, null, mapOf("windows" to w, "max_windows" to mx, "target_speech_s" to d(t), "window_s" to d(ws)))
    }

    /** fields: accepted, code, message, speech_s, total_speech_s, level_dbfs (or null), remaining. */
    fun parseSample(reply: String) = parse(reply) { r ->
        val lv = r["level_dbfs"]
        if (r["accepted"] !is Boolean || !text(r["code"]) || !text(r["message"]) || !seconds(r["speech_s"]) ||
            !seconds(r["total_speech_s"]) || !count(r["remaining"]) ||
            !(lv == null || lv is Long || (lv is Double && lv.isFinite()))) return@parse null
        LinkOps.Reply(true, null, mapOf("accepted" to r["accepted"], "code" to r["code"], "message" to r["message"],
            "speech_s" to d(r["speech_s"]), "total_speech_s" to d(r["total_speech_s"]), "level_dbfs" to lv?.let { d(it) },
            "remaining" to r["remaining"]))
    }

    /** fields: saved, code, message, total_speech_s, remaining. */
    fun parseFinish(reply: String) = parse(reply) { r ->
        if (r["saved"] !is Boolean || !text(r["code"]) || !text(r["message"]) || !seconds(r["total_speech_s"]) ||
            !count(r["remaining"])) return@parse null
        LinkOps.Reply(true, null, mapOf("saved" to r["saved"], "code" to r["code"], "message" to r["message"],
            "total_speech_s" to d(r["total_speech_s"]), "remaining" to r["remaining"]))
    }

    /** voice_cancel. */
    fun parseAck(reply: String) = parse(reply) { LinkOps.Reply(true, null) }
}

/** Phone-guided voice enrollment, independent of Android views (tested on the JVM). The phone shows each designed
 *  sentence, records it and sends it; the pod checks it and answers with feedback. In the order of
 *  `oiris voice-link enroll`: prompts, start; one sentence per window, the same sentence again after a refusal;
 *  the next sentence (cycling) while the pod asks for more total speech; finish when the pod says nothing remains,
 *  or at the window maximum; MAX_REFUSED refusals cancel (nothing saved). Holds no audio: every ShortArray passed
 *  to submit is zeroed before submit returns. The microphone screen (Android SDK) comes later. */
class VoiceEnrollmentFlow(private val send: (String) -> String) {
    enum class Stage(val wire: String) { IDLE("idle"), RECORDING("recording"), SAVED("saved"), STOPPED("stopped") }
    var stage = Stage.IDLE; private set
    var prompts: List<VoiceEnrollment.Prompt> = emptyList(); private set
    var person: String? = null; private set
    var windows = 0L; private set
    var maxWindows = 0L; private set
    var targetSpeechS = 0.0; private set
    var windowS = 0.0; private set
    var accepted = 0L; private set
    var refused = 0; private set
    var shown = 0; private set              // sentences started; the current one is prompts[(shown - 1) % size]
    var lastSample: LinkOps.Reply? = null; private set
    var lastFinish: LinkOps.Reply? = null; private set
    var lastMessage = ""; private set

    val current: VoiceEnrollment.Prompt? get() =
        if (stage == Stage.RECORDING && prompts.isNotEmpty() && shown > 0) prompts[(shown - 1) % prompts.size] else null

    private fun stop(message: String) { stage = Stage.STOPPED; lastMessage = message }

    /** Fetches the sentences and opens the pod's voice capture. Returns RECORDING, or STOPPED with the reason. */
    fun begin(person: String, consent: String): Stage {
        check(stage == Stage.IDLE) { "one flow per person" }
        val p = VoiceEnrollment.parsePrompts(send(EnrollmentProtocol.voicePrompts()))
        if (!p.ok) { stop(p.error ?: LinkOps.BAD_REPLY); return stage }
        val s = VoiceEnrollment.parseStart(send(EnrollmentProtocol.voiceStart(person, consent)))
        if (!s.ok) { stop(s.error ?: LinkOps.BAD_REPLY); return stage }
        prompts = VoiceEnrollment.prompts(p); this.person = person
        windows = s.fields["windows"] as Long; maxWindows = s.fields["max_windows"] as Long
        targetSpeechS = s.fields["target_speech_s"] as Double; windowS = s.fields["window_s"] as Double
        stage = Stage.RECORDING; shown = 1; lastMessage = current!!.text
        return stage
    }

    /** Sends one recording of the current sentence and acts on the pod's answer. `pcm` is zeroed before this returns. */
    fun submit(pcm: ShortArray, rate: Int): LinkOps.Reply {
        val r = try {
            check(stage == Stage.RECORDING) { "not recording" }
            VoiceEnrollment.parseSample(send(EnrollmentProtocol.voiceSample(person!!, current!!.id, rate, pcm)))
        } finally { pcm.fill(0) }
        lastSample = r
        when {
            r.ok && r.fields["accepted"] == true -> {
                accepted++
                lastMessage = r.fields["message"] as String
                if (r.fields["remaining"] == 0L || accepted >= maxWindows) finish() else { shown++; lastMessage = current!!.text }
            }
            r.ok && r.fields["code"] == "enough" -> finish()
            else -> {
                refused++
                lastMessage = (if (r.ok) r.fields["message"] else r.error) as String
                if (refused >= VoiceEnrollment.MAX_REFUSED) cancel("Too many recordings were refused; nothing saved.")
            }
        }
        return r
    }

    private fun finish() {
        val f = VoiceEnrollment.parseFinish(send(EnrollmentProtocol.voiceFinish(person!!)))
        lastFinish = f
        when {
            f.ok && f.fields["saved"] == true -> { stage = Stage.SAVED; lastMessage = f.fields["message"] as String }
            f.ok && (f.fields["remaining"] as Long) > 0 && accepted < maxWindows -> { shown++; lastMessage = current!!.text }
            else -> stop((if (f.ok) f.fields["message"] else f.error) as String)
        }
    }

    /** Closes the pod's capture (its embeddings are zeroed there); nothing is saved. */
    fun cancel(message: String = "Voice enrollment cancelled; nothing saved.") {
        if (stage != Stage.RECORDING) return
        VoiceEnrollment.parseAck(send(EnrollmentProtocol.voiceCancel(person!!)))
        stop(message)
    }
}

/** UI state machine for the enrollment screen, independent of Android views (tested on the JVM). */
class EnrollmentFlow(private val send: (String) -> String) {
    enum class Stage { LOCKED, WARNING, READY, CAPTURING, COMPLETE }
    var stage = Stage.LOCKED; private set
    var warningText = ""; private set
    var session: String? = null; private set
    var person: String? = null; private set
    var personLabel = ""; private set
    val accepted = mutableMapOf<Angle, Int>()
    var lastMessage = ""; private set
    var enrolledThisSession = 0; private set

    @Suppress("UNCHECKED_CAST")
    private fun call(req: String): Map<String, Any?> {
        val r = Json.decode(send(req)) as Map<String, Any?>
        if (r["ok"] != true) throw IllegalStateException(r["error"] as? String ?: "pod refused")
        return r
    }

    /** On open: unlocked only if the pod reports M-16 on and opted in. */
    fun open(): Stage {
        val r = call(EnrollmentProtocol.status())
        stage = if (r["opted_in"] == true && (r["modes"] as List<*>).contains("personal")) Stage.READY else Stage.LOCKED
        return stage
    }

    fun showWarning(jurisdiction: String = "strictest") {
        warningText = call(EnrollmentProtocol.warning(jurisdiction))["text"] as String
        stage = Stage.WARNING
    }

    /** Typed opt-in. Anything but the exact phrase keeps the module off. */
    fun optIn(typed: String, jurisdiction: String = "strictest"): Boolean {
        check(stage == Stage.WARNING) { "show the warning first" }
        if (typed.trim() != CONFIRM_PHRASE) { lastMessage = "Type \"$CONFIRM_PHRASE\" to turn face matching on"; return false }
        call(EnrollmentProtocol.optIn(typed, listOf("personal"), warningText, jurisdiction))
        stage = Stage.READY; return true
    }

    fun startPerson(label: String, consentId: String?, guardian: Boolean = false) {
        check(stage == Stage.READY || stage == Stage.COMPLETE) { "enrollment is locked" }
        val s = session ?: (call(EnrollmentProtocol.startSession())["session"] as String).also { session = it }
        person = call(EnrollmentProtocol.addPerson(s, label, consentId, guardian))["person"] as String
        personLabel = label; accepted.clear(); stage = Stage.CAPTURING
        lastMessage = nextAngle()?.prompt ?: ""
    }

    fun nextAngle(): Angle? = Angle.values().firstOrNull { (accepted[it] ?: 0) == 0 }
    fun checklist(): List<Pair<Angle, Boolean>> = Angle.values().map { it to ((accepted[it] ?: 0) > 0) }
    fun shotsAccepted() = accepted.values.sum()

    /** Submit one captured frame for the requested angle; returns whether it was accepted. */
    fun submit(angle: Angle, w: Int, h: Int, bgr: ByteArray): Boolean {
        check(stage == Stage.CAPTURING)
        if (shotsAccepted() >= 10) { lastMessage = "10 shots captured; finish this person"; return false }
        val r = call(EnrollmentProtocol.shot(person!!, angle, w, h, bgr))
        bgr.fill(0)   // the frame leaves RAM as soon as it is sent
        return if (r["accepted"] == true) {
            accepted[angle] = (accepted[angle] ?: 0) + 1
            lastMessage = nextAngle()?.prompt ?: "All angles captured. Add extras or finish."
            true
        } else { lastMessage = r["reason"] as? String ?: "shot rejected"; false }
    }

    fun finish(): Boolean {
        val r = call(EnrollmentProtocol.finish(person!!))
        return if (r["complete"] == true) {
            lastMessage = "Template complete for $personLabel"; stage = Stage.COMPLETE; enrolledThisSession++; person = null; true
        } else { lastMessage = r["message"] as? String ?: "template incomplete"; false }
    }

    fun end() {
        session?.let { call(EnrollmentProtocol.endSession(it)) }
        session = null; person = null; accepted.clear()
        stage = if (stage == Stage.LOCKED || stage == Stage.WARNING) stage else Stage.READY
    }
}
