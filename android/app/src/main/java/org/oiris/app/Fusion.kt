package org.oiris.app

import org.oiris.core.FusionCore

/** The app's view of fusion, delegating to :core FusionCore (the JVM-checked twin of oiris/fusion/stack.py):
 *  >=2 active families for ATTENTION, 1 = FYI, more than one unhealthy family suppresses. A fallback node has only
 *  optical + RF and is advisory. */
enum class Health { OK, DEGRADED, FAILED }
enum class Level { FYI, ATTENTION, CONFIRMED }

private const val SCORE_MIN = 0.0
private const val SCORE_MAX = 1.0

data class Score(val module: String, val value: Double, val confidence: Double, val health: Health = Health.OK) {
    init { require(value in SCORE_MIN..SCORE_MAX && confidence in SCORE_MIN..SCORE_MAX) { "C1: scores are in [0,1]" } }
}

object Fusion {
    val family: Map<String, String> = FusionCore.SCORE_FAMILY

    /** active: families with a contributing score, in FusionCore.FAMILIES order. */
    data class Result(val level: Level?, val active: List<String>, val suppressed: Boolean, val advisory: Boolean)

    fun fuse(scores: List<Score>, fallbackNode: Boolean): Result {
        val core = scores.map { FusionCore.Score(it.module, it.value, it.confidence, FusionCore.Health.valueOf(it.health.name)) }
        val r = FusionCore.fuse(core, fallbackNode = fallbackNode)
        return Result(r.level?.let { Level.valueOf(it.name) }, r.familiesActive, r.suppressed, r.advisory)
    }
}
