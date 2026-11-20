package org.oiris.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.view.View
import org.oiris.core.GlassesHud
import org.oiris.core.HudFrame
import org.oiris.core.HudPainter
import org.oiris.core.ThermalRender

/** The glasses display: onDraw paints the current display list through HudPainter on an AndroidHudCanvas. The list
 *  comes from the pod (a HudFrame, already cleaned by HudFrame.parse) or is built here (GlassesHud). The G3 image
 *  placeholder is filled from the latest thermal frame (ThermalRender, grayscale or colour by the op's mode). Setters
 *  may be called from the link thread; state changes on the UI thread only. */
class HudView(ctx: Context) : View(ctx) {
    private var ops: List<Map<String, Any?>> = emptyList()
    private var thermal: Triple<DoubleArray, Int, Int>? = null
    private val thermalBitmaps = HashMap<String, Bitmap>()      // by mode, rebuilt when the frame changes
    private val hud = AndroidHudCanvas { source, mode -> image(source, mode) }
    var version = NO_VERSION; private set

    /** A polled pod frame; an unchanged or refused one keeps what is shown. */
    fun setFrame(frame: HudFrame) {
        if (!frame.ok || frame.unchanged) return
        post { ops = frame.ops; version = frame.version; invalidate() }
    }

    /** A display list built on this device (GlassesHud.displayList / guideDisplayList). */
    fun setOps(list: List<GlassesHud.Op>) {
        val maps = list.map { it.toMap() }
        post { ops = maps; invalidate() }
    }

    /** The thermal frame (degrees C, row-major w x h) for the G3 window. */
    fun setThermal(celsius: DoubleArray, w: Int, h: Int) {
        require(celsius.size == w * h && w > 0 && h > 0) { "frame size" }
        val copy = celsius.copyOf()
        post { thermal = Triple(copy, w, h); thermalBitmaps.clear(); invalidate() }
    }

    private fun image(source: String, mode: String): Bitmap? {
        if (source != GlassesHud.SOURCE_THERMAL) return null
        val (celsius, w, h) = thermal ?: return null
        return thermalBitmaps.getOrPut(mode) {
            Bitmap.createBitmap(ThermalRender.argb(celsius, mode == MODE_COLOUR), w, h, Bitmap.Config.ARGB_8888)
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val saved = canvas.save()
        try {
            hud.begin(canvas, width, height)
            HudPainter.paint(ops, hud)
        } finally {
            canvas.restoreToCount(saved)
        }
    }

    companion object {
        private const val NO_VERSION = -1L
        private const val MODE_COLOUR = "colour"
    }
}
