package org.oiris.app

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Typeface
import org.oiris.core.HudCanvas
import org.oiris.core.HudPainter

/** HudCanvas on android.graphics.Canvas: executes HudPainter's draw calls (the same calls the JVM checks record).
 *  The logical frame (w x h) is scaled uniformly into the view and centred, as the SVG's viewBox is. Colours are the
 *  design-token hex, parsed by HudPainter.colour; "" or anything unparsable is not painted. The frame's opacity (the
 *  brightness cap) is applied as the alpha of every paint. A stroke width of 0 draws nothing (Android would draw a
 *  hairline). Reused across onDraw calls: [begin] binds the view's canvas; paints, path and typefaces are kept. */
class AndroidHudCanvas(private val images: (source: String, mode: String) -> Bitmap?) : HudCanvas {
    private var c: Canvas? = null
    private var viewW = 0f
    private var viewH = 0f
    private var alpha = OPAQUE
    private var family: String? = null
    private var normalFace: Typeface = Typeface.DEFAULT
    private var boldFace: Typeface = Typeface.DEFAULT_BOLD
    private val fill = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val stroke = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeCap = Paint.Cap.BUTT; strokeJoin = Paint.Join.MITER }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val bitmapPaint = Paint().apply { isFilterBitmap = false }
    private val path = Path()
    private val box = RectF()

    /** Binds the view's canvas for one onDraw; the caller saves and restores it around HudPainter.paint. */
    fun begin(canvas: Canvas, width: Int, height: Int) {
        c = canvas; viewW = width.toFloat(); viewH = height.toFloat()
        alpha = OPAQUE
    }

    /** Sets `p` to the colour at the frame's opacity; false = do not paint. */
    private fun paintWith(p: Paint, colour: String): Boolean {
        val argb = HudPainter.colour(colour) ?: return false
        p.color = argb
        p.alpha = alpha
        return true
    }

    private fun strokeWith(colour: String, width: Double): Boolean {
        if (width <= NO_WIDTH || !paintWith(stroke, colour)) return false
        stroke.strokeWidth = width.toFloat()
        return true
    }

    private fun tracePath(points: List<Double>, close: Boolean) {
        path.reset()
        var i = 0
        while (i + 1 < points.size) {
            val x = points[i].toFloat(); val y = points[i + 1].toFloat()
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
            i += 2
        }
        if (close) path.close()
    }

    override fun frame(w: Double, h: Double, background: String, opacity: Double, fontFamily: String) {
        val cv = c ?: return
        cv.drawColor(Color.BLACK)                       // additive display: black is transparent
        if (w <= NO_WIDTH || h <= NO_WIDTH) return
        val s = minOf(viewW / w.toFloat(), viewH / h.toFloat())
        cv.translate((viewW - w.toFloat() * s) / 2f, (viewH - h.toFloat() * s) / 2f)
        cv.scale(s, s)
        cv.clipRect(0f, 0f, w.toFloat(), h.toFloat())
        alpha = (opacity.coerceIn(MIN_OPACITY, MAX_OPACITY) * OPAQUE).toInt()
        if (fontFamily != family) {                      // the token family, e.g. "sans-serif"; unknown names fall back
            family = fontFamily
            normalFace = Typeface.create(fontFamily, Typeface.NORMAL)
            boldFace = Typeface.create(fontFamily, Typeface.BOLD)
        }
        if (paintWith(fill, background)) cv.drawRect(0f, 0f, w.toFloat(), h.toFloat(), fill)
    }

    override fun rect(x: Double, y: Double, w: Double, h: Double, fill: String, stroke: String, strokeWidth: Double, radius: Double) {
        val cv = c ?: return
        box.set(x.toFloat(), y.toFloat(), (x + w).toFloat(), (y + h).toFloat())
        val r = radius.toFloat()
        if (paintWith(this.fill, fill)) { if (radius > NO_WIDTH) cv.drawRoundRect(box, r, r, this.fill) else cv.drawRect(box, this.fill) }
        if (strokeWith(stroke, strokeWidth)) { if (radius > NO_WIDTH) cv.drawRoundRect(box, r, r, this.stroke) else cv.drawRect(box, this.stroke) }
    }

    override fun circle(cx: Double, cy: Double, r: Double, fill: String, stroke: String, strokeWidth: Double) {
        val cv = c ?: return
        if (paintWith(this.fill, fill)) cv.drawCircle(cx.toFloat(), cy.toFloat(), r.toFloat(), this.fill)
        if (strokeWith(stroke, strokeWidth)) cv.drawCircle(cx.toFloat(), cy.toFloat(), r.toFloat(), this.stroke)
    }

    override fun polygon(points: List<Double>, fill: String) {
        val cv = c ?: return
        if (!paintWith(this.fill, fill)) return
        tracePath(points, close = true)
        cv.drawPath(path, this.fill)
    }

    override fun polyline(points: List<Double>, stroke: String, strokeWidth: Double) {
        val cv = c ?: return
        if (!strokeWith(stroke, strokeWidth)) return
        tracePath(points, close = false)
        cv.drawPath(path, this.stroke)
    }

    override fun text(x: Double, y: Double, size: Double, colour: String, weight: String, anchor: String, text: String) {
        val cv = c ?: return
        if (size <= NO_WIDTH || !paintWith(textPaint, colour)) return
        textPaint.textSize = size.toFloat()
        textPaint.typeface = if (weight == WEIGHT_BOLD) boldFace else normalFace
        textPaint.textAlign = when (anchor) { "middle" -> Paint.Align.CENTER; "end" -> Paint.Align.RIGHT; else -> Paint.Align.LEFT }
        cv.drawText(text, x.toFloat(), y.toFloat(), textPaint)   // y is the baseline, as in the SVG
    }

    override fun image(x: Double, y: Double, w: Double, h: Double, source: String, mode: String) {
        val cv = c ?: return
        val bmp = images(source, mode) ?: return               // no pixels yet: nothing drawn in the window
        box.set(x.toFloat(), y.toFloat(), (x + w).toFloat(), (y + h).toFloat())
        bitmapPaint.alpha = alpha
        cv.drawBitmap(bmp, null, box, bitmapPaint)
    }

    companion object {
        private const val OPAQUE = 255
        private const val NO_WIDTH = 0.0
        private const val WEIGHT_BOLD = "bold"
        private const val MIN_OPACITY = 0.0
        private const val MAX_OPACITY = 1.0
    }
}
