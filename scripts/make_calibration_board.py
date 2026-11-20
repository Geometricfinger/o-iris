#!/usr/bin/env python3
"""Write a printable ArUco calibration board as SVG (ADR-0064, docs/CALIBRATION.md).

The board is the one oiris/spatial/calib_sim.Board describes: a rows x cols grid of DICT_4X4_50 markers, ids
row by row from the top left. The SVG is plain text written here, without any SVG library, and the same
arguments always give the same bytes. Printers scale pages, so the printed size is only nominal: measure the
true printed marker side with a ruler and enter it when calibrating.

  python scripts/make_calibration_board.py --ring ahead --out board_ahead.svg            # A3 landscape, 70 mm markers
  python scripts/make_calibration_board.py --marker-mm 40 --paper a4 --out board.svg     # a board for intrinsics
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oiris.features.aruco import DICT_4X4_50, code_bits  # noqa: E402
from oiris.spatial.calib_sim import (DEFAULT_COLS, DEFAULT_GAP_M, DEFAULT_MARKER_M, DEFAULT_ROWS, MARKER_CELLS, RING_NAMES,  # noqa: E402
                                     Board, ring_layout)

MM_PER_M = 1000.0
PAPER_MM = {"a4": (210.0, 297.0), "a3": (297.0, 420.0), "letter": (215.9, 279.4), "tabloid": (279.4, 431.8)}
SIDE_MARGIN_MM = 5.0          # most printers cannot print the outer 5 mm
TITLE_BAND_MM = 12.0          # room above (and, for centring, below) the board for the title line
TITLE_FONT_MM = 3.5
TITLE_BASELINE_MM = 8.0


def _n(v: float) -> str:
    """A fixed, short number format so the output is byte-for-byte repeatable."""
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def title_line(board: Board, ring: str | None) -> str:
    where = f" for the ring position '{ring}'" if ring else ""
    ids = board.ids
    return (f"O Iris calibration board{where}: DICT_4X4_50 ids {ids[0]}-{ids[-1]}, {board.rows} x {board.cols} markers, "
            f"marker side {_n(board.marker_m * MM_PER_M)} mm and gap {_n(board.gap_m * MM_PER_M)} mm as designed. "
            "Print at 100 % (no fit to page), then measure the true printed marker side with a ruler "
            "and enter it when calibrating.")


def make_svg(board: Board, paper: str = "a3", landscape: bool = True, ring: str | None = None) -> str:
    """The SVG text. Raises ValueError when the board does not fit the paper."""
    pw, ph = PAPER_MM[paper]
    if landscape:
        pw, ph = ph, pw
    bw, bh = (x * MM_PER_M for x in board.size_m)
    if bw > pw - 2 * SIDE_MARGIN_MM or bh > ph - 2 * TITLE_BAND_MM:
        raise ValueError(f"a {_n(bw)} x {_n(bh)} mm board does not fit {paper} {'landscape' if landscape else 'portrait'} "
                         f"({_n(pw)} x {_n(ph)} mm with margins): use smaller markers or larger paper")
    cell = board.marker_m * MM_PER_M / MARKER_CELLS
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           f'<svg xmlns="http://www.w3.org/2000/svg" width="{_n(pw)}mm" height="{_n(ph)}mm" viewBox="0 0 {_n(pw)} {_n(ph)}" '
           'shape-rendering="crispEdges">',
           f"<title>{title_line(board, ring)}</title>",
           f'<rect x="0" y="0" width="{_n(pw)}" height="{_n(ph)}" fill="#ffffff"/>',
           f'<text x="{_n(SIDE_MARGIN_MM + 2)}" y="{_n(TITLE_BASELINE_MM)}" font-family="sans-serif" font-size="{_n(TITLE_FONT_MM)}" '
           f'fill="#000000">{title_line(board, ring)}</text>']
    for mid in board.ids:
        tl = board.marker_corners(mid)[0]                     # board frame, metres, y up, origin at the centre
        x0, y0 = pw / 2 + tl[0] * MM_PER_M, ph / 2 - tl[1] * MM_PER_M
        r, c = divmod(mid - board.first_id, board.cols)
        out.append(f'<g id="marker-{mid}"><!-- DICT_4X4_50 id {mid}, row {r}, column {c} -->')
        out.append(f'<rect x="{_n(x0)}" y="{_n(y0)}" width="{_n(MARKER_CELLS * cell)}" height="{_n(MARKER_CELLS * cell)}" '
                   'fill="#000000"/>')
        bits = code_bits(DICT_4X4_50[mid])
        for i in range(4):
            for j in range(4):
                if bits[i, j]:
                    out.append(f'<rect x="{_n(x0 + (j + 1) * cell)}" y="{_n(y0 + (i + 1) * cell)}" width="{_n(cell)}" '
                               f'height="{_n(cell)}" fill="#ffffff"/>')
        out.append("</g>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--marker-mm", type=float, default=DEFAULT_MARKER_M * MM_PER_M, help="marker side as designed, border included")
    ap.add_argument("--gap-mm", type=float, default=DEFAULT_GAP_M * MM_PER_M, help="white gap between markers (at least marker/6)")
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    ap.add_argument("--cols", type=int, default=DEFAULT_COLS)
    ids = ap.add_mutually_exclusive_group()
    ids.add_argument("--first-id", type=int, default=0, help="id of the top-left marker")
    ids.add_argument("--ring", choices=RING_NAMES, help="a calibration ring board: its ids follow the ring layout")
    ap.add_argument("--paper", choices=sorted(PAPER_MM), default="a3")
    ap.add_argument("--portrait", action="store_true", help="portrait instead of landscape")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        board = Board(a.rows, a.cols, a.marker_mm / MM_PER_M, a.gap_mm / MM_PER_M, a.first_id)
        if a.ring:
            board = ring_layout(board=board)[a.ring][1]
        svg = make_svg(board, a.paper, not a.portrait, a.ring)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    a.out.write_text(svg, encoding="utf-8")
    print(title_line(board, a.ring))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
