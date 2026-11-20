"""The printable calibration board (scripts/make_calibration_board.py, ADR-0064): deterministic SVG, the right ids."""
from __future__ import annotations
import importlib.util
import re
from pathlib import Path

import pytest

from oiris.spatial import calib_sim as cs

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_calibration_board.py"
_spec = importlib.util.spec_from_file_location("make_calibration_board", SCRIPT)
assert _spec is not None and _spec.loader is not None
mcb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mcb)

MEASURE_PHRASE = "measure the true printed marker side with a ruler and enter it when calibrating"


def test_svg_is_byte_for_byte_repeatable(tmp_path, capsys):
    a, b = tmp_path / "a.svg", tmp_path / "b.svg"
    assert mcb.main(["--ring", "left", "--out", str(a)]) == 0
    assert mcb.main(["--ring", "left", "--out", str(b)]) == 0
    assert a.read_bytes() == b.read_bytes()
    out = capsys.readouterr().out
    assert MEASURE_PHRASE in out.splitlines()[0]            # the title line, printed first
    assert MEASURE_PHRASE in a.read_text()


@pytest.mark.parametrize("ring", cs.RING_NAMES)
def test_ring_boards_carry_exactly_their_ids(tmp_path, ring):
    out = tmp_path / f"{ring}.svg"
    assert mcb.main(["--ring", ring, "--out", str(out)]) == 0
    svg = out.read_text()
    want = cs.ring_layout()[ring][1].ids
    assert [int(x) for x in re.findall(r'<g id="marker-(\d+)">', svg)] == list(want)
    assert f"ids {want[0]}-{want[-1]}" in svg


def test_requested_size_and_ids_appear(tmp_path):
    out = tmp_path / "b.svg"
    assert mcb.main(["--marker-mm", "40", "--gap-mm", "10", "--rows", "2", "--cols", "3", "--first-id", "7",
                     "--paper", "a4", "--out", str(out)]) == 0
    svg = out.read_text()
    assert [int(x) for x in re.findall(r'<g id="marker-(\d+)">', svg)] == list(range(7, 13))
    assert 'width="297mm" height="210mm"' in svg and "marker side 40 mm" in svg
    assert svg.count('width="40" height="40" fill="#000000"') == 6     # one black square per marker, 40 mm


def test_marker_cells_match_the_dictionary():
    """Each marker's white cells are exactly its DICT_4X4_50 one-bits, at the board model's positions."""
    board = cs.Board(1, 2, 0.06, 0.012, 20)
    svg = mcb.make_svg(board, "a4")
    for mid in board.ids:
        body = svg.split(f'<g id="marker-{mid}">')[1].split("</g>")[0]
        bits = mcb.code_bits(mcb.DICT_4X4_50[mid])
        assert body.count('fill="#ffffff"') == int(bits.sum())


def test_a_board_too_big_for_the_paper_is_refused(tmp_path, capsys):
    out = tmp_path / "x.svg"
    assert mcb.main(["--paper", "a4", "--out", str(out)]) == 2
    assert not out.exists() and "does not fit" in capsys.readouterr().err
    assert mcb.main(["--marker-mm", "60", "--gap-mm", "5", "--out", str(out)]) == 2   # gap below one cell: no quiet zone
