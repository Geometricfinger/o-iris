"""record_clip never writes imagery of people to disk (face, RGB, NIR or eye frames), whatever a scenario holds."""
from pathlib import Path

import pytest

from oiris import headless
from oiris.adapters import sim

SCENARIOS = ("quiet", "camera_lens", "warm_prop", "fixed_emitter_drift", "replay_screen")


class _FakeCamera:
    id, caps = "fake_face_camera", {"face_frame"}

    def read(self, timeout_s):  # pragma: no cover - must never be reached
        raise AssertionError("record_clip read a person-imagery adapter")


@pytest.mark.parametrize("cap", sorted(headless.PERSON_IMAGERY_CAPS))
def test_person_imagery_is_refused_before_anything_is_written(monkeypatch, tmp_path: Path, cap):
    cam = _FakeCamera()
    cam.caps = {cap}
    monkeypatch.setattr(sim, "make", lambda scenario, seed=0: [cam])
    out = tmp_path / "clip"
    with pytest.raises(ValueError, match="person imagery"):
        headless.record_clip("anything", out, steps=2)
    assert not out.exists()                      # refused before the folder or any file was made


def test_the_sim_scenarios_carry_no_person_imagery_and_still_record(tmp_path: Path):
    for scen in SCENARIOS:
        caps = {c for a in sim.make(scen) for c in a.caps}
        assert not caps & headless.PERSON_IMAGERY_CAPS
    headless.record_clip("camera_lens", tmp_path / "clip", steps=2)
    assert (tmp_path / "clip" / "caps.json").exists()
