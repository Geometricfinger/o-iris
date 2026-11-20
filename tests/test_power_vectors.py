"""Spec E-27 Python-Kotlin replay parity for the power governor (ADR-0061): tests/vectors/power_governor.json
(scripts/make_android_vectors.py) is current, covers every threshold edge, reason and level, its power_sim readings
replay to the closed loop's levels, and android/core PowerGovernor.kt / Checks.kt are wired to it. The Kotlin side
itself runs in `:core:check` (Checks.kt powerChecks)."""
import importlib.util, json, re
from pathlib import Path

from oiris.adapters import power_sim
from oiris.power import duty, governor as G

ROOT = Path(__file__).resolve().parents[1]
VECTORS = ROOT / "tests/vectors/power_governor.json"
KOTLIN = ROOT / "android/core/src/main/kotlin/org/oiris/core/PowerGovernor.kt"
CHECKS = ROOT / "android/core/src/jvmCheck/kotlin/org/oiris/core/Checks.kt"


def _gen():
    spec = importlib.util.spec_from_file_location("make_android_vectors", ROOT / "scripts/make_android_vectors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_power_vectors_are_current():
    mod = _gen()
    on_disk = json.loads(VECTORS.read_text())
    assert mod.differences(on_disk, json.loads(json.dumps(mod.power_governor()))) == []


def test_constants_match_python():
    c = json.loads(VECTORS.read_text())["constants"]
    assert c["defaults"] == G.POWER_DEFAULTS and c["reasons"] == list(G.REASONS) and c["levels"] == list(duty.LEVELS)
    assert c["default_plans"] == duty.DEFAULT_PLANS and c["plan_keys"] == list(duty.PLAN_KEYS)
    block = re.search(r"val POWER_DEFAULTS: Map<String, Double> = linkedMapOf\((.*?)\n    \)", KOTLIN.read_text(), re.S)
    assert block
    assert {k: float(x) for k, x in re.findall(r'"(\w+)" to ([\d.]+)', block.group(1))} == G.POWER_DEFAULTS


def test_timelines_cover_edges_reasons_and_levels():
    v = json.loads(VECTORS.read_text())
    steps = [s for tl in v["timelines"] for s in tl["steps"]]
    assert {s["level"] for s in steps} == set(duty.LEVELS)
    assert {r for s in steps for r in s["reasons"]} == set(G.REASONS)
    seen = {float(x) for s in steps for k, x in s.items() if k in ("battery_pct", "skin_c", "device_c")
            and isinstance(x, (int, float)) and not isinstance(x, bool)}
    for k, x in G.POWER_DEFAULTS.items():   # every threshold edge is hit exactly
        if k.endswith(("_pct", "_c")):
            assert x in seen, k
    assert {s.get("thermal_status") for s in steps} >= set(range(G.THERMAL_STATUS_NONE, G.THERMAL_STATUS_SHUTDOWN + 1))
    assert any(s.get("now") is not None for s in steps) and any(s.get("charging") is True for s in steps)
    assert {t["level"] for t in v["thermal_status"]} == {None, duty.NORMAL, duty.HOT, duty.CRITICAL}
    assert any(not e["errors"] for e in v["validation"]) and any(e["errors"] for e in v["validation"])


def test_sim_readings_replay_to_the_closed_loop():
    v = json.loads(VECTORS.read_text())
    assert [s["scene"] for s in v["sim"]] == list(power_sim.SCENES)
    for s in v["sim"]:
        closed = power_sim.run(s["scene"], _gen().PG_SIM_DURATION_S)
        assert s["levels"] == [r["level"] for r in closed]
        gov = G.Governor()
        replay = [gov.update(G.PowerReading(t, b, c, sk, d)) for t, b, c, sk, d in s["readings"]]
        assert [d.level for d in replay] == s["levels"] and [list(d.reasons) for d in replay] == s["reasons"]
    assert {lv for s in v["sim"] for lv in s["levels"]} == {duty.NORMAL, duty.SAVER, duty.HOT, duty.CRITICAL}


def test_gate_and_duty_cases_include_refusals():
    v = json.loads(VECTORS.read_text())
    for key in ("frame_gate", "listen_duty"):
        assert {c["rejected"] for c in v[key]} == {True, False}, key


def test_kotlin_checks_are_wired():
    checks = CHECKS.read_text()
    assert "powerChecks(root)" in checks and "tests/vectors/power_governor.json" in checks
    src = KOTLIN.read_text()
    for name in ("class PowerGovernor", "class FrameGate", "class ListenDuty", "fun thermalStatusLevel", "fun validatePlans"):
        assert name in src, name
