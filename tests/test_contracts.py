"""Swap-contract tests: every adapter variant (real, replay, sim) and every score module
satisfies the same port contract, so they can be swapped without touching core."""
import json
import pytest

from oiris.adapters import drivers, sim
from oiris.adapters.base import ReplayAdapter, SensorAdapter
from oiris.bus import default_validator
from oiris.registry import default_registry, Registry, RegistryError
from oiris.schema import Health, Sample, Score
from oiris.scores import builtin as sb

REQUIRED_ATTRS = ("id", "bus", "caps", "license")


def all_sim():
    return sim.make("camera_lens")


@pytest.mark.parametrize("cls", drivers.REAL, ids=lambda c: c.id)
def test_real_drivers_declare_contract(cls):
    a = cls() if cls is not drivers.AndroidBridgeAdapter else cls(recv=lambda t: None)
    assert isinstance(a, SensorAdapter)
    for attr in REQUIRED_ATTRS:
        assert getattr(a, attr)
    assert a.health()[0] == Health.OK


@pytest.mark.parametrize("cls", [c for c in drivers.REAL if c is not drivers.AndroidBridgeAdapter], ids=lambda c: c.id)
def test_real_drivers_fail_cleanly_without_hardware(cls):
    a = cls()
    try:
        a.open({"site_key": b"k" * 32})
    except (drivers.DriverUnavailable, Exception) as e:   # must raise, never hang or exit
        assert not isinstance(e, SystemExit)


@pytest.mark.parametrize("a", all_sim(), ids=lambda a: a.id)
def test_sim_adapters_contract(a):
    assert isinstance(a, SensorAdapter)
    a.open({}); a.start()
    s = a.read(0.1)
    assert default_validator(s) and s.adapter_id == a.id
    s2 = a.read(0.1)
    assert s2.seq == s.seq + 1 and s2.t_mono_ns >= s.t_mono_ns
    a.stop(); a.close()


def test_replay_is_indistinguishable(tmp_path):
    from oiris.headless import record_clip, run_replay, run_scenario
    record_clip("camera_lens", tmp_path, steps=2)
    caps = json.loads((tmp_path / "caps.json").read_text())
    for aid, c in caps.items():
        r = ReplayAdapter(aid, tmp_path / f"{aid}.jsonl", set(c))
        r.open({})
        s = r.read(0)
        assert default_validator(s) and s.replay and s.t_mono_ns == 0
    rep = run_replay(tmp_path)
    live = run_scenario("camera_lens", 2)
    assert [x["signatures"] for x in rep] == [x["signatures"] for x in live]
    assert [x["level"] for x in rep] == [x["level"] for x in live]


# Inputs that no simulated adapter produces. A sim adapter for them would change the golden scenarios, so the
# contract test feeds one synthetic sample of the real shape instead (made-up values, no real identifiers).
SYNTHETIC_INPUTS = {
    # oiris/adapters/macos.py WifiMacosAdapter: the HMAC'd key, never a BSSID; the class worked out in memory.
    "ap_scan": {"aps": [{"key": "0" * 32, "tsf_us": None, "rssi": -50, "local_s": 0.0, "class": "camera_like"},
                        {"key": "1" * 32, "tsf_us": None, "rssi": -80, "local_s": 0.0, "class": "access_point"}]},
}


def contract_sample(cap: str) -> Sample:
    """One sample for capability `cap`: from the replay_screen sim scenario, else from SYNTHETIC_INPUTS.
    A capability with neither fails here (KeyError), so a new score cannot skip the contract silently."""
    adapters = {c: a for a in sim.make("replay_screen") for c in a.caps}
    if cap in adapters:
        src = adapters[cap]
        src.open({})
        return src.read(0)
    return Sample(f"synthetic_{cap}", 0, 0.0, 1, "", 1.0, SYNTHETIC_INPUTS[cap])


def test_synthetic_inputs_are_valid_samples_no_sim_adapter_produces():
    sim_caps = {c for a in sim.make("replay_screen") for c in a.caps}
    for cap in SYNTHETIC_INPUTS:
        assert cap not in sim_caps          # otherwise the sim adapter is the source and this entry is stale
        assert default_validator(contract_sample(cap))


@pytest.mark.parametrize("cls", sb.ALL, ids=lambda c: c.name)
def test_score_modules_contract(cls):
    m = cls()
    assert m.name.startswith("S_") and m.inputs
    s = m.update([])
    assert isinstance(s, Score) and s.health == Health.DEGRADED
    s = m.update([contract_sample(next(iter(m.inputs)))])
    assert 0 <= s.value <= 1 and 0 <= s.confidence <= 1


def test_registry_rejects_bad_plugins():
    r = Registry()
    with pytest.raises(RegistryError):
        r.register("sensor", "x", object, port_version="2.0")
    with pytest.raises(RegistryError):
        r.register("sensor", "x", object, port_version="1.0", license="GPL-3.0")
    with pytest.raises(RegistryError):
        r.register("nope", "x", object, port_version="1.0")
    with pytest.raises(RegistryError):
        r.get("sensor", "missing")


def test_default_registry_has_everything():
    r = default_registry()
    assert "thermal_mlx90640" in r.names("sensor") and "replay" in r.names("sensor")
    assert set(r.names("score")) == {c.name for c in sb.ALL}


def test_broken_entry_point_does_not_stop_startup(monkeypatch):
    class EP:
        name = "broken"
        def load(self):
            raise ImportError("bad plugin")

    class EPS:
        def select(self, group):
            return [EP()]
    import oiris.registry as reg
    monkeypatch.setattr(reg.metadata, "entry_points", lambda: EPS())
    assert Registry().load_entry_points() == []


def test_core_never_imports_concrete_drivers():
    import ast, pathlib
    core = ["oiris/engine.py", "oiris/bus.py", "oiris/fusion/stack.py"]
    for f in core:
        tree = ast.parse(pathlib.Path(f).read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module:
                assert "drivers" not in n.module and "sim" not in n.module.split("."), f


def test_schema_files_match_code():
    from oiris.schema import SIGNATURES
    v = json.load(open("schema/vocabulary.json"))
    assert set(v["signatures"]) == set(SIGNATURES)
    sc = json.load(open("schema/score.json"))
    assert sc["properties"]["value"]["maximum"] == 1 and "class" not in sc["properties"]


# ---- M-16 FaceMatchPort contract -----------------------------------------------------------
def _face_backends():
    from oiris.modules.face_match.backends.opencv_zoo import OpenCvZooBackend
    from oiris.modules.face_match.backends.test_deterministic import TestDeterministicBackend
    return [TestDeterministicBackend, OpenCvZooBackend]


@pytest.mark.parametrize("cls", _face_backends(), ids=lambda c: c.name)
def test_face_backends_declare_port_contract(cls):
    from oiris.registry import ALLOWED_LICENSES
    b = cls()
    for attr in ("name", "license", "model_id", "model_version", "open", "close", "health", "detect", "embed", "compare"):
        assert hasattr(b, attr), attr
    assert b.license in ALLOWED_LICENSES and isinstance(b.test_only, bool)
    # backends only compute: no storage/export/log methods on the port
    assert not [n for n in dir(b) if any(w in n.lower() for w in ("save", "store", "export", "upload", "persist", "write"))]


def test_face_test_backend_ram_only_embedding_and_health():
    from oiris.modules.face_match.backends.test_deterministic import TestDeterministicBackend, synth_face_frame
    b = TestDeterministicBackend(); b.open({})
    assert b.health()[0] == Health.OK and b.test_only
    f = synth_face_frame(3); box = b.detect(f, 1.0)[0]
    e = b.embed(f, box)
    assert e.valid(64)
    e.zero()
    assert not e.vec.any() and not e.valid()


def test_face_port_registered_lazily_with_flag_and_license():
    r = default_registry()
    p = r.get("face_match", "opencv_zoo")
    assert p.module_flag == "M-16" and p.license == "Apache-2.0" and p.port_version == "1.0"
    assert "test_deterministic" not in r.names("face_match")    # test-only backend is never registered
    with pytest.raises(RegistryError):
        r.register("face_match", "nc_model", object, port_version="1.0", license="CC-BY-NC-4.0")
    with pytest.raises(RegistryError):
        r.register("face_match", "v2", object, port_version="2.0")
