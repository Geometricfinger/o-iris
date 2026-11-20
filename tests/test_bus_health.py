"""The bus surfaces an adapter's own health()/reason even while read() returns None
(e.g. BLE waiting for the macOS Bluetooth permission prompt), not only after a sample."""
import time

from oiris.adapters.base import AdapterBase
from oiris.bus import SensorBus
from oiris.schema import Health

PERMISSION = "waiting for Bluetooth permission: answer the macOS prompt for O Iris"


class Waiting(AdapterBase):
    id, caps = "ble_wait", {"rf_adv"}
    def read(self, t):
        time.sleep(0.005)
        return None
    def health(self):
        return (Health.DEGRADED, PERMISSION)


class BadHealth(AdapterBase):
    id, caps = "ble_badhealth", {"rf_adv"}
    def read(self, t):
        time.sleep(0.005)
        return None
    def health(self):
        raise RuntimeError("status lookup failed")


def wait_for(cond, s=3.0):
    end = time.time() + s
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_none_read_surfaces_adapter_reason():
    bus = SensorBus(timeout_s=0.2)
    bus.add(Waiting())
    bus.start()
    try:
        assert wait_for(lambda: bus.health()["ble_wait"]["reason"] == PERMISSION)
        h = bus.health()["ble_wait"]
        assert h["health"] == "degraded"
        assert h["samples"] == 0 and h["restarts"] == 0
    finally:
        bus.stop()


def test_none_read_health_raising_is_degraded():
    bus = SensorBus(timeout_s=0.2)
    bus.add(BadHealth())
    bus.start()
    try:
        assert wait_for(lambda: bus.health()["ble_badhealth"]["reason"] == "health() raised")
        assert bus.health()["ble_badhealth"]["health"] == "degraded"
    finally:
        bus.stop()
