"""Capture rules (ADR-0063): the bystander recording light, the hardware microphone mute and the interlock that lets
the camera and the microphone run only while both allow it. Ports in indicator.py and mute.py, the pure interlock in
interlock.py; the simulated and Pi GPIO adapters are oiris/adapters/capture_sim.py and capture_gpio.py."""
from .indicator import INDICATOR_STATES, IndicatorPort, NoIndicator
from .interlock import CAMERA, KINDS, MIC, REFUSAL_CODES, CaptureInterlock, Grant, Refusal
from .mute import MUTE_STATES, MuteSense, UnknownMute

__all__ = ["CAMERA", "INDICATOR_STATES", "KINDS", "MIC", "MUTE_STATES", "REFUSAL_CODES", "CaptureInterlock", "Grant",
           "IndicatorPort", "MuteSense", "NoIndicator", "Refusal", "UnknownMute"]
