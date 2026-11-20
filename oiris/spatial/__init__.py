"""Spatial sensing (ADR-0060): where each sensor sits on the head (the rig), which way a detection lies
(head-frame azimuth/elevation and a direction word), and merging detections from every sensor into events.
Built and tested in simulation only; no mount has run on real hardware yet."""
from __future__ import annotations

from .detections import DETECTIONS_KEY, SPATIAL_CAP, SpatialDetection
from .frames import DirectionTracker, direction_word
from .merge import EventTracker, Located, SpatialEvent, locate, merge
from .rig import Mount, Rig

__all__ = ["SpatialDetection", "SPATIAL_CAP", "DETECTIONS_KEY", "DirectionTracker", "direction_word", "EventTracker",
           "Located", "SpatialEvent", "locate", "merge", "Mount", "Rig"]
