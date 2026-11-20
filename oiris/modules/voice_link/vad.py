"""Energy-based voice-activity detection (ADR-0057 decision 7: energy VAD first, no model). numpy only.

30 ms frames; frame level = RMS in dBFS; noise floor = 10th percentile of the frame levels in the
buffer; a frame is speech when its level is at least max(floor + snr_db, floor_dbfs); speech frames
keep the detector on for a 200 ms hangover; segments with less than 250 ms from their first to their
last speech frame are dropped (the hangover does not count, so short clicks never pass).
RAM only: the speech samples returned are a copy the caller zeroes after use (`VadResult.zero()`)."""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

FRAME_S = 0.030
HANGOVER_S = 0.200
MIN_SEGMENT_S = 0.250
FLOOR_PERCENTILE = 10
POWER_EPS = 1e-12           # keeps log10 finite on digital silence (-120 dBFS)
SILENCE_DBFS = -120.0


@dataclass
class VadResult:
    segments: list[tuple[int, int]] = field(default_factory=list)   # [start, end) sample indices
    speech_s: float = 0.0
    speech: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    level_dbfs: float = SILENCE_DBFS   # RMS level of the speech (of the whole buffer when there is none)

    def zero(self) -> None:
        self.speech.fill(0.0)


def frame_levels(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Per-frame RMS level in dBFS, and the frame length in samples. No audio copy is made."""
    n = max(1, int(round(FRAME_S * sample_rate)))
    frames = audio.size // n
    if frames == 0:
        return np.zeros(0), n
    x = audio[:frames * n].reshape(frames, n)                 # a view
    power = np.einsum("ij,ij->i", x, x, dtype=np.float64) / n  # mean square without a squared copy
    return 10.0 * np.log10(power + POWER_EPS), n


def _level(audio: np.ndarray, ranges: list[tuple[int, int]]) -> float:
    total = sum(float(np.dot(audio[a:b], audio[a:b])) for a, b in ranges)   # views: no audio copy
    count = sum(b - a for a, b in ranges)
    return SILENCE_DBFS if count == 0 else float(10.0 * np.log10(total / count + POWER_EPS))


def detect(audio: np.ndarray, sample_rate: int, snr_db: float, floor_dbfs: float) -> VadResult:
    """`audio`: mono float32 in [-1, 1]. Returns speech segments, total speech seconds and speech samples."""
    levels, n = frame_levels(audio, sample_rate)
    if levels.size == 0:
        return VadResult(level_dbfs=_level(audio, [(0, audio.size)]))
    floor = float(np.percentile(levels, FLOOR_PERCENTILE))
    on = levels >= max(floor + snr_db, floor_dbfs)
    hang = int(round(HANGOVER_S / FRAME_S))
    active = on.copy()
    for i in np.flatnonzero(on):
        active[i + 1:i + 1 + hang] = True
    min_frames = int(np.ceil(MIN_SEGMENT_S / FRAME_S))
    segments: list[tuple[int, int]] = []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], active.astype(np.int8), [0]))))
    for start, end in zip(edges[::2], edges[1::2]):
        speech_end = int(np.flatnonzero(on[start:end])[-1]) + start + 1   # the hangover does not count toward the minimum
        if speech_end - start >= min_frames:
            segments.append((int(start) * n, int(end) * n))
    if not segments:
        return VadResult(level_dbfs=_level(audio, [(0, audio.size)]))
    speech = np.concatenate([audio[a:b] for a, b in segments]).astype(np.float32, copy=False)
    return VadResult(segments, speech.size / sample_rate, speech, _level(audio, segments))
