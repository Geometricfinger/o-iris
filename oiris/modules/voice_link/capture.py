"""Microphone audio for M-17 (ADR-0057): the one place that talks to an audio input, so another capture
backend (for example Apple's voice-processing input) can be added without touching the listener.

A capture backend has `sample_rate`, `start(callback)` and `stop()`. While started it calls
`callback(block)` from its own thread with a float32 copy of each block: 1-D (mono, the default), or (n, N)
for an N-channel microphone array (ADR-0065: live mode opens one when the rig has a mic_array mount, channel
order = the mount's mic_positions_m). The copy belongs to the callee, which zeroes it after use. The stream's
own block is zeroed as soon as it is copied.
Audio is held in RAM only: nothing here writes, logs or sends a sample.

The sounddevice (PortAudio, MIT) package is imported lazily, only when a microphone is opened."""
from __future__ import annotations
import importlib, logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "float32"
BLOCK_S = 0.1                 # one callback block
SILENCE_NOTICE_S = 3.0        # this long of exact zeros => digital silence (macOS: microphone permission denied)
MIC_SETTINGS = "System Settings > Privacy & Security > Microphone"
SILENCE_NOTICE = f"no sound from the microphone: check {MIC_SETTINGS}, or the input device"
NO_SOUNDDEVICE = "the sounddevice package is not installed"
# Microphone check before enrollment window 1: opening the input makes macOS ask for microphone permission
# then, not during a real window (the first live check lost window 1 to the prompt). Nothing is kept.
MIC_CHECK_S = 1.5
MIC_QUIET_DBFS = -60.0        # below this level the input is too quiet: warn, and go on
MIC_CHECK_OK, MIC_CHECK_QUIET, MIC_CHECK_SILENT = "ok", "quiet", "silent"
MIC_PERMISSION_FIX = (f"no sound from the microphone (digital silence). In {MIC_SETTINGS}, allow the app running O Iris "
                      "(for example Terminal), or choose another input with --input (see: oiris voice-link devices)")
MIC_QUIET_HINT = "the microphone is very quiet: move closer or speak up"


class CaptureUnavailable(RuntimeError):
    """The microphone cannot be opened; the message is plain and says what to do."""


def load_sounddevice(sd: Any = None) -> Any:
    """The sounddevice module (or the injected stand-in). Raises CaptureUnavailable when not installed."""
    if sd is not None:
        return sd
    try:
        return importlib.import_module("sounddevice")
    except (ImportError, OSError) as e:   # OSError: the PortAudio library itself is missing
        raise CaptureUnavailable(NO_SOUNDDEVICE) from e


def inputs(sd: Any) -> list[tuple[int, str]]:
    """(index, name) of every device with an input channel. Names are shown, never stored."""
    return [(i, str(d.get("name", "?"))) for i, d in enumerate(sd.query_devices()) if d.get("max_input_channels", 0) > 0]


def resolve_device(value: Any, sd: Any) -> int | None:
    """None = system default; an index; or the first input device whose name contains the text."""
    if value is None or value == "":
        return None
    if isinstance(value, int) or str(value).strip().isdigit():
        return int(value)
    want = str(value).strip().lower()
    for i, name in inputs(sd):
        if want in name.lower():
            return i
    raise ValueError(f"no input device matches {value!r}; see: oiris voice-link devices")


def record(seconds: float, device: Any = None, sample_rate: int = SAMPLE_RATE, sd: Any = None) -> np.ndarray:
    """Record one window into RAM (enrollment). Returns a view of sounddevice's buffer, so zeroing it
    zeroes the buffer. `device` is an index or None (system default); see resolve_device."""
    dev = load_sounddevice(sd)
    buf = dev.rec(int(round(seconds * sample_rate)), samplerate=sample_rate, channels=CHANNELS, dtype=DTYPE, device=device)
    dev.wait()
    return buf.reshape(-1)


@dataclass
class MicCheck:
    """Result of mic_check. `ok` False only for digital silence (enrollment must not start)."""
    ok: bool
    status: str                  # MIC_CHECK_OK | MIC_CHECK_QUIET | MIC_CHECK_SILENT
    level_dbfs: float | None     # RMS level; None for digital silence
    message: str


def level_dbfs(x: np.ndarray) -> float | None:
    """RMS level in dBFS, or None when every sample is zero."""
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0
    return None if rms <= 0.0 else 20.0 * float(np.log10(rms))


def mic_check(grab: Callable[[float, int], Any], seconds: float = MIC_CHECK_S, sample_rate: int = SAMPLE_RATE,
              quiet_dbfs: float = MIC_QUIET_DBFS) -> MicCheck:
    """Open the input for `seconds` through `grab(seconds, sample_rate)` (for example record), measure the level
    and zero the buffer. Digital silence (exact zeros, as DigitalSilence detects) => not ok, with the permission
    fix; below `quiet_dbfs` => ok with a warning. Not used for enrollment; nothing is kept."""
    raw = grab(seconds, sample_rate)
    x = None
    try:
        a = raw if isinstance(raw, np.ndarray) else np.asarray(raw)
        # a multi-channel input: the level of channel 0 (the listener's reference channel)
        x = np.asarray(a[:, 0] if a.ndim == 2 else a, dtype=np.float32).reshape(-1)
        silence = DigitalSilence(sample_rate, notice_s=x.size / sample_rate if x.size else 0.0)
        if silence.feed(x) or x.size == 0:
            return MicCheck(False, MIC_CHECK_SILENT, None, MIC_PERMISSION_FIX)
        level = level_dbfs(x)
        if level is None or level < quiet_dbfs:
            shown = "n/a" if level is None else f"{level:.0f} dBFS"
            return MicCheck(True, MIC_CHECK_QUIET, level, f"{MIC_QUIET_HINT} (level {shown})")
        return MicCheck(True, MIC_CHECK_OK, level, f"microphone ok (level {level:.0f} dBFS)")
    finally:
        if isinstance(raw, np.ndarray) and raw.flags.writeable:
            raw.fill(0)
        if x is not None and x.flags.writeable:
            x.fill(0)


class DigitalSilence:
    """Tracks runs of exact-zero samples. macOS delivers all-zero audio when microphone permission is
    denied; a quiet room still has some noise, so any non-zero sample resets the run."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, notice_s: float = SILENCE_NOTICE_S):
        self.sample_rate, self.notice_s = sample_rate, notice_s
        self.zero_samples = 0

    def feed(self, block: np.ndarray) -> bool:
        """True while the latest run of exact zeros is at least notice_s long."""
        self.zero_samples = 0 if np.any(block) else self.zero_samples + int(block.size)
        return self.silent

    @property
    def silent(self) -> bool:
        return self.zero_samples >= self.notice_s * self.sample_rate

    def reset(self) -> None:
        self.zero_samples = 0


class SoundDeviceCapture:
    """A continuous 16 kHz float32 input stream through PortAudio: mono by default, or `channels` > 1 for a
    microphone array, delivered as (n, channels) blocks. `device`: an index, a name substring or None/"" for
    the system default. Raises CaptureUnavailable / ValueError with a plain reason when sounddevice is
    missing, the device is unknown or it has fewer input channels than asked for. When a multi-channel
    stream cannot be opened at start, it opens mono instead (logged; `channels` becomes 1): the listener
    then has no direction, never no voice."""
    name = "sounddevice"

    def __init__(self, device: Any = None, sample_rate: int = SAMPLE_RATE, block_s: float = BLOCK_S, sd: Any = None,
                 channels: int = CHANNELS):
        self.sd = load_sounddevice(sd)
        self.device = resolve_device(device, self.sd)
        if self.device is not None and self.device not in dict(inputs(self.sd)):
            raise ValueError(f"no input device {self.device}; see: oiris voice-link devices")
        if isinstance(channels, bool) or not isinstance(channels, int) or channels < CHANNELS:
            raise ValueError("channels must be a whole number of at least 1")
        if channels > CHANNELS and self.device is not None:
            have = int(self.sd.query_devices()[self.device].get("max_input_channels", 0))
            if have < channels:
                raise CaptureUnavailable(f"input device {self.device} has {have} channels, {channels} needed")
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = max(1, int(round(block_s * sample_rate)))
        self._stream: Any = None

    def start(self, callback: Callable[[np.ndarray], None]) -> None:
        if self._stream is not None:
            return

        def on_block(indata, frames, time_info, status) -> None:
            if self.channels == CHANNELS:
                block = np.array(indata[:, 0] if indata.ndim == 2 else indata, dtype=np.float32, copy=True)
            else:
                block = np.array(indata, dtype=np.float32, copy=True)
            indata.fill(0)
            callback(block)

        try:
            stream = self._open(on_block)
        except Exception as e:
            if self.channels == CHANNELS:
                raise
            log.warning("the microphone could not open %d channels (%s); listening in mono, without direction",
                        self.channels, e)
            self.channels = CHANNELS
            stream = self._open(on_block)
        self._stream = stream

    def _open(self, on_block: Callable[..., None]) -> Any:
        stream = self.sd.InputStream(samplerate=self.sample_rate, channels=self.channels, dtype=DTYPE, device=self.device,
                                     blocksize=self.blocksize, callback=on_block)
        if self.channels == CHANNELS:
            stream.start()                      # mono: exactly as before
            return stream
        try:
            stream.start()
        except Exception:
            stream.close()                      # the failed multi-channel stream, before mono is tried
            raise
        return stream

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()


__all__ = ["MIC_CHECK_S", "MIC_QUIET_DBFS", "SAMPLE_RATE", "SILENCE_NOTICE", "CaptureUnavailable", "DigitalSilence", "MicCheck",
           "SoundDeviceCapture", "inputs", "level_dbfs", "load_sounddevice", "mic_check", "record", "resolve_device"]
