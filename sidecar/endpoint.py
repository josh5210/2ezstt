"""Speech endpointing using WebRTC VAD and a two-state ring-buffer-backed machine.

The endpoint consumes fixed-size PCM16 frames (20 ms by default), monitors voice
activity with WebRTC VAD, and emits Utterance records when enough contiguous
speech followed by sufficient silence is observed. A small ring buffer preserves
recent audio so pre-roll can be prepended when speech starts. The only public
surface is the Endpoint class, which maintains deterministic, O(1) per-frame
behaviour designed for streaming use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

import numpy as np

try:  # pragma: no cover - exercised implicitly when dependency exists
    import webrtcvad  # type: ignore
except ImportError:  # pragma: no cover - test environments may not ship webrtcvad
    webrtcvad = None  # type: ignore


@dataclass
class EndpointConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    min_speech_ms: int = 120
    end_silence_ms: int = 250
    preroll_ms: int = 120
    vad_aggressiveness: int = 2


@dataclass
class Utterance:
    t0_ms: int
    t1_ms: int
    audio: np.ndarray
    voiced_ms: int
    silence_tail_ms: int
    frames: int


class _State(Enum):
    IDLE = "IDLE"
    IN_SPEECH = "IN_SPEECH"


class _FrameRingBuffer:
    def __init__(self, capacity_frames: int) -> None:
        self._capacity = max(1, capacity_frames)
        self._frames: List[np.ndarray] = [None] * self._capacity  # type: ignore[list-item]
        self._size = 0
        self._head = 0

    def push(self, frame: np.ndarray) -> None:
        self._frames[self._head] = frame
        self._head = (self._head + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    def dump_last_frames(self, count: int) -> List[np.ndarray]:
        if self._size == 0:
            return []
        count = min(count, self._size)
        frames: List[np.ndarray] = []
        start = (self._head - count) % self._capacity
        for i in range(count):
            idx = (start + i) % self._capacity
            frame = self._frames[idx]
            if frame is None:
                continue
            frames.append(frame)
        return frames

    def dump_last_ms(self, ms: int, frame_ms: int) -> List[np.ndarray]:
        needed = math.ceil(ms / frame_ms) if ms > 0 else 0
        return self.dump_last_frames(needed)

    def clear(self) -> None:
        self._frames = [None] * self._capacity  # type: ignore[list-item]
        self._size = 0
        self._head = 0


class Endpoint:
    def __init__(self, cfg: EndpointConfig, debug: bool = False, vad: Optional[Any] = None) -> None:
        self.cfg = cfg
        self._debug = debug
        if cfg.frame_ms not in {10, 20, 30}:
            raise ValueError("frame_ms must be one of {10, 20, 30} for WebRTC VAD")
        samples_per_frame = cfg.sample_rate * cfg.frame_ms
        if samples_per_frame % 1000 != 0:
            raise ValueError("sample_rate * frame_ms must produce an integer number of samples")
        self._frame_samples = samples_per_frame // 1000
        self._preroll_frame_count = math.ceil(cfg.preroll_ms / cfg.frame_ms)
        self._min_speech_frame_count = max(1, math.ceil(cfg.min_speech_ms / cfg.frame_ms))
        cap_frames = self._preroll_frame_count + self._min_speech_frame_count + 4
        self._ring_buffer = _FrameRingBuffer(cap_frames)
        self._state = _State.IDLE
        self._clock_ms = 0
        self._voiced_run_ms = 0
        self._silence_run_ms = 0
        self._voiced_accum_ms = 0
        self._utter_start_clock_ms = 0
        self._active_frames: List[np.ndarray] = []
        if vad is not None:
            self._vad = vad
        else:
            if webrtcvad is None:
                raise ImportError("webrtcvad is required unless a custom VAD instance is supplied")
            self._vad = webrtcvad.Vad(cfg.vad_aggressiveness)

    def reset(self) -> None:
        self._state = _State.IDLE
        self._clock_ms = 0
        self._voiced_run_ms = 0
        self._silence_run_ms = 0
        self._voiced_accum_ms = 0
        self._utter_start_clock_ms = 0
        self._active_frames.clear()
        self._ring_buffer.clear()

    def feed_frame(self, frame_pcm16: np.ndarray) -> Optional[Utterance]:
        frame = self._validate_frame(frame_pcm16)
        frame_bytes = frame.tobytes()
        vad_flag = bool(self._vad.is_speech(frame_bytes, self.cfg.sample_rate))
        self._ring_buffer.push(frame)
        frame_end_ms = self._clock_ms + self.cfg.frame_ms
        utterance: Optional[Utterance] = None
        if self._state is _State.IDLE:
            utterance = self._process_idle_frame(frame, vad_flag, frame_end_ms)
        else:
            utterance = self._process_in_speech_frame(frame, vad_flag, frame_end_ms)
        self._clock_ms = frame_end_ms
        if self._debug:
            self._debug_log(
                f"clock={self._clock_ms}, state={self._state.value}, vad={int(vad_flag)}, "
                f"voiced_run_ms={self._voiced_run_ms}, silence_run_ms={self._silence_run_ms}"
            )
        return utterance

    def flush(self) -> Optional[Utterance]:
        if self._state is _State.IDLE:
            return None
        audio_frames = list(self._active_frames)
        audio = self._concat_frames(audio_frames)
        utterance = Utterance(
            t0_ms=self._utter_start_clock_ms,
            t1_ms=self._clock_ms,
            audio=audio,
            voiced_ms=self._voiced_accum_ms,
            silence_tail_ms=self._silence_run_ms,
            frames=len(audio_frames),
        )
        if self._debug:
            dur_ms = utterance.t1_ms - utterance.t0_ms
            self._debug_log(f"FINAL t1_ms={utterance.t1_ms} dur_ms={dur_ms} (flush)")
        self._reset_after_utterance()
        return utterance

    def _process_idle_frame(self, frame: np.ndarray, vad_flag: bool, frame_end_ms: int) -> Optional[Utterance]:
        if vad_flag:
            self._voiced_run_ms += self.cfg.frame_ms
        else:
            self._voiced_run_ms = 0
        if self._voiced_run_ms < self.cfg.min_speech_ms:
            return None
        self._start_utterance(frame_end_ms)
        return None

    def _process_in_speech_frame(self, frame: np.ndarray, vad_flag: bool, frame_end_ms: int) -> Optional[Utterance]:
        self._active_frames.append(frame)
        if vad_flag:
            self._silence_run_ms = 0
            self._voiced_run_ms += self.cfg.frame_ms
            self._voiced_accum_ms += self.cfg.frame_ms
            return None
        self._voiced_run_ms = 0
        self._silence_run_ms += self.cfg.frame_ms
        if self._silence_run_ms < self.cfg.end_silence_ms:
            return None
        return self._finalize(frame_end_ms)

    def _start_utterance(self, frame_end_ms: int) -> None:
        voiced_frames = max(1, self._voiced_run_ms // self.cfg.frame_ms)
        preroll_frames = self._preroll_frame_count
        frames_to_copy = self._ring_buffer.dump_last_frames(preroll_frames + voiced_frames)
        if len(frames_to_copy) < voiced_frames:
            voiced_frames = len(frames_to_copy)
        self._active_frames = list(frames_to_copy)
        actual_preroll_frames = max(0, len(frames_to_copy) - voiced_frames)
        preroll_used_ms = actual_preroll_frames * self.cfg.frame_ms
        start_clock = frame_end_ms - self._voiced_run_ms - preroll_used_ms
        if start_clock < 0:
            start_clock = 0
        self._utter_start_clock_ms = start_clock
        self._silence_run_ms = 0
        self._voiced_accum_ms = self._voiced_run_ms
        self._state = _State.IN_SPEECH
        if self._debug:
            self._debug_log(f"START t0_ms={self._utter_start_clock_ms}")

    def _finalize(self, frame_end_ms: int) -> Utterance:
        silence_frames = min(len(self._active_frames), self._silence_run_ms // self.cfg.frame_ms)
        if silence_frames > 0:
            audio_frames = self._active_frames[:-silence_frames]
        else:
            audio_frames = self._active_frames
        audio = self._concat_frames(audio_frames)
        t1_ms = frame_end_ms - self._silence_run_ms
        utterance = Utterance(
            t0_ms=self._utter_start_clock_ms,
            t1_ms=t1_ms,
            audio=audio,
            voiced_ms=self._voiced_accum_ms,
            silence_tail_ms=self._silence_run_ms,
            frames=len(audio_frames),
        )
        if self._debug:
            dur_ms = utterance.t1_ms - utterance.t0_ms
            self._debug_log(f"FINAL t1_ms={utterance.t1_ms} dur_ms={dur_ms}")
        self._reset_after_utterance()
        return utterance

    def _concat_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(frames)

    def _reset_after_utterance(self) -> None:
        self._state = _State.IDLE
        self._voiced_run_ms = 0
        self._silence_run_ms = 0
        self._voiced_accum_ms = 0
        self._utter_start_clock_ms = 0
        self._active_frames = []

    def _validate_frame(self, frame: np.ndarray) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.dtype != np.int16:
            raise ValueError("frame must be PCM16 (dtype=int16)")
        if arr.ndim != 1:
            raise ValueError("frame must be a 1-D array")
        if arr.shape[0] != self._frame_samples:
            raise ValueError(f"frame must have {self._frame_samples} samples")
        return np.array(arr, copy=True)


    @property
    def is_in_speech(self) -> bool:
        """Return True when the endpoint is currently inside a speech segment."""
        return self._state is _State.IN_SPEECH

    @property
    def frame_samples(self) -> int:
        """Number of samples expected per frame for the configured SR/frame_ms."""
        return self._frame_samples

    def current_utterance_start_ms(self) -> Optional[int]:
        """Clock timestamp (ms) where the active utterance started, if any."""
        if self._state is _State.IN_SPEECH:
            return self._utter_start_clock_ms
        return None

    def _debug_log(self, message: str) -> None:
        if self._debug:
            print(message)
