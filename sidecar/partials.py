from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from .events import PartialEvent


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    return True


@dataclass
class PartialsConfig:
    enabled: bool = True
    sr: int = 16000
    interval_ms: int = 300
    window_ms: int = 800
    min_ms: int = 220
    max_chars: int = 160
    require_change: bool = True
    low_confidence_floor: float = 0.2

    @classmethod
    def from_env(cls) -> "PartialsConfig":
        return cls(
            enabled=_env_bool("EZSTT_PARTIALS_ENABLE", True),
            sr=int(os.getenv("EZSTT_SR", "16000")),
            interval_ms=int(os.getenv("EZSTT_PARTIAL_INTERVAL_MS", "300")),
            window_ms=int(os.getenv("EZSTT_PARTIAL_WINDOW_MS", "800")),
            min_ms=int(os.getenv("EZSTT_PARTIAL_MIN_MS", "220")),
            max_chars=int(os.getenv("EZSTT_PARTIAL_MAX_CHARS", "160")),
            require_change=_env_bool("EZSTT_PARTIAL_REQUIRE_CHANGE", True),
            low_confidence_floor=float(os.getenv("EZSTT_PARTIAL_LOW_CONFIDENCE_FLOOR", "0.2")),
        )


class PartialsEngine:
    def __init__(
        self,
        cfg: PartialsConfig,
        whisper_backend,
        emit_partial: Callable[[PartialEvent], None],
    ) -> None:
        self.cfg = cfg
        self._whisper_backend = whisper_backend
        self._emit_partial = emit_partial

        self._pad_short_ms = int(getattr(whisper_backend, "pad_short_ms", 0) or 0)

        self._session_id: Optional[str] = None
        self._speaker_id: Optional[str] = None

        self._buffer: List[np.ndarray] = []
        self._total_samples = 0
        self._utterance_id: Optional[str] = None
        self._utterance_start_ms: Optional[int] = None
        self._last_attempt_ms: Optional[int] = None
        self._revision = 0
        self._last_partial_text: Optional[str] = None
        self._decoding = False
        self._active = False

    def set_context(self, *, session_id: Optional[str] = None, speaker_id: Optional[str] = None) -> None:
        if session_id is not None:
            self._session_id = session_id
        if speaker_id is not None:
            self._speaker_id = speaker_id

    def on_utterance_start(self, utterance_id: str, start_ms: int) -> None:
        if not self.cfg.enabled:
            return

        self._buffer = []
        self._total_samples = 0
        self._utterance_id = utterance_id
        self._utterance_start_ms = start_ms
        self._last_attempt_ms = None
        self._revision = 0
        self._last_partial_text = None
        self._decoding = False
        self._active = True

    def on_frame(self, frame_pcm16: np.ndarray, now_ms: int) -> None:
        if not self.cfg.enabled or not self._active:
            return
        if self._utterance_id is None or self._utterance_start_ms is None:
            return

        frame = self._coerce_frame(frame_pcm16)
        self._buffer.append(frame)
        self._total_samples += frame.shape[0]

        self._maybe_attempt(now_ms)

    def on_utterance_end(self) -> None:
        self._active = False
        self._buffer = []
        self._total_samples = 0
        self._utterance_id = None
        self._utterance_start_ms = None
        self._last_attempt_ms = None
        self._revision = 0
        self._last_partial_text = None
        self._decoding = False

    def _coerce_frame(self, frame: np.ndarray) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise ValueError("frame_pcm16 must be a numpy ndarray")
        if frame.ndim != 1:
            raise ValueError("frame_pcm16 must be 1-D")
        if frame.dtype != np.int16:
            frame = frame.astype(np.int16, copy=False)
        return np.array(frame, copy=True)

    def _maybe_attempt(self, now_ms: int) -> None:
        if self._decoding or self._total_samples == 0:
            return
        if self._utterance_start_ms is None or self._utterance_id is None:
            return

        voiced_ms = max(0, now_ms - self._utterance_start_ms)
        if voiced_ms < self.cfg.min_ms:
            return

        if self._last_attempt_ms is not None and now_ms - self._last_attempt_ms < self.cfg.interval_ms:
            return

        audio = self._concat_audio()
        slice_end_ms = max(now_ms, self._utterance_start_ms)
        slice_start_ms = max(self._utterance_start_ms, slice_end_ms - self.cfg.window_ms)

        start_offset_ms = slice_start_ms - self._utterance_start_ms
        end_offset_ms = slice_end_ms - self._utterance_start_ms

        sample_start = self._ms_to_samples(start_offset_ms)
        sample_end = self._ms_to_samples(end_offset_ms)
        sample_end = min(sample_end, audio.shape[0])
        sample_start = max(0, min(sample_start, sample_end))

        window_audio = np.array(audio[sample_start:sample_end], copy=True)
        if window_audio.size == 0:
            self._last_attempt_ms = now_ms
            return

        context_left = self._build_left_context(audio, sample_start)

        self._decode_and_emit(window_audio, context_left, slice_start_ms, slice_end_ms, now_ms)

    def _concat_audio(self) -> np.ndarray:
        if not self._buffer:
            return np.zeros(0, dtype=np.int16)
        if len(self._buffer) == 1:
            return np.array(self._buffer[0], copy=True)
        return np.concatenate(self._buffer)

    def _build_left_context(self, audio: np.ndarray, sample_start: int) -> Optional[np.ndarray]:
        if sample_start <= 0:
            return None
        pad_samples = max(0, int(self._pad_short_ms * self.cfg.sr / 1000))
        if pad_samples <= 0:
            return None
        start = max(0, sample_start - pad_samples)
        if start == sample_start:
            return None
        return np.array(audio[start:sample_start], copy=True)

    def _decode_and_emit(
        self,
        window_audio: np.ndarray,
        context_left: Optional[np.ndarray],
        slice_start_ms: int,
        slice_end_ms: int,
        now_ms: int,
    ) -> None:
        self._decoding = True
        self._last_attempt_ms = now_ms
        try:
            start_time = time.perf_counter()
            result = self._whisper_backend.transcribe(
                window_audio,
                sr=self.cfg.sr,
                time_offset_ms=slice_start_ms,
                context_left_pcm16=context_left,
                context_right_pcm16=None,
            )
            latency_ms = int(round((time.perf_counter() - start_time) * 1000))
        finally:
            self._decoding = False

        text = (getattr(result, "text", "") or "").strip()
        if not text:
            return

        confidence = float(getattr(result, "confidence", 0.0) or 0.0)
        low_confidence = confidence < self.cfg.low_confidence_floor

        truncated = text[: self.cfg.max_chars]
        if self.cfg.require_change and not low_confidence and truncated == self._last_partial_text:
            return

        self._revision += 1
        event = PartialEvent(
            session_id=self._session_id or "",
            speaker_id=self._speaker_id or "",
            utterance_id=self._utterance_id or "",
            revision=self._revision,
            text=truncated,
            t0_ms=slice_start_ms,
            t1_ms=slice_end_ms,
            confidence=confidence,
            latency_ms=latency_ms,
            low_confidence=low_confidence,
        )
        self._emit_partial(event)
        if not low_confidence:
            self._last_partial_text = truncated

    def _ms_to_samples(self, ms: int) -> int:
        if ms <= 0:
            return 0
        return int(ms * self.cfg.sr // 1000)


__all__ = ["PartialsConfig", "PartialsEngine"]
