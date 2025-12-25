from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from sidecar.partials import PartialsConfig, PartialsEngine

SR = 16000
FRAME_MS = 20
FRAME_SAMPLES = SR * FRAME_MS // 1000


def make_frame(value: int = 1) -> np.ndarray:
    return np.full(FRAME_SAMPLES, value, dtype=np.int16)


@dataclass
class FakeResult:
    text: str
    confidence: float = 0.8


class StubBackend:
    def __init__(
        self,
        texts: Optional[List[str]] = None,
        confidence: float = 0.8,
        pad_short_ms: int = 450,
    ) -> None:
        self.texts = texts or [""]
        self.confidence = confidence
        self.pad_short_ms = pad_short_ms
        self.calls: List[dict] = []
        self._index = 0

    def transcribe(
        self,
        audio_pcm16: np.ndarray,
        *,
        sr: int,
        time_offset_ms: int,
        context_left_pcm16: Optional[np.ndarray],
        context_right_pcm16: Optional[np.ndarray],
    ) -> FakeResult:
        call = {
            "audio": np.array(audio_pcm16, copy=True),
            "sr": sr,
            "time_offset_ms": time_offset_ms,
            "context_left": None if context_left_pcm16 is None else np.array(context_left_pcm16, copy=True),
            "context_right": context_right_pcm16,
        }
        self.calls.append(call)
        if self._index < len(self.texts):
            text = self.texts[self._index]
        else:
            text = self.texts[-1] if self.texts else ""
        self._index += 1
        return FakeResult(text=text, confidence=self.confidence)


class InjectingBackend(StubBackend):
    def __init__(self, texts: Optional[List[str]] = None) -> None:
        super().__init__(texts=texts)
        self.engine: Optional[PartialsEngine] = None
        self.pending_inject: Optional[Tuple[np.ndarray, int]] = None
        self.reentrant = False

    def transcribe(
        self,
        audio_pcm16: np.ndarray,
        *,
        sr: int,
        time_offset_ms: int,
        context_left_pcm16: Optional[np.ndarray],
        context_right_pcm16: Optional[np.ndarray],
    ) -> FakeResult:
        pre_calls = len(self.calls)
        result = super().transcribe(
            audio_pcm16,
            sr=sr,
            time_offset_ms=time_offset_ms,
            context_left_pcm16=context_left_pcm16,
            context_right_pcm16=context_right_pcm16,
        )
        if self.pending_inject and self.engine is not None:
            frame, now_ms = self.pending_inject
            self.pending_inject = None
            self.engine.on_frame(frame, now_ms)
            if len(self.calls) > pre_calls + 1:
                self.reentrant = True
        return result


def create_engine(backend, cfg: Optional[PartialsConfig] = None):
    emitted: List = []
    engine = PartialsEngine(cfg or PartialsConfig(), backend, emitted.append)
    engine.set_context(session_id="sess", speaker_id="spk")
    return engine, emitted


def test_no_partial_before_min_ms():
    backend = StubBackend(["hi there"])
    cfg = PartialsConfig(min_ms=220)
    engine, emitted = create_engine(backend, cfg)
    engine.on_utterance_start("utt-1", 0)
    frame = make_frame()
    now = 0
    for _ in range(10):  # 200 ms total < min_ms
        now += FRAME_MS
        engine.on_frame(frame, now)
    assert emitted == []
    assert len(backend.calls) == 0


def test_cadence_basic():
    backend = StubBackend(["hel", "hello", "hello there"])
    cfg = PartialsConfig(interval_ms=300, window_ms=800, min_ms=220)
    engine, emitted = create_engine(backend, cfg)
    engine.on_utterance_start("utt-2", 0)
    frame = make_frame()
    now = 0
    for _ in range(100):
        now += FRAME_MS
        engine.on_frame(frame, now)
    assert [event.revision for event in emitted] == [1, 2, 3]
    assert [event.text for event in emitted] == ["hel", "hello", "hello there"]
    assert emitted[0].t1_ms >= cfg.min_ms
    assert all(emitted[i].t1_ms <= emitted[i + 1].t1_ms for i in range(len(emitted) - 1))
    assert all((event.t1_ms - event.t0_ms) <= cfg.window_ms for event in emitted)


def test_coalesce_when_busy():
    backend = InjectingBackend(["first", "second"])
    cfg = PartialsConfig(interval_ms=200, min_ms=200)
    engine, emitted = create_engine(backend, cfg)
    backend.engine = engine
    engine.on_utterance_start("utt-3", 0)
    frame = make_frame()
    now = 0
    for _ in range(9):
        now += FRAME_MS
        engine.on_frame(frame, now)
    inject_ms = now + FRAME_MS
    backend.pending_inject = (make_frame(2), inject_ms)
    now = inject_ms
    engine.on_frame(frame, now)
    for _ in range(20):
        now += FRAME_MS
        engine.on_frame(frame, now)
    assert not backend.reentrant
    assert len(emitted) == 2
    assert (emitted[1].t1_ms - emitted[0].t1_ms) >= cfg.interval_ms


def test_require_change_filters_duplicates():
    backend = StubBackend(["same", "same", "same"])
    cfg = PartialsConfig(require_change=True)
    engine, emitted = create_engine(backend, cfg)
    engine.on_utterance_start("utt-4", 0)
    frame = make_frame()
    now = 0
    for _ in range(60):
        now += FRAME_MS
        engine.on_frame(frame, now)
    assert len(emitted) == 1
    assert emitted[0].text == "same"
    assert len(backend.calls) >= 2


def test_window_slice_alignment():
    backend = StubBackend(["short", "longer"])
    cfg = PartialsConfig(window_ms=800)
    engine, emitted = create_engine(backend, cfg)
    engine.on_utterance_start("utt-5", 0)
    frame = make_frame()
    now = 0
    for _ in range(70):
        now += FRAME_MS
        engine.on_frame(frame, now)
    assert len(emitted) >= 2
    call = backend.calls[1]
    event = emitted[1]
    slice_ms = event.t1_ms - event.t0_ms
    expected_samples = slice_ms * cfg.sr // 1000
    assert call["audio"].shape[0] == expected_samples
    assert call["time_offset_ms"] == event.t0_ms


def test_on_utterance_end_stops_partials():
    backend = StubBackend(["first", "second"])
    engine, emitted = create_engine(backend)
    engine.on_utterance_start("utt-6", 0)
    frame = make_frame()
    now = 0
    for _ in range(20):
        now += FRAME_MS
        engine.on_frame(frame, now)
    first_count = len(backend.calls)
    engine.on_utterance_end()
    for _ in range(20):
        now += FRAME_MS
        engine.on_frame(frame, now)
    assert len(emitted) >= 1
    assert len(backend.calls) == first_count


def test_truncation_max_chars():
    backend = StubBackend(["supercalifragilisticexpialidocious"])
    cfg = PartialsConfig(max_chars=5)
    engine, emitted = create_engine(backend, cfg)
    engine.on_utterance_start("utt-7", 0)
    frame = make_frame()
    now = 0
    for _ in range(30):
        now += FRAME_MS
        engine.on_frame(frame, now)
    assert emitted
    assert emitted[0].text == "super"
