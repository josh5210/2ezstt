from __future__ import annotations

import os
from typing import List, Tuple, Dict, Any

import numpy as np
import pytest

from sidecar.whisper_backend import WhisperBackend


class MockSegment:
    def __init__(
        self,
        start: float,
        end: float,
        text: str,
        avg_logprob: float | None = None,
        no_speech_prob: float | None = None,
        tokens: List[int] | None = None,
    ) -> None:
        self.start = start
        self.end = end
        self.text = text
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob
        self.tokens = tokens


class MockInfo:
    def __init__(self, language: str = "en") -> None:
        self.language = language
        self.language_probability = 1.0


def _base_audio(duration_ms: int = 500) -> np.ndarray:
    samples = int(16000 * (duration_ms / 1000.0))
    return np.zeros(samples, dtype=np.int16)


def test_validation() -> None:
    backend = WhisperBackend()
    bad_dtype = np.zeros(1600, dtype=np.int32)
    with pytest.raises(ValueError):
        backend.transcribe(bad_dtype)

    bad_shape = np.zeros((10, 2), dtype=np.int16)
    with pytest.raises(ValueError):
        backend.transcribe(bad_shape)

    with pytest.raises(ValueError):
        backend.transcribe(np.zeros(1600, dtype=np.int16), sr=8000)


def test_auto_device_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: List[Tuple[str, str]] = []

    class DummyModel:
        def transcribe(self, audio: np.ndarray, **kwargs):
            segment = MockSegment(0.0, 0.5, "ok", avg_logprob=0.5, no_speech_prob=0.1, tokens=[1, 2, 3])
            return [segment], MockInfo()

    def fake_create(self: WhisperBackend, device: str, compute_type: str) -> DummyModel:
        attempts.append((device, compute_type))
        if device == "cuda":
            raise RuntimeError("cuda failed")
        return DummyModel()

    monkeypatch.setattr(WhisperBackend, "_create_model", fake_create, raising=False)

    backend = WhisperBackend(device="auto", compute_type="auto")
    audio = _base_audio(600)
    result = backend.transcribe(audio)

    assert attempts[0][0] == "cuda"
    assert attempts[1][0] == "cpu"
    assert result.device == "cpu"
    assert result.compute_type == "int8"
    assert result.tokens == 3


def test_short_audio_padding_prefers_context(monkeypatch: pytest.MonkeyPatch) -> None:
    # Use more specific typing to avoid IDE errors
    captured: Dict[str, Any] = {}

    class PadModel:
        def transcribe(self, audio: np.ndarray, **kwargs):
            captured["audio"] = audio
            captured["kwargs"] = kwargs
            segment = MockSegment(0.125, 0.3, "hi", avg_logprob=0.4, no_speech_prob=0.1)
            return [segment], MockInfo()

    def fake_create(self: WhisperBackend, device: str, compute_type: str) -> PadModel:
        return PadModel()

    monkeypatch.setattr(WhisperBackend, "_create_model", fake_create, raising=False)

    backend = WhisperBackend(pad_short_ms=450)
    utterance = _base_audio(200)
    left_context = np.arange(int(0.3 * 16000), dtype=np.int16)
    right_context = -np.arange(int(0.3 * 16000), dtype=np.int16)

    result = backend.transcribe(
        utterance,
        context_left_pcm16=left_context,
        context_right_pcm16=right_context,
    )

    target_samples = int(16000 * 0.45)
    # Cast to np.ndarray for type safety
    audio_array = captured["audio"]
    assert isinstance(audio_array, np.ndarray)
    assert audio_array.shape[0] >= target_samples
    left_needed = (target_samples - utterance.shape[0]) // 2
    ctx_tail = left_context[-left_needed:] / 32768.0
    np.testing.assert_allclose(audio_array[:left_needed], ctx_tail, rtol=1e-5, atol=1e-5)
    assert result.used_padding_ms > 0
    assert result.segments[0].start_ms <= 5


def test_confidence_heuristic(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = [
        (
            [MockSegment(0.0, 0.2, "hello", avg_logprob=1.5, no_speech_prob=0.1, tokens=[1, 2])],
            MockInfo(),
        ),
        (
            [MockSegment(0.0, 0.2, "", avg_logprob=-2.0, no_speech_prob=0.9, tokens=[])],
            MockInfo(),
        ),
    ]

    class SequencedModel:
        def transcribe(self, audio: np.ndarray, **kwargs):
            return outputs.pop(0)

    def fake_create(self: WhisperBackend, device: str, compute_type: str) -> SequencedModel:
        return SequencedModel()

    monkeypatch.setattr(WhisperBackend, "_create_model", fake_create, raising=False)
    backend = WhisperBackend()

    audio = _base_audio(500)
    high = backend.transcribe(audio)
    low = backend.transcribe(audio)

    assert high.confidence > 0.7
    assert high.tokens == 2
    assert low.confidence < 0.5


def test_time_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    class OffsetModel:
        def transcribe(self, audio: np.ndarray, **kwargs):
            segment = MockSegment(0.0, 0.2, "ok", avg_logprob=0.3, no_speech_prob=0.1)
            return [segment], MockInfo()

    def fake_create(self: WhisperBackend, device: str, compute_type: str) -> OffsetModel:
        return OffsetModel()

    monkeypatch.setattr(WhisperBackend, "_create_model", fake_create, raising=False)

    backend = WhisperBackend(pad_short_ms=300)
    utterance = _base_audio(100)
    result = backend.transcribe(utterance, time_offset_ms=1000)

    assert result.segments[0].start_ms == 900
    assert result.used_padding_ms >= 190


def test_params_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    class ParamModel:
        def transcribe(self, audio: np.ndarray, **kwargs):
            captured["kwargs"] = kwargs
            segment = MockSegment(0.0, 0.2, "hola", avg_logprob=0.4, no_speech_prob=0.2)
            return [segment], MockInfo()

    def fake_create(self: WhisperBackend, device: str, compute_type: str) -> ParamModel:
        return ParamModel()

    monkeypatch.setattr(WhisperBackend, "_create_model", fake_create, raising=False)

    backend = WhisperBackend()
    audio = _base_audio(600)
    backend.transcribe(audio, lang="es")

    # Cast to dict for type safety
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["language"] == "es"
    assert kwargs["beam_size"] == backend.beam_size == 1
    assert kwargs["best_of"] == backend.best_of == 1
    assert kwargs["vad_filter"] is False
    assert kwargs["condition_on_previous_text"] is False


@pytest.mark.skipif(os.getenv("EZSTT_RUN_SLOW_TESTS") != "1", reason="slow whisper test disabled")
def test_integration_slow() -> None:
    backend = WhisperBackend(device="cpu", compute_type="int8")
    audio = _base_audio(600)
    result = backend.transcribe(audio)
    assert isinstance(result.text, str)
