from pathlib import Path
from typing import Optional
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sidecar.endpoint import Endpoint, EndpointConfig


_CFG = EndpointConfig()
SAMPLE_RATE = _CFG.sample_rate
FRAME_MS = _CFG.frame_ms
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


def tone(freq, ms, sr=16000, amp=0.3):
    t = np.arange(int(sr * ms / 1000)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def pcm16(x):
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16)


def silence(ms, sr=16000):
    return np.zeros(int(sr * ms / 1000), dtype=np.int16)


def to_frames(pcm):
    assert len(pcm) % FRAME_SAMPLES == 0
    return [pcm[i : i + FRAME_SAMPLES] for i in range(0, len(pcm), FRAME_SAMPLES)]


class AmplitudeVAD:
    def __init__(self, threshold: float = 800.0) -> None:
        self.threshold = threshold

    def is_speech(self, frame_bytes: bytes, sample_rate: int) -> bool:
        frame = np.frombuffer(frame_bytes, dtype=np.int16)
        return bool(np.mean(np.abs(frame)) > self.threshold)


def make_endpoint(cfg: Optional[EndpointConfig] = None) -> Endpoint:
    cfg = cfg or EndpointConfig()
    return Endpoint(cfg, vad=AmplitudeVAD())


def test_single_word_yes():
    endpoint = make_endpoint()
    audio = np.concatenate(
        [
            silence(100, SAMPLE_RATE),
            pcm16(tone(220, 260, SAMPLE_RATE, amp=0.4)),
            silence(300, SAMPLE_RATE),
        ]
    )
    results = []
    for frame in to_frames(audio):
        utterance = endpoint.feed_frame(frame)
        if utterance:
            results.append(utterance)
    assert len(results) == 1
    utterance = results[0]
    assert utterance.t0_ms == 0
    assert utterance.t1_ms == 360
    assert utterance.t1_ms - utterance.t0_ms == utterance.frames * FRAME_MS
    assert utterance.voiced_ms >= 240
    assert utterance.silence_tail_ms >= endpoint.cfg.end_silence_ms
    assert utterance.audio.dtype == np.int16
    assert len(utterance.audio) == utterance.frames * FRAME_SAMPLES


def test_back_to_back_commands():
    endpoint = make_endpoint()
    audio = np.concatenate(
        [
            silence(100, SAMPLE_RATE),
            pcm16(tone(200, 260, SAMPLE_RATE, amp=0.4)),
            silence(300, SAMPLE_RATE),
            pcm16(tone(250, 260, SAMPLE_RATE, amp=0.4)),
            silence(320, SAMPLE_RATE),
        ]
    )
    results = []
    for frame in to_frames(audio):
        utterance = endpoint.feed_frame(frame)
        if utterance:
            results.append(utterance)
    assert len(results) == 2
    first, second = results
    assert first.t0_ms == 0
    assert first.t1_ms == 360
    assert second.t0_ms == 540
    assert second.t1_ms == 920
    assert first.t1_ms <= second.t0_ms


def test_sentence_not_split():
    endpoint = make_endpoint()
    base = pcm16(tone(180, 4000, SAMPLE_RATE, amp=0.35))
    frames = to_frames(base)
    dips = [(20, 5), (60, 6), (120, 7), (150, 8)]
    for start, length in dips:
        for idx in range(start, min(start + length, len(frames))):
            frames[idx] = np.zeros_like(frames[idx])
    audio = np.concatenate([silence(100, SAMPLE_RATE), *frames, silence(320, SAMPLE_RATE)])
    results = []
    for frame in to_frames(audio):
        utterance = endpoint.feed_frame(frame)
        if utterance:
            results.append(utterance)
    assert len(results) == 1
    utterance = results[0]
    assert utterance.t0_ms == 0
    assert utterance.t1_ms == 4100
    assert utterance.frames * FRAME_MS == utterance.t1_ms - utterance.t0_ms


def test_finalize_on_flush():
    endpoint = make_endpoint()
    audio = np.concatenate(
        [
            silence(100, SAMPLE_RATE),
            pcm16(tone(220, 260, SAMPLE_RATE, amp=0.4)),
            silence(100, SAMPLE_RATE),
        ]
    )
    for frame in to_frames(audio):
        assert endpoint.feed_frame(frame) is None
    utterance = endpoint.flush()
    assert utterance is not None
    assert utterance.t0_ms == 0
    assert utterance.t1_ms == 460
    assert utterance.silence_tail_ms == 100
    assert utterance.frames * FRAME_MS == utterance.t1_ms - utterance.t0_ms


def test_strict_shape_and_dtype():
    endpoint = make_endpoint()
    with pytest.raises(ValueError):
        endpoint.feed_frame(np.zeros(FRAME_SAMPLES, dtype=np.float32))
    with pytest.raises(ValueError):
        endpoint.feed_frame(np.zeros(FRAME_SAMPLES + 1, dtype=np.int16))
    with pytest.raises(ValueError):
        endpoint.feed_frame(np.zeros((FRAME_SAMPLES, 1), dtype=np.int16))
