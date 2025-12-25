from __future__ import annotations

import anyio
import base64
import sys
import time
import types
from contextlib import contextmanager, suppress
from typing import Callable, Optional

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sidecar.events import PartialEvent
from sidecar.server import ServerConfig, app

FRAME_MS = 20
SAMPLE_RATE = 16000
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000


def make_frame(value: int = 1) -> np.ndarray:
    return np.full(FRAME_SAMPLES, value, dtype=np.int16)


class StubEndpoint:
    def __init__(self, sr: int = SAMPLE_RATE, frame_ms: int = FRAME_MS, frames_to_finalize: int = 5) -> None:
        self.cfg = types.SimpleNamespace(sample_rate=sr, frame_ms=frame_ms)
        self._frames_to_finalize = frames_to_finalize
        self._frames: list[np.ndarray] = []
        self._in_speech = False
        self._start_ms = 0
        self._clock_ms = 0

    @property
    def is_in_speech(self) -> bool:
        return self._in_speech

    def current_utterance_start_ms(self) -> Optional[int]:
        return self._start_ms if self._in_speech else None

    def feed_frame(self, frame: np.ndarray):
        frame = np.array(frame, copy=True)
        self._frames.append(frame)
        if not self._in_speech:
            self._in_speech = True
            self._start_ms = self._clock_ms
        self._clock_ms += self.cfg.frame_ms
        if len(self._frames) >= self._frames_to_finalize:
            return self._build_utterance()
        return None

    def flush(self):
        if not self._frames:
            self._in_speech = False
            return None
        return self._build_utterance()

    def _build_utterance(self):
        audio = np.concatenate(self._frames)
        frames = len(self._frames)
        utterance = types.SimpleNamespace(
            t0_ms=self._start_ms,
            t1_ms=self._start_ms + frames * self.cfg.frame_ms,
            audio=audio,
            voiced_ms=frames * self.cfg.frame_ms,
            silence_tail_ms=0,
            frames=frames,
        )
        self._frames = []
        self._in_speech = False
        return utterance


class StubWhisperBackend:
    def __init__(self, text: str = "stub-final", confidence: float = 0.9) -> None:
        self.text = text
        self.confidence = confidence
        self.calls: list[dict] = []

    def transcribe(self, audio_pcm16: np.ndarray, **kwargs):
        self.calls.append({"audio_len": len(audio_pcm16), **kwargs})
        return types.SimpleNamespace(
            text=self.text,
            confidence=self.confidence,
            tokens=5,
            time_ms=42,
            device="cpu",
            compute_type="int8",
        )


class StubKeywordSpotter:
    def __init__(self, canonical: Optional[str] = None, confidence: float = 0.95) -> None:
        self._canonical = canonical
        self.confidence = confidence
        self.calls = 0

    def spot(self, audio_pcm16: np.ndarray):
        self.calls += 1
        if self._canonical is None:
            return types.SimpleNamespace(match=None, alternatives=[], used_vocabulary=[], duration_ms=100)
        match = types.SimpleNamespace(
            canonical=self._canonical,
            surface=self._canonical,
            confidence=self.confidence,
            start_ms=0,
            end_ms=100,
        )
        return types.SimpleNamespace(match=match, alternatives=[match], used_vocabulary=[self._canonical], duration_ms=100)


class StubPartialsEngine:
    def __init__(self, cfg, backend, emit_partial: Callable[[PartialEvent], None]) -> None:
        self.emit_partial = emit_partial
        self._session_id = ""
        self._speaker_id = ""
        self._utterance_id: Optional[str] = None
        self._active = False
        self._counter = 0
        self._start_ms = 0

    def set_context(self, *, session_id: Optional[str] = None, speaker_id: Optional[str] = None) -> None:
        if session_id is not None:
            self._session_id = session_id
        if speaker_id is not None:
            self._speaker_id = speaker_id

    def on_utterance_start(self, utterance_id: str, start_ms: int) -> None:
        self._active = True
        self._counter = 0
        self._utterance_id = utterance_id
        self._start_ms = start_ms

    def on_frame(self, frame_pcm16: np.ndarray, now_ms: int) -> None:
        if not self._active or self._utterance_id is None:
            return
        self._counter += 1
        if self._counter % 2 == 0:
            event = PartialEvent(
                session_id=self._session_id,
                speaker_id=self._speaker_id,
                utterance_id=self._utterance_id,
                revision=self._counter,
                text=f"partial-{self._counter}",
                t0_ms=self._start_ms,
                t1_ms=now_ms,
                confidence=0.5,
                latency_ms=8,
            )
            self.emit_partial(event)

    def on_utterance_end(self) -> None:
        self._active = False
        self._utterance_id = None


@pytest.fixture
def ws_setup():
    attrs = [
        "server_config",
        "endpoint_factory",
        "endpoint_config_factory",
        "whisper_backend_factory",
        "partials_engine_factory",
        "keyword_spotter_factory",
    ]

    def _setup(
        config: Optional[ServerConfig] = None,
        *,
        endpoint_factory: Optional[Callable[[int], StubEndpoint]] = None,
        whisper_backend: Optional[StubWhisperBackend] = None,
        partials_factory: Optional[
            Callable[[object, StubWhisperBackend, Callable[[PartialEvent], None]], StubPartialsEngine]
        ] = None,
        keyword_spotter: Optional[StubKeywordSpotter] = None,
    ) -> ServerConfig:
        for attr in attrs:
            if hasattr(app.state, attr):
                delattr(app.state, attr)
        cfg = config or ServerConfig()
        cfg.accept_json_audio = True
        cfg.apply_logging()
        app.state.server_config = cfg
        app.state.endpoint_factory = endpoint_factory or (lambda sr: StubEndpoint(sr=sr))
        app.state.whisper_backend_factory = lambda: whisper_backend or StubWhisperBackend()
        if partials_factory is None:
            app.state.partials_engine_factory = lambda c, backend, emit: StubPartialsEngine(c, backend, emit)
        else:
            app.state.partials_engine_factory = partials_factory
        if keyword_spotter is not None:
            app.state.keyword_spotter_factory = lambda: keyword_spotter
        elif hasattr(app.state, "keyword_spotter_factory"):
            delattr(app.state, "keyword_spotter_factory")
        return cfg

    yield _setup

    for attr in attrs:
        if hasattr(app.state, attr):
            delattr(app.state, attr)


def open_client() -> TestClient:
    return TestClient(app)


def start_session(ws, *, session_id: str = "s1", speaker_id: str = "spk", transport: str = "binary") -> None:
    ws.send_json(
        {
            "type": "session.start",
            "session_id": session_id,
            "speaker_id": speaker_id,
            "sample_rate": SAMPLE_RATE,
            "format": "pcm_s16le",
            "transport": transport,
            "meta": {"test": True},
        }
    )


def collect_until_final(ws, limit: int = 20):
    partials = []
    final_event = None
    for _ in range(limit):
        payload = ws.receive_json()
        if payload["type"] == "partial":
            partials.append(payload)
        if payload["type"] == "final":
            final_event = payload
            break
    return partials, final_event


@contextmanager
def ws_session(client: TestClient, path: str, **kwargs):
    manager = client.websocket_connect(path, **kwargs)
    try:
        ws = manager.__enter__()
    except Exception:
        with suppress(Exception):
            manager.__exit__(*sys.exc_info())
        raise
    try:
        yield ws
    finally:
        with suppress(Exception):
            ws.close()
        with suppress(Exception):
            manager.__exit__(None, None, None)


def test_happy_path_binary(ws_setup):
    whisper = StubWhisperBackend(text="hello world", confidence=0.88)
    ws_setup(whisper_backend=whisper, endpoint_factory=lambda sr: StubEndpoint(sr=sr, frames_to_finalize=5))
    with open_client() as client:
        with ws_session(client, "/ws") as ws:
            start_session(ws)
            started = ws.receive_json()
            assert started["type"] == "session.started"

            frames = [make_frame(i) for i in range(5)]
            ws.send_bytes(b"".join(frame.tobytes() for frame in frames))

            partials, final_event = collect_until_final(ws)
            assert len(partials) == 2
            assert partials[0]["text"].startswith("partial-")
            assert final_event is not None
            assert final_event["type"] == "final"
            assert final_event["text"] == "hello world"
            assert final_event["source"] == "whisper"
            assert final_event["dropped_frames"] == 0

            ws.send_json({"type": "session.end"})


def test_session_end_keeps_connection_open(ws_setup):
    ws_setup(whisper_backend=StubWhisperBackend(text="phrase"), endpoint_factory=lambda sr: StubEndpoint(sr=sr, frames_to_finalize=4))
    with open_client() as client:
        with ws_session(client, "/ws") as ws:
            start_session(ws)
            ws.receive_json()

            first_frames = [make_frame(1) for _ in range(4)]
            ws.send_bytes(b"".join(frame.tobytes() for frame in first_frames))
            _, first_final = collect_until_final(ws)
            assert first_final is not None
            assert first_final["utterance_id"].endswith("-1")

            ws.send_json({"type": "session.end"})

            second_frames = [make_frame(2) for _ in range(4)]
            ws.send_bytes(b"".join(frame.tobytes() for frame in second_frames))
            _, second_final = collect_until_final(ws)
            assert second_final is not None
            assert second_final["utterance_id"].endswith("-2")


def test_backpressure_drop_oldest(ws_setup):
    config = ServerConfig()
    config.recv_queue_frames = 3
    config.recv_drop_policy = "oldest"
    ws_setup(config=config, endpoint_factory=lambda sr: StubEndpoint(sr=sr, frames_to_finalize=12))
    with open_client() as client:
        with ws_session(client, "/ws") as ws:
            start_session(ws)
            ws.receive_json()

            frames = [make_frame(i) for i in range(12)]
            ws.send_bytes(b"".join(frame.tobytes() for frame in frames))
            _, final_event = collect_until_final(ws)
            assert final_event is not None
            assert final_event["dropped_frames"] > 0


def test_idle_timeout_flushes(ws_setup):
    config = ServerConfig()
    config.session_idle_timeout_ms = 150
    config.heartbeat_interval_ms = 0
    ws_setup(config=config, endpoint_factory=lambda sr: StubEndpoint(sr=sr, frames_to_finalize=50))
    with open_client() as client:
        with ws_session(client, "/ws") as ws:
            start_session(ws)
            ws.receive_json()
            ws.send_bytes(make_frame().tobytes())

            time.sleep(0.25)
            final_event = ws.receive_json()
            assert final_event["type"] == "final"
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()


def test_json_base64_path(ws_setup):
    ws_setup(endpoint_factory=lambda sr: StubEndpoint(sr=sr, frames_to_finalize=3))
    audio_blob = b"".join(make_frame(i).tobytes() for i in range(3))
    with open_client() as client:
        with ws_session(client, "/ws") as ws:
            start_session(ws, transport="json")
            ws.receive_json()

            payload = {
                "type": "audio.chunk",
                "seq": 1,
                "pcm_base64": base64.b64encode(audio_blob).decode("ascii"),
            }
            ws.send_json(payload)
            _, final_event = collect_until_final(ws)
            assert final_event is not None
            assert final_event["frames"] == 3


def test_auth_required(ws_setup):
    config = ServerConfig()
    config.auth_token = "secret"
    ws_setup(config=config)
    with open_client() as client:
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with ws_session(client, "/ws"):
                pass
        assert excinfo.value.code == 4401

        with ws_session(client, "/ws", headers={"Authorization": "Bearer secret"}) as ws:
            start_session(ws)
            started = ws.receive_json()
            assert started["type"] == "session.started"


def test_outbox_bound(ws_setup):
    class ChattyPartials(StubPartialsEngine):
        def on_frame(self, frame_pcm16: np.ndarray, now_ms: int) -> None:
            if not self._active or self._utterance_id is None:
                return
            for _ in range(5):
                self._counter += 1
                event = PartialEvent(
                    session_id=self._session_id,
                    speaker_id=self._speaker_id,
                    utterance_id=self._utterance_id,
                    revision=self._counter,
                    text=f"burst-{self._counter}",
                    t0_ms=self._start_ms,
                    t1_ms=now_ms,
                    confidence=0.4,
                    latency_ms=5,
                )
                self.emit_partial(event)

    config = ServerConfig()
    config.outbox_max_events = 3
    ws_setup(
        config=config,
        endpoint_factory=lambda sr: StubEndpoint(sr=sr, frames_to_finalize=1),
        partials_factory=lambda c, backend, emit: ChattyPartials(c, backend, emit),
    )
    with open_client() as client:
        with ws_session(client, "/ws") as ws:
            start_session(ws)
            ws.receive_json()
            ws.send_bytes(make_frame().tobytes())
            events = []
            while True:
                try:
                    events.append(ws.receive_json())
                except WebSocketDisconnect:
                    break
                except Exception:
                    break
            partials = [e for e in events if e["type"] == "partial"]
            revisions = [p["revision"] for p in partials]
            assert revisions[-3:] == [3, 4, 5]
            assert any(e["type"] == "final" for e in events)


def test_error_on_wrong_sr(ws_setup):
    ws_setup()
    with open_client() as client:
        with ws_session(client, "/ws") as ws:
            ws.send_json(
                {
                    "type": "session.start",
                    "session_id": "bad",
                    "speaker_id": "spk",
                    "sample_rate": SAMPLE_RATE // 2,
                    "format": "pcm_s16le",
                    "transport": "binary",
                }
            )
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["code"] == "invalid_sample_rate"
            with pytest.raises(WebSocketDisconnect):
                ws.receive_json()
