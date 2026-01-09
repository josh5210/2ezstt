from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # Load .env file before any other imports that use env vars

import faulthandler, sys, signal
faulthandler.enable(all_threads=True)
for signame in ("SIGTERM","SIGINT","SIGBREAK"):
    sig = getattr(signal, signame, None)
    if sig:
        try:
            faulthandler.register(sig, file=sys.stderr, all_threads=True)  # type: ignore[attr-defined]
        except Exception:
            pass

import asyncio
import atexit
import base64
import contextlib
import hmac
import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from types import TracebackType
from typing import Any, Callable, Deque, Dict, Optional, Type, Union, cast, overload

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header
from fastapi.responses import PlainTextResponse
from starlette.websockets import WebSocketState
from starlette.types import Message

from .endpoint import Endpoint, EndpointConfig, Utterance
from .events import PartialEvent
from .kws_vosk import KWSGrammar, KeywordSpotter, load_grammar
from .partials import PartialsConfig, PartialsEngine
from .whisper_backend import WhisperBackend

logger = logging.getLogger(__name__)

# STT Backend selection
EZSTT_BACKEND = os.getenv("EZSTT_BACKEND", "whisper").lower().strip()


def _create_stt_backend() -> WhisperBackend:
    """Factory function to create the appropriate STT backend.
    
    Returns WhisperBackend or OpenAIBackend based on EZSTT_BACKEND env var.
    Both backends share the same interface (transcribe method returns WhisperResult).
    """
    if EZSTT_BACKEND == "openai":
        from .openai_backend import OpenAIBackend
        logger.info("Using OpenAI STT backend")
        return OpenAIBackend()  # type: ignore[return-value]
    else:
        logger.info("Using local Whisper STT backend")
        return WhisperBackend()


LoopExceptionHandler = Callable[[asyncio.AbstractEventLoop, Dict[str, Any]], object]

_previous_loop_exception_handler: Optional[LoopExceptionHandler] = None
_previous_sys_excepthook: Optional[
    Callable[[Type[BaseException], BaseException, TracebackType], None]
] = None
_process_exit_handler_registered = False


def _configure_logging(level: int) -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s.%(msecs)03dZ %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    root.setLevel(level)
    logger.setLevel(level)


def _config_log_repr(cfg: "ServerConfig") -> Dict[str, Any]:
    data = asdict(cfg)
    if data.get("auth_token"):
        data["auth_token"] = "***redacted***"
    return data


def _peer_description(websocket: WebSocket) -> str:
    client = websocket.client
    if client is None:
        return "unknown"
    host = getattr(client, "host", None)
    port = getattr(client, "port", None)
    if host is None:
        return "unknown"
    if port is None:
        return str(host)
    return f"{host}:{port}"


def _trim_text(text: str, limit: int = 120) -> str:
    clean = text.replace("\\n", " ").strip()
    if len(clean) <= limit:
        return clean
    return clean[:limit] + "..."


def _asyncio_exception_handler(loop: asyncio.AbstractEventLoop, context: Dict[str, Any]) -> None:
    message = context.get("message") or "Unhandled exception in event loop"
    exception = context.get("exception")
    extra_context = {k: repr(v) for k, v in context.items() if k not in {"message", "exception"}}
    if exception is not None:
        logger.error("%s; context=%s", message, extra_context, exc_info=exception)
    else:
        logger.error("%s; context=%s", message, extra_context)
    if _previous_loop_exception_handler is not None:
        try:
            _previous_loop_exception_handler(loop, context)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Previous asyncio exception handler raised", exc_info=True)


def _sys_excepthook(exc_type: Type[BaseException], value: BaseException, tb: TracebackType) -> None:
    logger.critical("Uncaught exception", exc_info=(exc_type, value, tb))
    if _previous_sys_excepthook and _previous_sys_excepthook is not _sys_excepthook:
        try:
            _previous_sys_excepthook(exc_type, value, tb)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Previous sys.excepthook raised", exc_info=True)


def _on_process_exit() -> None:
    logger.info("Sidecar server process exiting")


def _ensure_process_exit_hook() -> None:
    global _process_exit_handler_registered
    if not _process_exit_handler_registered:
        atexit.register(_on_process_exit)
        _process_exit_handler_registered = True


def _install_exception_logging(loop: asyncio.AbstractEventLoop) -> None:
    global _previous_loop_exception_handler, _previous_sys_excepthook
    if getattr(app.state, "exception_logging_installed", False):
        return
    _previous_loop_exception_handler = loop.get_exception_handler()
    loop.set_exception_handler(_asyncio_exception_handler)
    _previous_sys_excepthook = sys.excepthook
    sys.excepthook = _sys_excepthook
    _ensure_process_exit_hook()
    app.state.exception_logging_installed = True


def _remove_exception_logging(loop: asyncio.AbstractEventLoop) -> None:
    global _previous_loop_exception_handler, _previous_sys_excepthook
    if not getattr(app.state, "exception_logging_installed", False):
        return
    loop.set_exception_handler(_previous_loop_exception_handler)
    sys.excepthook = _previous_sys_excepthook or sys.__excepthook__
    app.state.exception_logging_installed = False



EndpointFactory = Callable[[int], Endpoint]
EndpointConfigFactory = Callable[[int], EndpointConfig]
WhisperBackendFactory = Callable[[], WhisperBackend]
PartialsConfigFactory = Callable[[], PartialsConfig]
PartialsEngineFactory = Callable[[PartialsConfig, WhisperBackend, Callable[[PartialEvent], None]], PartialsEngine]
KeywordSpotterFactory = Callable[[], KeywordSpotter]


def _session_task_done_callback(session: "Session", task_name: str) -> Callable[[asyncio.Task], None]:
    def _callback(task: asyncio.Task) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            session.log.debug("%s task cancelled", task_name)
        except Exception:  # pragma: no cover - defensive
            session.log.exception("%s task failed", task_name)
    return _callback


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid int for %s=%r; using default %s", name, value, default)
        return default


@overload
def _env_str(name: str, default: None) -> Optional[str]:
    ...

@overload
def _env_str(name: str, default: str) -> str:
    ...

def _env_str(name: str, default: Optional[str]) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    return default


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8766
    max_msg_bytes: int = 2_097_152
    session_idle_timeout_ms: int = 5000
    heartbeat_interval_ms: int = 10000
    recv_queue_frames: int = 200
    recv_drop_policy: str = "oldest"
    outbox_max_events: int = 100
    auth_token: Optional[str] = None
    accept_json_audio: bool = True
    log_level: str = "INFO"
    sample_rate: int = 16000
    frame_ms: int = 20
    kws_model_dir: Optional[str] = None
    kws_grammar_path: Optional[str] = None
    kws_enabled: bool = False

    @classmethod
    def from_env(cls) -> "ServerConfig":
        cfg = cls(
            host=_env_str("EZSTT_HOST", "127.0.0.1"),
            port=_env_int("EZSTT_PORT", 8766),
            max_msg_bytes=_env_int("EZSTT_WS_MAX_MSG_BYTES", 2_097_152),
            session_idle_timeout_ms=_env_int("EZSTT_SESSION_IDLE_TIMEOUT_MS", 5000),
            heartbeat_interval_ms=_env_int("EZSTT_HEARTBEAT_INTERVAL_MS", 10000),
            recv_queue_frames=_env_int("EZSTT_RECV_QUEUE_FRAMES", 200),
            recv_drop_policy=_env_str("EZSTT_RECV_DROP_POLICY", "oldest"),
            outbox_max_events=_env_int("EZSTT_OUTBOX_MAX_EVENTS", 100),
            auth_token=_env_str("EZSTT_AUTH_TOKEN", None),
            accept_json_audio=_env_bool("EZSTT_ACCEPT_JSON_AUDIO", True),
            log_level=_env_str("EZSTT_LOG_LEVEL", "INFO"),
        )
        cfg.kws_model_dir = _env_str("EZSTT_KWS_MODEL_DIR", None)
        cfg.kws_grammar_path = _env_str("EZSTT_KWS_GRAMMAR_FILE", None)
        cfg.kws_enabled = bool(cfg.kws_model_dir)
        return cfg

    def apply_logging(self) -> None:
        level_name = (self.log_level or "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        _configure_logging(level)


app = FastAPI()


def _get_server_config() -> ServerConfig:
    cfg = getattr(app.state, "server_config", None)
    if cfg is None:
        cfg = ServerConfig.from_env()
        cfg.apply_logging()
        app.state.server_config = cfg
        logger.info("Loaded server config: %s", _config_log_repr(cfg))
    return cfg


@app.on_event("startup")
async def _on_startup() -> None:
    cfg = _get_server_config()
    loop = asyncio.get_running_loop()
    _install_exception_logging(loop)
    logger.info("Sidecar server starting host=%s port=%s sample_rate=%s frame_ms=%s json_audio=%s", cfg.host, cfg.port, cfg.sample_rate, cfg.frame_ms, cfg.accept_json_audio)


@app.on_event("shutdown")
async def _on_shutdown() -> None:
    loop = asyncio.get_running_loop()
    _remove_exception_logging(loop)
    logger.info("Sidecar server shutdown")


@dataclass
class Session:
    websocket: WebSocket
    config: ServerConfig
    session_id: str
    speaker_id: str
    sr: int
    transport: str
    app: FastAPI
    start_ns: int = field(default_factory=time.monotonic_ns)
    recv_queue: Deque[np.ndarray] = field(default_factory=deque)
    outbox: Deque[Dict[str, Any]] = field(default_factory=deque)
    dropped_frames: int = 0
    utterance_id_seq: int = 0
    active_utterance_id: Optional[str] = None
    active_start_ms: Optional[int] = None
    endpoint: Endpoint = field(init=False)
    partials_engine: PartialsEngine = field(init=False)
    whisper_backend: WhisperBackend = field(init=False)
    keyword_spotter: Optional[KeywordSpotter] = field(init=False, default=None)
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    heartbeat_task: Optional[asyncio.Task] = None
    closed: bool = False
    last_activity_ns: int = field(init=False)
    total_frames_received: int = 0
    outbox_overflow_events: int = field(init=False, default=0)
    # Debug state tracking
    _lifecycle_state: str = field(init=False, default="created")
    _messages_received: int = field(init=False, default=0)
    _messages_sent: int = field(init=False, default=0)
    _binary_messages: int = field(init=False, default=0)
    _json_messages: int = field(init=False, default=0)
    _first_frame_ns: Optional[int] = field(init=False, default=None)
    _last_frame_ns: Optional[int] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.log = logger.getChild(f"session[{self.session_id}]")
        self.peer = _peer_description(self.websocket)
        self.last_activity_ns = self.start_ns
        self.frame_samples = self.sr * self.config.frame_ms // 1000
        self.frame_bytes = self.frame_samples * 2
        self._last_queue_warning_frames = 0
        self.outbox_overflow_events = 0
        # Initialize debug tracking
        self._lifecycle_state = "initializing"
        self._messages_received = 0
        self._messages_sent = 0
        self._binary_messages = 0
        self._json_messages = 0
        self._first_frame_ns = None
        self._last_frame_ns = None
        self._init_components()
        kws_status = "enabled" if self.keyword_spotter is not None else "disabled"
        self._set_lifecycle_state("created")
        self.log.info(
            "session created peer=%s speaker=%s transport=%s sr=%d frame_ms=%d kws=%s json_audio=%s",
            self.peer,
            self.speaker_id or "<none>",
            self.transport,
            self.sr,
            self.config.frame_ms,
            kws_status,
            self.config.accept_json_audio,
        )

    def _set_lifecycle_state(self, new_state: str) -> None:
        """Track session lifecycle state transitions for debugging."""
        old_state = self._lifecycle_state
        self._lifecycle_state = new_state
        elapsed_ms = (self.now_ns() - self.start_ns) // 1_000_000
        self.log.debug(
            "lifecycle: %s -> %s elapsed_ms=%d frames=%d msgs_in=%d msgs_out=%d",
            old_state, new_state, elapsed_ms, self.total_frames_received,
            self._messages_received, self._messages_sent,
        )

    def debug_stats(self) -> Dict[str, Any]:
        """Return current debug statistics for the session."""
        elapsed_ms = (self.now_ns() - self.start_ns) // 1_000_000
        frame_duration_ms = 0
        if self._first_frame_ns and self._last_frame_ns:
            frame_duration_ms = (self._last_frame_ns - self._first_frame_ns) // 1_000_000
        return {
            "session_id": self.session_id,
            "peer": self.peer,
            "lifecycle_state": self._lifecycle_state,
            "elapsed_ms": elapsed_ms,
            "total_frames": self.total_frames_received,
            "dropped_frames": self.dropped_frames,
            "queue_depth": len(self.recv_queue),
            "messages_received": self._messages_received,
            "messages_sent": self._messages_sent,
            "binary_messages": self._binary_messages,
            "json_messages": self._json_messages,
            "frame_duration_ms": frame_duration_ms,
            "outbox_pending": len(self.outbox),
            "endpoint_in_speech": self.endpoint.is_in_speech if hasattr(self, 'endpoint') else None,
        }

    def _init_components(self) -> None:
        endpoint_factory_raw = getattr(self.app.state, "endpoint_factory", None)
        if callable(endpoint_factory_raw):
            endpoint_factory = cast(EndpointFactory, endpoint_factory_raw)
            self.endpoint = endpoint_factory(self.sr)
        else:
            cfg_factory_raw = getattr(self.app.state, "endpoint_config_factory", None)
            if callable(cfg_factory_raw):
                cfg_factory = cast(EndpointConfigFactory, cfg_factory_raw)
                endpoint_cfg = cfg_factory(self.sr)
            else:
                endpoint_cfg = EndpointConfig(sample_rate=self.sr)
            self.endpoint = Endpoint(endpoint_cfg)

        whisper_factory_raw = getattr(self.app.state, "whisper_backend_factory", None)
        if callable(whisper_factory_raw):
            whisper_factory = cast(WhisperBackendFactory, whisper_factory_raw)
            self.whisper_backend = whisper_factory()
        else:
            self.whisper_backend = _create_stt_backend()

        partials_cfg_factory_raw = getattr(self.app.state, "partials_config_factory", None)
        if callable(partials_cfg_factory_raw):
            partials_cfg_factory = cast(PartialsConfigFactory, partials_cfg_factory_raw)
            partials_cfg = partials_cfg_factory()
        else:
            partials_cfg = PartialsConfig.from_env()

        partials_engine_factory_raw = getattr(self.app.state, "partials_engine_factory", None)
        if callable(partials_engine_factory_raw):
            partials_engine_factory = cast(PartialsEngineFactory, partials_engine_factory_raw)
            self.partials_engine = partials_engine_factory(partials_cfg, self.whisper_backend, self.emit_partial)
        else:
            self.partials_engine = PartialsEngine(partials_cfg, self.whisper_backend, self.emit_partial)

        self.partials_engine.set_context(session_id=self.session_id, speaker_id=self.speaker_id)

        kws_factory_raw = getattr(self.app.state, "keyword_spotter_factory", None)
        if callable(kws_factory_raw):
            kws_factory = cast(KeywordSpotterFactory, kws_factory_raw)
            self.keyword_spotter = kws_factory()
        elif self.config.kws_enabled and self.config.kws_model_dir:
            grammar: Optional[KWSGrammar] = None
            if self.config.kws_grammar_path:
                try:
                    grammar = load_grammar(self.config.kws_grammar_path)
                except Exception as exc:  # pragma: no cover - configuration issue
                    logger.warning("Failed to load keyword grammar: %s", exc)
            try:
                self.keyword_spotter = KeywordSpotter(
                    model_dir=self.config.kws_model_dir,
                    grammar=grammar,
                    grammar_path=self.config.kws_grammar_path if grammar is None else None,
                    sample_rate=self.sr,
                )
            except Exception as exc:  # pragma: no cover - configuration issue
                logger.warning("Keyword spotter disabled: %s", exc)
                self.keyword_spotter = None
        else:
            self.keyword_spotter = None

    def now_ns(self) -> int:
        return time.monotonic_ns()

    def now_ms(self) -> int:
        return (self.now_ns() - self.start_ns) // 1_000_000

    def touch_activity(self) -> None:
        self.last_activity_ns = self.now_ns()

    def should_timeout(self) -> bool:
        if self.config.session_idle_timeout_ms <= 0:
            return False
        idle_ns = self.config.session_idle_timeout_ms * 1_000_000
        return self.now_ns() - self.last_activity_ns >= idle_ns

    def emit_partial(self, event: PartialEvent) -> None:
        payload = {
            "type": "partial",
            "session_id": event.session_id,
            "speaker_id": event.speaker_id,
            "utterance_id": event.utterance_id,
            "revision": event.revision,
            "text": event.text,
            "t0_ms": event.t0_ms,
            "t1_ms": event.t1_ms,
            "confidence": event.confidence,
            "latency_ms": event.latency_ms,
            "low_confidence": getattr(event, "low_confidence", False),
            "source": getattr(event, "source", "whisper"),
        }
        self.enqueue_outbox(payload, is_final=False)

    def enqueue_outbox(self, event: Dict[str, Any], *, is_final: bool) -> None:
        if is_final:
            self.outbox.append(event)
            return
        limit = max(0, self.config.outbox_max_events)
        if limit:
            while len(self.outbox) >= limit and self.outbox:
                if self.outbox[0].get("type") == "final":
                    return
                dropped = self.outbox.popleft()
                self.outbox_overflow_events += 1
                if self.outbox_overflow_events in {1, 10} or self.outbox_overflow_events % 50 == 0:
                    self.log.warning(
                        "dropping partial due to outbox limit=%d dropped=%d oldest_type=%s",
                        limit,
                        self.outbox_overflow_events,
                        dropped.get("type"),
                    )
        self.outbox.append(event)

    async def flush_outbox(self) -> None:
        while self.outbox:
            payload = self.outbox[0]
            if not await self.send_json(payload):
                self.outbox.clear()
                break
            self.outbox.popleft()

    async def send_json(self, payload: Dict[str, Any]) -> bool:
        if self.closed:
            self.log.debug("send_json skipped (session closed) type=%s", payload.get("type"))
            return False
        text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        msg_type = payload.get("type", "unknown")
        should_close = False
        error: Optional[BaseException] = None
        async with self.send_lock:
            if self.closed:
                return False
            try:
                await self.websocket.send_text(text)
                self._messages_sent += 1
                self.log.debug("send_json ok type=%s size=%d total_sent=%d", msg_type, len(text), self._messages_sent)
            except (WebSocketDisconnect, RuntimeError) as exc:
                should_close = True
                error = exc
            except Exception as exc:  # pragma: no cover - defensive
                should_close = True
                error = exc
        if should_close:
            if error is not None:
                self.log.debug("send_json failed type=%s error=%s; marking session closed", msg_type, error)
            await self._handle_send_failure(reason="send_failed")
            return False
        return True

    async def send_error(self, code: str, message: str) -> None:
        self.log.warning("sending error code=%s message=%s", code, message)
        await self.send_json(
            {
                "type": "error",
                "code": code,
                "message": message,
                "session_id": self.session_id,
                "speaker_id": self.speaker_id,
                "timestamp_ms": self.now_ms(),
            }
        )

    def metrics_snapshot(self) -> Dict[str, Any]:
        return {
            "type": "metrics",
            "session_id": self.session_id,
            "speaker_id": self.speaker_id,
            "timestamp_ms": self.now_ms(),
            "recv_queue_frames": len(self.recv_queue),
            "dropped_frames": self.dropped_frames,
            "total_frames": self.total_frames_received,
        }

    async def close(self, *, code: int = 1000, reason: str = "unspecified") -> None:
        if self.closed:
            self.log.debug("close() called but already closed reason=%s", reason)
            return
        self._set_lifecycle_state(f"closing:{reason}")
        elapsed_ms = (self.now_ns() - self.start_ns) // 1_000_000
        stats = self.debug_stats()
        self.log.info(
            "closing session reason=%s code=%d peer=%s dropped_frames=%d total_frames=%d outbox_pending=%d "
            "elapsed_ms=%d msgs_in=%d msgs_out=%d binary=%d json=%d",
            reason, code, self.peer, self.dropped_frames, self.total_frames_received, len(self.outbox),
            elapsed_ms, self._messages_received, self._messages_sent, self._binary_messages, self._json_messages,
        )
        # Log warning if session ended with no frames (potential issue)
        if self.total_frames_received == 0 and elapsed_ms > 100:
            self.log.warning(
                "session ended with ZERO frames after %dms - possible client/transport issue peer=%s",
                elapsed_ms, self.peer
            )
        self.closed = True
        self._set_lifecycle_state("closed")
        if self.heartbeat_task:
            if not self.heartbeat_task.done():
                self.heartbeat_task.cancel()
            if self.heartbeat_task is not asyncio.current_task():
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self.heartbeat_task
        try:
            if self.websocket.application_state is not WebSocketState.DISCONNECTED:
                await self.websocket.close(code=code)
        except RuntimeError:
            pass
        except Exception:  # pragma: no cover - defensive
            self.log.debug("Error closing websocket", exc_info=True)

    async def _handle_send_failure(self, *, reason: str) -> None:
        if self.closed:
            return
        await self.close(code=1011, reason=reason)


async def _heartbeat_loop(session: Session) -> None:
    interval_ms = session.config.heartbeat_interval_ms
    if interval_ms <= 0:
        return
    interval = interval_ms / 1000.0
    try:
        while not session.closed:
            await asyncio.sleep(interval)
            if session.closed:
                break
            if not await session.send_json(session.metrics_snapshot()):
                break
    except asyncio.CancelledError:
        raise
    except Exception:  # pragma: no cover - defensive
        if not session.closed:
            session.log.exception("Heartbeat loop failed")


async def _receive_session_start(websocket: WebSocket, config: ServerConfig) -> Dict[str, Any]:
    message = await websocket.receive()
    if message.get("text") is None:
        raise ValueError("Expected text session.start message")
    data = message["text"]
    if isinstance(data, str):
        if len(data.encode("utf-8")) > config.max_msg_bytes:
            raise ValueError("session.start exceeds maximum size")
    else:
        raise ValueError("Expected text payload")
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON payload") from exc
    if payload.get("type") != "session.start":
        raise ValueError("First message must be type session.start")
    return payload


def _authorization_ok(header: Optional[str], token: str) -> bool:
    if not header:
        return False
    scheme, _, candidate = header.partition(" ")
    if not candidate:
        return False
    if scheme.strip().lower() != "bearer":
        return False
    return hmac.compare_digest(candidate.strip(), token)


def _enqueue_frame(session: Session, frame: np.ndarray) -> None:
    limit = max(0, session.config.recv_queue_frames)
    queue = session.recv_queue
    # Track first/last frame timing for debugging
    now_ns = session.now_ns()
    if session._first_frame_ns is None:
        session._first_frame_ns = now_ns
        session.log.debug("first audio frame received peer=%s", session.peer)
    session._last_frame_ns = now_ns
    
    if limit and len(queue) >= limit:
        session.dropped_frames += 1
        queue_len = len(queue)
        if (
            session.dropped_frames in {1, 10}
            or session.dropped_frames % 100 == 0
            or queue_len > getattr(session, "_last_queue_warning_frames", 0)
        ):
            session._last_queue_warning_frames = queue_len
            session.log.warning(
                "recv queue overflow policy=%s dropped_frames=%d queue_len=%d limit=%d",
                session.config.recv_drop_policy,
                session.dropped_frames,
                queue_len,
                limit,
            )
        if session.config.recv_drop_policy.lower() == "oldest":
            if queue:
                queue.popleft()
            queue.append(frame)
        else:
            return
    else:
        queue.append(frame)
    session.total_frames_received += 1
    # Periodic debug logging for frame reception
    if session.total_frames_received in {1, 10, 50} or session.total_frames_received % 100 == 0:
        session.log.debug(
            "frame stats: total=%d queue=%d dropped=%d in_speech=%s",
            session.total_frames_received, len(queue), session.dropped_frames,
            session.endpoint.is_in_speech if hasattr(session, 'endpoint') else "N/A"
        )


async def _process_frames(session: Session) -> None:
    while session.recv_queue:
        frame = session.recv_queue.popleft()
        await _handle_frame(session, frame)
    await session.flush_outbox()


async def _handle_frame(session: Session, frame: np.ndarray) -> None:
    endpoint = session.endpoint
    was_in_speech = endpoint.is_in_speech
    utterance = endpoint.feed_frame(frame)
    now_in_speech = endpoint.is_in_speech
    if session.active_utterance_id is None:
        if now_in_speech or (utterance is not None and not was_in_speech):
            session.utterance_id_seq += 1
            utterance_id = f"{session.session_id}-{session.utterance_id_seq}"
            session.active_utterance_id = utterance_id
            start_ms = endpoint.current_utterance_start_ms()
            if start_ms is None:
                start_ms = max(session.now_ms() - session.config.frame_ms, 0)
            session.active_start_ms = start_ms
            session.partials_engine.on_utterance_start(utterance_id, start_ms)
    if session.active_utterance_id is not None:
        session.partials_engine.on_frame(frame, session.now_ms())
    if utterance is not None:
        await _finalize_utterance(session, utterance)


async def _finalize_utterance(session: Session, utterance: Utterance) -> None:
    # Skip utterances with no frames or no voiced audio
    if utterance.frames == 0:
        session.log.debug("skipping utterance with 0 frames")
        return
    if utterance.voiced_ms == 0:
        session.log.debug("skipping utterance with 0 voiced_ms")
        return

    if session.active_utterance_id is None:
        session.utterance_id_seq += 1
        utterance_id = f"{session.session_id}-{session.utterance_id_seq}"
    else:
        utterance_id = session.active_utterance_id
        session.partials_engine.on_utterance_end()
    session.active_utterance_id = None
    session.active_start_ms = None
    final_event = await _build_final_event(session, utterance, utterance_id)

    # Skip empty transcriptions
    text = str(final_event.get("text", "")).strip()
    if not text:
        session.log.debug(
            "skipping empty transcription utterance_id=%s frames=%d voiced_ms=%d",
            utterance_id,
            utterance.frames,
            utterance.voiced_ms,
        )
        return

    session.enqueue_outbox(final_event, is_final=True)
    session.log.info(
        'final event utterance_id=%s frames=%d voiced_ms=%d text="%s" source=%s latency_ms=%s dropped_frames=%d',
        final_event.get("utterance_id"),
        final_event.get("frames"),
        final_event.get("voiced_ms"),
        _trim_text(text),
        final_event.get("source"),
        final_event.get("latency_ms"),
        session.dropped_frames,
    )


async def _build_final_event(session: Session, utterance: Utterance, utterance_id: str) -> Dict[str, Any]:
    text = ""
    confidence = 0.0
    source = "whisper"
    extras: Dict[str, Any] = {}
    if session.keyword_spotter is not None:
        try:
            kws_result = session.keyword_spotter.spot(utterance.audio)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Keyword spotter failed: %s", exc)
            kws_result = None
        if kws_result and kws_result.match:
            match = kws_result.match
            text = match.canonical
            confidence = float(match.confidence)
            source = "kws"
            extras["kws_match"] = {
                "canonical": match.canonical,
                "surface": match.surface,
                "confidence": match.confidence,
                "start_ms": match.start_ms,
                "end_ms": match.end_ms,
            }
            extras["kws_duration_ms"] = kws_result.duration_ms
    if not text:
        start_time = time.perf_counter()
        result = session.whisper_backend.transcribe(
            utterance.audio,
            sr=session.sr,
            time_offset_ms=utterance.t0_ms,
            context_left_pcm16=None,
            context_right_pcm16=None,
        )
        latency_ms = int(round((time.perf_counter() - start_time) * 1000))
        text = (getattr(result, "text", "") or "").strip()
        confidence = float(getattr(result, "confidence", 0.0) or 0.0)
        extras["latency_ms"] = latency_ms
        if getattr(result, "tokens", None) is not None:
            extras["tokens"] = result.tokens
        if getattr(result, "time_ms", None) is not None:
            extras["decode_time_ms"] = result.time_ms
        extras["device"] = getattr(result, "device", None)
        extras["compute_type"] = getattr(result, "compute_type", None)
    payload = {
        "type": "final",
        "session_id": session.session_id,
        "speaker_id": session.speaker_id,
        "utterance_id": utterance_id,
        "t0_ms": utterance.t0_ms,
        "t1_ms": utterance.t1_ms,
        "frames": utterance.frames,
        "voiced_ms": utterance.voiced_ms,
        "silence_tail_ms": utterance.silence_tail_ms,
        "text": text,
        "confidence": confidence,
        "source": source,
        "dropped_frames": session.dropped_frames,
    }
    for key, value in extras.items():
        if value is not None:
            payload[key] = value
    return payload


async def _handle_binary_message(session: Session, data: bytes) -> None:
    session._binary_messages += 1
    session._messages_received += 1
    if not data:
        session.log.debug("received empty binary message")
        return
    if len(data) > session.config.max_msg_bytes:
        session.log.warning("binary message too large: %d > %d", len(data), session.config.max_msg_bytes)
        await session.send_error("message_too_large", "Binary payload exceeds limit")
        return
    if len(data) % session.frame_bytes != 0:
        session.log.warning(
            "bad frame size: %d bytes not divisible by frame_bytes=%d",
            len(data), session.frame_bytes
        )
        await session.send_error("bad_frame_size", "Audio chunk must be a multiple of a frame")
        return
    frame_bytes = session.frame_bytes
    num_frames = len(data) // frame_bytes
    session.log.debug(
        "binary message: %d bytes = %d frames, total_binary_msgs=%d",
        len(data), num_frames, session._binary_messages
    )
    for offset in range(0, len(data), frame_bytes):
        frame = np.frombuffer(data[offset : offset + frame_bytes], dtype=np.int16)
        frame = np.array(frame, copy=True)
        _enqueue_frame(session, frame)
    await _process_frames(session)


async def _handle_json_message(session: Session, payload: Dict[str, Any]) -> bool:
    session._json_messages += 1
    session._messages_received += 1
    msg_type = payload.get("type")
    session.log.debug(
        "json message type=%s total_json_msgs=%d total_msgs=%d",
        msg_type, session._json_messages, session._messages_received
    )
    if msg_type == "audio.chunk":
        if not session.config.accept_json_audio:
            session.log.debug("rejecting audio.chunk - json_audio disabled")
            await session.send_error("json_audio_disabled", "JSON audio transport disabled")
            return True
        pcm_b64 = payload.get("pcm_base64")
        if not isinstance(pcm_b64, str):
            session.log.warning("audio.chunk missing pcm_base64 field")
            await session.send_error("invalid_chunk", "pcm_base64 missing")
            return True
        try:
            audio_bytes = base64.b64decode(pcm_b64, validate=True)
        except Exception as exc:
            session.log.warning("audio.chunk invalid base64: %s", exc)
            await session.send_error("invalid_chunk", "pcm_base64 invalid")
            return True
        session.log.debug("audio.chunk decoded: %d bytes", len(audio_bytes))
        await _handle_binary_message(session, audio_bytes)
        return True
    if msg_type == "session.end":
        session.log.debug("session.end received, frames_so_far=%d", session.total_frames_received)
        await _finalize_session_end(session)
        return True
    session.log.warning("unknown json message type: %r", msg_type)
    await session.send_error("unknown_type", f"unknown message type {msg_type!r}")
    return True


async def _finalize_session_end(session: Session) -> None:
    session.log.info("session.end received; finalizing")
    session.touch_activity()
    # Skip finalization if no frames were ever received
    if session.total_frames_received == 0:
        session.log.debug("skipping finalization; no frames received")
        return
    utterance = session.endpoint.flush()
    if utterance is not None:
        await _finalize_utterance(session, utterance)
    await session.flush_outbox()


async def _handle_idle_timeout(session: Session) -> None:
    session.log.warning("idle timeout reached after %d ms", session.config.session_idle_timeout_ms)
    # Skip finalization if no frames were ever received
    if session.total_frames_received > 0:
        utterance = session.endpoint.flush()
        if utterance is not None:
            await _finalize_utterance(session, utterance)
        await session.flush_outbox()
    else:
        session.log.debug("skipping finalization; no frames received")
    await session.close(code=4408, reason="idle_timeout")


async def _finalize_disconnect(session: Session) -> None:
    session.log.info("websocket disconnect; finalizing session")
    # Skip finalization if no frames were ever received
    if session.total_frames_received > 0:
        utterance = session.endpoint.flush()
        if utterance is not None:
            await _finalize_utterance(session, utterance)
        await session.flush_outbox()
    else:
        session.log.debug("skipping finalization; no frames received")
    await session.close(code=1001, reason="client_disconnect")
async def _receive_with_timeout(session: Session) -> Optional[Message]:
    timeout_sec = session.config.session_idle_timeout_ms / 1000.0
    try:
        if timeout_sec > 0:
            message = cast(Message, await asyncio.wait_for(session.websocket.receive(), timeout=timeout_sec))
        else:
            message = cast(Message, await session.websocket.receive())
        return message
    except asyncio.TimeoutError:
        session.log.warning("receive timeout after %.2f seconds", timeout_sec)
        await _handle_idle_timeout(session)
        return None



@app.websocket("/ws")
async def websocket_handler(websocket: WebSocket, authorization: Optional[str] = Header(default=None)) -> None:
    peer = _peer_description(websocket)
    config = _get_server_config()
    logger.info("WebSocket connection opened peer=%s", peer)
    if config.auth_token:
        if not _authorization_ok(authorization, config.auth_token):
            logger.warning("Authorization failed for peer=%s", peer)
            await websocket.close(code=4401)
            return
    await websocket.accept()
    try:
        start_payload = await _receive_session_start(websocket, config)
    except Exception as exc:
        logger.warning("Invalid session.start payload from peer=%s: %s", peer, exc)
        error = {
            "type": "error",
            "code": "invalid_start",
            "message": "session.start required",
        }
        await websocket.send_text(json.dumps(error))
        await websocket.close(code=4400)
        return

    session_id = str(start_payload.get("session_id") or "").strip()
    if not session_id:
        error = {"type": "error", "code": "missing_session_id", "message": "session_id required"}
        logger.warning("Rejecting session: missing session_id peer=%s", peer)
        await websocket.send_text(json.dumps(error))
        await websocket.close(code=4400)
        return

    speaker_id = str(start_payload.get("speaker_id") or "").strip()
    sample_rate_raw = start_payload.get("sample_rate")
    if isinstance(sample_rate_raw, bool) or sample_rate_raw is None:
        error = {"type": "error", "code": "invalid_sample_rate", "message": "sample_rate must be an integer"}
        logger.warning("Rejecting session: sample_rate missing/invalid peer=%s value=%r", peer, sample_rate_raw)
        await websocket.send_text(json.dumps(error))
        await websocket.close(code=4400)
        return
    if not isinstance(sample_rate_raw, (int, str)):
        error = {"type": "error", "code": "invalid_sample_rate", "message": "sample_rate must be an integer"}
        logger.warning("Rejecting session: sample_rate wrong type peer=%s value=%r", peer, sample_rate_raw)
        await websocket.send_text(json.dumps(error))
        await websocket.close(code=4400)
        return
    try:
        sample_rate = int(cast(Union[int, str], sample_rate_raw))
    except (TypeError, ValueError):
        error = {"type": "error", "code": "invalid_sample_rate", "message": "sample_rate must be an integer"}
        logger.warning("Rejecting session: sample_rate parse failure peer=%s value=%r", peer, sample_rate_raw)
        await websocket.send_text(json.dumps(error))
        await websocket.close(code=4400)
        return
    if sample_rate != config.sample_rate:
        error = {
            "type": "error",
            "code": "invalid_sample_rate",
            "message": f"sample_rate must be {config.sample_rate}",
        }
        logger.warning("Rejecting session: sample_rate %s expected %s peer=%s", sample_rate, config.sample_rate, peer)
        await websocket.send_text(json.dumps(error))
        await websocket.close(code=4400)
        return

    transport = str(start_payload.get("transport") or "binary").strip().lower()
    if transport not in {"binary", "json"}:
        error = {"type": "error", "code": "invalid_transport", "message": "transport must be binary or json"}
        logger.warning("Rejecting session: invalid transport %s peer=%s", transport, peer)
        await websocket.send_text(json.dumps(error))
        await websocket.close(code=4400)
        return

    session = Session(
        websocket=websocket,
        config=config,
        session_id=session_id,
        speaker_id=speaker_id,
        sr=sample_rate,
        transport=transport,
        app=app,
    )
    session.touch_activity()
    session._set_lifecycle_state("handshake_complete")
    session.heartbeat_task = asyncio.create_task(_heartbeat_loop(session))
    session.heartbeat_task.add_done_callback(_session_task_done_callback(session, "heartbeat"))
    session.log.info("session handshake completed peer=%s transport=%s", peer, transport)
    session._set_lifecycle_state("active")
    if not await session.send_json(
        {
            "type": "session.started",
            "session_id": session.session_id,
            "speaker_id": session.speaker_id,
            "timestamp_ms": session.now_ms(),
        }
    ):
        session.log.warning("failed to send session.started - aborting")
        return

    try:
        session.log.debug("entering main message loop")
        loop_iterations = 0
        while True:
            loop_iterations += 1
            message = await _receive_with_timeout(session)
            if message is None:
                session.log.debug("receive timeout delivered None; closing loop after %d iterations", loop_iterations)
                break
            msg_ws_type = message.get("type", "unknown")
            if msg_ws_type == "websocket.disconnect":
                session.log.info("websocket disconnect event received from peer")
                session._set_lifecycle_state("peer_disconnect")
                break
            if message.get("text") is not None:
                session.touch_activity()
                raw_text = message["text"]
                session.log.debug("received text message len=%d", len(raw_text) if raw_text else 0)
                try:
                    payload = json.loads(raw_text)
                except json.JSONDecodeError as exc:
                    session.log.warning("JSON decode error: %s (text[:100]=%r)", exc, raw_text[:100] if raw_text else "")
                    await session.send_error("invalid_json", "Failed to decode JSON message")
                    continue
                keep_running = await _handle_json_message(session, payload)
                if not keep_running:
                    session.log.debug("json handler returned False; exiting loop")
                    break
            elif message.get("bytes") is not None:
                session.touch_activity()
                data = message["bytes"]
                session.log.debug("received binary message len=%d", len(data) if data else 0)
                await _handle_binary_message(session, data)
            else:
                session.log.debug("received message with no text/bytes: type=%s", msg_ws_type)
                continue
    except WebSocketDisconnect as exc:
        session.log.info("websocket disconnect from peer: %s", exc)
        session._set_lifecycle_state("ws_disconnect")
    except Exception:  # pragma: no cover - defensive
        session.log.exception("Unhandled websocket error")
        session._set_lifecycle_state("error")
        if not session.closed:
            await session.send_error("internal_error", "Unhandled server error")
    finally:
        session.log.debug(
            "exiting websocket handler: frames=%d msgs_in=%d msgs_out=%d",
            session.total_frames_received, session._messages_received, session._messages_sent
        )
        if not session.closed:
            await _finalize_disconnect(session)




@app.get("/healthz", response_class=PlainTextResponse)
async def healthz() -> str:
    return "ok"


@app.get("/version", response_class=PlainTextResponse)
async def version() -> str:
    return "ezstt-0.1"


class StreamingSession:
    """Minimal wiring between the audio loop and the partials engine.

    The real websocket server is expected to own one StreamingSession per
    active speaker and call these hooks from its endpointing loop. Partials are
    emitted synchronously through the supplied callback so the caller can relay
    them to clients immediately.
    """

    def __init__(
        self,
        session_id: str,
        speaker_id: str,
        whisper_backend,
        emit_partial: Callable[[PartialEvent], None],
        partials_cfg: Optional[PartialsConfig] = None,
    ) -> None:
        cfg = partials_cfg or PartialsConfig.from_env()
        self._partials = PartialsEngine(cfg, whisper_backend, emit_partial)
        self._partials.set_context(session_id=session_id, speaker_id=speaker_id)

    def on_in_speech_start(self, utterance_id: str, start_ms: int) -> None:
        self._partials.on_utterance_start(utterance_id, start_ms)

    def on_in_speech_frame(self, frame_pcm16: np.ndarray, now_ms: int) -> None:
        self._partials.on_frame(frame_pcm16, now_ms)

    def on_in_speech_end(self) -> None:
        self._partials.on_utterance_end()


__all__ = ["app", "ServerConfig", "Session", "StreamingSession"]


