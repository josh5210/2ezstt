"""Thin wrapper around faster-whisper for single-utterance decoding."""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, List, NamedTuple, Optional

import numpy as np
from faster_whisper import WhisperModel

MODEL_NAME = os.getenv("EZSTT_MODEL", "medium")
DEVICE = os.getenv("EZSTT_DEVICE", "auto")
COMPUTE_TYPE = os.getenv("EZSTT_COMPUTE_TYPE", "auto")
CPU_THREADS = int(os.getenv("EZSTT_CPU_THREADS", "4"))
BEAM_SIZE = int(os.getenv("EZSTT_BEAM_SIZE", "1"))
BEST_OF = int(os.getenv("EZSTT_BEST_OF", "1"))
LANG = os.getenv("EZSTT_LANG", "en")
PAD_SHORT_MS = int(os.getenv("EZSTT_PAD_SHORT_MS", "450"))
MAX_UTTER_MS = int(os.getenv("EZSTT_MAX_UTTER_MS", "30000"))
MODEL_CACHE_DIR = os.getenv("EZSTT_MODEL_DIR", "")  # Required: path to local Whisper model directory


class _SharedModelConfig(NamedTuple):
    model_dir: str
    device: str
    compute_type: str
    cpu_threads: int


_SHARED_MODEL: Optional[WhisperModel] = None
_SHARED_MODEL_CONFIG: Optional[_SharedModelConfig] = None
_SHARED_MODEL_LOCK = Lock()


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class WhisperSegment:
    start_ms: int
    end_ms: int
    text: str
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None


@dataclass
class WhisperResult:
    text: str
    confidence: float
    segments: List[WhisperSegment]
    tokens: Optional[int]
    time_ms: Optional[int]
    device: str
    compute_type: str
    used_padding_ms: int


class WhisperBackend:
    """Thin runtime wrapper ensuring deterministic, low-latency decoding."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = DEVICE,
        compute_type: str = COMPUTE_TYPE,
        model_cache_dir: Optional[str] = MODEL_CACHE_DIR,
        cpu_threads: int = CPU_THREADS,
        beam_size: int = BEAM_SIZE,
        best_of: int = BEST_OF,
        default_lang: str = LANG,
        pad_short_ms: int = PAD_SHORT_MS,
        max_utter_ms: int = MAX_UTTER_MS,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.model_cache_dir = model_cache_dir.strip() if isinstance(model_cache_dir, str) else model_cache_dir
        self.cpu_threads = cpu_threads
        self.beam_size = beam_size
        self.best_of = best_of
        self.default_lang = default_lang
        self.pad_short_ms = pad_short_ms
        self.max_utter_ms = max_utter_ms

        self._model: Optional[WhisperModel] = None
        self._model_lock = Lock()
        self._resolved_device: Optional[str] = None
        self._resolved_compute_type: Optional[str] = None
        self._resolved_model_dir: Optional[Path] = None

    def transcribe(
        self,
        audio_pcm16: np.ndarray,
        sr: int = 16000,
        lang: Optional[str] = None,
        time_offset_ms: int = 0,
        context_left_pcm16: Optional[np.ndarray] = None,
        context_right_pcm16: Optional[np.ndarray] = None,
    ) -> WhisperResult:
        if sr != 16000:
            raise ValueError("WhisperBackend expects 16 kHz audio")
        audio = self._validate_audio(audio_pcm16, "audio_pcm16")
        context_left = self._validate_context(context_left_pcm16, "context_left_pcm16")
        context_right = self._validate_context(context_right_pcm16, "context_right_pcm16")

        max_samples = int(sr * (self.max_utter_ms / 1000.0))
        if len(audio) > max_samples:
            raise ValueError("Utterance exceeds maximum duration")

        padded_audio, used_padding_ms, left_shift_ms = self._apply_padding(
            audio, sr, context_left, context_right
        )

        float_audio = padded_audio.astype(np.float32) / 32768.0
        np.clip(float_audio, -1.0, 1.0, out=float_audio)

        model = self._ensure_model_loaded()
        language = lang or self.default_lang

        start_time = time.perf_counter()
        segments_iter, info = model.transcribe(
            float_audio,
            beam_size=self.beam_size,
            best_of=self.best_of,
            language=language,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        elapsed_ms = int(round((time.perf_counter() - start_time) * 1000))

        raw_segments = list(segments_iter)

        segments: List[WhisperSegment] = []
        logprobs: List[float] = []
        no_speech_probs: List[float] = []
        total_tokens = 0

        for seg in raw_segments:
            start_ms = int(round(seg.start * 1000.0)) if seg.start is not None else 0
            end_ms = int(round(seg.end * 1000.0)) if seg.end is not None else start_ms
            adjusted_start = start_ms - left_shift_ms + time_offset_ms
            adjusted_end = end_ms - left_shift_ms + time_offset_ms
            if adjusted_start < 0:
                adjusted_start = 0
            if adjusted_end < 0:
                adjusted_end = 0
            if adjusted_end < adjusted_start:
                adjusted_end = adjusted_start

            avg_logprob = getattr(seg, "avg_logprob", None)
            no_speech_prob = getattr(seg, "no_speech_prob", None)
            tokens = getattr(seg, "tokens", None)
            if avg_logprob is not None:
                logprobs.append(avg_logprob)
            if no_speech_prob is not None:
                no_speech_probs.append(no_speech_prob)
            if tokens is not None:
                total_tokens += len(tokens)

            segments.append(
                WhisperSegment(
                    start_ms=adjusted_start,
                    end_ms=adjusted_end,
                    text=getattr(seg, "text", ""),
                    avg_logprob=avg_logprob,
                    no_speech_prob=no_speech_prob,
                )
            )

        text = " ".join(seg.text.strip() for seg in segments if seg.text).strip()

        confidence = 0.0
        if logprobs:
            confidence = _sigmoid(sum(logprobs) / len(logprobs))
        if not text:
            confidence *= 0.3
        if confidence and no_speech_probs:
            avg_no_speech = sum(no_speech_probs) / len(no_speech_probs)
            if avg_no_speech > 0.5:
                penalty = max(0.0, 1.0 - (avg_no_speech - 0.5) * 2.0)
                confidence *= penalty
        confidence = max(0.0, min(1.0, confidence))

        return WhisperResult(
            text=text,
            confidence=confidence,
            segments=segments,
            tokens=total_tokens if total_tokens > 0 else None,
            time_ms=elapsed_ms,
            device=self._resolved_device or "unknown",
            compute_type=self._resolved_compute_type or "unknown",
            used_padding_ms=used_padding_ms,
        )

    def _ensure_model_loaded(self) -> WhisperModel:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model

            model, device_option, compute_type = self._acquire_shared_model()
            self._model = model
            self._resolved_device = device_option
            self._resolved_compute_type = compute_type

        return self._model

    def _device_attempts(self) -> List[str]:
        device_lower = (self.device or "auto").lower()
        if device_lower == "auto":
            return ["cuda", "cpu"]
        if device_lower not in {"cuda", "cpu"}:
            raise ValueError(f"Unsupported device: {self.device}")
        return [device_lower]

    def _resolve_compute_type(self, device: str) -> str:
        if (self.compute_type or "auto").lower() == "auto":
            return "float16" if device == "cuda" else "int8"
        return self.compute_type

    def _create_model(self, device: str, compute_type: str) -> WhisperModel:
        model_dir = self._get_model_dir()
        return WhisperModel(
            str(model_dir),
            device=device,
            compute_type=compute_type,
            cpu_threads=self.cpu_threads,
            local_files_only=True,
        )

    def _get_model_dir(self) -> Path:
        if self._resolved_model_dir is not None:
            return self._resolved_model_dir
        if not self.model_cache_dir:
            raise RuntimeError(
                "WhisperBackend requires EZSTT_MODEL_DIR to point to a local Whisper model directory"
            )
        path = Path(self.model_cache_dir).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"Whisper model directory does not exist: {path}")
        required = ["model.bin"]
        missing = [name for name in required if not (path / name).exists()]
        if missing:
            missing_str = ", ".join(missing)
            raise FileNotFoundError(
                f"Whisper model directory {path} is missing required files: {missing_str}"
            )
        self._resolved_model_dir = path
        return path

    def _acquire_shared_model(self) -> tuple[WhisperModel, str, str]:
        model_dir = self._get_model_dir()
        return _get_or_create_shared_model(
            model_dir=model_dir,
            device_attempts=self._device_attempts(),
            resolve_compute_type=self._resolve_compute_type,
            create_model=self._create_model,
            cpu_threads=self.cpu_threads,
        )

    def _validate_audio(self, audio: np.ndarray, name: str) -> np.ndarray:
        if not isinstance(audio, np.ndarray):
            raise ValueError(f"{name} must be a numpy ndarray")
        if audio.dtype != np.int16:
            raise ValueError(f"{name} must be int16")
        if audio.ndim != 1:
            raise ValueError(f"{name} must be mono (1-D)")
        return audio

    def _validate_context(self, context: Optional[np.ndarray], name: str) -> Optional[np.ndarray]:
        if context is None:
            return None
        return self._validate_audio(context, name)

    def _apply_padding(
        self,
        audio: np.ndarray,
        sr: int,
        context_left: Optional[np.ndarray],
        context_right: Optional[np.ndarray],
    ) -> tuple[np.ndarray, int, int]:
        target_samples = int(sr * (self.pad_short_ms / 1000.0))
        if len(audio) >= target_samples or target_samples <= 0:
            return audio, 0, 0

        missing = target_samples - len(audio)
        left_samples = missing // 2
        right_samples = missing - left_samples

        left_pad = self._build_pad(context_left, left_samples, from_left=True)
        right_pad = self._build_pad(context_right, right_samples, from_left=False)

        padded = np.concatenate((left_pad, audio, right_pad)) if missing else audio
        used_padding_ms = int(round((left_samples + right_samples) * 1000.0 / sr))
        left_shift_ms = int(round(left_samples * 1000.0 / sr))
        return padded, used_padding_ms, left_shift_ms

    def _build_pad(
        self,
        context: Optional[np.ndarray],
        samples: int,
        *,
        from_left: bool,
    ) -> np.ndarray:
        if samples <= 0:
            return np.zeros(0, dtype=np.int16)
        if context is None or len(context) == 0:
            return np.zeros(samples, dtype=np.int16)

        if len(context) >= samples:
            slice_ = context[-samples:] if from_left else context[:samples]
            return np.array(slice_, dtype=np.int16, copy=True)

        pad = np.zeros(samples, dtype=np.int16)
        if from_left:
            pad[-len(context) :] = context
        else:
            pad[: len(context)] = context
        return pad

    @classmethod
    def _reset_shared_model_for_test(cls) -> None:
        with _SHARED_MODEL_LOCK:
            global _SHARED_MODEL, _SHARED_MODEL_CONFIG
            _SHARED_MODEL = None
            _SHARED_MODEL_CONFIG = None


def _normalize_model_dir(path: Path) -> str:
    resolved = path.resolve(strict=True)
    return os.path.normcase(str(resolved))


def _ensure_shared_config_compatible(
    normalized_dir: str,
    device_attempts: List[str],
    resolve_compute_type: Callable[[str], str],
    cpu_threads: int,
    config: _SharedModelConfig,
) -> None:
    if config.model_dir != normalized_dir:
        raise RuntimeError(
            "Whisper model already initialized with a different directory: "
            f"{config.model_dir} vs {normalized_dir}"
        )
    if config.cpu_threads != cpu_threads:
        raise RuntimeError(
            "Whisper model already initialized with cpu_threads="
            f"{config.cpu_threads}, requested {cpu_threads}"
        )
    if config.device not in device_attempts:
        raise RuntimeError(
            "Whisper model already initialized for device="
            f"{config.device}, but this backend is configured for {device_attempts}"
        )
    expected_compute = resolve_compute_type(config.device)
    if expected_compute != config.compute_type:
        raise RuntimeError(
            "Whisper model already initialized with compute_type="
            f"{config.compute_type}, expected {expected_compute} for this backend"
        )


def _get_or_create_shared_model(
    *,
    model_dir: Path,
    device_attempts: List[str],
    resolve_compute_type: Callable[[str], str],
    create_model: Callable[[str, str], WhisperModel],
    cpu_threads: int,
) -> tuple[WhisperModel, str, str]:
    normalized_dir = _normalize_model_dir(model_dir)

    with _SHARED_MODEL_LOCK:
        global _SHARED_MODEL, _SHARED_MODEL_CONFIG
        if _SHARED_MODEL is not None and _SHARED_MODEL_CONFIG is not None:
            _ensure_shared_config_compatible(
                normalized_dir, device_attempts, resolve_compute_type, cpu_threads, _SHARED_MODEL_CONFIG
            )
            return _SHARED_MODEL, _SHARED_MODEL_CONFIG.device, _SHARED_MODEL_CONFIG.compute_type

        last_error: Optional[Exception] = None
        for device_option in device_attempts:
            compute_type = resolve_compute_type(device_option)
            try:
                model = create_model(device_option, compute_type)
            except Exception as exc:  # pragma: no cover - depends on local runtime
                last_error = exc
                continue

            _SHARED_MODEL = model
            _SHARED_MODEL_CONFIG = _SharedModelConfig(
                model_dir=normalized_dir,
                device=device_option,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
            )
            return model, device_option, compute_type

    error_message = "Failed to load WhisperModel from local directory"
    if last_error is not None:
        raise RuntimeError(error_message) from last_error
    raise RuntimeError(error_message)
