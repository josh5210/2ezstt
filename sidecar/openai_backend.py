"""OpenAI Whisper API backend for cloud-based transcription."""
from __future__ import annotations

import io
import os
import wave
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# Import shared types from whisper_backend
from .whisper_backend import WhisperResult, WhisperSegment

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "whisper-1")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")


def _pcm16_to_wav_bytes(audio_pcm16: np.ndarray, sr: int = 16000) -> bytes:
    """Convert PCM16 numpy array to WAV bytes in memory."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(audio_pcm16.tobytes())
    buffer.seek(0)
    return buffer.read()


def _sigmoid(x: float) -> float:
    """Sigmoid function for confidence mapping."""
    import math
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class OpenAIBackend:
    """OpenAI Whisper API backend for cloud-based transcription.
    
    This backend sends complete utterances to the OpenAI API for transcription.
    It is designed as a drop-in replacement for WhisperBackend.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = OPENAI_MODEL,
        default_lang: str = "en",
        base_url: str = OPENAI_BASE_URL,
    ) -> None:
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        self.default_lang = default_lang
        self.base_url = base_url.rstrip("/")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

    def transcribe(
        self,
        audio_pcm16: np.ndarray,
        sr: int = 16000,
        lang: Optional[str] = None,
        time_offset_ms: int = 0,
        context_left_pcm16: Optional[np.ndarray] = None,
        context_right_pcm16: Optional[np.ndarray] = None,
    ) -> WhisperResult:
        """Transcribe audio using OpenAI Whisper API.
        
        Args:
            audio_pcm16: Audio samples as int16 numpy array
            sr: Sample rate (must be 16000)
            lang: Language code (default: "en")
            time_offset_ms: Time offset for segment timestamps
            context_left_pcm16: Ignored (API doesn't support context)
            context_right_pcm16: Ignored (API doesn't support context)
            
        Returns:
            WhisperResult compatible with local backend
        """
        import time
        
        if sr != 16000:
            raise ValueError("OpenAIBackend expects 16 kHz audio")
        
        if not isinstance(audio_pcm16, np.ndarray) or audio_pcm16.dtype != np.int16:
            raise ValueError("audio_pcm16 must be int16 numpy array")
        
        if audio_pcm16.ndim != 1:
            raise ValueError("audio_pcm16 must be mono (1-D)")
        
        # Convert to WAV bytes
        wav_bytes = _pcm16_to_wav_bytes(audio_pcm16, sr)
        
        # Make API request
        import urllib.request
        import json
        
        language = lang or self.default_lang
        url = f"{self.base_url}/audio/transcriptions"
        
        # Build multipart form data
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        
        body_parts = []
        
        # File part
        body_parts.append(f"--{boundary}".encode())
        body_parts.append(b'Content-Disposition: form-data; name="file"; filename="audio.wav"')
        body_parts.append(b"Content-Type: audio/wav")
        body_parts.append(b"")
        body_parts.append(wav_bytes)
        
        # Model part
        body_parts.append(f"--{boundary}".encode())
        body_parts.append(b'Content-Disposition: form-data; name="model"')
        body_parts.append(b"")
        body_parts.append(self.model.encode())
        
        # Language part
        body_parts.append(f"--{boundary}".encode())
        body_parts.append(b'Content-Disposition: form-data; name="language"')
        body_parts.append(b"")
        body_parts.append(language.encode())
        
        # Response format part
        body_parts.append(f"--{boundary}".encode())
        body_parts.append(b'Content-Disposition: form-data; name="response_format"')
        body_parts.append(b"")
        body_parts.append(b"verbose_json")
        
        # Timestamp granularities part
        body_parts.append(f"--{boundary}".encode())
        body_parts.append(b'Content-Disposition: form-data; name="timestamp_granularities[]"')
        body_parts.append(b"")
        body_parts.append(b"segment")
        
        # End boundary
        body_parts.append(f"--{boundary}--".encode())
        body_parts.append(b"")
        
        body = b"\r\n".join(body_parts)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        
        start_time = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"OpenAI API error {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Network error: {e.reason}") from e
        
        elapsed_ms = int(round((time.perf_counter() - start_time) * 1000))
        
        # Parse response
        text = response_data.get("text", "").strip()
        raw_segments = response_data.get("segments", [])
        
        segments: List[WhisperSegment] = []
        for seg in raw_segments:
            start_ms = int(round(seg.get("start", 0) * 1000)) + time_offset_ms
            end_ms = int(round(seg.get("end", 0) * 1000)) + time_offset_ms
            seg_text = seg.get("text", "").strip()
            
            # OpenAI doesn't provide avg_logprob in standard response
            avg_logprob = seg.get("avg_logprob")
            no_speech_prob = seg.get("no_speech_prob")
            
            segments.append(WhisperSegment(
                start_ms=start_ms,
                end_ms=end_ms,
                text=seg_text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
            ))
        
        # Calculate confidence
        # OpenAI verbose_json may include avg_logprob per segment
        logprobs = [s.avg_logprob for s in segments if s.avg_logprob is not None]
        confidence = 0.8  # Default confidence for API responses
        if logprobs:
            confidence = _sigmoid(sum(logprobs) / len(logprobs))
        if not text:
            confidence *= 0.3
        
        return WhisperResult(
            text=text,
            confidence=confidence,
            segments=segments,
            tokens=None,  # API doesn't return token count in basic response
            time_ms=elapsed_ms,
            device="openai-api",
            compute_type="cloud",
            used_padding_ms=0,
        )
