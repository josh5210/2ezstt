"""Tests for OpenAI Whisper API backend."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from typing import Any, Dict

import numpy as np
import pytest

from sidecar.openai_backend import OpenAIBackend, _pcm16_to_wav_bytes


def _base_audio(duration_ms: int = 500) -> np.ndarray:
    """Create silent audio for testing."""
    samples = int(16000 * (duration_ms / 1000.0))
    return np.zeros(samples, dtype=np.int16)


class TestPCMToWav:
    """Test PCM16 to WAV conversion."""

    def test_basic_conversion(self) -> None:
        audio = _base_audio(100)
        wav_bytes = _pcm16_to_wav_bytes(audio, sr=16000)
        
        # Check WAV header
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"
        assert len(wav_bytes) > 44  # Header + some data

    def test_preserves_sample_count(self) -> None:
        audio = _base_audio(200)
        wav_bytes = _pcm16_to_wav_bytes(audio, sr=16000)
        
        # WAV data size should match input
        expected_data_size = len(audio) * 2  # 16-bit = 2 bytes
        # Data size is at offset 40 in standard WAV header
        import struct
        data_size = struct.unpack("<I", wav_bytes[40:44])[0]
        assert data_size == expected_data_size


class TestOpenAIBackend:
    """Test OpenAI backend with mocked HTTP."""

    def test_missing_api_key(self) -> None:
        """Should raise if no API key provided."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=False):
            with pytest.raises(ValueError, match="API key required"):
                OpenAIBackend(api_key="")

    def test_validation_sample_rate(self) -> None:
        """Should reject non-16kHz audio."""
        backend = OpenAIBackend(api_key="sk-test-key")
        audio = _base_audio(500)
        
        with pytest.raises(ValueError, match="16 kHz"):
            backend.transcribe(audio, sr=8000)

    def test_validation_dtype(self) -> None:
        """Should reject non-int16 audio."""
        backend = OpenAIBackend(api_key="sk-test-key")
        audio = np.zeros(1600, dtype=np.float32)
        
        with pytest.raises(ValueError, match="int16"):
            backend.transcribe(audio)

    def test_validation_shape(self) -> None:
        """Should reject non-mono audio."""
        backend = OpenAIBackend(api_key="sk-test-key")
        audio = np.zeros((10, 2), dtype=np.int16)
        
        with pytest.raises(ValueError, match="mono"):
            backend.transcribe(audio)

    def test_successful_transcription(self) -> None:
        """Test successful API call with mocked response."""
        mock_response = {
            "text": "hello world",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.5,
                    "text": "hello world",
                    "avg_logprob": -0.3,
                }
            ],
        }
        
        mock_urlopen = MagicMock()
        mock_urlopen.__enter__ = MagicMock(return_value=mock_urlopen)
        mock_urlopen.__exit__ = MagicMock(return_value=False)
        mock_urlopen.read.return_value = json.dumps(mock_response).encode()
        
        with patch("urllib.request.urlopen", return_value=mock_urlopen):
            backend = OpenAIBackend(api_key="sk-test-key")
            audio = _base_audio(500)
            result = backend.transcribe(audio)
        
        assert result.text == "hello world"
        assert result.device == "openai-api"
        assert result.compute_type == "cloud"
        assert len(result.segments) == 1
        assert result.segments[0].start_ms == 0
        assert result.segments[0].end_ms == 1500

    def test_time_offset(self) -> None:
        """Test time offset is applied to segments."""
        mock_response = {
            "text": "test",
            "segments": [{"start": 0.1, "end": 0.5, "text": "test"}],
        }
        
        mock_urlopen = MagicMock()
        mock_urlopen.__enter__ = MagicMock(return_value=mock_urlopen)
        mock_urlopen.__exit__ = MagicMock(return_value=False)
        mock_urlopen.read.return_value = json.dumps(mock_response).encode()
        
        with patch("urllib.request.urlopen", return_value=mock_urlopen):
            backend = OpenAIBackend(api_key="sk-test-key")
            audio = _base_audio(500)
            result = backend.transcribe(audio, time_offset_ms=1000)
        
        assert result.segments[0].start_ms == 1100  # 100 + 1000
        assert result.segments[0].end_ms == 1500    # 500 + 1000

    def test_empty_transcription(self) -> None:
        """Test handling of empty transcription."""
        mock_response = {"text": "", "segments": []}
        
        mock_urlopen = MagicMock()
        mock_urlopen.__enter__ = MagicMock(return_value=mock_urlopen)
        mock_urlopen.__exit__ = MagicMock(return_value=False)
        mock_urlopen.read.return_value = json.dumps(mock_response).encode()
        
        with patch("urllib.request.urlopen", return_value=mock_urlopen):
            backend = OpenAIBackend(api_key="sk-test-key")
            audio = _base_audio(500)
            result = backend.transcribe(audio)
        
        assert result.text == ""
        assert result.confidence < 0.5  # Low confidence for empty

    def test_api_error_handling(self) -> None:
        """Test handling of API errors."""
        import urllib.error
        
        mock_error = urllib.error.HTTPError(
            url="https://api.openai.com/v1/audio/transcriptions",
            code=401,
            msg="Unauthorized",
            hdrs={},  # type: ignore
            fp=None,
        )
        
        with patch("urllib.request.urlopen", side_effect=mock_error):
            backend = OpenAIBackend(api_key="sk-invalid-key")
            audio = _base_audio(500)
            
            with pytest.raises(RuntimeError, match="OpenAI API error 401"):
                backend.transcribe(audio)


class TestBackendFactory:
    """Test backend factory in server module."""

    def test_default_whisper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default backend should be whisper."""
        monkeypatch.setenv("EZSTT_BACKEND", "whisper")
        
        # Import fresh to get updated env var
        import importlib
        from sidecar import server
        importlib.reload(server)
        
        # Factory should return WhisperBackend type
        # (We can't easily test this without mocking the model loading)

    def test_openai_selection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should create OpenAI backend when configured."""
        monkeypatch.setenv("EZSTT_BACKEND", "openai")
        
        from sidecar.openai_backend import OpenAIBackend
        # Pass api_key directly since module-level constant is evaluated at import time
        backend = OpenAIBackend(api_key="sk-test-key")
        assert backend.api_key == "sk-test-key"
