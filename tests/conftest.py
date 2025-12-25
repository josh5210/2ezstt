from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load env vars from .env
load_dotenv()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sidecar.whisper_backend import WhisperBackend


@pytest.fixture(autouse=True)
def reset_whisper_backend_singleton():
    WhisperBackend._reset_shared_model_for_test()
    yield
    WhisperBackend._reset_shared_model_for_test()


@pytest.fixture(autouse=True)
def ensure_local_model_dir(tmp_path, monkeypatch):
    existing = os.getenv("EZSTT_MODEL_DIR")
    if existing:
        path = Path(existing).expanduser()
        if path.is_dir() and (path / "model.bin").exists():
            yield
            return

    model_dir = tmp_path / "whisper-model"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"")
    monkeypatch.setenv("EZSTT_MODEL_DIR", str(model_dir))
    yield
