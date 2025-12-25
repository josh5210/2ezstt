from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PartialEvent:
    session_id: str
    speaker_id: str
    utterance_id: str
    revision: int
    text: str
    t0_ms: int
    t1_ms: int
    confidence: float
    latency_ms: int
    low_confidence: bool = False
    source: str = "whisper"


__all__ = ["PartialEvent"]
