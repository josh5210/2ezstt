"""Vosk-based keyword spotter wired for short command recognition.

This module loads a constrained grammar from JSON and performs lightweight
keyword spotting on trimmed utterances. It is optimized for the ezstt phase 4
requirements where very short clips should be resolved via KWS before
delegating to Whisper.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

try:  # pragma: no cover - import guard exercised indirectly via tests
    import vosk  # type: ignore
except ImportError:  # pragma: no cover - handled during init
    vosk = None  # type: ignore

if TYPE_CHECKING:
    from vosk import Model as VoskModel  # type: ignore
else:  # lightweight alias so runtime import guard is preserved
    VoskModel = Any


@dataclass
class KWSMatch:
    canonical: str
    surface: str
    confidence: float
    start_ms: int
    end_ms: int
    engine: str = "vosk"


@dataclass
class KWSResult:
    match: Optional[KWSMatch]
    alternatives: List[KWSMatch]
    used_vocabulary: List[str]
    duration_ms: int


@dataclass
class KWSGrammar:
    language: str
    mapping: Dict[str, List[str]]


def load_grammar(path: str) -> KWSGrammar:
    """Load grammar JSON and normalize entries.

    All surface forms and canonical IDs are lower-cased and deduplicated. A
    RuntimeError is raised if the file cannot be read or contains invalid data.
    """

    gram_path = Path(path)
    if not gram_path.is_file():
        raise RuntimeError(f"KWS grammar file not found: {gram_path}")

    try:
        payload = json.loads(gram_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - rare IO
        raise RuntimeError(f"Failed to load KWS grammar from {gram_path}") from exc

    language = str(payload.get("language", "")).strip().lower()
    entries = payload.get("entries", [])
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("KWS grammar JSON must contain a non-empty 'entries' list")

    mapping: Dict[str, List[str]] = {}
    seen_surfaces: Dict[str, str] = {}

    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("Invalid grammar entry; expected an object")

        canonical_raw = entry.get("id")
        surfaces_raw = entry.get("surfaces", [])
        if not canonical_raw or not surfaces_raw:
            raise RuntimeError("Grammar entry must provide 'id' and non-empty 'surfaces'")

        canonical = str(canonical_raw).strip().lower()
        if not canonical:
            raise RuntimeError("Grammar entry canonical ID must be non-empty after normalization")

        if canonical not in mapping:
            mapping[canonical] = []

        for surface in surfaces_raw:
            surface_norm = str(surface).strip().lower()
            if not surface_norm:
                continue
            existing = seen_surfaces.get(surface_norm)
            if existing and existing != canonical:
                raise RuntimeError(
                    f"Surface '{surface_norm}' reused for different canonical entries"
                )
            if surface_norm not in mapping[canonical]:
                mapping[canonical].append(surface_norm)
                seen_surfaces[surface_norm] = canonical

    if not mapping:
        raise RuntimeError("KWS grammar produced an empty mapping")

    return KWSGrammar(language=language or "", mapping=mapping)


class KeywordSpotter:
    """High-level wrapper around Vosk keyword spotting with constrained grammar."""

    _model_cache: Dict[str, VoskModel] = {}
    _model_lock = threading.Lock()

    def __init__(
        self,
        model_dir: Optional[str],
        grammar: Optional[KWSGrammar] = None,
        grammar_path: Optional[str] = None,
        sample_rate: int = 16000,
        conf_min: float = 0.70,
    ) -> None:
        if vosk is None:
            raise RuntimeError("Vosk is not installed; cannot initialize keyword spotter")

        if not model_dir:
            raise RuntimeError("EZSTT_KWS_MODEL_DIR is required for keyword spotter")

        model_path = Path(model_dir)
        if not model_path.is_dir():
            raise RuntimeError(f"KWS model directory not found: {model_path}")

        if grammar is None:
            if not grammar_path:
                raise RuntimeError("KWS grammar must be provided via object or path")
            grammar = load_grammar(grammar_path)

        if sample_rate <= 0:
            raise ValueError("Sample rate must be a positive integer")

        self.sample_rate = int(sample_rate)
        self.conf_min = float(conf_min)
        self.grammar = grammar
        self.model_dir = str(model_path)

        self._model = self._get_or_load_model(self.model_dir)
        self._surface_to_canonical: Dict[str, str] = {}

        used_vocab: List[str] = []
        for canonical, surfaces in self.grammar.mapping.items():
            for surface in surfaces:
                self._surface_to_canonical[surface] = canonical
                used_vocab.append(surface)

        self._used_vocabulary = used_vocab
        self._grammar_json = json.dumps(self._used_vocabulary)

    @classmethod
    def _get_or_load_model(cls, model_dir: str):
        with cls._model_lock:
            model = cls._model_cache.get(model_dir)
            if model is None:
                model = vosk.Model(model_dir)  # type: ignore[attr-defined]
                cls._model_cache[model_dir] = model
        return model

    def spot(self, audio_pcm16: np.ndarray) -> KWSResult:
        """Run keyword spotting for a single utterance."""

        if not isinstance(audio_pcm16, np.ndarray):
            raise ValueError("audio_pcm16 must be a numpy.ndarray")
        if audio_pcm16.dtype != np.int16:
            raise ValueError("audio_pcm16 must be np.int16")
        if audio_pcm16.ndim != 1:
            raise ValueError("audio_pcm16 must be a 1-D mono buffer")

        num_samples = int(audio_pcm16.size)
        duration_ms = int(round((num_samples / float(self.sample_rate)) * 1000.0))

        if num_samples == 0:
            return KWSResult(
                match=None,
                alternatives=[],
                used_vocabulary=list(self._used_vocabulary),
                duration_ms=duration_ms,
            )

        recognizer = vosk.KaldiRecognizer(  # type: ignore[attr-defined]
            self._model, self.sample_rate, self._grammar_json
        )
        recognizer.AcceptWaveform(audio_pcm16.tobytes())
        raw_result = recognizer.Result()

        try:
            result_dict = json.loads(raw_result) if raw_result else {}
        except json.JSONDecodeError:  # pragma: no cover - unexpected from vosk
            result_dict = {}

        text = str(result_dict.get("text", "")).strip()
        if not text:
            return KWSResult(
                match=None,
                alternatives=[],
                used_vocabulary=list(self._used_vocabulary),
                duration_ms=duration_ms,
            )

        tokens_in_text = {token.strip().lower() for token in text.split() if token}

        best_payload = None
        best_conf = -1.0

        for entry in result_dict.get("result", []) or []:
            word = str(entry.get("word", "")).strip().lower()
            if not word or word not in tokens_in_text:
                continue
            if word not in self._surface_to_canonical:
                continue
            conf_raw = entry.get("conf", 0.0)
            try:
                conf = float(conf_raw)
            except (TypeError, ValueError):
                conf = 0.0

            if conf > best_conf:
                start_ms = _seconds_to_ms(entry.get("start"))
                end_ms = _seconds_to_ms(entry.get("end"))
                best_conf = conf
                best_payload = (word, conf, start_ms, end_ms)

        best_match: Optional[KWSMatch] = None
        if best_payload is not None:
            surface, conf, start_ms, end_ms = best_payload
            canonical = self._surface_to_canonical[surface]
            best_match = KWSMatch(
                canonical=canonical,
                surface=surface,
                confidence=conf,
                start_ms=start_ms,
                end_ms=end_ms,
            )

        is_confident = bool(best_match and best_match.confidence >= self.conf_min)
        match_obj = best_match if is_confident else None
        alternatives = [best_match] if best_match else []

        return KWSResult(
            match=match_obj,
            alternatives=alternatives,
            used_vocabulary=list(self._used_vocabulary),
            duration_ms=duration_ms,
        )


def _seconds_to_ms(value: Optional[float]) -> int:
    if value is None:
        return 0
    try:
        return int(round(float(value) * 1000.0))
    except (TypeError, ValueError):
        return 0
