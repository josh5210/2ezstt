import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sidecar.kws_vosk import KWSGrammar, KeywordSpotter, load_grammar


@pytest.fixture(autouse=True)
def reset_model_cache():
    KeywordSpotter._model_cache.clear()
    yield
    KeywordSpotter._model_cache.clear()


@pytest.fixture
def fake_vosk(monkeypatch):
    namespace = types.SimpleNamespace()

    class FakeModel:
        def __init__(self, model_dir: str) -> None:
            self.model_dir = model_dir

    class FakeRecognizer:
        def __init__(self, model, sample_rate, grammar_json) -> None:
            self.model = model
            self.sample_rate = sample_rate
            self.grammar_json = grammar_json

        def AcceptWaveform(self, audio_bytes):
            self.audio_bytes = audio_bytes
            return True

        def Result(self):
            return json.dumps(namespace.result_payload)

    namespace.Model = FakeModel
    namespace.KaldiRecognizer = FakeRecognizer
    namespace.result_payload = {}

    def set_result(payload):
        namespace.result_payload = payload

    namespace.set_result = set_result
    monkeypatch.setattr("sidecar.kws_vosk.vosk", namespace)
    return namespace


def _basic_grammar() -> KWSGrammar:
    return KWSGrammar(
        language="en",
        mapping={
            "yes": ["yes", "yeah", "yep"],
            "no": ["no"],
            "ok": ["ok", "okay"],
        },
    )


def test_load_grammar_file(tmp_path):
    grammar_payload = {
        "language": "EN",
        "entries": [
            {"id": "Yes", "surfaces": ["Yes", "Yeah", "yeah"]},
            {"id": "No", "surfaces": ["no", "No"]},
        ],
    }
    gram_path = tmp_path / "grammar.json"
    gram_path.write_text(json.dumps(grammar_payload), encoding="utf-8")

    grammar = load_grammar(str(gram_path))

    assert grammar.language == "en"
    assert grammar.mapping["yes"] == ["yes", "yeah"]
    assert grammar.mapping["no"] == ["no"]


def test_valid_yes_single_token(tmp_path, fake_vosk):
    fake_vosk.set_result({
        "text": "yeah",
        "result": [{"word": "yeah", "conf": 0.82, "start": 0.05, "end": 0.2}],
    })

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    spotter = KeywordSpotter(model_dir=str(model_dir), grammar=_basic_grammar())

    audio = np.zeros(1600, dtype=np.int16)
    result = spotter.spot(audio)

    assert result.match is not None
    assert result.match.canonical == "yes"
    assert result.match.surface == "yeah"
    assert result.match.confidence == pytest.approx(0.82)
    assert result.duration_ms == 100
    assert result.alternatives


def test_no_match_below_threshold(tmp_path, fake_vosk):
    fake_vosk.set_result({"text": "", "result": []})

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    spotter = KeywordSpotter(model_dir=str(model_dir), grammar=_basic_grammar())

    audio = np.zeros(800, dtype=np.int16)
    result = spotter.spot(audio)

    assert result.match is None
    assert result.alternatives == []


def test_maps_to_canonical(tmp_path, fake_vosk):
    fake_vosk.set_result({
        "text": "okay",
        "result": [{"word": "okay", "conf": 0.88, "start": 0.0, "end": 0.1}],
    })

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    spotter = KeywordSpotter(model_dir=str(model_dir), grammar=_basic_grammar())

    audio = np.zeros(400, dtype=np.int16)
    result = spotter.spot(audio)

    assert result.match is not None
    assert result.match.canonical == "ok"


def test_invalid_audio_raises(tmp_path, fake_vosk):
    fake_vosk.set_result({"text": "yeah", "result": []})

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    spotter = KeywordSpotter(model_dir=str(model_dir), grammar=_basic_grammar())

    with pytest.raises(ValueError):
        spotter.spot(np.zeros((10, 2), dtype=np.int16))

    with pytest.raises(ValueError):
        spotter.spot(np.zeros(10, dtype=np.float32))


def test_missing_files_raise(tmp_path, fake_vosk):
    with pytest.raises(RuntimeError):
        load_grammar(str(tmp_path / "missing.json"))

    with pytest.raises(RuntimeError):
        KeywordSpotter(model_dir=str(tmp_path / "missing_model"), grammar=_basic_grammar())


def test_duration_reported(tmp_path, fake_vosk):
    fake_vosk.set_result({
        "text": "yes",
        "result": [{"word": "yes", "conf": 0.9}],
    })

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    spotter = KeywordSpotter(model_dir=str(model_dir), grammar=_basic_grammar())

    audio = np.zeros(3200, dtype=np.int16)
    result = spotter.spot(audio)

    assert result.duration_ms == 200


@pytest.mark.skipif(
    os.getenv("EZSTT_RUN_SLOW_TESTS", "0") != "1",
    reason="slow integration test disabled",
)
def test_integration_slow(tmp_path):
    model_dir = os.getenv("EZSTT_KWS_MODEL_DIR")
    grammar_path = os.getenv("EZSTT_KWS_GRAMMAR_FILE")
    if not model_dir or not grammar_path:
        pytest.skip("model or grammar path unavailable")

    audio_path = Path(os.getenv("EZSTT_KWS_INTEGRATION_AUDIO", ""))
    if not audio_path.is_file():
        pytest.skip("integration audio clip missing")

    try:
        import soundfile as sf  # type: ignore
    except ImportError:  # pragma: no cover - depends on optional deps
        pytest.skip("soundfile not installed")

    audio, sr = sf.read(str(audio_path), dtype="int16")
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = np.asarray(audio, dtype=np.int16)
    if sr != 16000:
        pytest.skip("audio not resampled to 16k")

    grammar = load_grammar(grammar_path)
    spotter = KeywordSpotter(model_dir=model_dir, grammar=grammar)
    result = spotter.spot(audio)

    assert result.duration_ms > 0
