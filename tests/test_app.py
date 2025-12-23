import io
import json
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi import UploadFile
from starlette.requests import Request

# Provide a lightweight stub for python-dotenv to avoid dependency downloads in tests.
if "dotenv" not in sys.modules:
    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None)

from app.app import DEFAULT_ADAPTER_DIR, api_transcribe, get_service, guess_audio_format, index


class DummyService:
    def __init__(self):
        self.called_with = None

    def transcribe(self, audio_16k, language, beam, temperature):
        self.called_with = (audio_16k, language, beam, temperature)
        return "dummy text"


@pytest.fixture(autouse=True)
def clear_service():
    # Reset singleton between tests
    import app.app as app_module

    app_module.SERVICE = None
    yield
    app_module.SERVICE = None


@pytest.fixture
def dummy_service(monkeypatch):
    dummy = DummyService()
    normalize_calls = []

    def fake_get_service(adapter_dir, config_path, device):
        _ = (adapter_dir, config_path, device)
        return dummy

    def make_wav_bytes():
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(np.zeros(1600, dtype=np.int16).tobytes())
        return buf.getvalue()

    def fake_normalize(upload_bytes: bytes, filename: str):
        normalize_calls.append(filename)
        _ = upload_bytes
        return make_wav_bytes()

    def fake_loader_bytes(data: bytes):
        _ = data
        return np.zeros(16000, dtype=np.float32)

    def fake_loader_bytes(data: bytes):
        _ = data
        return np.zeros(16000, dtype=np.float32)

    ffmpeg_calls = []

    def fake_run(cmd, check, env=None, stdout=None, stderr=None):
        _ = (check, env, stdout, stderr)
        ffmpeg_calls.append(cmd)
        output_path = Path(cmd[-1])
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(np.zeros(1600, dtype=np.int16).tobytes())
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("app.app.get_service", fake_get_service)
    monkeypatch.setattr("app.app.normalize_to_wav16k_mono", fake_normalize)
    monkeypatch.setattr("app.app.load_wav_bytes_to_float32_mono_16k", fake_loader_bytes)
    dummy.normalize_calls = normalize_calls
    return dummy


def test_index_page():
    scope = {"type": "http", "method": "GET", "headers": [], "path": "/"}
    req = Request(scope)
    resp = index(req)
    assert resp.template.name == "index.html"
    assert resp.context["adapter_dir_default"] == DEFAULT_ADAPTER_DIR


def test_api_transcribe_success(dummy_service):
    upload = UploadFile(filename="audio.wav", file=io.BytesIO(b"fakewav"))
    import asyncio

    resp = asyncio.get_event_loop().run_until_complete(
        api_transcribe(
            audio=upload,
            adapter_dir="adapter",
            config_path="config.yaml",
            device=None,
            language=None,
            beam_size=5,
            temperature=0.0,
        )
    )
    assert resp.status_code == 200
    parsed = json.loads(resp.body.decode())
    assert parsed["text"] == "dummy text"
    assert "timestamp" in parsed


def test_api_transcribe_converts_non_wav(dummy_service):
    upload = UploadFile(filename="audio.webm", file=io.BytesIO(b"fakewebm"))
    import asyncio

    resp = asyncio.get_event_loop().run_until_complete(
        api_transcribe(
            audio=upload,
            adapter_dir="adapter",
            config_path="config.yaml",
            device=None,
            language=None,
            beam_size=5,
            temperature=0.0,
        )
    )
    assert resp.status_code == 200
    parsed = json.loads(resp.body.decode())
    assert parsed["text"] == "dummy text"
    assert "timestamp" in parsed
    assert dummy_service.normalize_calls, "normalization should be invoked for non-wav uploads"


def test_api_transcribe_rejects_non_upload():
    import asyncio

    resp = asyncio.get_event_loop().run_until_complete(
        api_transcribe(
            audio=Path("clip.wav"),
            adapter_dir="adapter",
            config_path="config.yaml",
            device=None,
            language=None,
            beam_size=5,
            temperature=0.0,
        )
    )
    assert resp.status_code == 400
    parsed = json.loads(resp.body.decode())
    assert "Unsupported audio type" in parsed["error"]


def test_get_service_singleton(monkeypatch):
    dummy = DummyService()

    def fake_init(cfg_path, adapter_dir, device_str=None):
        _ = (cfg_path, adapter_dir, device_str)
        return dummy

    monkeypatch.setattr("app.app.STTService", fake_init)
    svc1 = get_service("dir1", "cfg", None)
    svc2 = get_service("dir2", "cfg2", "cpu")
    assert svc1 is svc2


def test_guess_audio_format_prefers_content_type():
    assert guess_audio_format("clip.webm", "audio/webm;codecs=opus") == "webm"
    assert guess_audio_format("clip.wav", None) == "wav"
    assert guess_audio_format("clip", None) is None
