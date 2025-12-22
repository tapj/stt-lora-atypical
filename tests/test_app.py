import io
import json
import sys
import types

import numpy as np
import pytest
from fastapi import UploadFile
from starlette.requests import Request

# Provide a lightweight stub for python-dotenv to avoid dependency downloads in tests.
if "dotenv" not in sys.modules:
    sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None)

from app.app import (
    DEFAULT_ADAPTER_DIR,
    api_download_model,
    api_transcribe,
    get_service,
    guess_audio_format,
    index,
)


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

    def fake_get_service(adapter_dir, config_path, device):
        _ = (adapter_dir, config_path, device)
        return dummy

    def fake_loader(data: bytes, fmt=None):
        _ = (data, fmt)
        return np.zeros(16000, dtype=np.float32)

    monkeypatch.setattr("app.app.get_service", fake_get_service)
    monkeypatch.setattr("app.app.load_wav_bytes_to_16k_mono", fake_loader)
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


def test_api_download_model(monkeypatch):
    called = {}

    def fake_snapshot_download(repo_id, cache_dir=None, token=None):
        called["repo_id"] = repo_id
        called["cache_dir"] = cache_dir
        called["token"] = token
        return "/tmp/fake"

    monkeypatch.setattr("app.app.snapshot_download", fake_snapshot_download)

    import asyncio

    resp = asyncio.get_event_loop().run_until_complete(api_download_model(model_id="openai/whisper-small"))
    assert resp.status_code == 200
    data = json.loads(resp.body.decode())
    assert data["model_id"] == "openai/whisper-small"
    assert called["repo_id"] == "openai/whisper-small"
