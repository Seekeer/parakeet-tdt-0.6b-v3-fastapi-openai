from __future__ import annotations

import io
import wave
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import TEST_API_KEY
from parakeet_service import config
from parakeet_service.main import create_app


class _Worker:
    async def submit_many(self, wavs, model_name):
        return [SimpleNamespace(text="hello world", tokens=[], timestamps=[]) for _ in wavs]


def _wav_bytes(seconds: float = 0.5) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(config.TARGET_SR)
        wav_file.writeframes(np.zeros(int(seconds * config.TARGET_SR), dtype=np.int16).tobytes())
    return buffer.getvalue()


@pytest.fixture
def client():
    # TestClient is not used as a context manager, so the model-loading
    # lifespan never runs; the state below stands in for it.
    app = create_app()
    app.state.ready = True
    app.state.worker = _Worker()
    app.state.audio_pool = ThreadPoolExecutor(max_workers=1)
    yield TestClient(app)
    app.state.audio_pool.shutdown()


def _transcribe(client, headers=None):
    return client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", _wav_bytes(), "audio/wav")},
        headers=headers or {},
    )


def test_missing_key_is_rejected(client):
    response = _transcribe(client)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "header",
    [f"Bearer {TEST_API_KEY}x", f"Basic {TEST_API_KEY}", TEST_API_KEY, "Bearer "],
)
def test_wrong_key_or_scheme_is_rejected(client, header):
    assert _transcribe(client, {"Authorization": header}).status_code == 401


def test_valid_key_transcribes(client):
    response = _transcribe(client, {"Authorization": f"Bearer {TEST_API_KEY}"})
    assert response.status_code == 200
    assert response.json() == {"text": "hello world"}


def test_batch_endpoint_requires_key(client):
    files = [("files", ("a.wav", _wav_bytes(), "audio/wav"))]
    assert client.post("/v1/audio/transcriptions/batch", files=files).status_code == 401
    response = client.post(
        "/v1/audio/transcriptions/batch",
        files=files,
        headers={"Authorization": f"Bearer {TEST_API_KEY}"},
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["text"] == "hello world"


def test_disabled_auth_allows_anonymous_requests(client, monkeypatch):
    monkeypatch.setattr(config, "AUTH_DISABLED", True)
    assert _transcribe(client).status_code == 200


def test_health_is_public_and_reflects_readiness(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ready"] is True

    client.app.state.ready = False
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "starting"


def test_healthz_alias_is_gone(client):
    headers = {"Authorization": f"Bearer {TEST_API_KEY}"}
    assert client.get("/healthz", headers=headers).status_code == 404


def test_unknown_paths_require_key(client):
    assert client.get("/healthz").status_code == 401


def test_docs_are_public(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_public_paths_work_behind_root_path(client):
    proxied = TestClient(client.app, root_path="/asr")
    assert proxied.get("/asr/health").status_code == 200
    assert proxied.get("/asr/v1/audio/transcriptions").status_code == 401


def test_unauthenticated_body_is_never_parsed(client, monkeypatch):
    from starlette.requests import Request

    parsed = []
    original = Request._get_form

    async def spy(self, *args, **kwargs):
        parsed.append(self.url.path)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(Request, "_get_form", spy)
    assert _transcribe(client).status_code == 401
    assert _transcribe(client, {"Authorization": "Bearer wrong-key-0123456789"}).status_code == 401
    assert parsed == []

    assert _transcribe(client, {"Authorization": f"Bearer {TEST_API_KEY}"}).status_code == 200
    assert parsed == ["/v1/audio/transcriptions"]
