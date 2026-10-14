import base64

import pytest
from fastapi.testclient import TestClient

from image_agent.api import create_app

from conftest import FAKE_JPEG


@pytest.fixture
def client(generator):
    return TestClient(create_app(generator))


def test_generation_returns_url_then_file(client):
    response = client.post("/v1/images/generations", json={"prompt": "un phare", "aspect_ratio": "16:9"})

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "gemini-nano-banana-2.1"
    assert len(body["data"]) == 1
    url = body["data"][0]["url"]
    assert body["data"][0]["b64_json"] is None

    image = client.get(url)
    assert image.status_code == 200
    assert image.content == FAKE_JPEG


def test_generation_b64(client):
    response = client.post("/v1/images/generations", json={"prompt": "un phare", "n": 2, "response_format": "b64_json"})
    data = response.json()["data"]
    assert len(data) == 2
    assert base64.b64decode(data[0]["b64_json"]) == FAKE_JPEG
    assert data[0]["url"] is None


def test_edit(client, fake_client):
    payload = {
        "prompt": "ajoute un chapeau",
        "images": [{"data": base64.b64encode(b"png").decode(), "mime_type": "image/png"}],
    }
    response = client.post("/v1/images/edits", json=payload)
    assert response.status_code == 200
    assert fake_client.interactions.calls[0]["input"][1]["mime_type"] == "image/png"


def test_edit_rejects_bad_base64(client):
    payload = {"prompt": "x", "images": [{"data": "pas du base64!", "mime_type": "image/png"}]}
    response = client.post("/v1/images/edits", json=payload)
    assert response.status_code == 400
    assert "base64" in response.json()["error"]["message"]


def test_generation_error_is_json(client):
    response = client.post("/v1/images/generations", json={"prompt": "x", "aspect_ratio": "7:3"})
    assert response.status_code == 400
    assert "aspect_ratio" in response.json()["error"]["message"]


def test_files_route_rejects_unknown_names(client):
    assert client.get("/files/..%2Fpyproject.toml").status_code == 404
    assert client.get("/files/random.jpg").status_code == 404


def test_models(client):
    body = client.get("/v1/models").json()
    assert body["default"] == "gemini-nano-banana-2.1"
    assert any(m["id"] == "gemini-3-pro-image" for m in body["data"])


def test_token_is_required_when_configured(generator):
    client = TestClient(create_app(generator, api_token="secret"))

    assert client.get("/health").status_code == 200
    assert client.post("/v1/images/generations", json={"prompt": "x"}).status_code == 401
    assert (
        client.post("/v1/images/generations", json={"prompt": "x"}, headers={"Authorization": "Bearer nope"}).status_code
        == 401
    )
    ok = client.post("/v1/images/generations", json={"prompt": "x"}, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
