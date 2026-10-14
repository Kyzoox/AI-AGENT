"""Faux client Gemini : reproduit la forme des réponses de l'API Interactions
sans appel réseau ni clé."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from typing import Any

import pytest

from image_agent.generator import ImageGenerator

FAKE_JPEG = b"\xff\xd8\xff\xe0fake-jpeg-bytes"


def interaction_with_image(data: bytes = FAKE_JPEG, text: str | None = None) -> SimpleNamespace:
    content = [SimpleNamespace(type="image", data=base64.b64encode(data).decode(), mime_type="image/jpeg")]
    if text:
        content.insert(0, SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        steps=[
            SimpleNamespace(type="user_input", content=[]),
            SimpleNamespace(type="model_output", content=content),
        ],
        output_image=None,
    )


class FakeInteractions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response: Any = interaction_with_image()
        self.error: Exception | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self) -> None:
        self.interactions = FakeInteractions()
        self.aio = SimpleNamespace(interactions=self.interactions)


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def generator(fake_client: FakeClient, tmp_path) -> ImageGenerator:
    return ImageGenerator(fake_client, output_dir=tmp_path / "out", default_model="gemini-nano-banana-2.1", max_images=4)
