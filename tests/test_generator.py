from types import SimpleNamespace

import pytest

from image_agent.generator import GenerationError, InputImage

from conftest import FAKE_JPEG, interaction_with_image


async def test_generate_saves_image_and_builds_request(generator, fake_client):
    result = await generator.generate("un chat astronaute", aspect_ratio="16:9", image_size="2K")

    assert result.model == "gemini-nano-banana-2.1"
    assert len(result.images) == 1
    image = result.images[0]
    assert image.path.read_bytes() == FAKE_JPEG
    assert image.path.suffix == ".jpg"
    assert image.mime_type == "image/jpeg"

    call = fake_client.interactions.calls[0]
    assert call == {
        "model": "gemini-nano-banana-2.1",
        "input": "un chat astronaute",
        "response_format": {"type": "image", "aspect_ratio": "16:9", "image_size": "2K"},
        "store": False,
    }


async def test_count_sends_parallel_requests(generator, fake_client):
    result = await generator.generate("un phare", count=3, model="gemini-3-pro-image")

    assert len(fake_client.interactions.calls) == 3
    assert all(c["model"] == "gemini-3-pro-image" for c in fake_client.interactions.calls)
    assert len({img.path for img in result.images}) == 3


async def test_edit_sends_text_and_images(generator, fake_client):
    await generator.generate("ajoute un chapeau", input_images=[InputImage(data=b"png-bytes", mime_type="image/png")])

    user_input = fake_client.interactions.calls[0]["input"]
    assert user_input[0] == {"type": "text", "text": "ajoute un chapeau"}
    assert user_input[1] == {"type": "image", "data": "cG5nLWJ5dGVz", "mime_type": "image/png"}


async def test_model_text_is_returned(generator, fake_client):
    fake_client.interactions.response = interaction_with_image(text="Voici votre image.")
    result = await generator.generate("un phare")
    assert result.text == "Voici votre image."


async def test_falls_back_to_output_image(generator, fake_client):
    fake_client.interactions.response = SimpleNamespace(
        steps=None, output_image=SimpleNamespace(data="aGVsbG8=", mime_type="image/png")
    )
    result = await generator.generate("un phare")
    assert result.images[0].path.suffix == ".png"
    assert result.images[0].data == b"hello"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prompt": "   "}, "vide"),
        ({"prompt": "x", "count": 0}, "count"),
        ({"prompt": "x", "count": 5}, "count"),
        ({"prompt": "x", "aspect_ratio": "7:3"}, "aspect_ratio"),
        ({"prompt": "x", "image_size": "8K"}, "image_size"),
    ],
)
async def test_validation(generator, fake_client, kwargs, message):
    with pytest.raises(GenerationError, match=message) as info:
        await generator.generate(**kwargs)
    assert info.value.status_code == 400
    assert fake_client.interactions.calls == []


async def test_no_image_is_reported(generator, fake_client):
    fake_client.interactions.response = SimpleNamespace(
        steps=[SimpleNamespace(type="model_output", content=[SimpleNamespace(type="text", text="Je ne peux pas.")])],
        output_image=None,
    )
    with pytest.raises(GenerationError, match="Je ne peux pas") as info:
        await generator.generate("x")
    assert info.value.status_code == 422


async def test_api_error_is_mapped(generator, fake_client):
    error = Exception("Error code: 429")
    error.status_code = 429
    error.body = [{"error": {"code": 429, "message": "Quota exceeded."}}]
    fake_client.interactions.error = error

    with pytest.raises(GenerationError, match="Quota exceeded") as info:
        await generator.generate("x")
    assert info.value.status_code == 429


async def test_unexpected_error_propagates(generator, fake_client):
    fake_client.interactions.error = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        await generator.generate("x")
