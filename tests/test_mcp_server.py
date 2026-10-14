import base64

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from image_agent.mcp_server import build_server

from conftest import FAKE_JPEG


@pytest.fixture
def server(generator):
    return build_server(generator)


async def test_lists_tools(server):
    names = {tool.name for tool in await server.list_tools()}
    assert names == {"generate_image", "edit_image", "list_image_models"}


async def test_generate_image_returns_path_and_image(server, generator):
    result = await server.call_tool("generate_image", {"prompt": "un phare", "aspect_ratio": "1:1"})

    assert not result.is_error
    text, image = result.content
    assert text.type == "text"
    assert str(generator.output_dir) in text.text
    assert image.type == "image"
    assert image.mime_type == "image/jpeg"
    assert base64.b64decode(image.data) == FAKE_JPEG


async def test_generate_image_without_inline_images(server):
    result = await server.call_tool("generate_image", {"prompt": "un phare", "count": 2, "return_images": False})
    assert [block.type for block in result.content] == ["text"]
    assert result.content[0].text.startswith("2 image(s)")


async def test_edit_image_reads_local_file(server, fake_client, tmp_path):
    source = tmp_path / "chat.png"
    source.write_bytes(b"png-bytes")

    result = await server.call_tool("edit_image", {"prompt": "ajoute un chapeau", "image_paths": [str(source)]})

    assert not result.is_error
    sent = fake_client.interactions.calls[0]["input"][1]
    assert sent["mime_type"] == "image/png"
    assert base64.b64decode(sent["data"]) == b"png-bytes"


async def test_edit_image_missing_file(server, tmp_path):
    with pytest.raises(ToolError, match="introuvable"):
        await server.call_tool("edit_image", {"prompt": "x", "image_paths": [str(tmp_path / "absent.png")]})


async def test_generation_error_becomes_tool_error(server):
    with pytest.raises(ToolError, match="aspect_ratio"):
        await server.call_tool("generate_image", {"prompt": "x", "aspect_ratio": "7:3"})


async def test_list_models(server):
    result = await server.call_tool("list_image_models", {})
    assert "gemini-nano-banana-2.1 (défaut)" in result.content[0].text
