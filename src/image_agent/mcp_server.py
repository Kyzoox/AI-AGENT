"""Serveur MCP : expose la génération d'images comme outils pour un LLM
(Claude Code, Claude Desktop, Cursor…)."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from image_agent.config import Settings
from image_agent.generator import (
    ASPECT_RATIOS,
    IMAGE_SIZES,
    MODELS,
    GenerationError,
    GenerationResult,
    ImageGenerator,
    InputImage,
)

MAX_INPUT_IMAGE_BYTES = 20 * 1024 * 1024
INPUT_MIME_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif", "image/heic", "image/heif"}

AspectRatio = Annotated[
    str | None, Field(description=f"Format de l'image. Valeurs : {', '.join(ASPECT_RATIOS)}. Défaut : choisi par le modèle.")
]
ImageSize = Annotated[str | None, Field(description=f"Résolution : {', '.join(IMAGE_SIZES)}. Défaut : 1K.")]
Model = Annotated[str | None, Field(description="Modèle Gemini (voir list_image_models). Défaut : celui de la configuration.")]


def build_server(generator: ImageGenerator) -> MCPServer:
    server = MCPServer(
        name="image-agent",
        instructions=(
            "Génère et modifie des images avec les modèles d'image Gemini. "
            "Chaque image est enregistrée sur disque ; son chemin est renvoyé pour pouvoir la réutiliser avec edit_image."
        ),
    )

    # Contenu non structuré : texte + images, que le client affiche au LLM.
    @server.tool(structured_output=False)
    async def generate_image(
        prompt: Annotated[str, Field(description="Description détaillée de l'image à produire.")],
        aspect_ratio: AspectRatio = None,
        image_size: ImageSize = None,
        count: Annotated[int, Field(ge=1, description="Nombre d'images (variantes) à générer.")] = 1,
        model: Model = None,
        return_images: Annotated[bool, Field(description="Renvoyer aussi les images dans la réponse pour les voir.")] = True,
    ) -> list[Any]:
        """Génère une ou plusieurs images à partir d'un texte."""
        result = await _run(generator, prompt, aspect_ratio, image_size, count, model, ())
        return _to_content(result, return_images)

    @server.tool(structured_output=False)
    async def edit_image(
        prompt: Annotated[str, Field(description="Modification à appliquer, ou consigne de composition.")],
        image_paths: Annotated[
            list[str], Field(min_length=1, description="Chemins locaux des images de référence (ex. images générées précédemment).")
        ],
        aspect_ratio: AspectRatio = None,
        image_size: ImageSize = None,
        model: Model = None,
        return_images: Annotated[bool, Field(description="Renvoyer aussi l'image dans la réponse pour la voir.")] = True,
    ) -> list[Any]:
        """Modifie une image existante, ou en combine plusieurs, selon une consigne texte."""
        inputs = [_read_input_image(path) for path in image_paths]
        result = await _run(generator, prompt, aspect_ratio, image_size, 1, model, inputs)
        return _to_content(result, return_images)

    @server.tool()
    def list_image_models() -> str:
        """Liste les modèles d'image disponibles et le modèle par défaut."""
        lines = [f"- {name}{' (défaut)' if name == generator.default_model else ''} : {desc}" for name, desc in MODELS.items()]
        return "\n".join(lines)

    return server


async def _run(
    generator: ImageGenerator,
    prompt: str,
    aspect_ratio: str | None,
    image_size: str | None,
    count: int,
    model: str | None,
    inputs: list[InputImage] | tuple[()],
) -> GenerationResult:
    try:
        return await generator.generate(
            prompt, aspect_ratio=aspect_ratio, image_size=image_size, count=count, model=model, input_images=inputs
        )
    except GenerationError as exc:
        raise ToolError(exc.message) from exc


def _read_input_image(raw_path: str) -> InputImage:
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise ToolError(f"Image introuvable : {raw_path}")
    mime_type = mimetypes.guess_type(path.name)[0]
    if mime_type not in INPUT_MIME_TYPES:
        raise ToolError(f"Format non pris en charge pour {raw_path} (attendu : PNG, JPEG, WebP, GIF, HEIC).")
    if path.stat().st_size > MAX_INPUT_IMAGE_BYTES:
        raise ToolError(f"Image trop lourde (> 20 Mo) : {raw_path}")
    return InputImage(data=path.read_bytes(), mime_type=mime_type)


def _to_content(result: GenerationResult, return_images: bool) -> list[Any]:
    summary = [f"{len(result.images)} image(s) générée(s) avec {result.model} :"]
    summary += [f"- {img.path}" for img in result.images]
    if result.text:
        summary.append(f"Commentaire du modèle : {result.text}")
    content: list[Any] = ["\n".join(summary)]
    if return_images:
        content += [Image(data=img.data, format=img.mime_type.split("/")[1]) for img in result.images]
    return content


def main() -> None:
    from google import genai

    settings = Settings.from_env()
    if not settings.gemini_api_key:
        raise SystemExit("GEMINI_API_KEY n'est pas défini (clé à créer sur https://aistudio.google.com/apikey).")
    generator = ImageGenerator(
        genai.Client(api_key=settings.gemini_api_key),
        output_dir=settings.output_dir,
        default_model=settings.default_model,
        max_images=settings.max_images,
    )
    build_server(generator).run("stdio")


if __name__ == "__main__":
    main()
