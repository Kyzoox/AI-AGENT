"""Cœur de la génération d'images : appelle l'API Gemini (Interactions) et
enregistre les images produites sur disque."""

from __future__ import annotations

import asyncio
import base64
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Valeurs acceptées par `response_format` de l'API Interactions (SDK google-genai 2.30).
ASPECT_RATIOS = (
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
    "9:16", "16:9", "21:9", "1:4", "4:1", "1:8", "8:1",
)
IMAGE_SIZES = ("512", "1K", "2K", "4K")

MODELS: dict[str, str] = {
    "gemini-nano-banana-2.1": "Nano Banana 2.1 — modèle recommandé : qualité, rendu du texte, cohérence en édition.",
    "gemini-3.1-flash-image": "Nano Banana 2 — génération rapide à grande échelle.",
    "gemini-3.1-flash-lite-image": "Nano Banana 2 Lite — le plus rapide et le moins cher.",
    "gemini-3-pro-image": "Nano Banana Pro — qualité studio, jusqu'à 4K, texte précis.",
}

_EXTENSIONS = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

FILE_NAME_PATTERN = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{12}-[0-9]+\.(jpg|png|webp|gif)$")


class GenerationError(Exception):
    """Erreur de génération présentable à l'appelant (LLM ou client HTTP)."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class InputImage:
    data: bytes
    mime_type: str


@dataclass(frozen=True)
class GeneratedImage:
    path: Path
    mime_type: str
    data: bytes = field(repr=False)

    @property
    def file_name(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class GenerationResult:
    model: str
    images: list[GeneratedImage]
    text: str


class ImageGenerator:
    """Génère des images avec un client `google.genai.Client` (ou compatible)."""

    def __init__(
        self,
        client: Any,
        output_dir: Path,
        default_model: str,
        max_images: int = 4,
    ) -> None:
        self._client = client
        self.output_dir = output_dir
        self.default_model = default_model
        self.max_images = max_images

    async def generate(
        self,
        prompt: str,
        *,
        aspect_ratio: str | None = None,
        image_size: str | None = None,
        count: int = 1,
        model: str | None = None,
        input_images: Sequence[InputImage] = (),
    ) -> GenerationResult:
        prompt = prompt.strip()
        if not prompt:
            raise GenerationError("Le prompt est vide.", 400)
        if not 1 <= count <= self.max_images:
            raise GenerationError(f"count doit être entre 1 et {self.max_images}.", 400)
        if aspect_ratio is not None and aspect_ratio not in ASPECT_RATIOS:
            raise GenerationError(
                f"aspect_ratio invalide : {aspect_ratio!r}. Valeurs possibles : {', '.join(ASPECT_RATIOS)}.", 400
            )
        if image_size is not None and image_size not in IMAGE_SIZES:
            raise GenerationError(
                f"image_size invalide : {image_size!r}. Valeurs possibles : {', '.join(IMAGE_SIZES)}.", 400
            )

        model = model or self.default_model
        request = _build_request(model, prompt, aspect_ratio, image_size, input_images)

        # Une requête renvoie une image : on parallélise pour en obtenir plusieurs.
        interactions = await asyncio.gather(*(self._create(request) for _ in range(count)))

        images: list[GeneratedImage] = []
        texts: list[str] = []
        batch = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}"
        for interaction in interactions:
            blobs, text = _extract_output(interaction)
            if text:
                texts.append(text)
            for data, mime_type in blobs:
                images.append(self._save(batch, len(images), data, mime_type))

        if not images:
            reason = " ".join(texts) or "aucune image renvoyée (contenu probablement bloqué par les filtres de sécurité)."
            raise GenerationError(f"Le modèle n'a pas produit d'image : {reason}", 422)

        return GenerationResult(model=model, images=images, text="\n".join(texts))

    async def _create(self, request: dict[str, Any]) -> Any:
        try:
            return await self._client.aio.interactions.create(**request)
        except Exception as exc:  # erreurs HTTP du SDK : elles exposent `status_code`
            status = getattr(exc, "status_code", None)
            if status is None:
                raise
            raise GenerationError(f"Erreur de l'API Gemini ({status}) : {_short_error(exc)}", _map_status(status)) from exc

    def _save(self, batch: str, index: int, data: bytes, mime_type: str) -> GeneratedImage:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{batch}-{index}{_EXTENSIONS.get(mime_type, '.jpg')}"
        path.write_bytes(data)
        return GeneratedImage(path=path, mime_type=mime_type, data=data)


def _build_request(
    model: str,
    prompt: str,
    aspect_ratio: str | None,
    image_size: str | None,
    input_images: Sequence[InputImage],
) -> dict[str, Any]:
    response_format: dict[str, Any] = {"type": "image"}
    if aspect_ratio:
        response_format["aspect_ratio"] = aspect_ratio
    if image_size:
        response_format["image_size"] = image_size

    user_input: Any = prompt
    if input_images:
        user_input = [{"type": "text", "text": prompt}] + [
            {"type": "image", "data": base64.b64encode(img.data).decode("ascii"), "mime_type": img.mime_type}
            for img in input_images
        ]

    return {
        "model": model,
        "input": user_input,
        "response_format": response_format,
        # Pas besoin d'historique côté serveur : chaque génération est indépendante.
        "store": False,
    }


def _extract_output(interaction: Any) -> tuple[list[tuple[bytes, str]], str]:
    images: list[tuple[bytes, str]] = []
    texts: list[str] = []
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) != "model_output":
            continue
        for block in getattr(step, "content", None) or []:
            block_type = getattr(block, "type", None)
            if block_type == "image" and getattr(block, "data", None):
                images.append((base64.b64decode(block.data), block.mime_type or "image/jpeg"))
            elif block_type == "text" and getattr(block, "text", None):
                texts.append(block.text)

    output_image = getattr(interaction, "output_image", None)
    if not images and output_image is not None and getattr(output_image, "data", None):
        images.append((base64.b64decode(output_image.data), output_image.mime_type or "image/jpeg"))

    return images, "\n".join(texts).strip()


def _map_status(status: int) -> int:
    # 400/403/404/429 sont des erreurs de l'appelant ou de quota : on les relaie telles quelles.
    return status if status in (400, 401, 403, 404, 429) else 502


def _short_error(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, list) and body:
        body = body[0]
    if isinstance(body, dict):
        message = (body.get("error") or {}).get("message")
        if isinstance(message, str):
            return message
    return str(exc)[:300]
