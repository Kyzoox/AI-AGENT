"""API REST : même génération que le serveur MCP, pour un LLM ou un agent
qui passe par des appels HTTP (function calling, n8n, LangChain…)."""

from __future__ import annotations

import base64
import binascii
import hmac
import time
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from image_agent.config import Settings
from image_agent.generator import (
    FILE_NAME_PATTERN,
    MODELS,
    GenerationError,
    GenerationResult,
    ImageGenerator,
    InputImage,
)

MAX_INPUT_IMAGE_BYTES = 20 * 1024 * 1024


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, description="Description de l'image à produire.")
    aspect_ratio: str | None = Field(default=None, examples=["16:9"])
    image_size: str | None = Field(default=None, examples=["2K"])
    n: int = Field(default=1, ge=1, description="Nombre d'images.")
    model: str | None = None
    response_format: Literal["url", "b64_json"] = "url"


class InputImagePayload(BaseModel):
    data: str = Field(description="Image encodée en base64.")
    mime_type: Literal["image/png", "image/jpeg", "image/webp", "image/gif", "image/heic", "image/heif"]


class EditRequest(BaseModel):
    prompt: str = Field(min_length=1, description="Modification à appliquer.")
    images: list[InputImagePayload] = Field(min_length=1)
    aspect_ratio: str | None = None
    image_size: str | None = None
    model: str | None = None
    response_format: Literal["url", "b64_json"] = "url"


class ImageData(BaseModel):
    url: str | None = None
    b64_json: str | None = None
    mime_type: str


class ImagesResponse(BaseModel):
    created: int
    model: str
    data: list[ImageData]
    text: str = ""


def create_app(generator: ImageGenerator, api_token: str | None = None) -> FastAPI:
    app = FastAPI(title="image-agent", version="0.1.0", description="Génération d'images via l'API Gemini.")
    bearer = HTTPBearer(auto_error=False)

    def require_token(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        if api_token is None:
            return
        if credentials is None or not hmac.compare_digest(credentials.credentials, api_token):
            raise HTTPException(status_code=401, detail="Jeton d'accès manquant ou invalide.")

    @app.exception_handler(GenerationError)
    async def _generation_error(_: Request, exc: GenerationError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": {"message": exc.message}})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(require_token)])
    def list_models() -> dict[str, object]:
        return {
            "default": generator.default_model,
            "data": [{"id": name, "description": desc} for name, desc in MODELS.items()],
        }

    @app.post("/v1/images/generations", dependencies=[Depends(require_token)])
    async def generate(body: GenerationRequest, request: Request) -> ImagesResponse:
        result = await generator.generate(
            body.prompt, aspect_ratio=body.aspect_ratio, image_size=body.image_size, count=body.n, model=body.model
        )
        return _response(result, body.response_format, request)

    @app.post("/v1/images/edits", dependencies=[Depends(require_token)])
    async def edit(body: EditRequest, request: Request) -> ImagesResponse:
        inputs = [_decode_input(img) for img in body.images]
        result = await generator.generate(
            body.prompt,
            aspect_ratio=body.aspect_ratio,
            image_size=body.image_size,
            model=body.model,
            input_images=inputs,
        )
        return _response(result, body.response_format, request)

    @app.get("/files/{file_name}", dependencies=[Depends(require_token)])
    def get_file(file_name: str) -> FileResponse:
        # Nom strictement contrôlé : aucun accès possible hors du dossier de sortie.
        path = generator.output_dir / file_name
        if not FILE_NAME_PATTERN.match(file_name) or not path.is_file():
            raise HTTPException(status_code=404, detail="Image introuvable.")
        return FileResponse(path)

    return app


def _decode_input(payload: InputImagePayload) -> InputImage:
    try:
        data = base64.b64decode(payload.data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise GenerationError("Image d'entrée : base64 invalide.", 400) from exc
    if len(data) > MAX_INPUT_IMAGE_BYTES:
        raise GenerationError("Image d'entrée trop lourde (> 20 Mo).", 413)
    return InputImage(data=data, mime_type=payload.mime_type)


def _response(result: GenerationResult, response_format: str, request: Request) -> ImagesResponse:
    data = [
        ImageData(
            url=str(request.url_for("get_file", file_name=img.file_name)) if response_format == "url" else None,
            b64_json=base64.b64encode(img.data).decode("ascii") if response_format == "b64_json" else None,
            mime_type=img.mime_type,
        )
        for img in result.images
    ]
    return ImagesResponse(created=int(time.time()), model=result.model, data=data, text=result.text)


def main() -> None:
    import uvicorn
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
    if settings.api_token is None and settings.host not in ("127.0.0.1", "localhost"):
        raise SystemExit("IMAGE_API_TOKEN est obligatoire quand l'API écoute sur une autre adresse que 127.0.0.1.")
    uvicorn.run(create_app(generator, settings.api_token), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
