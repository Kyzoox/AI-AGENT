"""Configuration lue depuis les variables d'environnement."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "gemini-nano-banana-2.1"


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    default_model: str
    output_dir: Path
    max_images: int
    api_token: str | None
    host: str
    port: int

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            gemini_api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
            default_model=os.environ.get("IMAGE_MODEL", DEFAULT_MODEL),
            output_dir=Path(os.environ.get("IMAGE_OUTPUT_DIR", "generated")).resolve(),
            max_images=int(os.environ.get("IMAGE_MAX_PER_REQUEST", "4")),
            api_token=os.environ.get("IMAGE_API_TOKEN") or None,
            host=os.environ.get("IMAGE_API_HOST", "127.0.0.1"),
            port=int(os.environ.get("IMAGE_API_PORT", "8000")),
        )
