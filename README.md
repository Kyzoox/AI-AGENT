# image-agent

Ce projet permet à un LLM de générer et de modifier des images. Il passe par l'**API Gemini officielle** de Google, avec les modèles d'image « Nano Banana » de la même famille que Google Flow.

Le service s'utilise de deux façons :

- **Serveur MCP** : le LLM (Claude Code, Claude Desktop, Cursor…) appelle directement les outils `generate_image`, `edit_image` et `list_image_models`.
- **API REST** : `POST /v1/images/generations` et `POST /v1/images/edits`, pour un agent qui passe par des appels HTTP (function calling, n8n, LangChain…).

Chaque image est enregistrée dans `IMAGE_OUTPUT_DIR` et son chemin est renvoyé. Le LLM peut ainsi la retoucher avec `edit_image`.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # puis renseigner GEMINI_API_KEY
```

La clé se crée gratuitement sur <https://aistudio.google.com/apikey>. La génération d'images est facturée à l'usage selon la grille tarifaire de l'API Gemini.

## Configuration

| Variable | Défaut | Rôle |
|---|---|---|
| `GEMINI_API_KEY` | — (obligatoire) | Clé de l'API Gemini |
| `IMAGE_MODEL` | `gemini-nano-banana-2.1` | Modèle par défaut |
| `IMAGE_OUTPUT_DIR` | `./generated` | Dossier des images générées |
| `IMAGE_MAX_PER_REQUEST` | `4` | Nombre maximal d'images par appel |
| `IMAGE_API_TOKEN` | — | Jeton Bearer exigé par l'API REST (obligatoire hors `127.0.0.1`) |
| `IMAGE_API_HOST` / `IMAGE_API_PORT` | `127.0.0.1` / `8000` | Adresse d'écoute de l'API REST |

Modèles disponibles :

| Modèle | Usage |
|---|---|
| `gemini-nano-banana-2.1` | Choix recommandé |
| `gemini-3.1-flash-image` | Génération rapide |
| `gemini-3.1-flash-lite-image` | Le moins cher |
| `gemini-3-pro-image` | Qualité studio, jusqu'à 4K |

## Utilisation avec un LLM via MCP

### Claude Code

```bash
claude mcp add image-agent \
  -e GEMINI_API_KEY=votre_cle \
  -e IMAGE_OUTPUT_DIR=/chemin/absolu/vers/images \
  -- /chemin/vers/.venv/bin/image-agent-mcp
```

### Claude Desktop, Cursor et autres clients MCP

```json
{
  "mcpServers": {
    "image-agent": {
      "command": "/chemin/vers/.venv/bin/image-agent-mcp",
      "env": {
        "GEMINI_API_KEY": "votre_cle",
        "IMAGE_OUTPUT_DIR": "/chemin/absolu/vers/images"
      }
    }
  }
}
```

Une fois le serveur ajouté, il suffit de demander au LLM, par exemple : « Génère 3 visuels 16:9 pour la bannière de mon site, puis ajoute le logo sur le meilleur ».

Outils exposés :

| Outil | Paramètres |
|---|---|
| `generate_image` | `prompt`, `aspect_ratio`, `image_size` (`512`, `1K`, `2K`, `4K`), `count`, `model`, `return_images` |
| `edit_image` | `prompt`, `image_paths` (une ou plusieurs images locales), `aspect_ratio`, `image_size`, `model`, `return_images` |
| `list_image_models` | — |

Formats acceptés pour `aspect_ratio` : `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9`, `1:4`, `4:1`, `1:8`, `8:1`.

Par défaut (`return_images: true`), les images sont aussi renvoyées dans la réponse, ce qui permet au LLM de les voir. Avec `false`, seuls les chemins sont renvoyés, ce qui consomme moins de contexte.

## Utilisation via l'API REST

Lancer le serveur :

```bash
set -a && source .env && set +a   # charge la configuration dans l'environnement
image-agent-api                   # écoute sur http://127.0.0.1:8000, documentation interactive sur /docs
```

Générer des images :

```bash
curl -s http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "un phare au coucher du soleil, style aquarelle", "aspect_ratio": "16:9", "n": 2}'
```

```json
{
  "created": 1792000000,
  "model": "gemini-nano-banana-2.1",
  "data": [
    {"url": "http://127.0.0.1:8000/files/20261014T101500Z-3f2a9c1b7e4d-0.jpg", "b64_json": null, "mime_type": "image/jpeg"},
    {"url": "http://127.0.0.1:8000/files/20261014T101500Z-3f2a9c1b7e4d-1.jpg", "b64_json": null, "mime_type": "image/jpeg"}
  ],
  "text": ""
}
```

- `"response_format": "b64_json"` renvoie les images encodées en base64 dans la réponse.
- `POST /v1/images/edits` prend en plus `images: [{"data": "<base64>", "mime_type": "image/png"}]`.
- `GET /v1/models` liste les modèles.

Si `IMAGE_API_TOKEN` est défini, chaque requête doit envoyer `Authorization: Bearer <jeton>`. C'est obligatoire si l'API est exposée sur le réseau : elle consomme votre clé payante.

Les erreurs sont renvoyées au format `{"error": {"message": ...}}` :

| Code | Cause |
|---|---|
| 400 | Paramètre invalide |
| 401 | Jeton manquant ou invalide |
| 422 | Image refusée par les filtres de sécurité |
| 429 | Quota Gemini atteint |
| 502 | Erreur côté Google |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Les tests utilisent un faux client Gemini : ils ne demandent ni clé ni accès réseau.

## Limites

- Les images générées portent le filigrane invisible SynthID de Google.
- Les filtres de sécurité de Gemini s'appliquent : un prompt refusé renvoie une erreur 422 avec l'explication du modèle.
- La génération de vidéo (Veo) n'est pas incluse ; elle passe par une autre API.
