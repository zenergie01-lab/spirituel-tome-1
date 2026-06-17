"""Configuration centrale — lue depuis les variables d'environnement (.env)."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# La console Windows est souvent en cp1252 et plante sur les caractères
# unicode (⇄, →, ☯…). On force UTF-8 sur stdout/stderr une fois pour toutes.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

# Racine du projet (le dossier parent de src/)
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

SOURCE_MD = ROOT / "tome1_social_table.md"


@dataclass(frozen=True)
class ProviderSpec:
    """Caractéristiques d'un fournisseur d'embeddings."""
    name: str
    model: str
    dimension: int


# Dimensions natives de chaque modèle — sert à créer l'index Pinecone
# avec la bonne taille. Si tu changes de modèle, l'index doit correspondre.
_DIMENSIONS = {
    ("mistral", "mistral-embed"): 1024,
    ("openai", "text-embedding-3-large"): 3072,
    ("openai", "text-embedding-3-small"): 1536,
    ("cohere", "embed-multilingual-v3.0"): 1024,
}


def get_provider() -> ProviderSpec:
    provider = os.getenv("EMBEDDING_PROVIDER", "mistral").strip().lower()
    if provider == "mistral":
        model = os.getenv("MISTRAL_EMBED_MODEL", "mistral-embed")
    elif provider == "openai":
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    elif provider == "cohere":
        model = os.getenv("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")
    else:
        raise ValueError(
            f"EMBEDDING_PROVIDER inconnu : '{provider}'. "
            "Utilise mistral, openai ou cohere."
        )

    dim = _DIMENSIONS.get((provider, model))
    if dim is None:
        raise ValueError(
            f"Dimension inconnue pour {provider}/{model}. "
            f"Ajoute-la dans config._DIMENSIONS."
        )
    return ProviderSpec(name=provider, model=model, dimension=dim)


def require_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise RuntimeError(
            f"Variable d'environnement manquante : {key}. "
            "Copie .env.example en .env et remplis-la."
        )
    return value


# --- Pinecone ---
def pinecone_index_name() -> str:
    return os.getenv("PINECONE_INDEX", "dictionnaire-vivant-tome1")


def pinecone_namespace() -> str:
    """Namespace où vivent nos vecteurs (isole le Tome 1 dans un index partagé)."""
    return os.getenv("PINECONE_NAMESPACE", "tome1")


def pinecone_cloud() -> str:
    return os.getenv("PINECONE_CLOUD", "aws")


def pinecone_region() -> str:
    return os.getenv("PINECONE_REGION", "us-east-1")
