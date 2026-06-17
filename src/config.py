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


def _get(key: str, default: str | None = None) -> str | None:
    """Lit une clé de config : variables d'env (.env) d'abord, puis les
    secrets Streamlit (st.secrets) quand on tourne dans Streamlit Cloud.

    Permet au même code de marcher en local (.env) et en ligne (secrets).
    """
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st  # import paresseux : pas requis hors Streamlit

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


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
    provider = (_get("EMBEDDING_PROVIDER", "mistral") or "mistral").strip().lower()
    if provider == "mistral":
        model = _get("MISTRAL_EMBED_MODEL", "mistral-embed")
    elif provider == "openai":
        model = _get("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    elif provider == "cohere":
        model = _get("COHERE_EMBED_MODEL", "embed-multilingual-v3.0")
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
    value = _get(key)
    if not value:
        raise RuntimeError(
            f"Configuration manquante : {key}. "
            "En local : copie .env.example en .env et remplis-la. "
            "Sur Streamlit Cloud : ajoute-la dans les Secrets de l'app."
        )
    return value


# --- Pinecone ---
def pinecone_index_name() -> str:
    return _get("PINECONE_INDEX", "dictionnaire-vivant-tome1")


def pinecone_namespace() -> str:
    """Namespace où vivent nos vecteurs (isole le Tome 1 dans un index partagé)."""
    return _get("PINECONE_NAMESPACE", "tome1")


def pinecone_cloud() -> str:
    return _get("PINECONE_CLOUD", "aws")


def pinecone_region() -> str:
    return _get("PINECONE_REGION", "us-east-1")
