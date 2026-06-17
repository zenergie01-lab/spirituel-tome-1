"""Couche d'embeddings agnostique.

Un seul point d'entrée — embed(texts) — qui route vers Mistral, OpenAI
ou Cohere selon EMBEDDING_PROVIDER. Le reste du code ne dépend jamais
d'un fournisseur précis.
"""
from __future__ import annotations

from typing import List, Sequence

from config import ProviderSpec, get_provider, require_env

# Type d'entrée d'un embedding : query (recherche) ou document (stockage).
# Cohere distingue les deux ; les autres l'ignorent.
InputType = str  # "search_document" | "search_query"


def _embed_mistral(texts: Sequence[str], spec: ProviderSpec) -> List[List[float]]:
    from mistralai import Mistral

    client = Mistral(api_key=require_env("MISTRAL_API_KEY"))
    resp = client.embeddings.create(model=spec.model, inputs=list(texts))
    return [d.embedding for d in resp.data]


def _embed_openai(texts: Sequence[str], spec: ProviderSpec) -> List[List[float]]:
    from openai import OpenAI

    client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
    resp = client.embeddings.create(model=spec.model, input=list(texts))
    return [d.embedding for d in resp.data]


def _embed_cohere(
    texts: Sequence[str], spec: ProviderSpec, input_type: InputType
) -> List[List[float]]:
    import cohere

    client = cohere.ClientV2(api_key=require_env("COHERE_API_KEY"))
    resp = client.embed(
        model=spec.model,
        texts=list(texts),
        input_type=input_type,
        embedding_types=["float"],
    )
    return resp.embeddings.float_


def embed(
    texts: Sequence[str],
    *,
    input_type: InputType = "search_document",
    batch_size: int = 64,
) -> List[List[float]]:
    """Vectorise une liste de textes. Renvoie une liste de vecteurs (floats).

    input_type : "search_document" pour l'ingestion, "search_query" pour
    une requête de recherche (utilisé seulement par Cohere).
    """
    if not texts:
        return []

    spec = get_provider()
    out: List[List[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        if spec.name == "mistral":
            out.extend(_embed_mistral(batch, spec))
        elif spec.name == "openai":
            out.extend(_embed_openai(batch, spec))
        elif spec.name == "cohere":
            out.extend(_embed_cohere(batch, spec, input_type))
        else:  # pragma: no cover - get_provider valide déjà
            raise ValueError(f"Fournisseur non géré : {spec.name}")
    return out


def embed_one(text: str, *, input_type: InputType = "search_query") -> List[float]:
    """Raccourci pour vectoriser un seul texte (typiquement une requête)."""
    return embed([text], input_type=input_type)[0]
