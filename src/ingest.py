"""Création de l'index Pinecone + ingestion des fiches.

Chaque fiche produit 3 vecteurs :
  - {id}#A  → le Pôle A
  - {id}#B  → le Pôle B
  - {id}#E  → l'Équilibre (le point d'origine de la fiche)

Métadonnées portées par chaque vecteur :
  fiche_id, role (A|B|E), pole_a, pole_b, equilibre, domaine, text
Cela permet de filtrer (par domaine, par rôle) et de reconstruire une fiche.
"""
from __future__ import annotations

import time
from typing import Dict, List

from pinecone import Pinecone, ServerlessSpec

import config
from embeddings import embed
from parser import Fiche, parse


def _client() -> Pinecone:
    return Pinecone(api_key=config.require_env("PINECONE_API_KEY"))


def _open_index():
    """Ouvre l'index existant (ne le crée pas) — pour lecture/append rapide."""
    return _client().Index(config.pinecone_index_name())


def ensure_index(pc: Pinecone, dimension: int) -> str:
    name = config.pinecone_index_name()
    existing = {ix["name"] for ix in pc.list_indexes()}
    if name in existing:
        desc = pc.describe_index(name)
        if desc.dimension != dimension:
            raise RuntimeError(
                f"L'index '{name}' existe en dimension {desc.dimension}, "
                f"mais le modèle courant produit du {dimension}. "
                f"Supprime l'index ou change de modèle/d'index."
            )
        return name

    print(f"Création de l'index '{name}' (dim={dimension})…")
    pc.create_index(
        name=name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=config.pinecone_cloud(),
            region=config.pinecone_region(),
        ),
    )
    while not pc.describe_index(name).status["ready"]:
        time.sleep(1)
    print("Index prêt.")
    return name


def _build_records(fiches: List[Fiche]) -> List[Dict]:
    """Construit (id, texte, métadonnées) pour les 3 vecteurs de chaque fiche."""
    records: List[Dict] = []
    for f in fiches:
        common = {
            "fiche_id": f.id,
            "pole_a": f.pole_a,
            "pole_b": f.pole_b,
            "equilibre": f.equilibre,
            "domaine": f.domaine,
            "titre": f.titre,
            # Présentation + punchlines stockées en metadata : permet de
            # reconstruire une fiche entièrement depuis Pinecone (utile pour
            # les nouveaux mots qui ne sont pas dans le .md).
            "presentation": f.presentation,
            "punchlines": [p for p in f.punchlines if p.strip()],
        }
        records.append(
            {"id": f"{f.id}#A", "text": f.pole_a_text(),
             "meta": {**common, "role": "A", "label": f.pole_a}}
        )
        records.append(
            {"id": f"{f.id}#B", "text": f.pole_b_text(),
             "meta": {**common, "role": "B", "label": f.pole_b}}
        )
        records.append(
            {"id": f"{f.id}#E", "text": f.equilibre_text(),
             "meta": {**common, "role": "E", "label": f.equilibre}}
        )
    return records


def ingest() -> None:
    spec = config.get_provider()
    print(f"Modèle d'embeddings : {spec.name}/{spec.model} ({spec.dimension}d)")

    fiches = parse()
    print(f"{len(fiches)} fiches → {len(fiches) * 3} vecteurs à créer.")

    records = _build_records(fiches)
    texts = [r["text"] for r in records]

    print("Calcul des embeddings…")
    vectors = embed(texts, input_type="search_document")

    pc = _client()
    name = ensure_index(pc, spec.dimension)
    index = pc.Index(name)
    namespace = config.pinecone_namespace()

    payload = [
        {"id": r["id"], "values": v, "metadata": {**r["meta"], "text": r["text"]}}
        for r, v in zip(records, vectors)
    ]

    print(f"Upsert vers Pinecone (index '{name}', namespace '{namespace}')…")
    for start in range(0, len(payload), 100):
        index.upsert(vectors=payload[start : start + 100], namespace=namespace)
    stats = index.describe_index_stats()
    ns_count = stats.get("namespaces", {}).get(namespace, {})
    count = getattr(ns_count, "vector_count", None) or (
        ns_count.get("vector_count") if isinstance(ns_count, dict) else None
    )
    print(f"Terminé. Vecteurs dans le namespace '{namespace}' : {count}")


# ─────────────────── Append / lecture d'une seule fiche ───────────────────────
def upsert_fiche(fiche: Fiche) -> None:
    """Ajoute (ou remplace) une seule fiche dans Pinecone : 3 vecteurs.

    Réutilise _build_records() et embed(). Idempotent grâce à l'upsert : si l'id
    existe déjà, les vecteurs sont écrasés ; sinon ils sont ajoutés.
    """
    spec = config.get_provider()
    records = _build_records([fiche])
    vectors = embed([r["text"] for r in records], input_type="search_document")

    pc = _client()
    index = pc.Index(ensure_index(pc, spec.dimension))
    payload = [
        {"id": r["id"], "values": v, "metadata": {**r["meta"], "text": r["text"]}}
        for r, v in zip(records, vectors)
    ]
    index.upsert(vectors=payload, namespace=config.pinecone_namespace())


def list_fiche_ids() -> List[str]:
    """Tous les fiche_id distincts présents dans le namespace Pinecone.

    Pinecone pagine les ids ; chaque fiche a 3 vecteurs (#A/#B/#E) → on retire
    le suffixe et on déduplique.
    """
    index = _open_index()
    namespace = config.pinecone_namespace()
    ids: set[str] = set()
    for page in index.list(namespace=namespace):
        # Chaque page est un ListResponse avec .vectors = [ListItem(id=...), …].
        # Selon la version du SDK, on peut aussi recevoir directement une liste.
        items = getattr(page, "vectors", page)
        for item in items:
            vid = getattr(item, "id", item)  # ListItem → .id, sinon str brut
            ids.add(str(vid).split("#", 1)[0])
    return sorted(ids)


def fiche_from_pinecone(fiche_id: str) -> Fiche | None:
    """Reconstruit une Fiche depuis ses métadonnées Pinecone (#A/#B/#E)."""
    index = _open_index()
    namespace = config.pinecone_namespace()
    res = index.fetch(
        ids=[f"{fiche_id}#A", f"{fiche_id}#B", f"{fiche_id}#E"], namespace=namespace
    )
    vectors = res.vectors
    rec = vectors.get(f"{fiche_id}#E") or next(iter(vectors.values()), None)
    if rec is None:
        return None
    m = rec.metadata or {}
    punch = list(m.get("punchlines", []))
    punch += [""] * (3 - len(punch))
    return Fiche(
        id=fiche_id,
        pole_a=str(m.get("pole_a", "")),
        pole_b=str(m.get("pole_b", "")),
        equilibre=str(m.get("equilibre", "")),
        domaine=str(m.get("domaine", "")),
        presentation=str(m.get("presentation", "")),
        punchlines=[str(p) for p in punch[:3]],
    )


def next_fiche_id(known_ids: List[str]) -> str:
    """Prochain id libre (max + 1), formaté sur 3 chiffres comme le parser."""
    nums = [int(i) for i in known_ids if i.isdigit()]
    return str((max(nums) + 1) if nums else 1).zfill(3)


if __name__ == "__main__":
    ingest()
