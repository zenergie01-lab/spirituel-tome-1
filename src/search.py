"""Recherche sémantique dans le dictionnaire vectoriel.

Exemples :
    python src/search.py "j'ai peur d'avancer"
    python src/search.py "trop de vitesse" --role E --domaine "Temps & Rythme"
"""
from __future__ import annotations

import argparse
from typing import List, Optional

from pinecone import Pinecone

import config
from embeddings import embed_one


def _index():
    pc = Pinecone(api_key=config.require_env("PINECONE_API_KEY"))
    return pc.Index(config.pinecone_index_name())


def search(
    query: str,
    *,
    top_k: int = 5,
    role: Optional[str] = None,
    domaine: Optional[str] = None,
) -> List[dict]:
    """Renvoie les vecteurs les plus proches de la requête.

    role : restreindre à "A", "B" ou "E" (équilibres).
    domaine : restreindre à un domaine ("Temps & Rythme", etc.).
    """
    vec = embed_one(query, input_type="search_query")

    flt = {}
    if role:
        flt["role"] = role.upper()
    if domaine:
        flt["domaine"] = domaine

    res = _index().query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        filter=flt or None,
        namespace=config.pinecone_namespace(),
    )
    return res.get("matches", [])


def _format(match: dict) -> str:
    m = match["metadata"]
    score = match["score"]
    role_fr = {"A": "Pôle A", "B": "Pôle B", "E": "Équilibre"}[m["role"]]
    return (
        f"  {score:.3f}  [{m['fiche_id']}] {role_fr}: {m['label']}\n"
        f"         {m['titre']}  ({m['domaine']})"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Recherche sémantique Tome 1")
    p.add_argument("query", help="texte de la requête")
    p.add_argument("-k", "--top-k", type=int, default=5)
    p.add_argument("--role", choices=["A", "B", "E"], help="filtrer par rôle")
    p.add_argument("--domaine", help="filtrer par domaine")
    args = p.parse_args()

    matches = search(args.query, top_k=args.top_k, role=args.role, domaine=args.domaine)
    print(f"\nRequête : « {args.query} »\n")
    if not matches:
        print("  (aucun résultat)")
        return
    for m in matches:
        print(_format(m))
        print()


if __name__ == "__main__":
    main()
