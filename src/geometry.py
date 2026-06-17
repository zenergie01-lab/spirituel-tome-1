"""Géométrie de la dualité — le cœur conceptuel du Tome 1.

Une fois les fiches vectorisées, chaque dualité devient un triangle dans
l'espace : Pôle A, Pôle B, et Équilibre. On peut alors mesurer :

  • la tension de la dualité   = distance entre A et B
  • la complémentarité          = à quel point A et B sont opposés (cos < 0)
  • la justesse de l'équilibre  = l'Équilibre est-il vraiment "au milieu" de A et B ?
  • le POINT ORIGINE            = le barycentre de tous les équilibres
                                  (le centre commun de toutes les dualités — le Tao)

Ce module recharge les vecteurs depuis Pinecone (fetch par id) pour ne pas
recalculer les embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from pinecone import Pinecone

import config
from parser import parse


# ─────────────────────────── utilitaires vectoriels ───────────────────────────
def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize(a), _normalize(b)))


def euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ──────────────────────────── chargement des vecteurs ─────────────────────────
def _index():
    pc = Pinecone(api_key=config.require_env("PINECONE_API_KEY"))
    return pc.Index(config.pinecone_index_name())


def _all_fiche_ids() -> List[str]:
    """IDs de fiches = union du .md (canonique) et de Pinecone (nouveaux mots)."""
    from ingest import list_fiche_ids

    md_ids = [f.id for f in parse()]
    try:
        pc_ids = list_fiche_ids()
    except Exception:
        pc_ids = []
    return sorted(set(md_ids) | set(pc_ids))


def _titres() -> Dict[str, str]:
    """Map fiche_id → titre, pour le .md ET les fiches ajoutées via Pinecone."""
    from ingest import fiche_from_pinecone

    titres = {f.id: f.titre for f in parse()}
    for fid in _all_fiche_ids():
        if fid not in titres:
            f = fiche_from_pinecone(fid)
            if f:
                titres[fid] = f.titre
    return titres


def load_vectors() -> Dict[str, np.ndarray]:
    """Récupère tous les vecteurs (A/B/E de chaque fiche) depuis Pinecone.

    Inclut les fiches ajoutées via l'app (présentes dans Pinecone mais pas
    dans le .md), pour qu'elles comptent dans le point origine.
    """
    ids: List[str] = []
    for fid in _all_fiche_ids():
        ids += [f"{fid}#A", f"{fid}#B", f"{fid}#E"]

    index = _index()
    namespace = config.pinecone_namespace()
    vectors: Dict[str, np.ndarray] = {}
    # fetch par lots (limite de l'API)
    for start in range(0, len(ids), 100):
        batch = ids[start : start + 100]
        res = index.fetch(ids=batch, namespace=namespace)
        for vid, rec in res.vectors.items():
            vectors[vid] = np.asarray(rec.values, dtype=np.float64)
    if not vectors:
        raise RuntimeError(
            "Aucun vecteur trouvé. Lance d'abord l'ingestion : python src/ingest.py"
        )
    return vectors


# ─────────────────────────────── analyses ─────────────────────────────────────
@dataclass
class DualityReport:
    fiche_id: str
    titre: str
    tension: float          # norme de (A−B) : amplitude de l'écart entre les pôles
    eq_projection: float    # position de l'Équilibre sur l'axe A−B, ∈ ~[−1, +1]
                            #   +1 = collé à A, −1 = collé à B, 0 = parfait milieu
    biais: str              # lecture humaine de eq_projection


def analyze_duality(fiche_id: str, vectors: Dict[str, np.ndarray]) -> DualityReport:
    a = vectors[f"{fiche_id}#A"]
    b = vectors[f"{fiche_id}#B"]
    e = vectors[f"{fiche_id}#E"]

    milieu = (a + b) / 2.0
    d = a - b                      # axe de dualité
    tension = float(np.linalg.norm(d))

    # Projection de (E − milieu) sur l'axe d, normalisée par la demi-longueur |d|/2.
    # → coordonnée de l'équilibre le long de l'axe : +1 = en A, −1 = en B, 0 = milieu.
    e_rel = e - milieu
    half = tension / 2.0
    if half > 1e-9:
        proj = float(np.dot(e_rel, d) / np.linalg.norm(d)) / half
    else:
        proj = 0.0

    if abs(proj) < 0.10:
        biais = "centré (juste milieu)"
    elif proj > 0:
        biais = f"penche vers le pôle A ({+proj:.2f})"
    else:
        biais = f"penche vers le pôle B ({proj:.2f})"

    titre = _titres().get(fiche_id, fiche_id)
    return DualityReport(
        fiche_id=fiche_id,
        titre=titre,
        tension=tension,
        eq_projection=proj,
        biais=biais,
    )


def equilibres_centres(
    vectors: Dict[str, np.ndarray], top_k: int = 51
) -> List[Tuple[str, float]]:
    """Classe les fiches selon que l'Équilibre tombe au juste milieu de A et B.

    |projection| proche de 0 = équilibre parfaitement centré entre les pôles.
    """
    fiches = _titres()
    reports = []
    for fid in fiches:
        try:
            r = analyze_duality(fid, vectors)
        except KeyError:
            continue
        reports.append((f"[{fid}] {fiches[fid]}", r.eq_projection))
    reports.sort(key=lambda x: abs(x[1]))  # les plus centrés d'abord
    return reports[:top_k]


def point_origine(vectors: Dict[str, np.ndarray]) -> np.ndarray:
    """Le barycentre de TOUS les équilibres : le centre commun des dualités.

    C'est la traduction géométrique du "point origine" / Tao : l'endroit de
    l'espace sémantique vers lequel convergent tous les points d'équilibre.
    """
    eq = [v for vid, v in vectors.items() if vid.endswith("#E")]
    return np.mean(np.stack(eq), axis=0)


def closest_to_origine(
    vectors: Dict[str, np.ndarray], top_k: int = 5
) -> List[Tuple[str, float]]:
    """Quels équilibres sont les plus proches du point origine commun ?

    Les fiches qui le « contiennent » le mieux (Yin/Yang→Tao devrait bien
    figurer) remontent en tête.
    """
    origine = point_origine(vectors)
    fiches = _titres()
    scored = [
        (vid.split("#", 1)[0], cosine(v, origine))
        for vid, v in vectors.items()
        if vid.endswith("#E")
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [
        (f"[{fid}] {fiches.get(fid, fid)}", score)
        for fid, score in scored[:top_k]
    ]


# ─────────────────────────────────── CLI ──────────────────────────────────────
def _print_section(title: str) -> None:
    print(f"\n{'─' * 60}\n{title}\n{'─' * 60}")


def main(fiche_id: Optional[str] = None) -> None:
    vectors = load_vectors()

    if fiche_id:
        fid = fiche_id.zfill(3)
        r = analyze_duality(fid, vectors)
        _print_section(f"Dualité {r.fiche_id} — {r.titre}")
        print(f"  Tension (amplitude A−B)         : {r.tension:.4f}")
        print(f"  Position de l'équilibre sur l'axe: {r.eq_projection:+.3f}")
        print(f"  Lecture                          : {r.biais}")
        print("\n  (+1 = collé au pôle A, −1 = collé au pôle B, 0 = juste milieu)")
        return

    _print_section("POINT ORIGINE — équilibres les plus proches du centre commun")
    for label, score in closest_to_origine(vectors):
        print(f"  {score:+.4f}  {label}")

    _print_section("ÉQUILIBRES LES PLUS CENTRÉS (au juste milieu de leurs pôles)")
    for label, proj in equilibres_centres(vectors, top_k=8):
        print(f"  {proj:+.3f}  {label}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
