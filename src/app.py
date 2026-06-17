"""Menu interactif — Dictionnaire Vivant Tome 1.

Lance simplement :  python src/app.py
Tape une phrase (ton ressenti, ta situation) → reçois la dualité qui répond,
son point d'équilibre, et les punchlines de la fiche.

Commandes spéciales :
    /origine        affiche le point origine (centre commun des dualités)
    /fiche 021      affiche la fiche complète n°021
    /aide           rappel des commandes
    /quitter        sortir (ou Ctrl-C)
"""
from __future__ import annotations

from typing import Dict, Optional

from parser import Fiche, parse
from search import search

ROLE_FR = {"A": "Pôle A", "B": "Pôle B", "E": "Équilibre"}


def _fiches_index() -> Dict[str, Fiche]:
    return {f.id: f for f in parse()}


def _print_fiche(f: Fiche, *, score: Optional[float] = None, role: Optional[str] = None) -> None:
    barre = "═" * 64
    print(f"\n{barre}")
    entete = f"  [{f.id}]  {f.pole_a}  ⇄  {f.pole_b}   →   ✨ {f.equilibre}"
    print(entete)
    print(f"  Domaine : {f.domaine}", end="")
    if score is not None and role is not None:
        print(f"   ·   trouvé via {ROLE_FR.get(role, role)} (proximité {score:.0%})")
    else:
        print()
    print(barre)
    print(f"\n  {f.presentation}\n")
    for i, punch in enumerate(f.punchlines, 1):
        if punch.strip():
            print(f"   {i}. {punch}")
    print()


def _handle_command(cmd: str, fiches: Dict[str, Fiche]) -> bool:
    """Traite une commande /…  Renvoie False si on doit quitter."""
    parts = cmd.split()
    name = parts[0].lower()

    if name in ("/quitter", "/quit", "/q", "/exit"):
        print("\nÀ bientôt. 🙏\n")
        return False

    if name in ("/aide", "/help", "/?"):
        print(__doc__)
        return True

    if name == "/origine":
        # Import tardif : évite de charger numpy/pinecone si on ne s'en sert pas.
        from geometry import closest_to_origine, load_vectors

        print("\n  POINT ORIGINE — équilibres les plus proches du centre commun :\n")
        for label, sc in closest_to_origine(load_vectors(), top_k=5):
            print(f"    {sc:+.3f}  {label}")
        print()
        return True

    if name == "/fiche":
        if len(parts) < 2 or not parts[1].strip().zfill(3) in fiches:
            print("  Usage : /fiche 021   (numéro entre 001 et 051)")
            return True
        _print_fiche(fiches[parts[1].strip().zfill(3)])
        return True

    print(f"  Commande inconnue : {name}.  Tape /aide pour la liste.")
    return True


def _do_search(query: str, fiches: Dict[str, Fiche]) -> None:
    matches = search(query, top_k=3)
    if not matches:
        print("  Aucune fiche trouvée. Reformule ta phrase ?")
        return
    # Meilleur match → on affiche la fiche complète.
    best = matches[0]
    meta = best["metadata"]
    fiche = fiches.get(meta["fiche_id"])
    if fiche:
        _print_fiche(fiche, score=best["score"], role=meta["role"])

    # Les autres pistes, en une ligne chacune.
    autres = matches[1:]
    if autres:
        print("  Autres pistes :")
        for m in autres:
            md = m["metadata"]
            print(f"    · [{md['fiche_id']}] {md['titre']}  ({m['score']:.0%})")
        print()


def main() -> None:
    fiches = _fiches_index()
    print("\n" + "─" * 64)
    print("  DICTIONNAIRE VIVANT — Tome 1")
    print("  Décris ce que tu ressens. Reçois le point d'équilibre.")
    print("  (/aide pour les commandes · /quitter pour sortir)")
    print("─" * 64)

    while True:
        try:
            entree = input("\n🌱 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÀ bientôt. 🙏\n")
            break

        if not entree:
            continue
        if entree.startswith("/"):
            if not _handle_command(entree, fiches):
                break
            continue

        _do_search(entree, fiches)


if __name__ == "__main__":
    main()
