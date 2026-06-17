"""Interface web — Dictionnaire Vivant Tome 1.

Lance :  streamlit run web.py

Décris ce que tu ressens → reçois la dualité, son point d'équilibre,
et les punchlines de la fiche. Onglet séparé pour le « point origine ».
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Rend les modules de src/ importables (config, parser, search, geometry…).
sys.path.insert(0, str(Path(__file__).parent / "src"))

from parser import Fiche, parse  # noqa: E402
from search import search  # noqa: E402

ROLE_FR = {"A": "Pôle A", "B": "Pôle B", "E": "Équilibre"}


# ── Données et ressources mises en cache (chargées une seule fois) ──────────────
@st.cache_data(show_spinner=False)
def load_fiches() -> dict[str, Fiche]:
    return {f.id: f for f in parse()}


@st.cache_data(show_spinner="Recherche en cours…")
def cached_search(query: str, role: str | None, domaine: str | None) -> list[dict]:
    """Recherche mise en cache : une même requête ne recalcule pas l'embedding."""
    matches = search(query, top_k=4, role=role, domaine=domaine)
    # On ne garde que ce qui est sérialisable (le cache Streamlit l'exige).
    return [{"score": m["score"], "metadata": dict(m["metadata"])} for m in matches]


@st.cache_data(show_spinner="Calcul du point origine…")
def cached_origine(top_k: int) -> list[tuple[str, float]]:
    from geometry import closest_to_origine, load_vectors

    return closest_to_origine(load_vectors(), top_k=top_k)


# ── Affichage d'une fiche ──────────────────────────────────────────────────────
def render_fiche(f: Fiche, *, score: float | None = None, role: str | None = None) -> None:
    st.markdown(f"### {f.pole_a} ⇄ {f.pole_b} → ✨ **{f.equilibre}**")
    meta_line = f"`{f.id}` · *{f.domaine}*"
    if score is not None and role is not None:
        meta_line += f" · trouvé via **{ROLE_FR.get(role, role)}** ({score:.0%})"
    st.caption(meta_line)
    st.write(f"> {f.presentation}")
    for punch in f.punchlines:
        if punch.strip():
            st.markdown(f"- {punch}")


# ── Pages ──────────────────────────────────────────────────────────────────────
def page_recherche(fiches: dict[str, Fiche]) -> None:
    st.subheader("Décris ce que tu ressens")
    query = st.text_input(
        "Ta phrase",
        placeholder="ex. je suis épuisé, je n'arrive plus à m'arrêter",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        domaines = sorted({f.domaine for f in fiches.values()})
        domaine = st.selectbox("Domaine", ["Tous"] + domaines)
    with col2:
        focus_eq = st.toggle("Cibler les points d'équilibre", value=False)

    if not query.strip():
        st.info("Tape une phrase ci-dessus pour trouver la dualité qui te répond.")
        return

    role = "E" if focus_eq else None
    dom = None if domaine == "Tous" else domaine
    matches = cached_search(query.strip(), role, dom)

    if not matches:
        st.warning("Aucune fiche trouvée. Reformule ta phrase ?")
        return

    best = matches[0]
    fiche = fiches.get(best["metadata"]["fiche_id"])
    if fiche:
        render_fiche(fiche, score=best["score"], role=best["metadata"]["role"])

    autres = matches[1:]
    if autres:
        st.divider()
        st.caption("Autres pistes :")
        for m in autres:
            md = m["metadata"]
            with st.expander(f"[{md['fiche_id']}] {md['titre']} · {m['score']:.0%}"):
                other = fiches.get(md["fiche_id"])
                if other:
                    render_fiche(other)


def page_origine() -> None:
    st.subheader("Point origine — le centre commun des 51 dualités")
    st.write(
        "Le barycentre de tous les points d'équilibre. C'est la traduction "
        "géométrique du Tao : l'endroit vers lequel convergent toutes les dualités."
    )
    top_k = st.slider("Nombre de fiches", 3, 15, 5)
    for label, score in cached_origine(top_k):
        st.markdown(f"**{score:+.3f}**  ·  {label}")


def page_explorer(fiches: dict[str, Fiche]) -> None:
    st.subheader("Explorer les 51 fiches")
    options = [f"{f.id} — {f.pole_a} ⇄ {f.pole_b} → {f.equilibre}" for f in fiches.values()]
    choix = st.selectbox("Choisis une dualité", options)
    fiche = fiches[choix[:3]]
    render_fiche(fiche)


# ── App ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="Dictionnaire Vivant — Tome 1", page_icon="🌱")
    st.title("🌱 Dictionnaire Vivant — Tome 1")
    st.caption("Entre deux pôles opposés, trouve ton point d'équilibre.")

    fiches = load_fiches()
    tab1, tab2, tab3 = st.tabs(["🔍 Recherche", "☯️ Point origine", "📖 Explorer"])
    with tab1:
        page_recherche(fiches)
    with tab2:
        page_origine()
    with tab3:
        page_explorer(fiches)


main()
