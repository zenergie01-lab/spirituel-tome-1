"""Génération d'une fiche du Dictionnaire Vivant à partir d'un seul mot.

L'utilisateur fournit un mot (qui devient le Pôle A) ; Mistral génère le pôle
opposé, le point d'équilibre, le domaine, la présentation et 3 punchlines, dans
le style exact du Tome 1, sous forme de JSON strict.

Note : ce module fait de la GÉNÉRATION de texte (chat), distincte des embeddings
de src/embeddings.py. Il utilise la même clé Mistral.
"""
from __future__ import annotations

import json

from config import require_env
from parser import Fiche

GEN_MODEL = "mistral-large-latest"

# Liste fermée des 10 domaines du Tome 1 (extraits de tome1_social_table.md).
# Le modèle doit choisir DANS cette liste — sinon le filtre Recherche se fausse.
DOMAINES = [
    "Éléments & Nature",
    "Temps & Rythme",
    "Espace & Mouvement",
    "Corps & Énergie",
    "Émotions & États",
    "Relations & Lien",
    "Mental & Pensée",
    "Action & Création",
    "Société & Valeurs",
    "Spirituel & Universel",
]

_SYSTEM = """Tu es l'auteur du « Dictionnaire Vivant — Tome 1 », un recueil de \
dualités. Chaque fiche oppose deux pôles antonymes (Pôle A ⇄ Pôle B) et révèle \
un point d'ÉQUILIBRE qui n'est ni l'un ni l'autre, mais leur juste rencontre — \
une troisième voie vivante.

Style à respecter scrupuleusement :
- « presentation » : 1 à 2 phrases contemplatives, imagées (métaphore concrète \
de la nature, du corps ou du quotidien), qui posent la dualité avec profondeur.
- « punchlines » : exactement 3 phrases courtes et percutantes, façon réseaux \
sociaux, chacune terminée par UN emoji pertinent. Elles interpellent le lecteur \
(« Tu… ? »), donnent une analogie concrète, ou résument la sagesse de l'équilibre.
- « pole_b » : l'antonyme le plus juste et naturel du mot donné.
- « equilibre » : un mot ou court terme nommant le point d'équilibre (ni A ni B).
- « domaine » : EXACTEMENT l'un de ces 10 domaines, sans en inventer d'autre :
{domaines}.

Réponds UNIQUEMENT par un objet JSON valide, sans texte autour, de la forme :
{{"pole_a": "...", "pole_b": "...", "equilibre": "...", "domaine": "...", \
"presentation": "...", "punchlines": ["...", "...", "..."]}}"""

# Quelques fiches réelles en few-shot pour caler le ton.
_EXEMPLES = [
    {
        "pole_a": "Chaud", "pole_b": "Froid", "equilibre": "Tempéré",
        "domaine": "Éléments & Nature",
        "presentation": "Le corps brûle ou se fige. Entre les deux, il y a "
                        "l'endroit où la vie circule librement.",
        "punchlines": [
            "Ni brûler, ni geler. Juste habiter sa vie à la bonne température. 🌡️",
            "Tu connais le bain parfait ? C'est exactement ça — appliqué à toute ta vie.",
            "Le tempéré n'est pas tiède. C'est où les deux extrêmes coexistent.",
        ],
    },
    {
        "pole_a": "Peur", "pole_b": "Courage", "equilibre": "Confiance",
        "domaine": "Émotions & États",
        "presentation": "Le courage n'est jamais l'absence de peur — c'est la "
                        "décision de ne pas la laisser décider à ta place.",
        "punchlines": [
            "La peur n'est pas ton ennemie. C'est de l'information. 🦁",
            "Attendre que la peur disparaisse pour agir ? Elle n'attend pas.",
            "Le courage, c'est la décision d'avancer avec elle.",
        ],
    },
]


def _coerce_domaine(value: str) -> str:
    """Force le domaine dans la liste fermée (sécurité contre l'hallucination)."""
    value = (value or "").strip()
    for d in DOMAINES:
        if d.lower() == value.lower():
            return d
    # repli : correspondance partielle, sinon le domaine le plus universel
    for d in DOMAINES:
        if value and (value.lower() in d.lower() or d.lower() in value.lower()):
            return d
    return "Spirituel & Universel"


def generate_fiche(mot: str) -> Fiche:
    """Génère une Fiche complète à partir d'un mot (qui devient le Pôle A).

    L'id est laissé vide ('') ; il sera attribué au moment de l'enregistrement.
    """
    mot = mot.strip()
    if not mot:
        raise ValueError("Le mot ne peut pas être vide.")

    from mistralai import Mistral

    client = Mistral(api_key=require_env("MISTRAL_API_KEY"))
    system = _SYSTEM.format(domaines=", ".join(DOMAINES))

    messages = [{"role": "system", "content": system}]
    # few-shot : on montre des paires (mot → fiche JSON)
    for ex in _EXEMPLES:
        messages.append({"role": "user", "content": f"Mot : {ex['pole_a']}"})
        messages.append({"role": "assistant", "content": json.dumps(ex, ensure_ascii=False)})
    messages.append({"role": "user", "content": f"Mot : {mot}"})

    resp = client.chat.complete(
        model=GEN_MODEL,
        messages=messages,
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)

    # On impose le mot saisi comme Pôle A (le modèle pourrait le reformuler).
    punchlines = list(data.get("punchlines", []))[:3]
    punchlines += [""] * (3 - len(punchlines))  # toujours 3 cases

    return Fiche(
        id="",
        pole_a=mot,
        pole_b=str(data.get("pole_b", "")).strip(),
        equilibre=str(data.get("equilibre", "")).strip(),
        domaine=_coerce_domaine(data.get("domaine", "")),
        presentation=str(data.get("presentation", "")).strip(),
        punchlines=[str(p).strip() for p in punchlines],
    )


if __name__ == "__main__":
    import sys

    mot = sys.argv[1] if len(sys.argv) > 1 else "Orgueil"
    f = generate_fiche(mot)
    print(f"{f.titre}  ({f.domaine})")
    print(f"\n{f.presentation}\n")
    for p in f.punchlines:
        if p:
            print(f"  • {p}")
