"""Lecture de tome1_social_table.md → liste de Fiche structurées.

Le fichier source est un tableau Markdown dont chaque ligne est une dualité :
| # | Pôle A | Pôle B | Équilibre | Domaine | Présentation | Punchline 1..3 |
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from config import SOURCE_MD

# Ordre des colonnes attendu dans le tableau Markdown.
_COLUMNS = [
    "id", "pole_a", "pole_b", "equilibre", "domaine",
    "presentation", "punchline_1", "punchline_2", "punchline_3",
]


@dataclass
class Fiche:
    id: str            # ex. "001"
    pole_a: str
    pole_b: str
    equilibre: str
    domaine: str
    presentation: str
    punchlines: List[str] = field(default_factory=list)

    @property
    def titre(self) -> str:
        return f"{self.pole_a} ⇄ {self.pole_b} → {self.equilibre}"

    def pole_a_text(self) -> str:
        """Texte enrichi pour l'embedding du pôle A (contexte = sa fiche)."""
        return f"{self.pole_a}. {self.presentation}"

    def pole_b_text(self) -> str:
        return f"{self.pole_b}. {self.presentation}"

    def equilibre_text(self) -> str:
        punch = " ".join(self.punchlines)
        return f"{self.equilibre}. {self.presentation} {punch}".strip()


def _split_row(line: str) -> List[str]:
    """Découpe une ligne de tableau Markdown en cellules nettoyées."""
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def _is_data_row(cells: List[str]) -> bool:
    """Une vraie ligne de données : 1re cellule = identifiant numérique."""
    return bool(cells) and cells[0].isdigit()


def parse(path: Path | None = None) -> List[Fiche]:
    path = path or SOURCE_MD
    if not path.exists():
        raise FileNotFoundError(f"Source introuvable : {path}")

    fiches: List[Fiche] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.lstrip().startswith("|"):
            continue
        cells = _split_row(raw)
        if not _is_data_row(cells):
            continue  # ignore l'en-tête et la ligne de séparation |---|
        if len(cells) < len(_COLUMNS):
            cells += [""] * (len(_COLUMNS) - len(cells))

        record = dict(zip(_COLUMNS, cells))
        fiches.append(
            Fiche(
                id=record["id"].zfill(3),
                pole_a=record["pole_a"],
                pole_b=record["pole_b"],
                equilibre=record["equilibre"],
                domaine=record["domaine"],
                presentation=record["presentation"],
                punchlines=[
                    record["punchline_1"],
                    record["punchline_2"],
                    record["punchline_3"],
                ],
            )
        )
    return fiches


if __name__ == "__main__":
    fiches = parse()
    print(f"{len(fiches)} fiches lues.")
    for f in fiches[:3]:
        print(f"  [{f.id}] {f.titre}  ({f.domaine})")
    if fiches:
        print(f"  ... [{fiches[-1].id}] {fiches[-1].titre}")
