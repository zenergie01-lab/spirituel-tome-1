# Dictionnaire Vivant — Tome 1 · Base vectorielle

Transforme les **51 dualités** de `tome1_social_table.md` en un dictionnaire
vectoriel (Pinecone) pour :

1. **Rechercher** des fiches par sens (« j'ai peur d'avancer » → la bonne dualité)
2. **Mesurer la dualité** géométriquement (tension, complémentarité des pôles)
3. **Trouver le point origine** : le barycentre de tous les équilibres — la
   traduction mathématique du Tao, le centre commun vers lequel convergent
   toutes les dualités.

## Concept

Chaque fiche devient **3 vecteurs** dans l'espace sémantique :

```
        Pôle A  ●─────────●  Pôle B      ← tension = distance A↔B
                 \       /
                  \     /
                   ● Équilibre            ← doit être "au milieu" de A et B
```

En empilant les 51 équilibres et en prenant leur **barycentre**, on obtient le
**point origine** — le centre de gravité de toutes les dualités.

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env      # puis remplis tes clés
```

Tu as besoin d'une clé **Pinecone** + une clé du fournisseur d'embeddings choisi
(Mistral par défaut). Le code est **agnostique** : change `EMBEDDING_PROVIDER`
dans `.env` (`mistral` | `openai` | `cohere`) sans toucher au code.

> ⚠️ Si tu changes de modèle, ses vecteurs ont une dimension différente :
> utilise un **nouveau** `PINECONE_INDEX` (l'ingestion refuse de mélanger).

## Utilisation

```bash
# 0a. Interface web (4 onglets : recherche · point origine · explorer · ajouter un mot)
streamlit run web.py

# 0b. Menu interactif en terminal (tape une phrase, reçois la fiche)
python src/app.py

# 1. Vérifier la lecture des fiches (aucune clé requise)
python src/parser.py

# 2. Créer l'index + envoyer les 153 vecteurs (3 × 51)
python src/ingest.py

# 3. Rechercher par sens
python src/search.py "j'ai peur d'avancer"
python src/search.py "trop de vitesse" --role E --domaine "Temps & Rythme"

# 4. Analyser la géométrie
python src/geometry.py              # point origine + dualités les plus complémentaires
python src/geometry.py 051          # analyse détaillée d'une fiche (Yin/Yang→Tao)

# 5. Générer une fiche à partir d'un mot (Mistral) — sans l'enregistrer
python src/generate.py "Orgueil"
```

## Ajouter un mot (le dictionnaire grandit)

L'onglet **« ➕ Ajouter un mot »** de l'app web : tu tapes un mot, Mistral génère
le pôle opposé, l'équilibre, le domaine, la présentation et 3 punchlines dans le
style du Tome 1. Tu prévisualises, puis **Enregistrer** ajoute la fiche à Pinecone.

Les mots ajoutés rejoignent les 51 originaux **partout** : recherche, Explorer et
calcul du point origine. Ils sont stockés dans **Pinecone** (pas dans le `.md`),
ce qui les rend persistants même sur Streamlit Cloud (filesystem éphémère).

## Structure

| Fichier            | Rôle                                                       |
|--------------------|------------------------------------------------------------|
| `web.py`           | **Interface web Streamlit** (recherche · origine · explorer · ajouter)|
| `src/generate.py`  | Génère une fiche complète à partir d'un mot (Mistral chat)  |
| `src/app.py`       | Menu interactif en terminal : phrase → fiche + punchlines   |
| `src/config.py`    | Config + choix du modèle via `.env`                        |
| `src/parser.py`    | `tome1_social_table.md` → objets `Fiche`                   |
| `src/embeddings.py`| Couche agnostique Mistral / OpenAI / Cohere                |
| `src/ingest.py`    | Crée l'index Pinecone, upsert 3 vecteurs/fiche             |
| `src/search.py`    | Recherche sémantique (filtres rôle/domaine)                |
| `src/geometry.py`  | Tension, complémentarité, **point origine** (barycentre)   |

## Métadonnées Pinecone

Chaque vecteur (`001#A`, `001#B`, `001#E`, …) porte :
`fiche_id`, `role` (A/B/E), `label`, `pole_a`, `pole_b`, `equilibre`,
`domaine`, `titre`, `text` — pour filtrer et reconstruire les fiches.
