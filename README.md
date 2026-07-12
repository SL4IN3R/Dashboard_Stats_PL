# ⚽ Dashboard — Performances en Premier League (2015-2023)

Dashboard interactif réalisé dans le cadre du projet d'Analyse de Données (EPT 2025-2026).
Données : [Understat.com](https://understat.com) — 3 420 matchs de Premier League sur 9 saisons.

## Aperçu

8 sections accessibles depuis la barre de navigation latérale, pilotées par un filtre global de saisons :

| Section | Contenu |
|---|---|
| 📊 Vue d'ensemble | KPIs, distribution des résultats, buts par saison, xG vs buts réels |
| 🏟️ Équipes | Classement des points cumulés, radar de profil par équipe |
| 📈 Descriptives | Tendance centrale, dispersion, boxplots, courbe de Lorenz & Gini |
| 🔗 Corrélations | Heatmap de Pearson, explorateur de relations |
| 🧭 ACP | Éboulis, cercle des corrélations, plan factoriel des matchs |
| 🎯 Clustering | K-Means (coude + silhouette), dendrogramme CAH |
| ⏱️ Séries temporelles | Série mensuelle des buts, moyenne mobile, décomposition |
| 🔮 Prédiction | Régression logistique (V/N/D), matrice de confusion, importance des variables, simulateur de match |

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run dashboard.py
```

Le dashboard s'ouvre sur `http://localhost:8501`. Le fichier `df_pretraite.csv`
(dataset nettoyé, généré par le notebook `projet AD propre.ipynb`) doit être
dans le même dossier que `dashboard.py`.

## Structure du projet

- `projet AD propre.ipynb` — notebook complet : prétraitement, analyse descriptive, ACP, clustering, séries temporelles
- `dashboard.py` — dashboard Streamlit reprenant toutes les analyses en version interactive
- `understat_match_1524.csv` — données brutes
- `df_pretraite.csv` — données nettoyées et enrichies (features engineering)
- `requirements.txt` — dépendances Python
