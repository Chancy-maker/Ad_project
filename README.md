# AI Advertising Inventory Optimizer

POC académique (mémoire de recherche Mastère 2 Big Data & IA) : application web de démonstration qui
illustre comment le Machine Learning peut aider une régie publicitaire vidéo à gérer son inventaire.

| Module | Question | Cible | Modèles comparés |
|---|---|---|---|
| **A — Inventory Forecasting** | Combien de requêtes vidéo dans les 24 h, et quel stock publicitaire éligible en découle ? | `requests` (Internos) | baseline naïve, Ridge, Random Forest, HistGradientBoosting, LightGBM (+ intervalles par régression quantile) |
| **B — Under-delivery Prediction** | Une campagne atteindra-t-elle son objectif d'impressions ? | `under_delivery_final` | Dummy, règle de rythme (métier), Régression logistique, Random Forest, HistGradientBoosting, LightGBM |

> **Aucune donnée interne de TF1+ n'est utilisée.** Le POC s'appuie sur des datasets publics et des
> données synthétiques ; c'est une simulation académique, pas une reproduction d'un système réel.

## Démarrage rapide (≈ 2 minutes)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -e ".[dev]"
python scripts/bootstrap_demo.py                          # données + entraînement des 2 modules
streamlit run src/ad_optimizer/app/streamlit_app.py       # ouvre http://localhost:8501
```

Alternative : lancer Streamlit directement et cliquer sur **« Initialiser la démo »** (page *Vue d'ensemble*).
L'entraînement n'est **jamais** déclenché automatiquement au chargement d'une page.

## Données réelles vs substituts (IMPORTANT pour le mémoire)

Sans fichiers réels, le POC fonctionne avec des **substituts synthétiques clairement étiquetés** dans
l'interface (bandeaux d'avertissement, page *Données et méthodologie*, statut des datasets).

| Dataset | À faire | Sans le fichier |
|---|---|---|
| **Internos** (Mendeley, CC BY 4.0) — module A | Télécharger <https://data.mendeley.com/datasets/69g55s83mn/1>. Deux formats acceptés : **(a)** `train.csv`, `validation.csv`, `test.csv` (déjà normalisés et découpés, séparés par des espaces) placés dans `data/raw/` — le découpage fourni est conservé ; **(b)** un fichier brut unique `data/raw/internos.csv` | Substitut synthétique au même schéma (`internos_proxy`) |
| **Criteo Attribution** (CC BY-NC-SA 4.0) — module B | `python scripts/download_data.py` (tente HuggingFace) ou téléchargement manuel dans `data/raw/criteo_attribution_dataset.tsv.gz` | Distributions de volume/variabilité paramétriques |
| YTLive (facultatif) | Non utilisé par défaut | — |

Puis relancer `python scripts/bootstrap_demo.py`.

**Fichiers train/validation/test (format a)** : colonnes `rep_one_day`, `reproductions`, encodages sin/cos, drapeaux `video_type_0..3`… sont renommées automatiquement. Ces fichiers n'ont ni date ni identifiant de contenu : la date est reconstruite depuis le jour de l'année (année fictive), l'application agrège donc par jour et par type, et la cible reste en **unités relatives 0–1** si son échelle d'origine n'est pas déductible (renseigner `internos_target_max_requests` dans `configs/inventory.yaml` pour des volumes réels). Impossible de vérifier l'absence de fuite entre contenus (pas d'identifiant) : à mentionner comme limite. Le loader normalise les noms de colonnes
(`Requests one week` → `requests_one_week`). **Vérifier le schéma réel** (notebook `01_inventory_eda.ipynb`) :
si le fichier ne contient ni date ni identifiant de ressource, le POC le signale (voir « Notes » dans la page
Données) et dérive une date à partir de `day_of_year` — dans ce cas, privilégier un split groupé.

## Commandes

| Action | Commande |
|---|---|
| Télécharger / vérifier les données | `python scripts/download_data.py` |
| Préparer Internos + générer campagnes synthétiques | `python scripts/generate_synthetic_data.py [--seed 42] [--n-campaigns 3000]` |
| Entraîner le module A | `python scripts/train_inventory.py [--models naive lightgbm ...]` |
| Entraîner le module B | `python scripts/train_under_delivery.py [--models ...]` |
| Tout enchaîner | `python scripts/bootstrap_demo.py` |
| Rapport d'expériences (Markdown + CSV) | `python scripts/make_report.py` → `reports/rapport_experiences.md` |
| Tests | `pytest` |
| Lint | `ruff check src tests scripts` |

Configuration centralisée dans `configs/*.yaml` (graine `random_seed`, hypothèses de conversion, tolérance de
sous-livraison, seuils, tailles). Chaque entraînement enregistre modèle, préprocesseur, métriques, split, graine
et **version des données** dans `models/<module>/<version>/` et ajoute une ligne à `reports/metrics/experiments.jsonl`.

## Structure

```
configs/            paramètres (app, inventory, under_delivery)
data/{raw,processed,synthetic}   données (non versionnées)
models/             modèles versionnés + active.json
reports/            métriques CSV, experiments.jsonl, rapport_experiences.md
notebooks/          EDA Internos et Criteo
scripts/            commandes CLI
src/ad_optimizer/
  data/             download, validation, preprocessing, synthetic_generator
  inventory/        features, train, predict, evaluation          (module A)
  under_delivery/   aggregation, snapshot_builder, train, predict, evaluation   (module B)
  common/           config, logging, model_registry, schemas, status, pipeline
  app/              streamlit_app.py, ui.py, views/ (5 pages)
tests/              43 tests dont test_no_leakage.py
```

## Choix méthodologiques

- **Deux modèles séparés**, cibles/métriques/validations différentes. Chaque modèle est comparé à une baseline.
- **Module A** : cible en `log1p` (puis `expm1`) ; split **temporel** (train < val < test) ; métriques MAE, RMSE, WAPE
  (MAPE seulement sur cible > 0) ; choix du modèle sur le WAPE de **validation** ; intervalle 10–90 % par régression
  quantile ; importance par permutation. Stock éligible = `requêtes × opportunités/lecture × taux d'éligibilité`
  (deux hypothèses **simulées**, réglables dans l'interface).
- **Module B** : snapshots à 25 / 50 / 75 % de la durée ; variables calculées **uniquement** sur les jours écoulés
  (`compute_snapshot_features` ne reçoit que des séries tronquées) ; split **par campagne** ; seuil d'alerte choisi sur la
  validation (rappel cible 0,80, modifiable) ; métriques Recall/Precision/F1/PR-AUC/ROC-AUC/Brier, matrice de confusion,
  calibration ; validation sur un **lot de campagnes généré séparément** ; recommandations = **règles déterministes**,
  explication locale par occlusion.
- **Anti-fuite** : `tests/test_no_leakage.py` corrompt le futur des séries et vérifie que les variables ne changent pas ;
  vérifie l'absence de colonnes interdites et la disjonction des splits (un test de mutation a confirmé qu'il détecte une fuite).
- **Reproductibilité** : générateurs déterministes (`random_seed`), mêmes splits pour tous les modèles.
- **Sobriété** : pas de Spark/Kafka/Kubernetes — le volume ne les justifie pas (extension PySpark possible en perspective).

## Écarts assumés par rapport au cahier des charges

- Le dossier `app/pages/` est nommé **`app/views/`** : Streamlit détecte `pages/` comme navigation multipage native
  (menu parasite et pages cassées).
- **MLflow** remplacé par un registre local léger (`model_registry.py` + `experiments.jsonl`), **SHAP** par l'importance
  par permutation et l'occlusion (méthodes agnostiques, sans dépendance), **FastAPI** non inclus (le cahier le jugeait
  optionnel ; Streamlit appelle directement les services Python), **Optuna** remplacé par une petite grille évaluée sur la validation.
- `YTLive` n'est pas intégré (facultatif).

## Limites à rappeler dans le mémoire

- **Internos** : intranet universitaire (2016–2018) ; la cible est une demande de lecture, pas des impressions ; non représentatif du marché actuel.
- **Criteo** : publicité display (pas instream vidéo), `timestamp` relatif, `cost`/`cpo` transformés, ni objectifs ni stock.
- Contrats, livraisons, éligibilité et catégories sont **synthétiques** : les performances mesurées dépendent des scénarios
  que le générateur simule et **ne sont pas des performances opérationnelles**.
- Ne pas confondre audience prévue, opportunités théoriques, inventaire éligible, réservé et diffusé.
- Observations sur le substitut : Ridge (sur cible log) est instable ; les gros volumes sont sous-estimés (biais négatif,
  conséquence de la transformation log) ; les faibles volumes ont un WAPE élevé. À analyser sur les vraies données.

## Licences et attribution

Internos : CC BY 4.0 (Mendeley Data, DOI 10.17632/69g55s83mn.1). Criteo : CC BY-NC-SA 4.0 — attribution à
Diemert, Meynet, Lefortier & Galland (2017), usage non commercial, partage à l'identique ; vérifier les conditions avant toute diffusion.
