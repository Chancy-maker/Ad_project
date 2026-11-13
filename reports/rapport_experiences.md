# Rapport d'expériences — AI Advertising Inventory Optimizer

> Généré automatiquement. Tous les chiffres proviennent de l'entraînement réel ; aucune donnée TF1+ n'est utilisée.

## 1. Module A — Prévision des requêtes vidéo à 24 h

- Données : **Substitut SYNTHÉTIQUE d'Internos (fichier réel absent)** (version 1940828a626d, 89372 lignes)
- Split temporal : train ≤ 2017-11-06, validation ≤ 2018-03-31, test ≥ 2018-04-01 (62648/13356/13368)
- Modèle retenu (WAPE validation) : **lightgbm** — version v20261112-230602, seed 42
- Couverture de l'intervalle 80% sur le test : 77.2%

| Modèle | MAE test | RMSE test | WAPE test | WAPE val | Gain WAPE vs naïve |
|---|---|---|---|---|---|
| naive | 18.292 | 50.750 | 0.679 | 0.675 | 0.000 |
| ridge | 20.017 | 77.485 | 0.743 | 1.171 | -0.094 |
| random_forest | 12.729 | 36.940 | 0.472 | 0.472 | 0.304 |
| hist_gb | 12.406 | 36.233 | 0.460 | 0.459 | 0.322 |
| lightgbm | 12.301 | 35.800 | 0.456 | 0.459 | 0.328 |

### Erreur par type de vidéo

| video_type | n | volume | mae | wape | bias |
|---|---|---|---|---|---|
| lecture | 4247 | 139986.000 | 14.583 | 0.442 | -0.194 |
| tutorial | 3111 | 88960.000 | 13.134 | 0.459 | -0.181 |
| event | 1523 | 61007.000 | 18.431 | 0.460 | -0.178 |
| seminar | 2719 | 46760.000 | 8.214 | 0.478 | -0.186 |
| other | 1768 | 23531.000 | 6.359 | 0.478 | -0.177 |

### Erreur par quantile de volume observé

| volume_bin | n | volume | mae | wape | bias |
|---|---|---|---|---|---|
| Q5 | 2674 | 265108.000 | 42.497 | 0.429 | -0.285 |
| Q4 | 2673 | 56750.000 | 9.631 | 0.454 | -0.017 |
| Q3 | 2674 | 24443.000 | 4.783 | 0.523 | 0.064 |
| Q2 | 2673 | 11174.000 | 2.737 | 0.655 | 0.302 |
| Q1 | 2674 | 2769.000 | 1.852 | 1.789 | 1.606 |

### Importance des variables (permutation)

| feature | importance | std |
|---|---|---|
| requests_one_week | 13.772 | 0.183 |
| requests_two_week | 3.025 | 0.142 |
| day_of_week | 2.196 | 0.207 |
| days_published | 1.698 | 0.049 |
| trend_week | 0.598 | 0.018 |
| requests_one_day | 0.425 | 0.054 |
| next_holidays | 0.172 | 0.029 |
| votes_in_section | 0.107 | 0.021 |
| comments | 0.073 | 0.011 |
| requests_two_days | 0.064 | 0.033 |
| comments_in_section | 0.053 | 0.018 |
| ratio_day_week | 0.041 | 0.007 |

### Sensibilité du stock éligible total (test) aux hypothèses de conversion

Lignes : opportunités par lecture ; colonnes : taux d'éligibilité.

| ad_opportunities_per_view | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|
| 1.000 | 175946.000 | 205270.000 | 234594.000 | 263919.000 | 293243.000 |
| 1.500 | 263919.000 | 307905.000 | 351891.000 | 395878.000 | 439864.000 |
| 2.000 | 351891.000 | 410540.000 | 469189.000 | 527837.000 | 586486.000 |
| 2.500 | 439864.000 | 513175.000 | 586486.000 | 659797.000 | 733107.000 |
| 3.000 | 527837.000 | 615810.000 | 703783.000 | 791756.000 | 879729.000 |

## 2. Module B — Risque de sous-impression

- Données : Synthétique (contrats/livraisons) + profils paramétrique (Criteo absent) (version 8eeb1cc60893), prévalence 30.5%, tolérance 0.98
- Split par campagne : 2100/450/450 campagnes (train/val/test)
- Modèle retenu (PR-AUC validation) : **random_forest** — seuil 0.316 (recall_target, rappel cible 0.8) — version v20261112-230619

| Modèle | Recall | Precision | F1 | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| dummy | 0.000 | 0.000 | 0.000 | 0.313 | 0.500 | 0.215 |
| pacing_rule | 0.806 | 0.626 | 0.705 | 0.742 | 0.880 | 0.134 |
| logistic_regression | 0.785 | 0.619 | 0.692 | 0.775 | 0.879 | 0.129 |
| random_forest | 0.820 | 0.661 | 0.732 | 0.819 | 0.904 | 0.115 |
| hist_gb | 0.780 | 0.645 | 0.706 | 0.809 | 0.893 | 0.122 |
| lightgbm | 0.783 | 0.661 | 0.716 | 0.813 | 0.898 | 0.119 |

Lot de campagnes généré séparément : recall 0.801, precision 0.654, PR-AUC 0.800, Brier 0.114.

### Performance par instant de suivi

| snapshot_pct | n | prevalence | recall | precision | pr_auc |
|---|---|---|---|---|---|
| 0.250 | 450.000 | 0.313 | 0.766 | 0.565 | 0.703 |
| 0.500 | 450.000 | 0.313 | 0.787 | 0.665 | 0.823 |
| 0.750 | 450.000 | 0.313 | 0.908 | 0.766 | 0.895 |

### Performance par type de campagne

| campaign_type | n | prevalence | recall | precision | pr_auc |
|---|---|---|---|---|---|
| awareness | 417 | 0.317 | 0.841 | 0.665 | 0.821 |
| branding | 447 | 0.309 | 0.790 | 0.690 | 0.799 |
| performance | 486 | 0.315 | 0.830 | 0.635 | 0.843 |

### Performance par scénario du générateur

| scenario | n | prevalence | recall | precision | pr_auc |
|---|---|---|---|---|---|
| ambitious | 150 | 0.500 | 0.973 | 0.652 | 0.901 |
| easy | 126 | 0.000 | 0.000 | 0.000 | nan |
| inventory_drop | 219 | 0.671 | 0.578 | 0.966 | 0.948 |
| irregular | 129 | 0.465 | 0.883 | 0.646 | 0.813 |
| narrow_targeting | 186 | 0.226 | 0.881 | 0.481 | 0.637 |
| normal | 399 | 0.000 | 0.000 | 0.000 | nan |
| slow_start | 141 | 0.702 | 1.000 | 0.723 | 0.893 |

### Comparaison de seuils d'alerte (test)

| threshold | recall | precision | f1 | alert_rate |
|---|---|---|---|---|
| 0.100 | 0.981 | 0.484 | 0.648 | 0.635 |
| 0.200 | 0.917 | 0.557 | 0.693 | 0.516 |
| 0.300 | 0.830 | 0.639 | 0.722 | 0.407 |
| 0.400 | 0.733 | 0.721 | 0.727 | 0.319 |
| 0.500 | 0.671 | 0.768 | 0.716 | 0.274 |
| 0.600 | 0.596 | 0.840 | 0.697 | 0.222 |
| 0.700 | 0.468 | 0.884 | 0.612 | 0.166 |
| 0.800 | 0.333 | 0.940 | 0.492 | 0.111 |
| 0.900 | 0.137 | 0.967 | 0.240 | 0.044 |

### Calibration (test)

| mean_predicted | fraction_positive |
|---|---|
| 0.000 | 0.000 |
| 0.002 | 0.000 |
| 0.026 | 0.037 |
| 0.090 | 0.067 |
| 0.170 | 0.193 |
| 0.253 | 0.244 |
| 0.362 | 0.378 |
| 0.544 | 0.474 |
| 0.733 | 0.800 |
| 0.896 | 0.941 |

### Importance des variables (permutation)

| feature | importance | std |
|---|---|---|
| recent_to_required_rate_ratio | 0.051 | 0.010 |
| projected_capacity_ratio | 0.039 | 0.005 |
| avg_to_required_rate_ratio | 0.018 | 0.004 |
| progress_gap | 0.014 | 0.003 |
| inventory_trend_ratio | 0.007 | 0.004 |
| delivery_progress_rate | 0.006 | 0.002 |
| required_daily_rate | 0.006 | 0.006 |
| observed_capture_rate | 0.004 | 0.001 |
| elapsed_days | 0.003 | 0.002 |
| n_allowed_categories | 0.002 | 0.002 |
| snapshot_pct | 0.002 | 0.003 |
| daily_cap_to_required_ratio | 0.002 | 0.001 |

## 3. Historique des expériences

| Date | Module | Version | Retenu | Seed |
|---|---|---|---|---|
| 2026-11-12T22:57:16 | inventory | v20261112-225716 | lightgbm | 42 |
| 2026-11-12T22:58:45 | under_delivery | v20261112-225845 | random_forest | 42 |
| 2026-11-12T23:06:02 | inventory | v20261112-230602 | lightgbm | 42 |
| 2026-11-12T23:06:19 | under_delivery | v20261112-230619 | random_forest | 42 |

## 4. Limites

- Résultats obtenus sur données publiques/synthétiques : ils illustrent la méthode, pas des performances opérationnelles.
- Internos : cible = requêtes vidéo (pas des impressions) ; Criteo : display, pas vidéo instream ; contrats et livraisons : synthétiques.
