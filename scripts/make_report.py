"""Génère reports/rapport_experiences.md à partir des métriques RÉELLEMENT obtenues (rien d'inventé)."""
import _bootstrap  # noqa: F401
import pandas as pd

from ad_optimizer.common.config import default_paths
from ad_optimizer.common.model_registry import active_metadata, read_experiments


def table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def csv(paths, name):
    f = paths.reports_metrics / name
    return pd.read_csv(f) if f.exists() else None


def main() -> None:
    paths = default_paths().ensure()
    a, b = active_metadata("inventory", paths), active_metadata("under_delivery", paths)
    if not a or not b:
        raise SystemExit("Entraînez d'abord les deux modules (scripts/bootstrap_demo.py).")
    out = ["# Rapport d'expériences — AI Advertising Inventory Optimizer", "",
           "> Généré automatiquement. Tous les chiffres proviennent de l'entraînement réel ; "
           "aucune donnée TF1+ n'est utilisée.", ""]
    out += ["## 1. Module A — Prévision des requêtes vidéo à 24 h", "",
            f"- Données : **{a['data_source']}** (version {a['data_version']}, {a['n_rows']} lignes)",
            f"- Split {a['split']['type']}"
            + (f" : train ≤ {a['split']['train_end']}, validation ≤ {a['split']['val_end']}, test ≥ {a['split']['test_start']}"
               if a['split']['train_end'] else " (découpage fourni avec les données)")
            + f" ({a['split']['n_train']}/{a['split']['n_val']}/{a['split']['n_test']})",
            f"- Modèle retenu (WAPE validation) : **{a['selected_model']}** — version {a['version']}, seed {a['seed']}",
            f"- Couverture de l'intervalle {a['interval_nominal']:.0%} sur le test : {a['interval_coverage_test']:.1%}", ""]
    rows = [{"Modèle": k, "MAE test": v["test"]["mae"], "RMSE test": v["test"]["rmse"], "WAPE test": v["test"]["wape"],
             "WAPE val": v["val"]["wape"], "Gain WAPE vs naïve": v.get("gain_vs_naive_wape") or 0.0}
            for k, v in a["metrics"].items()]
    out += [table(pd.DataFrame(rows)), ""]
    for title, name in (("Erreur par type de vidéo", "inventory_error_by_type.csv"),
                        ("Erreur par quantile de volume observé", "inventory_error_by_volume.csv"),
                        ("Importance des variables (permutation)", "inventory_feature_importance.csv")):
        t = csv(paths, name)
        if t is not None:
            out += [f"### {title}", "", table(t.head(12)), ""]
    t = csv(paths, "inventory_conversion_sensitivity.csv")
    if t is not None:
        piv = t.pivot(index="ad_opportunities_per_view", columns="eligibility_rate", values="eligible_impressions")
        piv = piv.round(0).astype(int).reset_index()
        piv.columns = [str(c) for c in piv.columns]
        out += ["### Sensibilité du stock éligible total (test) aux hypothèses de conversion", "",
                "Lignes : opportunités par lecture ; colonnes : taux d'éligibilité.", "", table(piv), ""]

    out += ["## 2. Module B — Risque de sous-impression", "",
            f"- Données : {b['data_source']} (version {b['data_version']}), prévalence {b['prevalence']:.1%}, "
            f"tolérance {b['tolerance']}",
            f"- Split par campagne : {b['split']['n_train_campaigns']}/{b['split']['n_val_campaigns']}/"
            f"{b['split']['n_test_campaigns']} campagnes (train/val/test)",
            f"- Modèle retenu (PR-AUC validation) : **{b['selected_model']}** — seuil {b['threshold']:.3f} "
            f"({b['threshold_strategy']['strategy']}, rappel cible {b['threshold_strategy']['recall_target']}) "
            f"— version {b['version']}", ""]
    rows = [{"Modèle": k, "Recall": v["test"]["recall"], "Precision": v["test"]["precision"], "F1": v["test"]["f1"],
             "PR-AUC": v["test"]["pr_auc"], "ROC-AUC": v["test"]["roc_auc"], "Brier": v["test"]["brier"]}
            for k, v in b["metrics"].items()]
    ho = b["holdout_batch"]
    out += [table(pd.DataFrame(rows)), "",
            f"Lot de campagnes généré séparément : recall {ho['recall']:.3f}, precision {ho['precision']:.3f}, "
            f"PR-AUC {ho['pr_auc']:.3f}, Brier {ho['brier']:.3f}.", ""]
    for title, name in (("Performance par instant de suivi", "under_delivery_by_snapshot.csv"),
                        ("Performance par type de campagne", "under_delivery_by_campaign_type.csv"),
                        ("Performance par scénario du générateur", "under_delivery_by_scenario.csv"),
                        ("Comparaison de seuils d'alerte (test)", "under_delivery_threshold_sweep.csv"),
                        ("Calibration (test)", "under_delivery_calibration.csv"),
                        ("Importance des variables (permutation)", "under_delivery_feature_importance.csv")):
        t = csv(paths, name)
        if t is not None:
            out += [f"### {title}", "", table(t.head(12)), ""]

    out += ["## 3. Historique des expériences", ""]
    exps = read_experiments(paths)
    out += [table(pd.DataFrame([{"Date": e["logged_at"], "Module": e["kind"], "Version": e["version"],
                                 "Retenu": e["selected"], "Seed": e["seed"]} for e in exps]))
            if exps else "Aucune.", ""]
    out += ["## 4. Limites", "",
            "- Résultats obtenus sur données publiques/synthétiques : ils illustrent la méthode, pas des performances opérationnelles.",
            "- Internos : cible = requêtes vidéo (pas des impressions) ; Criteo : display, pas vidéo instream ; "
            "contrats et livraisons : synthétiques.", ""]
    dest = paths.root / "reports" / "rapport_experiences.md"
    dest.write_text("\n".join(out), encoding="utf-8")
    print("Rapport écrit :", dest)


if __name__ == "__main__":
    main()
