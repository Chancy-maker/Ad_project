"""Page 4 — Entraînement et évaluation (déclenchement explicite uniquement)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ...common.model_registry import active_metadata, list_versions, read_experiments
from ...inventory import train as inv_train
from ...under_delivery import train as ud_train
from ..ui import BLUE, GREY, ORANGE, fmt, get_paths, pct, style

NAMES_A = {"naive": "Baseline naïve", "ridge": "Ridge", "random_forest": "Random Forest",
           "hist_gb": "HistGradientBoosting", "lightgbm": "LightGBM"}
NAMES_B = {"dummy": "Dummy (majoritaire)", "pacing_rule": "Règle de rythme (métier)",
           "logistic_regression": "Régression logistique", "random_forest": "Random Forest",
           "hist_gb": "HistGradientBoosting", "lightgbm": "LightGBM"}


def _clear():
    st.cache_data.clear()
    st.cache_resource.clear()


def _inventory_section(paths):
    st.subheader("Module A — Inventory Forecasting")
    c1, c2 = st.columns([2, 1])
    models = c1.multiselect("Modèles candidats", list(NAMES_A), default=list(NAMES_A),
                            format_func=NAMES_A.get, key="inv_models")
    seed = c2.number_input("Graine aléatoire", 0, 10_000, 42, key="inv_seed")
    proxy = st.checkbox("Forcer le substitut synthétique (ignorer data/raw/internos.csv)", key="inv_proxy")
    if st.button("Lancer l'entraînement — Inventaire", type="primary", disabled=not models):
        with st.status("Entraînement en cours…", expanded=True) as s:
            inv_train.run(paths, models=models, seed=int(seed), force_proxy=proxy, progress=st.write)
            s.update(label="Terminé", state="complete")
        _clear()
        st.rerun()
    m = active_metadata("inventory", paths)
    if not m:
        st.info("Aucun modèle entraîné.")
        return
    st.caption(f"Version active **{m['version']}** — entraîné le {m['trained_at'].replace('T', ' ')} — "
               f"split {m['split']['type']}"
               + (f" (train ≤ {m['split']['train_end']}, test ≥ {m['split']['test_start']})"
                  if m['split']['train_end'] else " (découpage fourni avec les données)") + " — "
               f"{m['split']['n_train']}/{m['split']['n_val']}/{m['split']['n_test']} lignes train/val/test — "
               f"données : {m['data_source']} (v. {m['data_version']}).")
    rows = [{"Modèle": NAMES_A.get(k, k), "MAE (test)": v["test"]["mae"], "RMSE (test)": v["test"]["rmse"],
             "WAPE (test)": v["test"]["wape"], "WAPE (validation)": v["val"]["wape"],
             "Gain WAPE vs naïve": v.get("gain_vs_naive_wape"), "Sélectionné": "✓" if k == m["selected_model"] else ""}
            for k, v in m["metrics"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                ["MAE (test)", "RMSE (test)", "WAPE (test)", "WAPE (validation)", "Gain WAPE vs naïve"]})
    fig = go.Figure(go.Bar(x=[r["Modèle"] for r in rows], y=[r["WAPE (test)"] for r in rows],
                           marker_color=[ORANGE if r["Sélectionné"] else (GREY if r["Modèle"] == NAMES_A["naive"] else BLUE)
                                         for r in rows]))
    fig.update_yaxes(title="WAPE (test) — plus bas = mieux")
    st.plotly_chart(style(fig, 300, "Comparaison à la baseline (WAPE sur le jeu de test)"), width="stretch")
    st.caption(f"Sélection sur la **validation** (WAPE) : le test n'intervient jamais dans le choix. "
               f"Couverture de l'intervalle {pct(m['interval_nominal'], 0)} nominal : {pct(m['interval_coverage_test'])} (test).")


def _under_delivery_section(paths):
    st.subheader("Module B — Under-delivery Prediction")
    c1, c2, c3 = st.columns([2, 1, 1])
    models = c1.multiselect("Modèles candidats", list(NAMES_B), default=list(NAMES_B),
                            format_func=NAMES_B.get, key="ud_models")
    seed = c2.number_input("Graine aléatoire", 0, 10_000, 42, key="ud_seed")
    n = c3.number_input("Campagnes synthétiques", 200, 10_000, 3000, 100, key="ud_n")
    if st.button("Générer les données et lancer l'entraînement — Sous-impression", type="primary",
                 disabled=not models):
        with st.status("Génération des campagnes et entraînement…", expanded=True) as s:
            ud_train.run(paths, models=models, seed=int(seed), n_campaigns=int(n), progress=st.write)
            s.update(label="Terminé", state="complete")
        _clear()
        st.rerun()
    m = active_metadata("under_delivery", paths)
    if not m:
        st.info("Aucun modèle entraîné.")
        return
    sp = m["split"]
    st.caption(f"Version active **{m['version']}** — entraîné le {m['trained_at'].replace('T', ' ')} — split par "
               f"campagne ({sp['n_train_campaigns']}/{sp['n_val_campaigns']}/{sp['n_test_campaigns']} campagnes "
               f"train/val/test) — prévalence de sous-impression {pct(m['prevalence'])} — seuil {m['threshold']:.2f} "
               f"({m['threshold_strategy']['strategy']}, rappel cible {m['threshold_strategy']['recall_target']}).")
    rows = [{"Modèle": NAMES_B.get(k, k), "Recall": v["test"]["recall"], "Precision": v["test"]["precision"],
             "F1": v["test"]["f1"], "PR-AUC": v["test"]["pr_auc"], "ROC-AUC": v["test"]["roc_auc"],
             "Brier": v["test"]["brier"], "Sélectionné": "✓" if k == m["selected_model"] else ""}
            for k, v in m["metrics"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                ["Recall", "Precision", "F1", "PR-AUC", "ROC-AUC", "Brier"]})
    ho = m["holdout_batch"]
    st.caption(f"Lot de campagnes généré séparément (autre graine, {fmt(ho['n'])} snapshots) : recall "
               f"{pct(ho['recall'])}, precision {pct(ho['precision'])}, PR-AUC {ho['pr_auc']:.3f}.")
    fig = go.Figure(go.Bar(x=[r["Modèle"] for r in rows], y=[r["PR-AUC"] for r in rows],
                           marker_color=[ORANGE if r["Sélectionné"] else (GREY if "Dummy" in r["Modèle"] else BLUE)
                                         for r in rows]))
    fig.update_yaxes(title="PR-AUC (test) — plus haut = mieux")
    st.plotly_chart(style(fig, 300, "Comparaison aux baselines (PR-AUC sur le jeu de test)"), width="stretch")
    imp = pd.DataFrame(m["importance"]).head(10).iloc[::-1]
    fig = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"], orientation="h", marker_color=BLUE))
    fig.update_xaxes(title="Baisse de PR-AUC si la variable est permutée (validation)")
    st.plotly_chart(style(fig, 330, "Importance des variables"), width="stretch")


def render() -> None:
    st.title("Entraînement et évaluation")
    paths = get_paths()
    st.caption("L'entraînement n'est jamais lancé au chargement de la page : uniquement sur action explicite. "
               "Les deux modules sont entraînés séparément (cibles et métriques différentes).")
    tab_a, tab_b, tab_h = st.tabs(["Inventaire", "Sous-impression", "Historique des expériences"])
    with tab_a:
        _inventory_section(paths)
    with tab_b:
        _under_delivery_section(paths)
    with tab_h:
        exps = read_experiments(paths)
        if not exps:
            st.info("Aucune expérience enregistrée.")
        else:
            rows = [{"Date": e["logged_at"].replace("T", " "), "Module": e["kind"], "Version": e["version"],
                     "Modèle retenu": e["selected"], "Graine": e["seed"], "Version données": e.get("data_version"),
                     "Métriques": str({k: round(v, 3) for k, v in
                                       (e.get("test_wape") or e.get("test_pr_auc") or {}).items() if v is not None})}
                    for e in reversed(exps)]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Historique complet : `reports/metrics/experiments.jsonl`. "
                       f"Versions de modèles stockées — inventaire : {len(list_versions('inventory', paths))}, "
                       f"sous-impression : {len(list_versions('under_delivery', paths))}.")
