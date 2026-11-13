"""Page 3 — Risque de sous-impression."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve

from ...under_delivery.evaluation import classification_metrics, risk_class
from ...under_delivery.predict import local_explanation, recommend, score_snapshots
from ..ui import (
    BLUE,
    GREY,
    ORANGE,
    RISK_COLORS,
    fmt,
    get_model,
    get_paths,
    need_training,
    pct,
    read_parquet,
    style,
    synthetic_banner,
)

RISKS = ["faible", "modéré", "élevé"]


@st.cache_data(show_spinner="Calcul des scores…")
def _score(version: str, root: str, mtime: float, _bundle) -> pd.DataFrame:
    from pathlib import Path
    snaps = pd.read_parquet(Path(root) / "data" / "synthetic" / "snapshots.parquet")
    return score_snapshots(snaps, _bundle)


def render() -> None:
    st.title("Risque de sous-impression")
    paths = get_paths()
    bundle, meta = get_model("under_delivery")
    f = paths.data_synthetic / "snapshots.parquet"
    if bundle is None or not f.exists():
        need_training("Under-delivery")
        return
    synthetic_banner(False, "campagnes, contrats et livraisons entièrement synthétiques")
    daily = read_parquet(paths.data_synthetic / "synthetic_campaign_delivery_daily.parquet")
    scored = _score(meta["version"], str(paths.root), f.stat().st_mtime, bundle)
    test_ids = set(meta["split"]["test_campaign_ids"])
    val_ids = set(meta["split"]["val_campaign_ids"])
    scored["jeu"] = np.where(scored["campaign_id"].isin(test_ids), "test",
                             np.where(scored["campaign_id"].isin(val_ids), "validation", "entraînement"))

    c1, c2, c3 = st.columns(3)
    pcts = sorted(scored["snapshot_pct"].unique())
    snap = c1.selectbox("Instant de suivi", pcts, index=min(1, len(pcts) - 1),
                        format_func=lambda x: f"{int(x * 100)} % de la durée")
    thr = c2.slider("Seuil d'alerte (probabilité)", 0.05, 0.95, float(round(meta["threshold"], 2)), 0.01,
                    help="Choisi sur la validation pour viser le rappel cible ; ajustable avec les métiers.")
    risks = c3.multiselect("Niveaux de risque", RISKS, default=RISKS)
    jeux = st.multiselect("Jeux de campagnes affichés", ["test", "validation", "entraînement"], default=["test"],
                          help="Seul le jeu de test mesure la performance ; les autres ont servi à entraîner/choisir.")

    b = meta["risk_bins"]
    view = scored[scored["snapshot_pct"] == snap].copy()
    view["risk_class"] = risk_class(view["proba"], b["low_max"], b["high_min"])
    view["alerte"] = (view["proba"] >= thr).astype(int)
    view = view[view["risk_class"].isin(risks) & view["jeu"].isin(jeux)]

    k = st.columns(4)
    k[0].metric("Campagnes affichées", fmt(len(view)))
    k[1].metric("Alertes (≥ seuil)", fmt(int(view["alerte"].sum())))
    k[2].metric("Risque élevé", fmt(int((view["risk_class"] == "élevé").sum())))
    k[3].metric("Probabilité moyenne", pct(view["proba"].mean()) if len(view) else "—")

    table = view[["campaign_id", "risk_class", "proba", "booked_impressions", "delivered_so_far",
                  "remaining_impressions", "remaining_days", "required_daily_rate", "alerte", "jeu"]].rename(columns={
        "campaign_id": "Campagne", "risk_class": "Risque", "proba": "Probabilité", "booked_impressions": "Objectif",
        "delivered_so_far": "Livré", "remaining_impressions": "Restant", "remaining_days": "Jours restants",
        "required_daily_rate": "Rythme nécessaire / j", "alerte": "Alerte", "jeu": "Jeu"}
    ).sort_values("Probabilité", ascending=False)
    st.dataframe(table, hide_index=True, width="stretch", height=260,
                 column_config={"Probabilité": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                                "Objectif": st.column_config.NumberColumn(format="%d"),
                                "Livré": st.column_config.NumberColumn(format="%d"),
                                "Restant": st.column_config.NumberColumn(format="%d"),
                                "Rythme nécessaire / j": st.column_config.NumberColumn(format="%d")})
    st.download_button("Exporter les alertes (CSV)",
                       table[table["Alerte"] == 1].to_csv(index=False).encode("utf-8"),
                       file_name="alertes_sous_impression.csv", mime="text/csv")

    if len(view) == 0:
        st.info("Aucune campagne ne correspond aux filtres.")
        return

    st.subheader("Détail d'une campagne")
    cid = st.selectbox("Campagne", table["Campagne"].tolist())
    row = view[view["campaign_id"] == cid].iloc[0]
    d1, d2 = st.columns([1, 1])
    with d1:
        m = st.columns(3)
        m[0].metric("Probabilité", pct(row["proba"]))
        m[1].metric("Risque", str(row["risk_class"]).capitalize())
        m[2].metric("Alerte", "Oui" if row["alerte"] else "Non")
        m = st.columns(3)
        m[0].metric("Objectif contractuel", fmt(row["booked_impressions"]))
        m[1].metric("Déjà diffusé", fmt(row["delivered_so_far"]))
        m[2].metric("Restant", fmt(row["remaining_impressions"]))
        m = st.columns(2)
        m[0].metric("Jours restants", fmt(row["remaining_days"]))
        m[1].metric("Rythme quotidien nécessaire", fmt(row["required_daily_rate"]))
        st.markdown("**Recommandation (règles transparentes)**")
        st.info(recommend(row, row["proba"], row["risk_class"]))
    with d2:
        exp = local_explanation(bundle, view[view["campaign_id"] == cid].iloc[[0]])
        exp = exp.iloc[::-1]
        fig = go.Figure(go.Bar(x=exp["contribution"], y=exp["variable"], orientation="h",
                               marker_color=[ORANGE if c > 0 else BLUE for c in exp["contribution"]]))
        fig.update_xaxes(title="Effet sur la probabilité (vs valeur de référence)")
        st.plotly_chart(style(fig, 300, "Principaux facteurs du score"), width="stretch")
        st.caption("Explication par occlusion : écart de probabilité si la variable prenait sa valeur de "
                   "référence (médiane du jeu d'entraînement). Orange = augmente le risque.")

    dd = daily[daily["campaign_id"] == cid].sort_values("day_index")
    D = len(dd)
    e = int(row["elapsed_days"])
    booked = float(row["booked_impressions"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd["day_index"] + 1, y=booked * (dd["day_index"] + 1) / D, name="Trajectoire idéale",
                             line=dict(color=GREY, dash="dot")))
    fig.add_trace(go.Scatter(x=dd["day_index"][:e] + 1, y=dd["cumulative_delivered_impressions"][:e],
                             name="Livré (connu au suivi)", line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=dd["day_index"][e - 1:] + 1, y=dd["cumulative_delivered_impressions"][e - 1:],
                             name="Suite simulée (inconnue au suivi)", line=dict(color=ORANGE, width=2, dash="dash")))
    fig.add_hline(y=booked, line_color="#d03b3b", annotation_text="Objectif", annotation_position="top left")
    fig.add_vline(x=e, line_color=GREY, line_dash="dot")
    fig.update_xaxes(title="Jour de campagne")
    fig.update_yaxes(title="Impressions cumulées")
    st.plotly_chart(style(fig, 340, "Progression de la livraison"), width="stretch")

    st.subheader("Vue d'ensemble du portefeuille et performance du modèle")
    g1, g2 = st.columns(2)
    with g1:
        fig = px.histogram(view, x="proba", nbins=25, color_discrete_sequence=[BLUE],
                           labels={"proba": "Probabilité de sous-impression"})
        fig.add_vline(x=thr, line_color=ORANGE, line_dash="dash", annotation_text="seuil")
        st.plotly_chart(style(fig, 320, "Distribution des probabilités"), width="stretch")
    with g2:
        cnt = view["risk_class"].value_counts().reindex(RISKS).fillna(0).reset_index()
        cnt.columns = ["Risque", "Campagnes"]
        fig = px.bar(cnt, x="Risque", y="Campagnes", color="Risque", color_discrete_map=RISK_COLORS)
        fig.update_layout(showlegend=False)
        st.plotly_chart(style(fig, 320, "Campagnes par niveau de risque"), width="stretch")

    tst = scored[scored["jeu"] == "test"]
    if len(tst):
        cm = classification_metrics(tst["under_delivery_final"], tst["proba"], thr)
        h1, h2 = st.columns(2)
        with h1:
            z = np.array(cm["confusion"])
            fig = px.imshow(z, text_auto=True, color_continuous_scale=["#ffffff", BLUE],
                            x=["Prédit : OK", "Prédit : risque"], y=["Réel : OK", "Réel : sous-impression"])
            fig.update_layout(coloraxis_showscale=False)
            fig.update_xaxes(side="top")
            st.plotly_chart(style(fig, 320, f"Matrice de confusion (test, seuil {thr:.2f})"), width="stretch")
        with h2:
            p, r, _ = precision_recall_curve(tst["under_delivery_final"], tst["proba"])
            fig = go.Figure(go.Scatter(x=r, y=p, line=dict(color=BLUE, width=2), name="Modèle"))
            fig.add_trace(go.Scatter(x=[cm["recall"]], y=[cm["precision"]], mode="markers", name="Seuil courant",
                                     marker=dict(color=ORANGE, size=11, line=dict(color="white", width=2))))
            fig.add_hline(y=cm["prevalence"], line_color=GREY, line_dash="dot", annotation_text="prévalence")
            fig.update_xaxes(title="Recall")
            fig.update_yaxes(title="Precision", range=[0, 1.02])
            st.plotly_chart(style(fig, 320, f"Courbe Precision-Recall (test, PR-AUC {cm['pr_auc']:.3f})"),
                            width="stretch")
        st.caption(f"Au seuil {thr:.2f} (tous instants de suivi, jeu de test) : recall {pct(cm['recall'])}, "
                   f"precision {pct(cm['precision'])}. Abaisser le seuil augmente le recall mais aussi les fausses alertes.")
