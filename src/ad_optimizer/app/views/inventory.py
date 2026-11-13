"""Page 2 — Prévision de l'inventaire."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ...common.schemas import ConversionAssumptions
from ...inventory.predict import estimate_inventory, predict_requests
from ..ui import (
    AQUA,
    BLUE,
    GREY,
    ORANGE,
    fmt,
    get_model,
    get_paths,
    need_training,
    pct,
    read_parquet,
    style,
    synthetic_banner,
)


def render() -> None:
    st.title("Prévision de l'inventaire")
    paths = get_paths()
    bundle, meta = get_model("inventory")
    df = read_parquet(paths.data_processed / "internos_clean.parquet")
    if bundle is None or df is None:
        need_training("Inventory")
        return
    te = read_parquet(paths.data_processed / "inventory_test_predictions.parquet")
    synthetic_banner(meta.get("is_real_data", False), meta["data_source"])
    st.caption("Horizon : requêtes vidéo des **24 heures suivant la date T** (seul horizon disponible dans "
               "Internos). Le stock publicitaire est une **estimation dérivée d'hypothèses configurables**.")

    if meta.get("target_normalised"):
        st.info("ℹ️ La cible des fichiers fournis est **normalisée (0–1)** : requêtes et stock sont exprimés en "
                "**unités relatives**, pas en nombre réel de requêtes. Pour des volumes réels, renseigner "
                "`internos_target_max_requests` dans `configs/inventory.yaml` (valeur max d'origine, voir la fiche Mendeley).")
    st.subheader("Hypothèses de conversion (simulées)")
    d = meta["conversion_defaults"]
    h1, h2 = st.columns(2)
    opps = h1.slider("Opportunités publicitaires par lecture", 0.5, 6.0, float(d["ad_opportunities_per_view"]), 0.1)
    elig = h2.slider("Taux d'éligibilité", 0.3, 1.0, float(d["eligibility_rate"]), 0.01)
    assum = ConversionAssumptions(ad_opportunities_per_view=opps, eligibility_rate=elig)
    st.caption("impressions éligibles estimées = requêtes prévues × opportunités par lecture × taux d'éligibilité. "
               "Ces deux paramètres sont des hypothèses de simulation, pas des paramètres réels.")

    has_media = meta.get("has_media_id", True)
    f1, f2, f3 = st.columns([1, 1, 1])
    cats = ["Toutes"] + sorted(df["content_category"].dropna().unique())
    cat = f1.selectbox("Catégorie" if has_media else "Type de vidéo", cats)
    sub = df if cat == "Toutes" else df[df["content_category"] == cat]
    jeu_sel = None
    if has_media:
        top = sub.groupby("media_id")["requests"].sum().sort_values(ascending=False).head(200).index.tolist()
        media = f2.selectbox("Contenu (200 plus consultés)", top)
        rows = df[df["media_id"] == media].sort_values("date")
    else:
        jeu_sel = f2.selectbox("Jeu de données", ["test", "val", "train"],
                               format_func={"test": "test", "val": "validation", "train": "entraînement"}.get)
        media = f"agrege_{cat}_{jeu_sel}".replace(" ", "_")
        rows = sub[sub["split"] == jeu_sel].sort_values("date") if "split" in sub else sub.sort_values("date")
        st.caption("Les données fournies n'ont pas d'identifiant de contenu : les courbes agrègent toutes les "
                   "observations du jeu et du type choisis, par jour de l'année (date reconstruite, année fictive).")
    f3.selectbox("Horizon", ["24 heures"], help="Seul horizon disponible dans Internos.")
    if len(rows) == 0:
        st.info("Aucune observation pour ce filtre.")
        return

    pred = predict_requests(rows, bundle)
    inv = estimate_inventory(pred, assum)
    row_wape = ((pred["y_pred"] - rows["requests"]).abs().sum() / rows["requests"].sum()
                if rows["requests"].sum() > 0 else float("nan"))
    out = pd.concat([rows[["media_id", "date", "content_category", "video_type", "requests"]], inv], axis=1)
    sp = meta["split"]
    if not has_media:
        sums = [c for c in out.columns if c in ("requests", "y_pred", "y_low", "y_high", "theoretical_ad_opportunities",
                                                 "eligible_impressions", "eligible_impressions_low",
                                                 "eligible_impressions_high")]
        out = out.groupby("date", as_index=False)[sums].sum()
        out["jeu"] = {"test": "test", "val": "validation", "train": "entraînement"}[jeu_sel]
    elif sp["train_end"]:
        out["jeu"] = ["entraînement" if t <= pd.Timestamp(sp["train_end"]) else
                      "validation" if t <= pd.Timestamp(sp["val_end"]) else "test" for t in out["date"]]
    else:
        out["jeu"] = "n/a"

    k = st.columns(4)
    k[0].metric("Requêtes prévues (somme)", fmt(out["y_pred"].sum()))
    k[1].metric("Opportunités théoriques", fmt(out["theoretical_ad_opportunities"].sum()))
    k[2].metric("Stock éligible estimé", fmt(out["eligible_impressions"].sum()))
    tcont = out[out["jeu"] == "test"]
    k[3].metric("Erreur (WAPE) sur la période de test", pct(row_wape) if len(tcont) else "—")

    st.subheader("Historique vs prévision")
    fig = go.Figure()
    if "y_low" in out:
        fig.add_trace(go.Scatter(x=out["date"], y=out["y_high"], line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=out["date"], y=out["y_low"], fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(42,120,214,0.15)", name="Intervalle 10–90 %", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=out["date"], y=out["requests"], name="Observé (24 h suivantes)",
                             line=dict(color=GREY, width=2)))
    fig.add_trace(go.Scatter(x=out["date"], y=out["y_pred"], name="Prévision", line=dict(color=BLUE, width=2)))
    if len(tcont) and has_media:
        fig.add_vrect(x0=tcont["date"].min(), x1=out["date"].max(), fillcolor="#f4efe6", opacity=0.5,
                      line_width=0, annotation_text="période de test", annotation_position="top left")
    fig.update_yaxes(title="Requêtes vidéo / 24 h")
    st.plotly_chart(style(fig), width="stretch")
    if has_media:
        st.caption("Hors période de test, la prévision est évaluée sur des données vues à l'entraînement : "
                   "seule la zone de test mesure la performance réelle du modèle.")

    st.subheader("Stock publicitaire estimé par jour")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=out["date"], y=out["eligible_impressions"], name="Impressions éligibles estimées",
                         marker_color=AQUA))
    fig.add_trace(go.Scatter(x=out["date"], y=out["theoretical_ad_opportunities"], name="Opportunités théoriques",
                             line=dict(color=ORANGE, width=2)))
    st.plotly_chart(style(fig, 320), width="stretch")

    if te is not None:
        st.subheader("Résultats sur le jeu de test")
        scope = te if cat == "Toutes" else te[te["content_category"] == cat]
        c1, c2 = st.columns(2)
        with c1:
            g = (te.groupby("content_category")[["y_true", "y_pred"]].sum().reset_index())
            g["eligible"] = g["y_pred"] * assum.ad_opportunities_per_view * assum.eligibility_rate
            fig = px.bar(g, x="content_category", y="eligible", color_discrete_sequence=[BLUE],
                         labels={"content_category": "Catégorie", "eligible": "Impressions éligibles estimées"})
            st.plotly_chart(style(fig, 340, "Stock éligible estimé par catégorie (test)"), width="stretch")
        with c2:
            s = scope.sample(min(3000, len(scope)), random_state=0)
            fig = px.scatter(s, x="y_true", y="y_pred", opacity=0.35, log_x=True, log_y=True,
                             color_discrete_sequence=[BLUE],
                             labels={"y_true": "Observé", "y_pred": "Prévu"})
            mx = float(max(s["y_true"].max(), s["y_pred"].max(), 1))
            fig.add_trace(go.Scatter(x=[1, mx], y=[1, mx], mode="lines", line=dict(color=ORANGE, dash="dash"),
                                     name="y = x"))
            st.plotly_chart(style(fig, 340, "Prévision vs observé — dispersion (test)"), width="stretch")

    with st.expander("Facteurs ayant contribué à la prévision (importance par permutation)"):
        imp = pd.DataFrame(meta["importance"]).head(10).iloc[::-1]
        fig = px.bar(imp, x="importance", y="feature", orientation="h", color_discrete_sequence=[BLUE],
                     labels={"importance": "Hausse de MAE si la variable est permutée", "feature": ""})
        st.plotly_chart(style(fig, 340), width="stretch")
        st.caption("Importance prédictive ≠ causalité : ces variables sont corrélées à la demande, "
                   "elles n'en sont pas démontrées comme cause.")

    st.download_button("Exporter les prévisions (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name=f"previsions_inventaire_{media}.csv", mime="text/csv")
