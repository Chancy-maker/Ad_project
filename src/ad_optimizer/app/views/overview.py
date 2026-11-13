"""Page 1 — Vue d'ensemble."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ...common import pipeline
from ...common.model_registry import active_metadata
from ...common.status import dataset_status, inventory_manifest, under_delivery_manifest
from ..ui import fmt, get_paths, pct


def render() -> None:
    paths = get_paths()
    st.title("AI Advertising Inventory Optimizer")
    st.caption("POC académique — mémoire de recherche Mastère 2 Big Data & IA. Outil d'aide à la "
               "décision : il ne modifie aucune campagne et n'utilise aucune donnée interne de TF1+.")

    inv_m, ud_m = active_metadata("inventory", paths), active_metadata("under_delivery", paths)
    inv_d, ud_d = inventory_manifest(paths), under_delivery_manifest(paths)

    if not inv_m or not ud_m:
        st.info("Les données et les modèles ne sont pas encore prêts. Cliquez sur le bouton ci-dessous "
                "pour générer les données et entraîner les deux modules (≈ 1 à 2 minutes). "
                "L'entraînement n'est jamais déclenché automatiquement.")
        if st.button("Initialiser la démo (données + entraînement)", type="primary"):
            with st.status("Initialisation en cours…", expanded=True) as status:
                pipeline.run_all(paths, progress=st.write)
                status.update(label="Terminé", state="complete")
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    vol = (inv_d["n_rows"] if inv_d else 0) + (ud_d["n_daily_rows"] + ud_d["n_snapshots"] if ud_d else 0)
    c = st.columns(4)
    c[0].metric("Volume de données chargées", fmt(vol), help="Lignes Internos + livraisons quotidiennes + snapshots")
    c[1].metric("Dernier entraînement — Inventaire", inv_m["trained_at"][:16].replace("T", " ") if inv_m else "—")
    c[2].metric("Dernier entraînement — Sous-impression", ud_m["trained_at"][:16].replace("T", " ") if ud_m else "—")
    c[3].metric("Versions actives", f"{inv_m['version'][1:9] if inv_m else '—'} / {ud_m['version'][1:9] if ud_m else '—'}")

    st.subheader("Performances du modèle actif (jeu de test)")
    a, b = st.columns(2)
    with a:
        st.markdown("**Module A — Inventory Forecasting**")
        if inv_m:
            t = inv_m["metrics"][inv_m["selected_model"]]["test"]
            g = inv_m["metrics"][inv_m["selected_model"]].get("gain_vs_naive_wape")
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", fmt(t["mae"], 2))
            m2.metric("WAPE", pct(t["wape"]))
            m3.metric("Gain vs naïve", pct(g) if g is not None else "—", help="Réduction relative du WAPE")
            st.caption(f"Modèle : **{inv_m['selected_model']}** — split {inv_m['split']['type']}")
        else:
            st.write("—")
    with b:
        st.markdown("**Module B — Under-delivery Prediction**")
        if ud_m:
            t = ud_m["metrics"][ud_m["selected_model"]]["test"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Recall (classe 1)", pct(t["recall"]))
            m2.metric("Precision", pct(t["precision"]))
            m3.metric("PR-AUC", fmt(t["pr_auc"], 3))
            st.caption(f"Modèle : **{ud_m['selected_model']}** — seuil {ud_m['threshold']:.2f} — split par campagne")
        else:
            st.write("—")

    st.subheader("Statut des datasets")
    ds = pd.DataFrame(dataset_status(paths))
    ds["Lignes"] = ds["Lignes"].map(lambda x: fmt(x) if pd.notna(x) else "—")
    st.dataframe(ds, hide_index=True, width="stretch")
    st.caption("Les datasets publics sont à placer dans `data/raw/` (voir README). En leur absence, des "
               "substituts synthétiques clairement étiquetés sont utilisés.")
