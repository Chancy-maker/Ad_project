"""Page 5 — Données et méthodologie."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ...common.config import load_config
from ...common.status import inventory_manifest, under_delivery_manifest
from ..ui import get_paths


def render() -> None:
    st.title("Données et méthodologie")
    paths = get_paths()
    inv_cfg, ud_cfg = load_config("inventory", paths), load_config("under_delivery", paths)
    inv, ud = inventory_manifest(paths), under_delivery_manifest(paths)

    st.warning("Ce POC n'utilise **aucune donnée interne de TF1+**. Il s'appuie sur des datasets publics et des "
               "données synthétiques : c'est une simulation académique, pas une reproduction du système réel.")

    st.subheader("Sources")
    st.dataframe(pd.DataFrame([
        {"Dataset": "Internos Video Popularity Forecasting", "Module": "A", "Licence": "CC BY 4.0",
         "Lien": "https://data.mendeley.com/datasets/69g55s83mn/1", "DOI": "10.17632/69g55s83mn.1",
         "Attribution": "Mendeley Data — Internos"},
        {"Dataset": "Criteo Attribution Modeling for Bidding", "Module": "B", "Licence": "CC BY-NC-SA 4.0",
         "Lien": "https://ailab.criteo.com/criteo-attribution-modeling-bidding-dataset/", "DOI": "—",
         "Attribution": "Diemert, Meynet, Lefortier & Galland (2017)"},
        {"Dataset": "YTLive (facultatif, non utilisé)", "Module": "A (option)", "Licence": "voir dépôt",
         "Lien": "https://github.com/ghalandar/YTLive", "DOI": "arXiv:2510.24769", "Attribution": "YTLive"},
        {"Dataset": "Données synthétiques métier", "Module": "A + B", "Licence": "générées (graine fixe)",
         "Lien": "—", "DOI": "—", "Attribution": "ad_optimizer.data.synthetic_generator"},
    ]), hide_index=True, width="stretch")
    st.caption("Criteo : respecter l'attribution, le partage à l'identique et la restriction non commerciale.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Module A — état du chargement")
        if inv:
            st.write({"Source": inv["source"], "Réel": inv["is_real"], "Lignes": inv["n_rows"],
                      "Contenus": inv["n_media"], "Période": f"{inv['date_min']} → {inv['date_max']}",
                      "Chargé le": inv["loaded_at"].replace("T", " "), "Version des données": inv["data_version"]})
            for n in inv.get("notes", []):
                st.warning(n)
            with st.expander("Rapport de qualité des données"):
                st.json(inv["validation"])
            with st.expander("Colonnes"):
                st.write(inv["columns"])
        else:
            st.info("Dataset non préparé.")
    with c2:
        st.subheader("Module B — état de la génération")
        if ud:
            st.write({"Campagnes": ud["n_campaigns"], "Snapshots": ud["n_snapshots"],
                      "Lignes de livraison quotidienne": ud["n_daily_rows"], "Graine": ud["seed"],
                      "Tolérance de sous-livraison": ud["tolerance"],
                      "Prévalence": round(ud["prevalence_under_delivery"], 3),
                      "Profils de diffusion": ud["criteo"]["source"],
                      "Généré le": ud["generated_at"].replace("T", " ")})
        else:
            st.info("Campagnes non générées.")

    st.subheader("Variables réelles vs synthétiques")
    st.dataframe(pd.DataFrame([
        {"Variable / table": "Requêtes vidéo, variables de popularité et de calendrier (Internos)",
         "Nature": "Réel si internos.csv présent, sinon SUBSTITUT synthétique"},
        {"Variable / table": "Catégorie de contenu, opportunités par lecture, éligibilité, remplissage",
         "Nature": "Synthétique (hypothèses de simulation)"},
        {"Variable / table": "Profils de volume / variabilité des campagnes", "Nature": "Réel (Criteo) si présent, sinon paramétrique"},
        {"Variable / table": "Contrats : objectif d'impressions, dates, priorité, ciblage, plafond", "Nature": "Synthétique"},
        {"Variable / table": "Livraisons quotidiennes, inventaire disponible, cible de sous-impression", "Nature": "Synthétique"},
    ]), hide_index=True, width="stretch")

    st.subheader("Hypothèses de simulation")
    st.markdown(f"""
- Conversion (modifiable dans la page Inventaire) : **{inv_cfg['conversion']['ad_opportunities_per_view']} opportunités par lecture**,
  **{inv_cfg['conversion']['eligibility_rate']:.0%} d'éligibilité** — valeurs de simulation, jamais des paramètres réels.
- Campagnes : {ud_cfg['generator']['n_campaigns']} par défaut, 7 scénarios (livraison normale, objectif ambitieux, baisse soudaine d'inventaire,
  ciblage restreint, démarrage lent, rythme irrégulier, objectif facile), rattrapage de pacing borné.
- Sous-impression : livraison finale < **{ud_cfg['generator']['tolerance']:.0%}** de l'objectif (seuil paramétrable).
- Snapshots à {', '.join(f'{int(p*100)} %' for p in ud_cfg['snapshots'])} de la durée ; seules les données antérieures au snapshot servent d'entrée.
""")

    st.subheader("Avertissements méthodologiques")
    st.markdown("""
- **Internos** : intranet universitaire (2016–2018) ; la cible est une *demande de lecture*, pas un nombre d'impressions publicitaires ; non représentatif du marché actuel.
- **Criteo** : publicité *display*, pas *instream vidéo* ; `timestamp` relatif (aucune date réelle) ; `cost`/`cpo` transformés ; pas d'objectifs contractuels ni de stock.
- Ne pas confondre **audience prévue**, **opportunités théoriques**, **inventaire éligible**, **inventaire réservé** et **impressions diffusées**.
- Les résultats sont ceux obtenus sur données simulées : ils illustrent la méthode, **pas** des performances opérationnelles.
- Anti-fuite : split par ressource/temps (A) et par campagne (B) ; variables du snapshot calculées sur données tronquées ; cible jamais en entrée.
- Explications : importance par permutation et occlusion ; recommandations = règles déterministes sur variables calculées. Corrélation ≠ causalité.
""")
