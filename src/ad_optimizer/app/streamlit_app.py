"""Point d'entrée Streamlit : streamlit run src/ad_optimizer/app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from ad_optimizer.app.views import (  # noqa: E402
    inventory,
    methodology,
    overview,
    training,
    under_delivery,
)

st.set_page_config(page_title="AI Advertising Inventory Optimizer", page_icon="📊", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
[data-testid="stMetricValue"] {font-size: 1.5rem;}
</style>""", unsafe_allow_html=True)

PAGES = {
    "Vue d'ensemble": overview.render,
    "Prévision de l'inventaire": inventory.render,
    "Risque de sous-impression": under_delivery.render,
    "Entraînement et évaluation": training.render,
    "Données et méthodologie": methodology.render,
}

with st.sidebar:
    st.markdown("### 📊 Ad Inventory Optimizer")
    choice = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("POC académique · données publiques et synthétiques · aucune donnée TF1+")

PAGES[choice]()
