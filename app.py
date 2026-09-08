"""
app.py
======
Entry point for the "Computer Aided Diagnosis of Heart Sounds for
Cardiovascular Diseases" Streamlit application.

Run locally:      streamlit run app.py
Deploy:            push this folder to GitHub and deploy on
                   Streamlit Community Cloud, pointing at app.py.

The trained model is loaded once via `st.cache_resource` in `model_utils.py`
the first time it is needed, then reused for every subsequent prediction -
no manual steps or notebook cells are required at runtime.
"""

import streamlit as st

from config import PROJECT_TITLE, PROJECT_SHORT
from styles import inject_global_css
from components import render_sidebar
from views import home, dataset, upload, diagnosis, recommendation, about

st.set_page_config(
    page_title=f"{PROJECT_SHORT} | Heart Sound CAD",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"

inject_global_css(st.session_state.dark_mode)

current_page = render_sidebar()

PAGE_RENDERERS = {
    "Home": home.render,
    "Dataset": dataset.render,
    "Upload Heart Sound": upload.render,
    "AI Diagnosis": diagnosis.render,
    "Clinical Recommendation": recommendation.render,
    "About & Team": about.render,
}

renderer = PAGE_RENDERERS.get(current_page, home.render)
renderer()
