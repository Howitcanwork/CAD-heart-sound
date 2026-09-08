import streamlit as st

from config import (
    PROJECT_TITLE, TEAM_STUDENTS, SUPERVISOR, DEPARTMENT, UNIVERSITY,
    PROJECT_YEAR, MODEL_NAME, MODEL_ACCURACY, DATASET_NAME,
)
from components import section_title, footer, team_card, info_card

# EDIT 4 (v2): clicking a student card opens that person's LinkedIn profile.
LINKEDIN_PROFILES = {
    "Nihad Hatim": "https://eg.linkedin.com/in/nihad-undefined-1a791540b",
    "Noon Abdalhady": "https://www.linkedin.com/in/noun-abdelhadi-85091a310",
    "Sjood Othman": "https://www.linkedin.com/in/sjood-othman-95ba64312",
}


def _initials(name: str) -> str:
    parts = name.split()
    return "".join(p[0] for p in parts[:2]).upper()


def render():
    section_title("About & Team", "The people and research behind this project.")

    st.markdown(
        f"""
        <div class="cad-card">
            <h4>📖 Project Overview</h4>
            <p>{PROJECT_TITLE} is a graduation project developed in the
            {DEPARTMENT} department at {UNIVERSITY} ({PROJECT_YEAR}). The system uses a
            hybrid {MODEL_NAME} deep-learning architecture, trained on the {DATASET_NAME},
            achieving {MODEL_ACCURACY} test accuracy in classifying phonocardiogram
            recordings into five diagnostic categories.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Student Team")
    cols = st.columns(len(TEAM_STUDENTS))
    for col, name in zip(cols, TEAM_STUDENTS):
        with col:
            team_card(name, "Student Researcher", _initials(name),
                      link=LINKEDIN_PROFILES.get(name))

    st.markdown("### Supervision")
    c1, c2 = st.columns(2)
    with c1:
        team_card(SUPERVISOR, "Project Supervisor", _initials(SUPERVISOR))
    with c2:
        info_card("Department", DEPARTMENT, "🏛️", UNIVERSITY)

    # EDIT 7: the "References" section was removed.

    footer()
