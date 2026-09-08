import streamlit as st

from config import CLASS_NAMES, CLASS_FULL_NAMES, CLASS_COLORS, CLINICAL_RECOMMENDATIONS, DISCLAIMER
from components import section_title, footer, pill, disclaimer_banner


def _urgency_pill_kind(urgency: str) -> str:
    if "Routine" in urgency:
        return "success"
    if "Priority" in urgency or "Urgent" in urgency:
        return "danger"
    return "warning"


def render():
    section_title("Clinical Recommendation", "Suggested next steps based on the AI-detected pattern.")

    disclaimer_banner(DISCLAIMER)

    active_class = st.session_state.get("prediction")
    if active_class:
        st.markdown(f"Showing the recommendation for the most recent diagnosis: "
                     f"**{active_class} — {CLASS_FULL_NAMES.get(active_class, '')}**")
    tabs = st.tabs([f"{c}" for c in CLASS_NAMES])

    for tab, cls in zip(tabs, CLASS_NAMES):
        with tab:
            rec = CLINICAL_RECOMMENDATIONS[cls]
            color = CLASS_COLORS[cls]
            st.markdown(
                f"""
                <div class="cad-card" style="border-left: 5px solid {color};">
                    <h4>{cls} — {CLASS_FULL_NAMES[cls]}</h4>
                    <p>{rec['summary']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("**Recommended actions:**")
            for a in rec["actions"]:
                st.markdown(f"- {a}")
            st.markdown(pill(f"Suggested urgency: {rec['urgency']}", _urgency_pill_kind(rec["urgency"])),
                         unsafe_allow_html=True)

    st.markdown("---")
    disclaimer_banner(DISCLAIMER)
    footer()
