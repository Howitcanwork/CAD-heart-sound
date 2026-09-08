"""
components.py
=============
Reusable, presentation-only building blocks shared across every page:
cards, badges, the confidence gauge, the sidebar, and the footer.
"""

import streamlit as st
import plotly.graph_objects as go

from config import (
    COLORS, PROJECT_SHORT, PROJECT_TITLE, PROJECT_YEAR, DEPARTMENT,
    UNIVERSITY, TEAM_STUDENTS, SUPERVISOR, MODEL_NAME, MODEL_ACCURACY,
    NUM_CLASSES, DATASET_NAME,
)


# ------------------------------------------------------------------------
# Cards & badges
# ------------------------------------------------------------------------
def feature_card(icon: str, title: str, description: str):
    st.markdown(
        f"""
        <div class="cad-card">
            <div style="font-size:1.9rem;">{icon}</div>
            <h4>{title}</h4>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, value: str, icon: str = "📊", sub: str = ""):
    st.markdown(
        f"""
        <div class="cad-card">
            <div style="font-size:1.5rem;">{icon}</div>
            <h4 style="margin-bottom:0.15rem;">{value}</h4>
            <p style="font-weight:600; color:{COLORS['text_muted']}; margin-bottom:0.1rem;">{title}</p>
            <p style="font-size:0.8rem;">{sub}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill(text: str, kind: str = "info"):
    return f'<span class="cad-pill pill-{kind}">{text}</span>'


def safe_image(image, caption: str = None):
    """st.image(..., use_container_width=True) only exists on newer Streamlit
    releases; older installs (e.g. a Colab session with a pre-cached
    streamlit import) only support use_column_width. Try the modern kwarg
    first and fall back automatically, so this never crashes either way."""
    try:
        st.image(image, use_container_width=True, caption=caption)
    except TypeError:
        st.image(image, use_column_width=True, caption=caption)


def section_title(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"<p style='color:{COLORS['text_muted']}; margin-top:-0.6rem;'>{subtitle}</p>",
                     unsafe_allow_html=True)


# ------------------------------------------------------------------------
# Confidence gauge
# ------------------------------------------------------------------------
def confidence_gauge(confidence: float, class_color: str = COLORS["primary"]):
    pct = confidence * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 40, "color": class_color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["text_muted"]},
            "bar": {"color": class_color, "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "#FEE4E2"},
                {"range": [50, 75], "color": "#FEF0C7"},
                {"range": [75, 100], "color": "#DCFAE6"},
            ],
        },
        title={"text": "Model Confidence", "font": {"size": 16}},
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def probability_bar_chart(probs_dict: dict, class_colors: dict):
    classes = list(probs_dict.keys())
    values = [probs_dict[c] * 100 for c in classes]
    colors = [class_colors.get(c, COLORS["primary"]) for c in classes]

    fig = go.Figure(go.Bar(
        x=values, y=classes, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_white",
        height=280,
        margin=dict(l=10, r=30, t=30, b=20),
        xaxis=dict(title="Probability (%)", range=[0, 105]),
        yaxis=dict(title=""),
        title="Class Probability Distribution",
    )
    return fig


# ------------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------------
PAGES = [
    ("🏠", "Home"),
    ("📊", "Dataset"),
    ("🎧", "Upload Heart Sound"),
    ("🩺", "AI Diagnosis"),
    ("📋", "Clinical Recommendation"),
    ("👥", "About & Team"),
]


def _on_nav_change():
    """EDIT 9: commit the sidebar selection before the page is re-rendered,
    so a click lands on the chosen page immediately instead of one rerun
    later (which is what made the app bounce back to the previous page)."""
    st.session_state.nav_page = st.session_state.nav_radio.split("  ", 1)[1]


def render_sidebar():
    with st.sidebar:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:1.2rem;">
                <div style="width:42px; height:42px; border-radius:12px;
                            background:linear-gradient(135deg, {COLORS['primary']}, {COLORS['accent']});
                            display:flex; align-items:center; justify-content:center;
                            font-size:1.3rem;">🫀</div>
                <div>
                    <div style="font-weight:800; font-size:1.02rem; line-height:1.1;">{PROJECT_SHORT}</div>
                    <div style="font-size:0.72rem; color:{COLORS['text_muted']};">Clinical Decision Support</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        labels = [f"{icon}  {name}" for icon, name in PAGES]
        if "nav_page" not in st.session_state:
            st.session_state.nav_page = "Home"

        # The radio keeps a stable key so its state is never rebuilt from a
        # changing `index`. When another page navigates programmatically
        # (e.g. the "Run AI Diagnosis" button) we push that page into the
        # widget state before the widget is created.
        current_label = next((l for l in labels if l.endswith(st.session_state.nav_page)), labels[0])
        if st.session_state.get("nav_radio") != current_label:
            st.session_state.nav_radio = current_label

        st.radio("Navigation", labels, key="nav_radio",
                 label_visibility="collapsed", on_change=_on_nav_change)

        st.markdown("---")
        st.markdown("**Quick Statistics**")
        stat("Diagnostic Classes", str(NUM_CLASSES))
        stat("Model Accuracy", MODEL_ACCURACY)
        stat("Architecture", MODEL_NAME)

        st.markdown("---")
        # EDIT 9: keyed toggle - session_state["dark_mode"] is already updated
        # when the script reruns, so the new theme is applied on the first
        # click instead of needing a second toggle.
        st.toggle("🌙 Dark mode", key="dark_mode")

        if st.button("🗑️ Clear Session", use_container_width=True):
            for key in ["audio_bytes", "audio_y", "audio_sr", "orig_sr", "prediction",
                        "confidence", "probs", "true_label", "signal_info",
                        "dataset_match", "scalogram", "file_name"]:
                st.session_state.pop(key, None)
            # Rebuild the file uploader so it releases the cleared recording.
            st.session_state.uploader_round = st.session_state.get("uploader_round", 0) + 1
            st.rerun()

        st.markdown("---")
        st.caption(f"© {PROJECT_YEAR} {UNIVERSITY}\n\n{DEPARTMENT}")

    return st.session_state.nav_page


def stat(label: str, value: str):
    st.markdown(
        f"""
        <div class="cad-stat">
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def team_card(name: str, role: str, initials: str, link: str = None):
    card = f"""
        <div class="cad-team-card">
            <div class="cad-avatar">{initials}</div>
            <div style="font-weight:700;">{name}</div>
            <div style="font-size:0.8rem; color:{COLORS['text_muted']};">{role}</div>
        </div>
        """
    # EDIT 4 (v2): when a profile link is supplied, the exact same card markup
    # is wrapped in an anchor. No underline and inherited colors, so the card
    # looks identical - it simply becomes clickable.
    if link:
        card = (f'<a href="{link}" target="_blank" rel="noopener noreferrer" '
                f'style="text-decoration:none; color:inherit; display:block;">{card}</a>')
    st.markdown(card, unsafe_allow_html=True)


def footer():
    st.markdown(
        f"""
        <div class="cad-footer">
            {PROJECT_TITLE}<br/>
            {DEPARTMENT} · {UNIVERSITY} · {PROJECT_YEAR}<br/>
            Built by {", ".join(TEAM_STUDENTS)} · Supervised by {SUPERVISOR}
        </div>
        """,
        unsafe_allow_html=True,
    )


def disclaimer_banner(text: str):
    st.markdown(f'<div class="cad-disclaimer">⚠️ {text}</div>', unsafe_allow_html=True)
