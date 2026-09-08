import streamlit as st
import numpy as np
import plotly.graph_objects as go

from config import (
    DATASET_NAME, CLASS_NAMES, CLASS_FULL_NAMES, DATASET_CLASS_COUNTS,
    CLASS_COLORS, MODEL_NAME, MODEL_ACCURACY, COLORS,
)
from components import info_card, section_title, footer
from audio_utils import waveform_figure


def _synthetic_example_waveform(seed: int = 7, sr: int = 4000, duration: float = 3.0):
    """A representative, illustrative PCG-like waveform for the Dataset page
    (used only for visual reference, not derived from real patient data)."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, duration, int(sr * duration))
    signal = np.zeros_like(t)
    beat_interval = 0.8
    for beat_start in np.arange(0.1, duration, beat_interval):
        s1 = np.exp(-((t - beat_start) ** 2) / (2 * 0.004 ** 2)) * np.sin(2 * np.pi * 60 * (t - beat_start))
        s2 = np.exp(-((t - beat_start - 0.3) ** 2) / (2 * 0.006 ** 2)) * np.sin(2 * np.pi * 45 * (t - beat_start - 0.3))
        signal += s1 + s2
    signal += rng.normal(0, 0.01, size=signal.shape)
    return signal, sr


def render():
    section_title("Dataset Overview", f"Details of the {DATASET_NAME} used to train and validate the model.")

    total_samples = sum(DATASET_CLASS_COUNTS.values())
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        info_card("Dataset", DATASET_NAME.split(" ")[0], "🗂️", DATASET_NAME)
    with c2:
        info_card("Diagnostic Classes", str(len(CLASS_NAMES)), "🏷️", ", ".join(CLASS_NAMES))
    with c3:
        info_card("Total Samples", str(total_samples), "📈", "Across all classes")
    with c4:
        info_card("Test Accuracy", MODEL_ACCURACY, "🎯", MODEL_NAME)

    st.markdown("### Class Distribution")
    left, right = st.columns([1.3, 1])
    with left:
        fig = go.Figure(go.Bar(
            x=list(DATASET_CLASS_COUNTS.keys()),
            y=list(DATASET_CLASS_COUNTS.values()),
            marker_color=[CLASS_COLORS[c] for c in DATASET_CLASS_COUNTS],
            text=list(DATASET_CLASS_COUNTS.values()),
            textposition="outside",
        ))
        fig.update_layout(
            template="plotly_white", height=360,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Class", yaxis_title="Number of Recordings",
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown('<div class="cad-card">', unsafe_allow_html=True)
        st.markdown("#### Class Legend")
        for c in CLASS_NAMES:
            st.markdown(
                f"""<div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem;">
                <div style="width:14px;height:14px;border-radius:4px;background:{CLASS_COLORS[c]};"></div>
                <div><b>{c}</b> — {CLASS_FULL_NAMES[c]} ({DATASET_CLASS_COUNTS[c]} samples)</div>
                </div>""",
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Example Waveform")
    st.caption("An illustrative phonocardiogram waveform pattern, shown for reference only.")
    y, sr = _synthetic_example_waveform()
    example_fig = waveform_figure(y, sr, color=COLORS["primary"], title="Representative PCG Waveform")
    # EDIT 3: keep the amplitude axis fixed to the [-1, 1] range.
    example_fig.update_yaxes(range=[-1, 1])
    st.plotly_chart(example_fig, use_container_width=True)

    # EDIT 4: the "Preprocessing Pipeline" section and its two cards were removed.

    footer()
