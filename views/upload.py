import streamlit as st

from config import COLORS
from components import info_card, section_title, footer, pill
from audio_utils import load_audio, get_signal_info, waveform_figure
from dataset_lookup import find_match

# Session keys that describe the currently uploaded recording, and the keys
# that hold a diagnosis produced FROM that recording. Keeping the two groups
# separate means a new upload can invalidate a stale diagnosis without
# touching anything else in the session.
UPLOAD_KEYS = ["audio_bytes", "audio_y", "audio_sr", "orig_sr", "signal_info",
               "file_name", "dataset_match", "true_label"]
DIAGNOSIS_KEYS = ["prediction", "confidence", "probs", "scalogram"]


def clear_upload():
    """EDIT 5: remove the uploaded recording (and any diagnosis derived from
    it). The file_uploader widget is rebuilt with a fresh key so it stops
    holding on to the deleted file."""
    for key in UPLOAD_KEYS + DIAGNOSIS_KEYS:
        st.session_state.pop(key, None)
    st.session_state.uploader_round = st.session_state.get("uploader_round", 0) + 1


def render():
    section_title("Upload Heart Sound", "Upload a phonocardiogram (.wav) recording to begin analysis.")

    uploader_key = f"pcg_uploader_{st.session_state.get('uploader_round', 0)}"
    uploaded = st.file_uploader("Drag and drop or browse a .wav heart sound recording",
                                type=["wav"], key=uploader_key)

    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        is_new_file = (st.session_state.get("file_name") != uploaded.name
                       or st.session_state.get("audio_bytes") != audio_bytes)

        if is_new_file:
            with st.spinner("Reading and analyzing the recording..."):
                try:
                    y, sr = load_audio(audio_bytes)
                    info = get_signal_info(y, sr)
                except Exception as e:
                    st.error(f"Could not process this file: {e}")
                    return

            # A new recording invalidates the previous result, so the
            # AI Diagnosis page never keeps showing the older dataset.
            for key in DIAGNOSIS_KEYS:
                st.session_state.pop(key, None)

            st.session_state.audio_bytes = audio_bytes
            st.session_state.audio_y = y
            st.session_state.audio_sr = sr
            st.session_state.signal_info = info
            st.session_state.file_name = uploaded.name

            # Look up this recording in the pre-extracted feature dataset (if
            # its Drive folders are mounted). A match gives us both the exact
            # features the model was trained on AND the ground-truth label,
            # automatically - no manual selection needed.
            match = find_match(uploaded.name)
            st.session_state.dataset_match = match
            st.session_state.true_label = match["class_name"] if match else None

    # Everything below is rendered from session state rather than from the
    # widget, so the uploaded recording stays visible when the user navigates
    # to another page and comes back. It is only removed via Delete.
    if "audio_bytes" not in st.session_state:
        st.markdown(
            """
            <div class="cad-card">
                <h4>📌 Recording Guidelines</h4>
                <p>For best results, use a recording of at least 3-5 seconds captured in a
                quiet environment with the stethoscope firmly positioned over the auscultation
                site. Supported format: WAV (mono or stereo, any standard sampling rate).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        footer()
        return

    audio_bytes = st.session_state.audio_bytes
    y = st.session_state.audio_y
    sr = st.session_state.audio_sr
    info = st.session_state.signal_info
    file_name = st.session_state.file_name
    match = st.session_state.get("dataset_match")

    st.success("File uploaded and analyzed successfully.")

    if match:
        st.info(
            f"🔎 This recording matches a known sample in the reference dataset. "
            f"**Ground-truth diagnosis: {match['class_name']}.** "
            f"The exact precomputed features will be used for diagnosis."
        )
    else:
        st.caption(
            "This file wasn't found in the reference dataset - no ground-truth "
            "label is available, and features will be extracted directly from the audio."
        )

    st.markdown("### Audio Player")
    st.audio(audio_bytes, format="audio/wav")

    st.markdown("### File Information")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        info_card("File Name", file_name[:20], "📁")
    with c2:
        info_card("Duration", f"{info['duration_sec']} s", "⏱️")
    with c3:
        info_card("Sampling Rate", f"{info['sampling_rate']} Hz", "📶")
    with c4:
        quality = info["quality"]
        kind = {"Excellent": "success", "Good": "success", "Fair": "warning", "Poor": "danger"}[quality]
        info_card("Recording Quality", quality, "🩻", pill(quality, kind))

    st.markdown("### Waveform Preview")
    # EDIT 3 (v2): shown naturally again - the fixed [-1, 1] amplitude range
    # was removed, so the plot autoscales to the signal as it did before.
    st.plotly_chart(waveform_figure(y, sr, color=COLORS["primary"]), use_container_width=True)

    st.markdown("---")
    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        if st.button("🩺 Run AI Diagnosis", type="primary"):
            st.session_state.nav_page = "AI Diagnosis"
            st.session_state.auto_run_diagnosis = True
            st.rerun()
    with colB:
        if st.button("🗑️ Delete Dataset"):
            clear_upload()
            st.rerun()
    with colC:
        st.caption("Running the diagnosis opens the AI Diagnosis page and analyzes this "
                    "recording automatically. Delete removes it from the session.")

    footer()
