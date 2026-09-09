import time
import streamlit as st

from config import CLASS_FULL_NAMES, CLASS_COLORS, CLINICAL_RECOMMENDATIONS, IMG_SIZE
from components import section_title, footer, pill, confidence_gauge, probability_bar_chart
from audio_utils import extract_mfcc, prepare_mfcc_sequence, compute_cwt_scalogram_image, \
    preprocess_image_for_xception, scalogram_preview_figure
from dataset_lookup import load_precomputed_image, load_precomputed_mfcc
from model_utils import load_mfcc_stats, run_inference
from report_utils import generate_pdf_report


def _urgency_pill_kind(urgency: str) -> str:
    if "Routine" in urgency:
        return "success"
    if "Priority" in urgency or "Urgent" in urgency:
        return "danger"
    return "warning"


def render():
    section_title("AI Diagnosis", "Run the hybrid Xception-LSTM model on the uploaded recording.")

    # EDIT 5: set by the "Run AI Diagnosis" button on the Upload page - the
    # diagnosis starts on its own here, with no second button press.
    auto_run = st.session_state.pop("auto_run_diagnosis", False)

    if "audio_y" not in st.session_state:
        st.info("⬅️ Please upload a heart sound recording first on the **Upload Heart Sound** page.")
        if st.button("Go to Upload Page"):
            st.session_state.nav_page = "Upload Heart Sound"
            st.rerun()
        footer()
        return

    match = st.session_state.get("dataset_match")
    true_label = st.session_state.get("true_label")

    if match:
        st.markdown(
            f"Ground-truth diagnosis (auto-detected from the reference dataset): "
            f"**{true_label} — {CLASS_FULL_NAMES.get(true_label, '')}**"
        )
    else:
        st.caption("No ground-truth label available for this recording — "
                    "it wasn't found in the reference dataset.")

    run = st.button("🚀 Run AI Diagnosis", type="primary") or auto_run

    if run:
        progress = st.progress(0, text="Preparing signal...")

        try:
            if match:
                # Use the EXACT precomputed features the model was trained on.
                progress.progress(30, text="Loading matched reference features...")
                mfcc_raw = load_precomputed_mfcc(match["mfcc_path"])
                scalogram = load_precomputed_image(match["image_path"], IMG_SIZE)
            else:
                # No dataset match — extract features directly from the audio.
                y, sr = st.session_state.audio_y, st.session_state.audio_sr
                progress.progress(20, text="Extracting MFCC sequence...")
                mfcc_raw = extract_mfcc(y, sr)
                progress.progress(50, text="Generating CWT scalogram...")
                scalogram = compute_cwt_scalogram_image(y, sr)

            mean, std, max_len = load_mfcc_stats()
            mfcc_tensor = prepare_mfcc_sequence(mfcc_raw, mean, std, max_len)
            st.session_state.scalogram = scalogram
            image_tensor = preprocess_image_for_xception(scalogram)

            progress.progress(75, text="Running hybrid Xception-LSTM inference...")
            pred_class, confidence, probs, err = run_inference(image_tensor, mfcc_tensor)

            if err:
                progress.empty()
                st.error(f"⚠️ {err}")
                st.info("The rest of the pipeline (upload, waveform, and report layout) is fully working - "
                        "add your trained `.keras` model to the `models/` folder to enable live predictions.")
                footer()
                return

            progress.progress(100, text="Done.")
            time.sleep(0.3)
            progress.empty()

            st.session_state.prediction = pred_class
            st.session_state.confidence = confidence
            st.session_state.probs = probs
            st.success("✅ Diagnosis complete.")

        except Exception as e:
            progress.empty()
            st.error(f"An error occurred during processing: {e}")
            footer()
            return

    if st.session_state.get("prediction"):
        pred_class = st.session_state.prediction
        confidence = st.session_state.confidence
        probs = st.session_state.probs
        color = CLASS_COLORS.get(pred_class, "#0B5ED7")

        st.markdown("### Diagnosis Result")
        left, right = st.columns([1.2, 1])
        with left:
            true_label = st.session_state.get("true_label")
            badges = pill(f"Confidence: {confidence*100:.1f}%", "info")
            if true_label:
                is_match = true_label == pred_class
                match_html = pill("MATCH" if is_match else "MISMATCH", "success" if is_match else "danger")
                badges += "&nbsp;" + pill(f"True: {true_label}", "info") + " " + match_html

            # NOTE: built as ONE single-line string (no blank lines inside the
            # <div>) before being handed to st.markdown - a multi-line f-string
            # with a blank line in the middle gets split by Streamlit's
            # Markdown parser and the stray closing tag shows up as literal
            # "</div>" text instead of being rendered.
            result_html = (
                f'<div class="cad-result">'
                f'<div style="font-size:0.85rem; color:#5B6B7F; font-weight:600;">PREDICTED DISEASE</div>'
                f'<div class="big-label" style="color:{color};">{pred_class} — {CLASS_FULL_NAMES.get(pred_class, "")}</div>'
                f'<div style="margin:0.6rem 0;">{badges}</div>'
                f'</div>'
            )
            st.markdown(result_html, unsafe_allow_html=True)

            # EDIT 2 (v2): the "CWT Scalogram (model input)" section was removed.

        with right:
            st.plotly_chart(confidence_gauge(confidence, color), use_container_width=True)

        st.plotly_chart(probability_bar_chart(probs, CLASS_COLORS), use_container_width=True)

        rec = CLINICAL_RECOMMENDATIONS.get(pred_class)
        if rec:
            st.markdown("### Quick Clinical Note")
            st.markdown(
                f"""
                <div class="cad-card">
                    <p>{rec['summary']}</p>
                    <div style="margin-top:0.6rem;">{pill(rec['urgency'], _urgency_pill_kind(rec['urgency']))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("See the **Clinical Recommendation** page for full guidance.")

        st.markdown("### Export & Save")
        c1, c2 = st.columns(2)
        with c1:
            pdf_bytes = generate_pdf_report(
                pred_class, confidence, probs,
                st.session_state.get("signal_info", {}),
                true_label=st.session_state.get("true_label"),
            )
            st.download_button("📄 Download PDF Report", data=pdf_bytes,
                                file_name=f"diagnosis_report_{pred_class}.pdf",
                                mime="application/pdf", use_container_width=True)
        with c2:
            if st.button("💾 Save Result to Session", use_container_width=True):
                st.session_state.setdefault("saved_results", [])
                st.session_state.saved_results.append({
                    "file": st.session_state.get("file_name", "recording.wav"),
                    "prediction": pred_class,
                    "confidence": confidence,
                })
                st.success("Result saved to this session's history.")

        if st.session_state.get("saved_results"):
            st.markdown("#### Session Result History")
            st.table(st.session_state.saved_results)

            # EDIT 1 (v2): delete a single saved result. The row numbers match
            # the index column of the table above, and removing one entry
            # leaves every other saved result untouched.
            del_options = [
                f"{i} - {r.get('file', 'recording.wav')} ({r.get('prediction', '')})"
                for i, r in enumerate(st.session_state.saved_results)
            ]
            d1, d2 = st.columns([3, 1])
            with d1:
                to_delete = st.selectbox("Select a saved result to delete", del_options)
            with d2:
                if st.button("🗑️ Delete Result", use_container_width=True):
                    st.session_state.saved_results.pop(del_options.index(to_delete))
                    st.rerun()

    footer()
