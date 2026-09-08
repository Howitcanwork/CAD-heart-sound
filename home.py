import streamlit as st
from config import PROJECT_TITLE, MODEL_ACCURACY, NUM_CLASSES, DATASET_NAME
from components import feature_card, footer, safe_image


def render():
    st.markdown(
        f"""
        <div class="cad-hero">
            <div class="cad-badge">AI-POWERED CLINICAL DECISION SUPPORT</div>
            <h1>{PROJECT_TITLE}</h1>
            <p>A deep-learning platform that analyzes phonocardiogram (PCG) recordings to
            support the detection of valvular heart disease.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.markdown("### Why CardioSound AI?")
        st.write(
            "Cardiovascular disease remains one of the leading causes of death worldwide, "
            "and many valvular abnormalities produce subtle murmurs that are easy to miss "
            "during routine auscultation. CardioSound AI gives clinicians a fast, objective "
            "second opinion by analyzing the acoustic signature of the heart sound itself - "
            "turning a single recording into an evidence-based diagnostic suggestion in seconds."
        )
        if st.button("🩺 Start Diagnosis", type="primary"):
            st.session_state.nav_page = "Upload Heart Sound"
            st.rerun()
    with col2:
        safe_image(
            "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?q=80&w=800&auto=format&fit=crop",
            caption="AI-assisted cardiac auscultation",
        )

    st.markdown("### Key Features")
    c1, c2, c3 = st.columns(3)
    with c1:
        feature_card("🧠", "Hybrid Deep Learning",
                      "A hybrid architecture combining Xception Convolutional Neural Network (CNN) with Long Short-Term Memory (LSTM) for advanced heart sound classification.")
        feature_card("⚡", "Real-Time Inference",
                      "Cached model loading and an optimized preprocessing pipeline deliver predictions in seconds.")
    with c2:
        feature_card("📊", f"{NUM_CLASSES}-Class Detection",
                      "Distinguishes Normal heart sounds from MVP, MR, MS, and AS with high accuracy.")
        feature_card("📄", "Downloadable Reports",
                      "Generate a clean, shareable PDF diagnosis report for clinical documentation.")
    with c3:
        feature_card("🎯", f"{MODEL_ACCURACY} Test Accuracy",
                      f"Validated on the {DATASET_NAME} using a stratified 70/15/15 train-validation-test split.")
        feature_card("🔒", "Clinician-in-the-Loop",
                      "Designed as a decision-support aid - final diagnosis always rests with a qualified physician.")

    st.markdown("### How It Works")
    steps = st.columns(4)
    labels = [
        ("1️⃣", "Upload", "Upload a heart sound (.wav) recording"),
        ("2️⃣", "Extract", "CWT scalogram + MFCC sequence are computed"),
        ("3️⃣", "Predict", "The hybrid model classifies the recording"),
        ("4️⃣", "Report", "Review results and download a PDF report"),
    ]
    for col, (icon, title, desc) in zip(steps, labels):
        with col:
            feature_card(icon, title, desc)

    footer()
