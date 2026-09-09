"""
model_utils.py
==============
Cached model + MFCC-statistics loading, and the single `run_inference`
entry point used by the Diagnosis page.

Everything here runs INFERENCE ONLY. No training code, no notebook cells
to execute - the moment the app starts, `load_model()` loads the trained
`.keras` file once and keeps it cached in memory for every subsequent
prediction, giving fast, production-ready response times.
"""

import os
import shutil
import numpy as np
import streamlit as st

from config import (
    MODEL_PATH, MODEL_PATH_FALLBACK, MFCC_STATS_PATH,
    HF_REPO_ID, HF_REPO_TYPE, HF_MODEL_FILENAME, HF_STATS_FILENAME, HF_TOKEN,
    N_MFCC, DEFAULT_MAX_LEN, CLASS_NAMES,
)


def _download_from_hf(filename: str, dest_path: str) -> bool:
    """Download a single file from the Hugging Face Hub repo (HF_REPO_ID)
    to `dest_path`. Returns True on success (or if the file is already
    there)."""
    if not filename:
        return False
    if os.path.exists(dest_path):
        return True
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return False

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename=filename,
            token=HF_TOKEN,
        )
        # hf_hub_download returns a path inside HF's local cache; copy it
        # to the exact path the rest of the app expects.
        if os.path.abspath(downloaded_path) != os.path.abspath(dest_path):
            shutil.copy(downloaded_path, dest_path)
        return os.path.exists(dest_path) and os.path.getsize(dest_path) > 0
    except Exception:
        return False


def _ensure_model_files():
    """Make sure the .keras model and the MFCC stats file are present
    locally, downloading them from Hugging Face Hub on first run. Cheap
    no-op on every run after the first, since the files then already
    exist on disk."""
    if not (os.path.exists(MODEL_PATH) or os.path.exists(MODEL_PATH_FALLBACK)):
        with st.spinner("Downloading trained model from Hugging Face (first run only)..."):
            _download_from_hf(HF_MODEL_FILENAME, MODEL_PATH_FALLBACK)

    if not os.path.exists(MFCC_STATS_PATH):
        _download_from_hf(HF_STATS_FILENAME, MFCC_STATS_PATH)


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained hybrid Xception+LSTM model once per server process.

    `st.cache_resource` ensures the (potentially large, GPU/CPU-heavy) Keras
    model is created a single time and shared across all user sessions,
    which is exactly what's needed for fast, production-grade inference on
    Streamlit Community Cloud.
    """
    from tensorflow import keras

    _ensure_model_files()

    path = MODEL_PATH if os.path.exists(MODEL_PATH) else MODEL_PATH_FALLBACK
    if not os.path.exists(path):
        return None, (
            "No trained model file was found. Either place your exported "
            "`best_stage2_finetuned.keras` (or `final_model.keras`) inside "
            "the `models/` folder of this app, or check that `HF_REPO_ID` "
            "(in config.py or as a Streamlit secret) points to a Hugging "
            "Face Hub repo containing the model file."
        )
    try:
        model = keras.models.load_model(path, compile=False)
        return model, None
    except Exception as e:  # pragma: no cover
        return None, f"Failed to load model from {path}: {e}"


@st.cache_resource(show_spinner=False)
def load_mfcc_stats():
    """Load the (mean, std, max_len) MFCC standardization stats saved during
    training. Falls back to safe defaults (zero-mean/unit-std, fixed length)
    if the stats file isn't bundled, so the demo still runs end-to-end."""
    _ensure_model_files()
    if os.path.exists(MFCC_STATS_PATH):
        stats = np.load(MFCC_STATS_PATH, allow_pickle=True)
        mean = stats["mean"].astype(np.float32)
        std = stats["std"].astype(np.float32)
        max_len = int(stats["max_len"])
        return mean, std, max_len

    mean = np.zeros(N_MFCC, dtype=np.float32)
    std = np.ones(N_MFCC, dtype=np.float32)
    return mean, std, DEFAULT_MAX_LEN


def run_inference(image_tensor: np.ndarray, mfcc_tensor: np.ndarray):
    """Run the dual-branch model and return (predicted_class, confidence, probs_dict)."""
    model, err = load_model()
    if model is None:
        return None, None, None, err

    probs = model.predict((image_tensor, mfcc_tensor), verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    probs_dict = {c: float(p) for c, p in zip(CLASS_NAMES, probs)}
    return pred_class, confidence, probs_dict, None
