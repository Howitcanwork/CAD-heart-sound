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
import numpy as np
import streamlit as st

from config import (
    MODEL_PATH, MODEL_PATH_FALLBACK, MFCC_STATS_PATH,
    GDRIVE_MODEL_FILE_ID, GDRIVE_STATS_FILE_ID,
    N_MFCC, DEFAULT_MAX_LEN, CLASS_NAMES,
)


def _download_from_drive(file_id: str, dest_path: str) -> bool:
    """Download a single file from Google Drive (by its file ID) to
    `dest_path`, using `gdown` (handles Drive's large-file confirmation
    redirect automatically). Returns True on success."""
    if not file_id:
        return False
    if os.path.exists(dest_path):
        return True
    try:
        import gdown
    except ImportError:
        return False

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        gdown.download(id=file_id, output=dest_path, quiet=False)
        return os.path.exists(dest_path) and os.path.getsize(dest_path) > 0
    except Exception:
        return False


def _ensure_model_files():
    """Make sure the .keras model (and, if configured, the MFCC stats file)
    are present locally, downloading them from Google Drive on first run.
    Cheap no-op on every run after the first, since the files then already
    exist on disk."""
    if not (os.path.exists(MODEL_PATH) or os.path.exists(MODEL_PATH_FALLBACK)):
        with st.spinner("Downloading trained model from Google Drive (first run only)..."):
            _download_from_drive(GDRIVE_MODEL_FILE_ID, MODEL_PATH)

    if not os.path.exists(MFCC_STATS_PATH):
        _download_from_drive(GDRIVE_STATS_FILE_ID, MFCC_STATS_PATH)


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
            "the `models/` folder of this app, or set `GDRIVE_MODEL_FILE_ID` "
            "(in config.py or as a Streamlit secret) to a Google Drive file "
            "shared as \"Anyone with the link\"."
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
