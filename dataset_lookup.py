"""
dataset_lookup.py
==================
Looks up a matching precomputed CWT scalogram (.png) and MFCC (.npy) pair
for an uploaded recording by filename stem, searching every class
subfolder of CWT_DIR / MFCC_DIR (mirroring how `final_hybrid.ipynb` indexed
the dataset in `index_paired_dataset`).

Why this exists: the offline preprocessing pipeline that generated the
training features is not part of `final_hybrid.ipynb`, so a *fresh*
raw-audio extraction (see `audio_utils.py`) can only approximate it. If the
uploaded file is one of the recordings already in the dataset, using the
REAL precomputed features instead guarantees the model sees exactly the
input it was trained on - and the folder name doubles as the ground-truth
label, so no manual "true diagnosis" selector is needed.
"""

import os
import numpy as np
import streamlit as st
from PIL import Image

from config import (
    CLASS_NAMES, CWT_DIR, MFCC_DIR, N_MFCC, DATASET_BASE_DRIVE_PATH,
    HF_DATASET_REPO_ID, HF_DATASET_REPO_TYPE, HF_TOKEN, HF_SYNC_DATASET,
)


@st.cache_resource(show_spinner=False)
def _sync_dataset_from_hf():
    """One-time download of the precomputed CWT/MFCC dataset folders from
    the Hugging Face Hub DATASET repo (HF_DATASET_REPO_ID in config.py -
    a separate repo from the model, since that's how the team uploaded
    them). Runs at most once per server process thanks to st.cache_resource.
    Silently does nothing (and the app keeps working via on-the-fly
    extraction) if the flag is off, `huggingface_hub` isn't installed, or
    the download fails."""
    if not HF_SYNC_DATASET:
        return False
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False

    ok = True
    if not os.path.isdir(CWT_DIR):
        try:
            snapshot_download(
                repo_id=HF_DATASET_REPO_ID, repo_type=HF_DATASET_REPO_TYPE, token=HF_TOKEN,
                local_dir=DATASET_BASE_DRIVE_PATH,
                allow_patterns=["CWT_Scalogram_new299/**"],
            )
        except Exception:
            ok = False
    if not os.path.isdir(MFCC_DIR):
        try:
            snapshot_download(
                repo_id=HF_DATASET_REPO_ID, repo_type=HF_DATASET_REPO_TYPE, token=HF_TOKEN,
                local_dir=DATASET_BASE_DRIVE_PATH,
                allow_patterns=["MFCC_Features_npy/**"],
            )
        except Exception:
            ok = False
    return ok


def find_match(filename: str):
    """Search CWT_DIR/<class>/*.png and MFCC_DIR/<class>/*.npy for a file
    whose stem matches `filename`'s stem.

    Returns a dict {"class_name", "image_path", "mfcc_path"} on a full
    match (both modalities found in the SAME class), or None if the
    dataset isn't available or no match is found.
    """
    if HF_SYNC_DATASET and not (os.path.isdir(CWT_DIR) and os.path.isdir(MFCC_DIR)):
        with st.spinner("Syncing precomputed CWT/MFCC dataset from Hugging Face (first run only)..."):
            _sync_dataset_from_hf()

    if not (os.path.isdir(CWT_DIR) and os.path.isdir(MFCC_DIR)):
        return None

    stem = os.path.splitext(os.path.basename(filename))[0].strip().lower()
    if not stem:
        return None

    for class_name in CLASS_NAMES:
        img_folder = os.path.join(CWT_DIR, class_name)
        mfcc_folder = os.path.join(MFCC_DIR, class_name)
        if not (os.path.isdir(img_folder) and os.path.isdir(mfcc_folder)):
            continue

        img_match = _find_stem_match(img_folder, stem, (".png",))
        mfcc_match = _find_stem_match(mfcc_folder, stem, (".npy",))

        if img_match and mfcc_match:
            return {
                "class_name": class_name,
                "image_path": os.path.join(img_folder, img_match),
                "mfcc_path": os.path.join(mfcc_folder, mfcc_match),
            }

    return None


def _find_stem_match(folder: str, stem: str, extensions: tuple):
    for f in os.listdir(folder):
        name, ext = os.path.splitext(f)
        if ext.lower() in extensions and name.strip().lower() == stem:
            return f
    return None


def load_precomputed_image(image_path: str, img_size) -> np.ndarray:
    """Load the exact PNG scalogram used during training, resized to the
    model's expected input size."""
    img = Image.open(image_path).convert("RGB").resize(img_size, Image.BILINEAR)
    return np.array(img)


def load_precomputed_mfcc(mfcc_path: str) -> np.ndarray:
    """Load the raw (pre-standardization) MFCC array exactly as saved by the
    offline preprocessing pipeline, shape (n_mfcc, time)."""
    arr = np.load(mfcc_path).astype(np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(N_MFCC, -1)
    return arr
