"""
config.py
=========
Single source of truth for the whole application: paths, class definitions,
color palette, team/university info and static clinical content.

Keeping every constant in one file means the rest of the codebase never
hard-codes a class name, color, or file path - change it once, here.
"""

import os

# ------------------------------------------------------------------------
# Base paths
# ------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(APP_DIR, "assets")

# Where the trained artifacts are expected to live at deploy time.
# Placing the .keras file (and the MFCC standardization stats produced
# during training) inside `models/` means the app "just works" the moment
# it is opened - no manual steps, no retraining, no notebook cells to run.
MODELS_DIR = os.path.join(APP_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_stage2_finetuned.keras")
MODEL_PATH_FALLBACK = os.path.join(MODELS_DIR, "final_model.keras")
MFCC_STATS_PATH = os.path.join(MODELS_DIR, "mfcc_standardization_stats.npz")

# ------------------------------------------------------------------------
# Google Drive auto-download (model files are too large for GitHub)
# ------------------------------------------------------------------------
# The trained .keras model (and, optionally, the MFCC standardization
# stats .npz) live on Google Drive instead of in this repo, because
# GitHub rejects files over 100 MB. On first run, `model_utils.py`
# downloads them from Drive into `models/` and caches them there for
# every following run.
#
# HOW TO FILL THESE IN:
#   1. In Google Drive, right-click each file -> "Share" -> "Anyone with
#      the link" (Viewer is enough).
#   2. Copy the link. It looks like:
#         https://drive.google.com/file/d/1AbCdeFGhIJKlmnoPQRstuVWxyz/view
#      The long part between /d/ and /view is the FILE ID.
#   3. Paste that ID below, OR (safer - keeps the ID out of the public
#      GitHub repo) set it as a secret when deploying on Streamlit
#      Community Cloud:
#         GDRIVE_MODEL_FILE_ID = "..."
#         GDRIVE_STATS_FILE_ID = "..."
#      Secrets always override the hardcoded values below.
#
# The IDs below are already filled in with Ayman's shared files/folders
# (all set to "Anyone with the link"), so the app works out of the box.
# Override via Streamlit secrets or env vars if you swap in your own files.
GDRIVE_MODEL_FILE_ID = os.environ.get("GDRIVE_MODEL_FILE_ID", "12RrMFoxWqAZlU2c8dD7C5o9q7uTwrFLB")
GDRIVE_STATS_FILE_ID = os.environ.get("GDRIVE_STATS_FILE_ID", "1WTCergkTSKSPpacLLkqsm_Le6eds6b6b")

# ------------------------------------------------------------------------
# Google Drive dataset sync (OPTIONAL - off by default)
# ------------------------------------------------------------------------
# CWT_Scalogram_new299/ and MFCC_Features_npy/ are the ORIGINAL precomputed
# feature folders (one subfolder per class) used by dataset_lookup.py to
# serve the exact features a training sample was trained on, instead of
# recomputing an approximation from raw audio. These folders can be large
# (thousands of PNG + .npy files), so syncing them is OPT-IN: leave
# GDRIVE_SYNC_DATASET as "false" to skip it entirely (the app then always
# falls back to on-the-fly extraction in audio_utils.py, which is what most
# deployments should use). Set it to "true" only if you've checked the
# folder size fits Streamlit Community Cloud's storage limit.
GDRIVE_CWT_FOLDER_ID = os.environ.get("GDRIVE_CWT_FOLDER_ID", "1HMpxSnnQzBP3DJaPLsLavDgPVbAQWPlb")
GDRIVE_MFCC_FOLDER_ID = os.environ.get("GDRIVE_MFCC_FOLDER_ID", "1bGmHEbC6cKFjuUhC1uKiuUjKrU4qprHR")
GDRIVE_SYNC_DATASET = os.environ.get("GDRIVE_SYNC_DATASET", "false").strip().lower() in ("1", "true", "yes")

try:  # pull from Streamlit secrets when running as a deployed app
    import streamlit as _st
    GDRIVE_MODEL_FILE_ID = _st.secrets.get("GDRIVE_MODEL_FILE_ID", GDRIVE_MODEL_FILE_ID)
    GDRIVE_STATS_FILE_ID = _st.secrets.get("GDRIVE_STATS_FILE_ID", GDRIVE_STATS_FILE_ID)
    GDRIVE_CWT_FOLDER_ID = _st.secrets.get("GDRIVE_CWT_FOLDER_ID", GDRIVE_CWT_FOLDER_ID)
    GDRIVE_MFCC_FOLDER_ID = _st.secrets.get("GDRIVE_MFCC_FOLDER_ID", GDRIVE_MFCC_FOLDER_ID)
    _sync_secret = _st.secrets.get("GDRIVE_SYNC_DATASET", None)
    if _sync_secret is not None:
        GDRIVE_SYNC_DATASET = str(_sync_secret).strip().lower() in ("1", "true", "yes")
except Exception:
    pass  # no secrets.toml locally / Streamlit not fully initialized yet - fine

# ------------------------------------------------------------------------
# Pre-extracted feature dataset (optional)
# ------------------------------------------------------------------------
# If your Google Drive with the ORIGINAL CWT_Scalogram_new299 / MFCC_Features_npy
# folders (produced by your offline preprocessing pipeline) is mounted, point
# these at them. When a user uploads a .wav whose filename stem matches a
# file already in this dataset, the app loads the EXACT precomputed features
# used during training (instead of recomputing an approximation from raw
# audio) - this guarantees the model sees the same input distribution it was
# trained on, and lets the app auto-detect the true label from the folder
# name the match was found in. Leave as-is (paths won't exist) when deploying
# without Drive access - the app falls back to on-the-fly extraction.
#
# NOTE: the original notebook pointed this at a Colab-only path
# ("/content/drive/MyDrive/..."), which doesn't exist on Streamlit
# Community Cloud. Here it points at a local `dataset/` folder inside the
# app instead - see GDRIVE_SYNC_DATASET above for how that folder gets
# populated from Google Drive (opt-in, off by default).
DATASET_BASE_DRIVE_PATH = os.path.join(APP_DIR, "dataset")
CWT_DIR = os.path.join(DATASET_BASE_DRIVE_PATH, "CWT_Scalogram_new299")
MFCC_DIR = os.path.join(DATASET_BASE_DRIVE_PATH, "MFCC_Features_npy")

# ------------------------------------------------------------------------
# Model / signal-processing configuration
# (must mirror the training notebook `final_hybrid.ipynb` exactly)
# ------------------------------------------------------------------------
CLASS_NAMES = ["Normal", "MVP", "MR", "MS", "AS"]
NUM_CLASSES = len(CLASS_NAMES)

CLASS_FULL_NAMES = {
    "Normal": "Normal Heart Sound",
    "MVP": "Mitral Valve Prolapse",
    "MR": "Mitral Regurgitation",
    "MS": "Mitral Stenosis",
    "AS": "Aortic Stenosis",
}

IMG_SIZE = (299, 299)          # Xception native input size
IMG_CHANNELS = 3
N_MFCC = 40                     # must match training extraction
DEFAULT_MAX_LEN = 400           # fallback sequence length if stats file is absent
TARGET_SR = 4000                # standard PCG (phonocardiogram) sampling rate

# ------------------------------------------------------------------------
# Brand / color palette - modern clinical blue & cyan on white
# ------------------------------------------------------------------------
COLORS = {
    "primary": "#0B5ED7",       # deep clinical blue
    "primary_dark": "#073E8A",
    "accent": "#00C2CB",        # cyan accent
    "accent_soft": "#E5FBFC",
    "success": "#12B76A",
    "warning": "#F79009",
    "danger": "#F04438",
    "bg": "#F7FAFC",
    "surface": "#FFFFFF",
    "text": "#0F1B2D",
    "text_muted": "#5B6B7F",
    "border": "#E4EAF1",
}

# A distinct color per diagnostic class, used consistently across every chart
CLASS_COLORS = {
    "Normal": "#12B76A",
    "MVP": "#0B5ED7",
    "MR": "#00C2CB",
    "MS": "#F79009",
    "AS": "#F04438",
}

# ------------------------------------------------------------------------
# Project / team metadata
# ------------------------------------------------------------------------
PROJECT_TITLE = "Computer Aided Diagnosis of Heart Sounds for Cardiovascular Diseases"
PROJECT_SHORT = "CardioSound AI"
PROJECT_YEAR = "2026"
DEPARTMENT = "Biomedical Engineering"
UNIVERSITY = "Sudan International University"

TEAM_STUDENTS = ["Nihad Hatim", "Noon Abdalhady", "Sjood Othman"]
SUPERVISOR = "Dr. Manahil Awad Bashari"

MODEL_NAME = "Hybrid Xception-CNN + LSTM"
MODEL_ACCURACY = "99.33%"

DATASET_NAME = "Yaseen PCG Heart Sound Dataset"
DATASET_CLASS_COUNTS = {
    "Normal": 200,
    "MVP": 200,
    "MR": 200,
    "MS": 200,
    "AS": 200,
}

# ------------------------------------------------------------------------
# Static clinical recommendation text per class
# ------------------------------------------------------------------------
CLINICAL_RECOMMENDATIONS = {
    "Normal": {
        "summary": "No pathological murmur pattern detected. Heart sounds fall within "
                    "the expected normal physiological range.",
        "actions": [
            "No immediate cardiac intervention indicated by this analysis.",
            "Continue routine health check-ups as per standard care schedule.",
            "Encourage a heart-healthy lifestyle: balanced diet, regular exercise, no smoking.",
        ],
        "urgency": "Routine",
    },
    "MVP": {
        "summary": "Pattern consistent with Mitral Valve Prolapse (MVP), where the mitral "
                    "valve leaflets bulge into the left atrium during systole.",
        "actions": [
            "Refer to a cardiologist for confirmatory echocardiography.",
            "Assess for symptoms such as palpitations, chest discomfort, or fatigue.",
            "Monitor for progression to mitral regurgitation over time.",
        ],
        "urgency": "Non-urgent follow-up",
    },
    "MR": {
        "summary": "Pattern consistent with Mitral Regurgitation (MR), indicating backward "
                    "leakage of blood through the mitral valve during systole.",
        "actions": [
            "Recommend echocardiogram to grade regurgitation severity.",
            "Evaluate left ventricular size and function.",
            "Consider cardiology referral, especially if symptomatic.",
        ],
        "urgency": "Prompt cardiology evaluation",
    },
    "MS": {
        "summary": "Pattern consistent with Mitral Stenosis (MS), suggesting narrowing of "
                    "the mitral valve orifice restricting blood flow.",
        "actions": [
            "Recommend echocardiography to quantify valve area and gradient.",
            "Assess for atrial fibrillation and pulmonary hypertension risk.",
            "Cardiology referral advised, particularly if dyspnea is present.",
        ],
        "urgency": "Prompt cardiology evaluation",
    },
    "AS": {
        "summary": "Pattern consistent with Aortic Stenosis (AS), suggesting narrowing of "
                    "the aortic valve restricting left-ventricular outflow.",
        "actions": [
            "Urgent echocardiography recommended to grade stenosis severity.",
            "Evaluate for symptoms: syncope, angina, exertional dyspnea.",
            "Timely cardiology referral is strongly advised.",
        ],
        "urgency": "Priority cardiology evaluation",
    },
}

DISCLAIMER = (
    "This system is intended to assist clinicians and is not a replacement "
    "for professional medical diagnosis."
)
