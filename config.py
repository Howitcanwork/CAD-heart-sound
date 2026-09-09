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
# Hugging Face auto-download (model + dataset files are too large for GitHub)
# ------------------------------------------------------------------------
# The trained .keras model, the MFCC standardization stats .npz, and the
# original precomputed CWT/MFCC dataset folders all live in a Hugging Face
# Hub repo instead of this GitHub repo (GitHub rejects files over 100 MB,
# and Google Drive links hit virus-scan interstitials / download quotas
# under repeated programmatic access - Hugging Face Hub is built for
# exactly this: serving model + dataset files to an app at scale).
#
# On first run, `model_utils.py` / `dataset_lookup.py` download what they
# need from this repo into local folders and cache them there for every
# following run (st.cache_resource -> at most once per server process).
#
# Everything below is already filled in with Ayman's team's repo:
#   https://huggingface.co/noonatu/cad-heart-sound
# so the app works out of the box. If you fork this and use your own HF
# repo, change HF_REPO_ID below (or override via a Streamlit secret /
# env var of the same name - secrets always win over the hardcoded value).
HF_REPO_ID = os.environ.get("HF_REPO_ID", "noonatu/cad-heart-sound")
HF_REPO_TYPE = os.environ.get("HF_REPO_TYPE", "model")  # it's a Model repo, not a Space
HF_MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "final_model.keras")
HF_STATS_FILENAME = os.environ.get("HF_STATS_FILENAME", "mfcc_standardization_stats.npz")

# Only needed if the HF repo is/becomes private. Public repos work fine
# with this left empty. Set as a Streamlit secret named HF_TOKEN, never
# hardcode a real token here if this repo is public.
HF_TOKEN = os.environ.get("HF_TOKEN", "") or None

# ------------------------------------------------------------------------
# Hugging Face dataset sync (precomputed CWT/MFCC folders - ON by default)
# ------------------------------------------------------------------------
# CWT_Scalogram_new299/ and MFCC_Features_npy/ are the ORIGINAL precomputed
# feature folders (one subfolder per class) used by dataset_lookup.py to
# serve the exact features a training sample was trained on, instead of
# recomputing an approximation from raw audio - this is what makes
# predictions match the Colab/training results exactly for samples that
# are part of the original dataset. They're stored in the same HF repo as
# the model. Unlike the old Google Drive setup, Hugging Face Hub handles
# many small files and repeated downloads without quota/virus-scan issues,
# so this is ON by default now. Set HF_SYNC_DATASET="false" (as a secret
# or env var) to disable it and always fall back to on-the-fly extraction.
HF_SYNC_DATASET = os.environ.get("HF_SYNC_DATASET", "true").strip().lower() in ("1", "true", "yes")

try:  # pull from Streamlit secrets when running as a deployed app
    import streamlit as _st
    HF_REPO_ID = _st.secrets.get("HF_REPO_ID", HF_REPO_ID)
    HF_REPO_TYPE = _st.secrets.get("HF_REPO_TYPE", HF_REPO_TYPE)
    HF_MODEL_FILENAME = _st.secrets.get("HF_MODEL_FILENAME", HF_MODEL_FILENAME)
    HF_STATS_FILENAME = _st.secrets.get("HF_STATS_FILENAME", HF_STATS_FILENAME)
    HF_TOKEN = _st.secrets.get("HF_TOKEN", HF_TOKEN)
    _sync_secret = _st.secrets.get("HF_SYNC_DATASET", None)
    if _sync_secret is not None:
        HF_SYNC_DATASET = str(_sync_secret).strip().lower() in ("1", "true", "yes")
except Exception:
    pass  # no secrets.toml locally / Streamlit not fully initialized yet - fine

# ------------------------------------------------------------------------
# Pre-extracted feature dataset (optional)
# ------------------------------------------------------------------------
# If the precomputed CWT/MFCC dataset is synced locally (see HF_SYNC_DATASET
# above), when a user uploads a .wav whose filename stem matches a file
# already in this dataset, the app loads the EXACT precomputed features
# used during training (instead of recomputing an approximation from raw
# audio) - this guarantees the model sees the same input distribution it was
# trained on, and lets the app auto-detect the true label from the folder
# name the match was found in.
#
# NOTE: the original notebook pointed this at a Colab-only path
# ("/content/drive/MyDrive/..."), which doesn't exist on Streamlit
# Community Cloud. Here it points at a local `dataset/` folder inside the
# app instead - see HF_SYNC_DATASET above for how that folder gets
# populated from Hugging Face Hub.
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
