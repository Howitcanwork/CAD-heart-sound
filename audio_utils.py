"""
audio_utils.py
==============
Everything related to turning a raw heart-sound recording (.wav) into the
two model inputs used by the Hybrid Xception-LSTM network:

    1. A 299x299x3 CWT (Continuous Wavelet Transform) scalogram image,
       preprocessed exactly like the Xception branch expects.
    2. A (MAX_LEN, 40) standardized MFCC sequence for the LSTM branch.

NOTE ON REPRODUCIBILITY
------------------------
`final_hybrid.ipynb` trains on *already-extracted* CWT PNGs and MFCC .npy
files - the raw-audio -> feature extraction script itself is a separate,
offline preprocessing step and is not part of that notebook. The functions
below implement the standard, widely-used recipe for PCG (phonocardiogram)
CWT scalograms (Morlet wavelet, log-power, 'jet' colormap) and MFCC
extraction (n_mfcc=40) so the app can run end-to-end on a fresh .wav file.
If your offline extraction script used different wavelet/colormap/hop
settings, update `CWT_WAVELET`, `CWT_SCALES`, and `CWT_CMAP` below to match
exactly - the rest of the pipeline (resizing, Xception preprocessing,
MFCC standardization) will keep working unchanged.
"""

import io
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config import IMG_SIZE, N_MFCC, TARGET_SR

try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    _HAS_PYWT = False

CWT_WAVELET = "morl"          # Morlet wavelet - standard choice for PCG scalograms
CWT_SCALES = np.arange(1, 128)
CWT_CMAP = "jet"


# ------------------------------------------------------------------------
# Loading & basic signal info
# ------------------------------------------------------------------------
def load_audio(file_bytes: bytes, target_sr: int = TARGET_SR):
    """Load a WAV file (as raw bytes) into a mono float32 waveform."""
    y, sr = librosa.load(io.BytesIO(file_bytes), sr=target_sr, mono=True)
    return y, sr


def get_signal_info(y: np.ndarray, sr: int, original_sr: int = None) -> dict:
    duration = len(y) / sr
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    snr_estimate = 20 * np.log10(peak / (rms + 1e-8)) if rms > 0 else 0.0

    if snr_estimate > 12 and duration >= 3:
        quality = "Excellent"
    elif snr_estimate > 8 and duration >= 2:
        quality = "Good"
    elif snr_estimate > 4:
        quality = "Fair"
    else:
        quality = "Poor"

    return {
        "duration_sec": round(duration, 2),
        "sampling_rate": original_sr or sr,
        "processed_sampling_rate": sr,
        "num_samples": len(y),
        "peak_amplitude": round(peak, 4),
        "rms_level": round(rms, 4),
        "snr_estimate_db": round(snr_estimate, 2),
        "quality": quality,
    }


# ------------------------------------------------------------------------
# Waveform plot (Plotly, used on Upload + Dataset pages)
# ------------------------------------------------------------------------
def waveform_figure(y: np.ndarray, sr: int, color: str = "#0B5ED7", title: str = "Heart Sound Waveform"):
    import plotly.graph_objects as go

    t = np.linspace(0, len(y) / sr, num=len(y))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", line=dict(color=color, width=1.2), name="Amplitude"))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
        height=280,
        margin=dict(l=30, r=20, t=45, b=30),
        showlegend=False,
    )
    return fig


# ------------------------------------------------------------------------
# MFCC extraction -> standardized (MAX_LEN, N_MFCC) sequence
# ------------------------------------------------------------------------
def extract_mfcc(y: np.ndarray, sr: int, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Returns raw MFCCs with shape (n_mfcc, time)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.astype(np.float32)


def prepare_mfcc_sequence(mfcc: np.ndarray, mean: np.ndarray, std: np.ndarray, max_len: int) -> np.ndarray:
    """Standardize with training statistics, then pad/truncate to max_len,
    matching `_load_mfcc_numpy` in the training notebook exactly."""
    arr = (mfcc - mean[:, None]) / std[:, None]
    t = arr.shape[1]
    if t >= max_len:
        arr = arr[:, :max_len]
    else:
        arr = np.pad(arr, ((0, 0), (0, max_len - t)), mode="constant", constant_values=0.0)
    arr = arr.T  # (time, n_mfcc)
    return arr.astype(np.float32)[None, ...]  # add batch dim


# ------------------------------------------------------------------------
# CWT scalogram -> 299x299x3 Xception-ready tensor
# ------------------------------------------------------------------------
def compute_cwt_scalogram_image(y: np.ndarray, sr: int, img_size=IMG_SIZE) -> np.ndarray:
    """Compute a CWT scalogram and render it as an RGB image array (H, W, 3),
    uint8, matching the visual format the Xception branch was trained on."""
    if not _HAS_PYWT:
        raise ImportError(
            "PyWavelets (pywt) is required for CWT scalogram generation. "
            "Add 'PyWavelets' to requirements.txt."
        )

    coeffs, _ = pywt.cwt(y, CWT_SCALES, CWT_WAVELET, sampling_period=1.0 / sr)
    power = np.abs(coeffs) ** 2
    power = np.log1p(power)

    fig = plt.figure(figsize=(img_size[0] / 100, img_size[1] / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(power, aspect="auto", cmap=CWT_CMAP, origin="lower")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB").resize(img_size, Image.BILINEAR)
    return np.array(img)


def preprocess_image_for_xception(img_array: np.ndarray) -> np.ndarray:
    """Apply Xception's own preprocess_input, matching `_load_image` exactly."""
    from tensorflow.keras.applications.xception import preprocess_input

    img = img_array.astype(np.float32)
    img = preprocess_input(img)
    return img[None, ...]  # add batch dim


def scalogram_preview_figure(img_array: np.ndarray, title: str = "CWT Scalogram"):
    import plotly.express as px

    fig = px.imshow(img_array, title=title)
    fig.update_layout(
        template="plotly_white",
        height=300,
        margin=dict(l=10, r=10, t=45, b=10),
        coloraxis_showscale=False,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
