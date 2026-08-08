"""
app.py


It simulates a live feed by replaying synthetic (or real, once you have
DEAP + a trained model) data chunk by chunk, exactly like your original
FileStreamAcquisition class did — same idea, now wired to the real pipeline
and displayed properly.
"""

import time
import numpy as np
import streamlit as st

from generate_synthetic_deap import generate_subject_data, N_SAMPLES, FS
from pipeline import EmotionPipeline
from theme import apply_theme

st.set_page_config(page_title="EEG/Cardiac Emotion Recognition", layout="wide")
apply_theme()
st.title("💗 EEG + Cardiac Emotion Recognition — Live Demo")
st.caption(
    "Currently replaying synthetic data for demo purposes. Swap the data "
    "source for a real DEAP trial or live hardware feed once available — "
    "the pipeline code doesn't need to change, only where chunks come from."
)

CHUNK_SEC = 2.0
CHUNK_SIZE = int(FS * CHUNK_SEC)

# --- session state so the demo signal persists across reruns ---
if "subject_data" not in st.session_state:
    st.session_state.subject_data = generate_subject_data(seed=1)
    st.session_state.trial_idx = 0
    st.session_state.sample_idx = 0

if "pipeline" not in st.session_state:
    try:
        st.session_state.pipeline = EmotionPipeline("trained_model.joblib")
        st.session_state.model_loaded = True
    except FileNotFoundError:
        st.session_state.model_loaded = False

if not st.session_state.model_loaded:
    st.error(
        "No trained_model.joblib found. Run `python train.py <data_dir> --target valence` "
        "first (real DEAP data or generate_synthetic_deap.py output) — this saves the "
        "model file this app loads."
    )
    st.stop()

col1, col2 = st.columns([2, 1])

placeholder_signal = col1.empty()
placeholder_result = col2.empty()

run = st.button("Start live demo")

if run:
    trial = st.session_state.subject_data["data"][st.session_state.trial_idx]
    eeg_full = trial[0:32, :]
    pleth_full = trial[38, :]

    for step in range(10):  # 10 chunks for the demo loop
        start = st.session_state.sample_idx
        end = start + CHUNK_SIZE
        if end > N_SAMPLES:
            break

        eeg_chunk = eeg_full[:, start:end]
        pleth_chunk = pleth_full[start:end]
        st.session_state.sample_idx = end

        result = st.session_state.pipeline.process_chunk(eeg_chunk, pleth_chunk)

        with placeholder_signal.container():
            st.line_chart(pleth_chunk, height=200, use_container_width=True)
            st.caption("Plethysmograph (cardiac) signal, current chunk")

        with placeholder_result.container():
            st.metric("Prediction", result["prediction"], f"{result['confidence']*100:.1f}% confidence")
            st.write("**Band powers (avg across channels):**")
            st.json(result["band_powers"])
            st.write("**HRV features:**")
            st.json(result["hrv"])
            st.write("**Fusion features:**")
            st.json(result["fusion_features"])

        time.sleep(1)

    st.success("Demo chunk sequence complete — click Start to run again.")

