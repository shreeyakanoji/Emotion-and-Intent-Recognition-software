"""
unified_app.py

One-page web app: get data (synthetic, DEAP, or DREAMER), train a model on
it, and test predictions — all from the browser, no separate CLI steps.

Run with:
    streamlit run unified_app.py

CHANGED FROM v2: data loading now goes through dataset_adapters.py's
ADAPTERS registry instead of hardcoding DEAP's .dat format. Adding support
for a new dataset later means writing one loader function in
dataset_adapters.py and adding it to ADAPTERS — nothing in this file needs
to change.

HONEST LIMIT, worth knowing before you rely on this: this is NOT a
universal "upload literally anything" parser — each dataset needs its own
adapter because raw EEG/physiological file formats genuinely differ
(channel counts, sampling rates, file structure). What this gives you is
a clean extension point, not format-agnostic magic. DEAP is fully tested
here (synthetic generator mimics its exact shape). DREAMER's adapter is
written to the published spec but hasn't been run against a real file —
verify shapes on your first real run (see dataset_adapters.py's note).
"""

import os
import tempfile

import numpy as np
import streamlit as st

from generate_synthetic_deap import generate_subject_data, N_SAMPLES, FS
import dataset_adapters
from train import train_model
from pipeline import EmotionPipeline
from theme import apply_theme

st.set_page_config(page_title="EEG/Cardiac Emotion Recognition — Studio", layout="wide")
apply_theme()
st.title("💗 EEG + Cardiac Emotion Recognition — Studio")
st.caption("Get data, train a model, and test it on new trials — all in one place.")

for key, default in [
    ("bundle", None), ("report", None), ("model_path", None), ("data_source_label", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# =========================================================================
# STEP 1 — DATA
# =========================================================================
st.header("1. Get data")

dataset_options = {"Synthetic (no real data needed)": "synthetic"}
for key, info in dataset_adapters.ADAPTERS.items():
    dataset_options[info["label"]] = key

data_mode_label = st.radio("Choose a data source", list(dataset_options.keys()))
data_mode = dataset_options[data_mode_label]

if data_mode == "synthetic":
    n_subjects = st.slider("Number of synthetic subjects", 1, 10, 3)
    if st.button("Generate synthetic data"):
        loaded = dataset_adapters.load_synthetic(n_subjects=n_subjects)
        st.session_state.loaded = loaded
        st.session_state.data_source_label = f"{n_subjects} synthetic subject(s), {loaded['eeg'].shape[0]} trials"
        st.success(f"Generated {loaded['eeg'].shape[0]} trials from {n_subjects} synthetic subject(s).")
        st.caption(
            "⚠️ Synthetic data has a deliberately exaggerated signal baked in so the "
            "pipeline has something to learn. Near-100% accuracy here is EXPECTED and "
            "means nothing about real-world performance — don't report these numbers "
            "anywhere. Use this only to confirm the pipeline runs end to end."
        )
else:
    adapter_info = dataset_adapters.ADAPTERS[data_mode]
    uploaded_files = st.file_uploader(
        f"Upload {adapter_info['label']}",
        type=None,
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Load uploaded file(s)"):
        try:
            file_bytes_list = [f.read() for f in uploaded_files]
            loaded = adapter_info["load_bytes"](file_bytes_list)
            st.session_state.loaded = loaded
            st.session_state.data_source_label = f"{len(uploaded_files)} uploaded file(s), {loaded['eeg'].shape[0]} trials"
            st.success(f"Loaded {loaded['eeg'].shape[0]} trials from {len(uploaded_files)} file(s).")
        except Exception as e:
            st.error(
                f"Couldn't parse file(s) as {adapter_info['label']}: {e}\n\n"
                "If this is DREAMER, the loader is written to the published spec but "
                "untested on a real file — check dataset_adapters.py's notes and adjust "
                "field names if the .mat structure differs."
            )

if "loaded" in st.session_state:
    loaded = st.session_state.loaded
    st.info(f"Current dataset: {st.session_state.data_source_label} — "
            f"{len(loaded['eeg_channel_names'])} EEG channels @ {loaded['eeg_fs']}Hz, "
            f"cardiac @ {loaded['cardiac_fs']}Hz")

    with st.container(border=True):
        st.markdown("### 🔎 Live scan — signal currently loaded")
        preview_trial = st.slider("Preview trial", 0, loaded["eeg"].shape[0] - 1, 0, key="preview_trial")
        preview_len = min(1000, loaded["eeg"].shape[2])
        pc1, pc2 = st.columns(2)
        with pc1:
            st.line_chart(loaded["eeg"][preview_trial, 0, :preview_len], height=180, use_container_width=True)
            st.caption(f"EEG channel {loaded['eeg_channel_names'][0]} — trial {preview_trial}")
        with pc2:
            cardiac_preview_len = min(1000, loaded["cardiac"].shape[1])
            st.line_chart(loaded["cardiac"][preview_trial, :cardiac_preview_len], height=180, use_container_width=True)
            st.caption(f"Cardiac signal — trial {preview_trial}")

st.divider()

# =========================================================================
# STEP 2 — TRAIN
# =========================================================================
st.header("2. Train")

if "loaded" not in st.session_state:
    st.write("Load data above first.")
else:
    loaded = st.session_state.loaded
    target = st.selectbox("Target label", ["valence", "arousal"])
    if st.button("Train model", type="primary"):
        progress_box = st.empty()
        log_lines = []

        def progress_cb(msg):
            log_lines.append(msg)
            progress_box.text("\n".join(log_lines))

        with st.spinner("Training..."):
            bundle, report = train_model(
                loaded["eeg"], loaded["cardiac"],
                loaded["eeg_channel_names"], loaded["eeg_fs"], loaded["cardiac_fs"],
                loaded["labels"][target], target,
                progress_cb=progress_cb,
            )

        st.session_state.bundle = bundle
        st.session_state.report = report

        import joblib
        model_path = os.path.join(tempfile.gettempdir(), "trained_model.joblib")
        joblib.dump(bundle, model_path)
        st.session_state.model_path = model_path
        st.success("Training complete.")

    if st.session_state.report is not None:
        report = st.session_state.report
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{report['accuracy']:.3f}")
        c2.metric("F1 score", f"{report['f1']:.3f}")
        c3.metric("5-fold CV accuracy", f"{report['cv_mean']:.3f}")
        c4.metric("CV std", f"±{report['cv_std']:.3f}")

        st.caption(
            f"{report['n_trials']} trials, {report['n_features']} features. "
            "On REAL data, 55-70% accuracy is the realistic range — much higher "
            "usually means an easy synthetic shortcut or a data leak, not a "
            "genuinely strong real-world model."
        )

        with st.expander("Confusion matrix"):
            st.write(np.array(report["confusion_matrix"]))
        with st.expander("Top 10 most important features (by index)"):
            for idx, importance in report["top_features"]:
                st.write(f"feature[{idx}]: importance={importance:.4f}")

        with open(st.session_state.model_path, "rb") as f:
            st.download_button("Download trained_model.joblib", data=f.read(), file_name="trained_model.joblib")

st.divider()

# =========================================================================
# STEP 3 — TEST
# =========================================================================
st.header("3. Test on new data")

if st.session_state.bundle is None:
    st.write("Train a model above first (or upload a `trained_model.joblib` below).")
    uploaded_model = st.file_uploader("...or upload an existing trained_model.joblib", type=None, key="model_upload")
    if uploaded_model is not None:
        import joblib
        model_path = os.path.join(tempfile.gettempdir(), "uploaded_trained_model.joblib")
        with open(model_path, "wb") as f:
            f.write(uploaded_model.read())
        bundle = joblib.load(model_path)
        st.session_state.bundle = bundle
        st.session_state.model_path = model_path
        st.success(f"Loaded model trained for target: {bundle['target']}")

if st.session_state.bundle is not None:
    pipeline = EmotionPipeline(st.session_state.model_path)
    st.write(f"Model ready — predicts **{pipeline.target}** (Low/High), "
             f"expects {len(pipeline.channel_names)} EEG channels @ {pipeline.eeg_fs}Hz.")

    if "loaded" not in st.session_state:
        st.write("No dataset loaded — load data in step 1 first to test on a trial from it.")
    else:
        loaded = st.session_state.loaded
        if len(loaded["eeg_channel_names"]) != len(pipeline.channel_names):
            st.warning(
                "⚠️ The currently loaded dataset has a different number of EEG channels "
                "than this model was trained on — predictions will fail. Load data from "
                "the same dataset type the model was trained on, or train a fresh model "
                "on the currently loaded data."
            )
        else:
            n_trials_avail = loaded["eeg"].shape[0]
            trial_idx = st.number_input("Trial index", 0, n_trials_avail - 1, 0)

            if st.button("Run prediction", type="primary"):
                eeg_trial = loaded["eeg"][trial_idx]
                cardiac_trial = loaded["cardiac"][trial_idx]

                eeg_chunk_size = int(pipeline.eeg_fs * 2.0)
                cardiac_chunk_size = int(pipeline.cardiac_fs * 2.0)
                eeg_chunk = eeg_trial[:, :eeg_chunk_size]
                cardiac_chunk = cardiac_trial[:cardiac_chunk_size]

                result = pipeline.process_chunk(eeg_chunk, cardiac_chunk)

                true_label = None
                if pipeline.target in loaded["labels"]:
                    true_val = loaded["labels"][pipeline.target][trial_idx]
                    true_label = "High " + pipeline.target if true_val == 1 else "Low " + pipeline.target

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.line_chart(cardiac_chunk, height=200, use_container_width=True)
                    st.caption("Cardiac signal, first 2s of this trial")
                with col2:
                    st.metric("Prediction", result["prediction"], f"{result['confidence']*100:.1f}% confidence")
                    if true_label is not None:
                        match = "✅ matches" if true_label == result["prediction"] else "❌ differs from"
                        st.write(f"True label: **{true_label}** ({match} prediction)")

                with st.expander("Band powers (avg across channels)"):
                    st.json(result["band_powers"])
                with st.expander("HRV features"):
                    st.json(result["hrv"])
                with st.expander("Fusion features"):
                    st.json(result["fusion_features"])
