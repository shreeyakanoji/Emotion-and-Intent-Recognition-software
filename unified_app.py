"""
unified_app.py

One-page web app that combines everything: get data (synthetic, DEAP-format
files, or your own EEG/cardiac device's CSV exports), train a model on it,
and test predictions — all from the browser, no separate CLI steps.

Run with:
    streamlit run unified_app.py

This does not replace app.py (the live-replay demo for an already-trained
model) — it's the all-in-one version for going from zero to a trained,
testable model without touching a terminal after launch.
"""

import io
import os
import pickle
import tempfile

import numpy as np
import streamlit as st

from deap_loader import SAMPLING_RATE as DEAP_FS
from generate_synthetic_deap import generate_subject_data
from generic_loader import load_trial_csv
from train import train_model
from pipeline import EmotionPipeline
from theme import apply_theme

st.set_page_config(page_title="EEG/Cardiac Emotion Recognition — Studio", layout="wide")
apply_theme()
st.title("💗 EEG + Cardiac Emotion Recognition — Studio")
st.caption(
    "Get data, train a model, and test it on new trials — all in one place. "
    "Works with synthetic data, real DEAP files, or your own EEG/cardiac device's CSV exports."
)

for key, default in [
    ("bundle", None), ("report", None), ("model_path", None),
    ("data_source_label", None), ("channel_names", None), ("fs", DEAP_FS),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def dat_bytes_to_arrays(file_bytes):
    """Parse an uploaded DEAP-format .dat file (pickle) into eeg/pleth/labels."""
    subject_dict = pickle.load(io.BytesIO(file_bytes), encoding="latin1")
    data = subject_dict["data"]        # (40, 40, 8064)
    labels = subject_dict["labels"]    # (40, 4)
    eeg = data[:, 0:32, :]
    pleth = data[:, 38, :]
    valence = (labels[:, 0] >= 5).astype(int)
    arousal = (labels[:, 1] >= 5).astype(int)
    return eeg, pleth, {"valence": valence, "arousal": arousal}


# =========================================================================
# STEP 1 — DATA
# =========================================================================
st.header("1. Get data")

data_mode = st.radio(
    "Choose a data source",
    [
        "Generate synthetic data (no DEAP access needed)",
        "Upload DEAP-format .dat file(s)",
        "Upload your own EEG/cardiac CSV files (any device)",
    ],
    horizontal=True,
)

if data_mode.startswith("Generate"):
    n_subjects = st.slider("Number of synthetic subjects", 1, 10, 3)
    if st.button("Generate synthetic data"):
        eeg_list, pleth_list, val_list, aro_list = [], [], [], []
        for i in range(1, n_subjects + 1):
            d = generate_subject_data(seed=i)
            data = d["data"]
            labels = d["labels"]
            eeg_list.append(data[:, 0:32, :])
            pleth_list.append(data[:, 38, :])
            val_list.append((labels[:, 0] >= 5).astype(int))
            aro_list.append((labels[:, 1] >= 5).astype(int))
        eeg_all = np.concatenate(eeg_list, axis=0)
        pleth_all = np.concatenate(pleth_list, axis=0)
        labels_all = {
            "valence": np.concatenate(val_list, axis=0),
            "arousal": np.concatenate(aro_list, axis=0),
        }
        st.session_state.eeg_all = eeg_all
        st.session_state.pleth_all = pleth_all
        st.session_state.labels_all = labels_all
        st.session_state.channel_names = None  # DEAP default names
        st.session_state.fs = DEAP_FS
        st.session_state.data_source_label = f"{n_subjects} synthetic subject(s), {eeg_all.shape[0]} trials"
        st.success(f"Generated {eeg_all.shape[0]} trials from {n_subjects} synthetic subject(s).")
        st.caption(
            "Note: this is noisy synthetic data on purpose — expect ~60-80% "
            "accuracy after training, not near-100%. Near-perfect accuracy "
            "on real data is usually a red flag (leakage), not a great model."
        )

elif data_mode.startswith("Upload DEAP"):
    uploaded_files = st.file_uploader(
        "Upload one or more DEAP subject files (sNN.dat)",
        type=None,
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Load uploaded file(s)"):
        eeg_list, pleth_list, val_list, aro_list = [], [], [], []
        errors = []
        for f in uploaded_files:
            try:
                eeg, pleth, labels = dat_bytes_to_arrays(f.read())
                eeg_list.append(eeg)
                pleth_list.append(pleth)
                val_list.append(labels["valence"])
                aro_list.append(labels["arousal"])
            except Exception as e:
                errors.append(f"{f.name}: {e}")
        if errors:
            st.error("Some files failed to load:\n" + "\n".join(errors))
        if eeg_list:
            eeg_all = np.concatenate(eeg_list, axis=0)
            pleth_all = np.concatenate(pleth_list, axis=0)
            labels_all = {
                "valence": np.concatenate(val_list, axis=0),
                "arousal": np.concatenate(aro_list, axis=0),
            }
            st.session_state.eeg_all = eeg_all
            st.session_state.pleth_all = pleth_all
            st.session_state.labels_all = labels_all
            st.session_state.channel_names = None  # DEAP default names
            st.session_state.fs = DEAP_FS
            st.session_state.data_source_label = f"{len(eeg_list)} uploaded DEAP file(s), {eeg_all.shape[0]} trials"
            st.success(f"Loaded {eeg_all.shape[0]} trials from {len(eeg_list)} file(s).")

else:
    st.caption(
        "Upload one CSV/TSV file per trial/recording — any device, any channel count. "
        "Each column is a channel; a cardiac (PPG/ECG/pulse) column is auto-detected by "
        "name, or falls back to the last numeric column. Non-numeric columns (like "
        "timestamps) are dropped automatically."
    )
    col_fs, col_target = st.columns(2)
    with col_fs:
        fs_generic = st.number_input("Sampling rate of your data (Hz)", min_value=1, value=128)
    with col_target:
        target_name = st.text_input("What are you classifying?", value="state", help="e.g. stress, focus, valence")

    generic_files = st.file_uploader(
        "Upload one file per trial/recording",
        type=None, accept_multiple_files=True, key="generic_files",
    )

    if generic_files:
        st.write(f"**{len(generic_files)} file(s) uploaded — label each one (Low/High {target_name}):**")
        with st.form("generic_label_form"):
            file_labels = []
            for f in generic_files:
                lbl = st.selectbox(f"{f.name}", ["Low", "High"], key=f"label_{f.name}")
                file_labels.append(lbl)
            submitted = st.form_submit_button("Load & label these files", type="primary")

        if submitted:
            eeg_list, pleth_list, y_list = [], [], []
            eeg_names_ref = None
            errors = []
            for f, lbl in zip(generic_files, file_labels):
                try:
                    eeg, cardiac, eeg_names, cardiac_col = load_trial_csv(f.getvalue())
                except Exception as e:
                    errors.append(f"{f.name}: {e}")
                    continue
                if eeg_names_ref is None:
                    eeg_names_ref = eeg_names
                elif len(eeg_names) != len(eeg_names_ref):
                    errors.append(
                        f"{f.name}: has {len(eeg_names)} channel(s), but {generic_files[0].name} "
                        f"had {len(eeg_names_ref)} — every file needs the same channel count."
                    )
                    continue
                eeg_list.append(eeg)
                pleth_list.append(cardiac)
                y_list.append(1 if lbl == "High" else 0)

            if errors:
                st.error("Some files had issues:\n\n" + "\n".join(errors))

            if eeg_list:
                min_len = min(e.shape[1] for e in eeg_list)
                eeg_all = np.stack([e[:, :min_len] for e in eeg_list])
                pleth_all = np.stack([p[:min_len] for p in pleth_list])
                y_arr = np.array(y_list)
                st.session_state.eeg_all = eeg_all
                st.session_state.pleth_all = pleth_all
                st.session_state.labels_all = {target_name: y_arr}
                st.session_state.channel_names = eeg_names_ref
                st.session_state.fs = int(fs_generic)
                st.session_state.data_source_label = (
                    f"{len(eeg_list)} uploaded file(s), {len(eeg_names_ref)} channel(s) "
                    f"at {fs_generic}Hz, target='{target_name}' (cardiac column: {cardiac_col})"
                )
                st.success(f"Loaded {len(eeg_list)} labeled trial(s) from your own data.")
                if y_arr.sum() == 0 or y_arr.sum() == len(y_arr):
                    st.warning(
                        "All your files got the same label — a model needs at least one "
                        "example of each class (Low and High) to learn anything."
                    )

if "eeg_all" in st.session_state:
    st.info(f"Current dataset: {st.session_state.data_source_label}")

    ch_names_display = st.session_state.channel_names
    first_ch_label = ch_names_display[0] if ch_names_display else "Fp1 (DEAP ch. 0)"

    with st.container(border=True):
        st.markdown("### 🔎 Live scan — signal currently loaded")
        preview_trial = st.slider(
            "Preview trial", 0, st.session_state.eeg_all.shape[0] - 1, 0, key="preview_trial"
        )
        preview_len = min(1000, st.session_state.eeg_all.shape[2])
        pc1, pc2 = st.columns(2)
        with pc1:
            st.line_chart(
                st.session_state.eeg_all[preview_trial, 0, :preview_len],
                height=180, use_container_width=True,
            )
            st.caption(f"EEG channel {first_ch_label} — trial {preview_trial}")
        with pc2:
            st.line_chart(
                st.session_state.pleth_all[preview_trial, :preview_len],
                height=180, use_container_width=True,
            )
            st.caption(f"Plethysmograph / cardiac — trial {preview_trial}")

st.divider()

# =========================================================================
# STEP 2 — TRAIN
# =========================================================================
st.header("2. Train")

if "eeg_all" not in st.session_state:
    st.write("Load data above first.")
else:
    target_options = list(st.session_state.labels_all.keys())
    target = st.selectbox("Target label", target_options)
    if st.button("Train model", type="primary"):
        progress_box = st.empty()
        log_lines = []

        def progress_cb(msg):
            log_lines.append(msg)
            progress_box.text("\n".join(log_lines))

        with st.spinner("Training..."):
            bundle, report = train_model(
                st.session_state.eeg_all,
                st.session_state.pleth_all,
                st.session_state.labels_all[target],
                target,
                channel_names=st.session_state.channel_names,
                fs=st.session_state.fs,
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

        if report["accuracy"] >= 0.95 or report["cv_mean"] >= 0.95:
            st.warning(
                "⚠️ Accuracy this high is suspicious rather than impressive — on real "
                "physiological data, genuine 95-100% almost always means a data leak "
                "(e.g. very few trials, or training/testing overlap) rather than a truly "
                "excellent model. Worth double-checking before reporting this number."
            )

        st.caption(
            f"{report['n_trials']} trials, {report['n_features']} features. "
            "55-70% accuracy is the realistic range on real DEAP data."
        )

        with st.expander("Confusion matrix"):
            st.write(np.array(report["confusion_matrix"]))

        with st.expander("Top 10 most important features (by index)"):
            for idx, importance in report["top_features"]:
                st.write(f"feature[{idx}]: importance={importance:.4f}")

        import joblib
        with open(st.session_state.model_path, "rb") as f:
            st.download_button(
                "Download trained_model.joblib",
                data=f.read(),
                file_name="trained_model.joblib",
            )

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
    st.write(
        f"Model ready — predicts **{pipeline.target}** (Low/High), "
        f"trained on **{pipeline.n_channels} channels** at **{pipeline.fs}Hz**."
    )

    test_mode = st.radio(
        "Test data source",
        [
            "Use a trial from the loaded dataset",
            "Upload a DEAP-format .dat file to test on",
            "Upload your own CSV to test on",
        ],
        horizontal=True,
        key="test_mode",
    )

    test_eeg_trial, test_pleth_trial, true_label, test_fs = None, None, None, pipeline.fs

    if test_mode.startswith("Use"):
        if "eeg_all" not in st.session_state:
            st.write("No dataset loaded — load data in step 1 first.")
        else:
            n_trials_avail = st.session_state.eeg_all.shape[0]
            trial_idx = st.number_input("Trial index", 0, n_trials_avail - 1, 0)
            test_eeg_trial = st.session_state.eeg_all[trial_idx]
            test_pleth_trial = st.session_state.pleth_all[trial_idx]
            test_fs = st.session_state.fs
            if pipeline.target in st.session_state.labels_all:
                true_val = st.session_state.labels_all[pipeline.target][trial_idx]
                true_label = "High " + pipeline.target if true_val == 1 else "Low " + pipeline.target

    elif test_mode.startswith("Upload a DEAP"):
        test_file = st.file_uploader("Upload a DEAP-format .dat file", type=None, key="test_file_dat")
        if test_file is not None:
            try:
                eeg, pleth, labels = dat_bytes_to_arrays(test_file.read())
                n_trials_avail = eeg.shape[0]
                trial_idx = st.number_input("Trial index within this file", 0, n_trials_avail - 1, 0)
                test_eeg_trial = eeg[trial_idx]
                test_pleth_trial = pleth[trial_idx]
                test_fs = DEAP_FS
                if pipeline.target in labels:
                    true_val = labels[pipeline.target][trial_idx]
                    true_label = "High " + pipeline.target if true_val == 1 else "Low " + pipeline.target
            except Exception as e:
                st.error(f"Couldn't parse that file: {e}")

    else:
        st.caption(
            f"This model expects {pipeline.n_channels} channel(s) at {pipeline.fs}Hz — "
            "your test file's channel count must match exactly, since that's what it was trained on."
        )
        test_file_csv = st.file_uploader("Upload a CSV/TSV recording", type=None, key="test_file_csv")
        if test_file_csv is not None:
            try:
                eeg, cardiac, eeg_names, cardiac_col = load_trial_csv(test_file_csv.getvalue())
                st.caption(f"Detected {len(eeg_names)} EEG channel(s), cardiac column: `{cardiac_col}`")
                test_eeg_trial = eeg
                test_pleth_trial = cardiac
                test_fs = pipeline.fs  # assume same rate as training unless told otherwise
            except Exception as e:
                st.error(f"Couldn't parse that file: {e}")

    if test_eeg_trial is not None and st.button("Run prediction", type="primary"):
        chunk_sec = 2.0
        chunk_size = int(test_fs * chunk_sec)
        chunk_size = min(chunk_size, test_eeg_trial.shape[1])
        eeg_chunk = test_eeg_trial[:, :chunk_size]
        pleth_chunk = test_pleth_trial[:chunk_size]

        try:
            result = pipeline.process_chunk(eeg_chunk, pleth_chunk)
        except ValueError as e:
            st.error(str(e))
            result = None

        if result is not None:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.line_chart(pleth_chunk, height=200, use_container_width=True)
                st.caption(f"Plethysmograph / cardiac signal, first {chunk_sec:.0f}s of this trial")
            with col2:
                st.metric(
                    "Prediction", result["prediction"],
                    f"{result['confidence']*100:.1f}% confidence",
                )
                if true_label is not None:
                    match = "✅ matches" if true_label == result["prediction"] else "❌ differs from"
                    st.write(f"True label: **{true_label}** ({match} prediction)")

            with st.expander("Band powers (avg across channels)"):
                st.json(result["band_powers"])
            with st.expander("HRV features"):
                st.json(result["hrv"])
            with st.expander("Fusion features"):
                st.json(result["fusion_features"])
