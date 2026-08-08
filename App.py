"""
app.py


It simulates a live feed by replaying synthetic (or real, once you have
DEAP + a trained model) data chunk by chunk, exactly like your original
FileStrea

CHUNK_SEC = 2.0
CHUNK_SIZE = int(FS * CHUNK_SEC)

# --- session state so the demo signal persists across reruns ---
if "subject_data" not in st.session_state:
  
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

