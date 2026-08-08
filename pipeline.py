"""
pipeline.py


    raw signal chunk -> filter -> extract features -> fuse -> classify -> feedback

Any interface (a CLI loop, a Streamlit dashboard, a future real hardware
reader) just calls pipeline.process_chunk() repeatedly and gets a result
back. That's the actual design lesson here: separate "the logic" from
"how it's displayed" so you can swap the display (terminal, dashboard,
eventually a real EEG headset feed) without touching the logic at all.
"""

import numpy as np
import joblib
from scipy.signal import butter, filtfilt, welch, find_peaks

from deap_loader import EEG_CHANNEL_NAMES, SAMPLING_RATE, PLETHYSMOGRAPH_CHANNEL_IDX
from features import BANDS, LEFT_FRONTAL, RIGHT_FRONTAL, band_power, eeg_band_features, hrv_features
from fusion_features import curated_cross_modal_interactions


class EmotionPipeline:
    def __init__(self, model_path="trained_model.joblib", fs=SAMPLING_RATE):
        self.fs = fs
        loaded = joblib.load(model_path)
        self.model = loaded["model"]
        self.scaler = loaded["scaler"]
        self.target = loaded["target"]
        self.class_names = {0: "Low " + self.target, 1: "High " + self.target}

    def _bandpass(self, signal, low, high, order=4):
        nyq = 0.5 * self.fs
        b, a = butter(order, [low / nyq, high / nyq], btype="band")
        try:
            return filtfilt(b, a, signal)
        except Exception:
            from scipy.signal import lfilter
            return lfilter(b, a, signal)

    def process_chunk(self, eeg_chunk, plethysmo_chunk):
        """
        eeg_chunk: (32, n_samples)
        plethysmo_chunk: (n_samples,)

        Returns a dict with the prediction, confidence, and the intermediate
        values worth displaying (band powers, HRV, fusion features) — a real
        dashboard should show its work, not just a final label.
        """
        # 1. Filter — clean 1-45Hz EEG range, matches your original v1 approach
        clean_eeg = np.array([self._bandpass(ch, 1.0, 45.0) for ch in eeg_chunk])
        clean_pleth = self._bandpass(plethysmo_chunk, 0.5, 8.0)

        # 2. Extract base features
        eeg_feats = eeg_band_features(clean_eeg, self.fs)
        hrv_feats_arr = hrv_features(clean_pleth, self.fs)
        base_features = np.concatenate([eeg_feats, hrv_feats_arr])

        # 3. Fusion features (curated, interpretable — see fusion_features.py
        #    for why we don't use the full 8,500-term combinatorial version here)
        avg_band_powers = {
            band: float(np.mean([band_power(ch, self.fs, rng) for ch in clean_eeg]))
            for band, rng in BANDS.items()
        }
        hrv_dict = {"rmssd": float(hrv_feats_arr[0]), "sdnn": float(hrv_feats_arr[1])}
        fusion = curated_cross_modal_interactions(avg_band_powers, hrv_dict)

        # 4. Classify — scale with the SAME scaler used at training time.
        #    This is a common bug source: fitting a new scaler at inference
        #    time instead of reusing the training one silently breaks results.
        X = self.scaler.transform(base_features.reshape(1, -1))
        pred_class = int(self.model.predict(X)[0])
        pred_proba = self.model.predict_proba(X)[0]
        confidence = float(pred_proba[pred_class])

        return {
            "prediction": self.class_names[pred_class],
            "confidence": confidence,
            "band_powers": avg_band_powers,
            "hrv": hrv_dict,
            "fusion_features": fusion,
        }

