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
    def __init__(self, model_path="trained_model.joblib", fs=None):
        """
        fs: sampling rate override. Leave as None (recommended) to
        automatically use the rate the model was trained at — stored in
        the model file itself. Only pass this explicitly if you have a
        good reason to reinterpret the model at a different rate.
        """
        loaded = joblib.load(model_path)
        self.model = loaded["model"]
        self.scaler = loaded["scaler"]
        self.target = loaded["target"]
        # Backward-compatible with older model files saved before channel/
        # rate info was tracked (assume DEAP's layout/rate in that case).
        self.fs = fs or loaded.get("fs") or SAMPLING_RATE
        self.channel_names = loaded.get("channel_names") or EEG_CHANNEL_NAMES
        self.n_channels = loaded.get("n_channels") or len(self.channel_names)
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
        eeg_chunk: (n_channels, n_samples) — n_channels MUST match the
            channel count this pipeline's model was trained on. Different
            hardware/datasets are fine as long as you train and test with
            the same channel count and (ideally) the same channel identity.
        plethysmo_chunk: (n_samples,)

        Returns a dict with the prediction, confidence, and the intermediate
        values worth displaying (band powers, HRV, fusion features) — a real
        dashboard should show its work, not just a final label.
        """
        actual_channels = eeg_chunk.shape[0]
        if actual_channels != self.n_channels:
            raise ValueError(
                f"This model was trained on {self.n_channels}-channel EEG data, "
                f"but the data you're testing on has {actual_channels} channels. "
                "Train a model on data with the same channel count as what "
                "you plan to test on — channel count can't be mismatched "
                "between training and inference."
            )

        # 1. Filter — clean 1-45Hz EEG range, matches your original v1 approach
        clean_eeg = np.array([self._bandpass(ch, 1.0, 45.0) for ch in eeg_chunk])
        clean_pleth = self._bandpass(plethysmo_chunk, 0.5, 8.0)

        # 2. Extract base features
        eeg_feats = eeg_band_features(clean_eeg, self.fs, channel_names=self.channel_names)
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

