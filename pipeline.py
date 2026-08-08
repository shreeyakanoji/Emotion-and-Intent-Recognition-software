"""
pipeline.py

    raw signal chunk -> filter -> extract features -> fuse -> classify -> feedback

Any interface (a CLI loop, a Streamlit dashboard, a future real hardware
reader) just calls pipeline.process_chunk() repeatedly and gets a result
back — separates "the logic" from "how it's displayed."

CHANGED FROM v2: no longer imports DEAP-specific channel names/sampling
rate at module load time. Instead it reads eeg_channel_names, eeg_fs, and
cardiac_fs out of the trained model bundle (train.py now saves these
alongside the model). This is the piece that makes the pipeline actually
dataset-agnostic: whatever montage/rate you trained on is what inference
uses, automatically, no matter which dataset that was.
"""

import numpy as np
import joblib
from scipy.signal import butter, filtfilt, welch, find_peaks

from features import BANDS, band_power, eeg_band_features, hrv_features
from fusion_features import curated_cross_modal_interactions


class EmotionPipeline:
    def __init__(self, model_path="trained_model.joblib"):
        loaded = joblib.load(model_path)
        self.model = loaded["model"]
        self.scaler = loaded["scaler"]
        self.target = loaded["target"]
        self.channel_names = loaded["eeg_channel_names"]
        self.eeg_fs = loaded["eeg_fs"]
        self.cardiac_fs = loaded["cardiac_fs"]
        self.class_names = {0: "Low " + self.target, 1: "High " + self.target}

    def _bandpass(self, signal, low, high, fs, order=4):
        nyq = 0.5 * fs
        b, a = butter(order, [low / nyq, high / nyq], btype="band")
        try:
            return filtfilt(b, a, signal)
        except Exception:
            from scipy.signal import lfilter
            return lfilter(b, a, signal)

    def process_chunk(self, eeg_chunk, cardiac_chunk):
        """
        eeg_chunk: (n_channels, n_samples) — n_channels must match the
                   montage this model was trained on (self.channel_names)
        cardiac_chunk: (n_samples,)

        Returns a dict with the prediction, confidence, and the intermediate
        values worth displaying (band powers, HRV, fusion features).
        """
        if eeg_chunk.shape[0] != len(self.channel_names):
            raise ValueError(
                f"This model expects {len(self.channel_names)} EEG channels "
                f"({self.channel_names}), got {eeg_chunk.shape[0]}. "
                "You're likely feeding it a chunk from a different dataset "
                "than it was trained on."
            )

        # 1. Filter — clean EEG range, cardiac range tuned looser since ECG
        #    QRS complexes need a wider passband than a plethysmograph pulse.
        clean_eeg = np.array([self._bandpass(ch, 1.0, 45.0, self.eeg_fs) for ch in eeg_chunk])
        clean_cardiac = self._bandpass(cardiac_chunk, 0.5, 40.0, self.cardiac_fs)

        # 2. Extract base features
        eeg_feats = eeg_band_features(clean_eeg, self.channel_names, self.eeg_fs)
        hrv_feats_arr = hrv_features(clean_cardiac, self.cardiac_fs)
        base_features = np.concatenate([eeg_feats, hrv_feats_arr])

        # 3. Fusion features (curated, interpretable)
        avg_band_powers = {
            band: float(np.mean([band_power(ch, self.eeg_fs, rng) for ch in clean_eeg]))
            for band, rng in BANDS.items()
        }
        hrv_dict = {"rmssd": float(hrv_feats_arr[0]), "sdnn": float(hrv_feats_arr[1])}
        fusion = curated_cross_modal_interactions(avg_band_powers, hrv_dict)

        # 4. Classify — scale with the SAME scaler used at training time.
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
