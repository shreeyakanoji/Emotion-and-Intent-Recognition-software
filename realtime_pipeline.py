import os
import time
import logging
import tempfile

import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import butter, lfilter, filtfilt, welch
from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------------------------
# Simple, safer example of the realtime pipeline from the Draft 1 commit.
# Improvements made:
# - Validates .mat loading and expected keys
# - Handles short/rolling chunks without raising IndexError
# - Validates filter bounds and falls back when signal is too short for filtfilt
# - Uses deterministic mock training (seeded RNG) and checks that the model is trained
# - Uses logging instead of prints
# - Writes generated sample dataset to a temp directory (not the repo root)
# -------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FileStreamAcquisition:
    def __init__(self, file_path, sampling_rate=250):
        self.file_path = file_path
        self.sampling_rate = sampling_rate
        self.current_sample_idx = 0
        self.raw_data = None
        self.load_dataset_file()

    def load_dataset_file(self):
        """Loads a .mat dataset and validates expected structure.

        Expected shape: (channels, samples) and a key named 'eeg_data' (or similar).
        """
        logger.info("[CORE 1] Extracting data from: %s", self.file_path)
        try:
            mat_contents = loadmat(self.file_path)
        except FileNotFoundError:
            logger.exception("Dataset file not found: %s", self.file_path)
            raise
        except Exception:
            logger.exception("Failed to read .mat file: %s", self.file_path)
            raise

        # Prefer 'eeg_data' but fall back to any 2D numeric array found in mat_contents
        candidate_keys = [k for k in mat_contents.keys() if not k.startswith("__")]
        data_key = None
        if "eeg_data" in mat_contents:
            data_key = "eeg_data"
        else:
            # Look for the first numpy ndarray with 2 dimensions
            for k in candidate_keys:
                v = mat_contents[k]
                if isinstance(v, np.ndarray) and v.ndim == 2:
                    data_key = k
                    break

        if data_key is None:
            logger.error("No suitable 2D data array found in .mat file. Keys: %s", candidate_keys)
            raise KeyError(".mat file does not contain a 2D data array like 'eeg_data'")

        self.raw_data = mat_contents[data_key]
        if self.raw_data.ndim != 2:
            logger.error("Loaded data has unexpected ndim=%d", self.raw_data.ndim)
            raise ValueError("Expected 2D array (channels x samples)")

        logger.info("[CORE 1] Loaded matrix shape: %s (Channels x Samples)", self.raw_data.shape)

    def get_live_chunk(self, chunk_duration_sec=2.0):
        """Return a chunk of length chunk_duration_sec for each channel.

        This function ensures the returned chunk has exactly chunk_size samples by
        wrapping around and concatenating if necessary instead of returning shorter
        arrays that could break downstream expectations.
        """
        chunk_size = int(self.sampling_rate * chunk_duration_sec)
        total_samples = self.raw_data.shape[1]
        channels = self.raw_data.shape[0]

        if channels < 2:
            raise IndexError("Dataset must contain at least 2 channels (EEG, ECG)")

        if total_samples == 0:
            raise ValueError("Dataset contains zero samples")

        start = self.current_sample_idx
        end = start + chunk_size

        if end <= total_samples:
            chunk = self.raw_data[:, start:end]
            self.current_sample_idx = end % total_samples
        else:
            # Wrap: take the remainder and then the required samples from the start
            need = chunk_size
            parts = []
            idx = start
            while need > 0:
                take = min(total_samples - idx, need)
                parts.append(self.raw_data[:, idx: idx + take])
                need -= take
                idx = 0  # subsequent reads start at 0
            chunk = np.concatenate(parts, axis=1)
            # Update index to the next position after the wrapped chunk
            self.current_sample_idx = (start + chunk_size) % total_samples

        # For our simple architecture, treat row 0 as EEG and row 1 as ECG
        eeg_channel = chunk[0, :]
        ecg_channel = chunk[1, :]
        return eeg_channel, ecg_channel


class Preprocessing:
    def __init__(self, sampling_rate=250):
        self.fs = sampling_rate

    def _butter_bandpass(self, lowcut, highcut, order=4):
        nyq = 0.5 * self.fs
        # Validate frequency bounds
        if not (0.0 < lowcut < highcut < nyq):
            raise ValueError("Invalid bandpass bounds: lowcut < highcut < Nyquist required")
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        return b, a

    def clean_signals(self, eeg_data, ecg_data):
        """Cleans signals with a bandpass filter. Uses filtfilt when feasible for
        zero-phase filtering, otherwise falls back to lfilter.
        """
        b_eeg, a_eeg = self._butter_bandpass(1.0, 45.0)
        try:
            cleaned_eeg = filtfilt(b_eeg, a_eeg, eeg_data)
        except Exception:
            # filtfilt can fail for very short signals; fall back to lfilter
            cleaned_eeg = lfilter(b_eeg, a_eeg, eeg_data)

        b_ecg, a_ecg = self._butter_bandpass(5.0, 35.0)
        try:
            cleaned_ecg = filtfilt(b_ecg, a_ecg, ecg_data)
        except Exception:
            cleaned_ecg = lfilter(b_ecg, a_ecg, ecg_data)

        return cleaned_eeg, cleaned_ecg


class FeatureExtraction:
    def __init__(self, sampling_rate=250):
        self.fs = sampling_rate

    def extract_features(self, clean_eeg, clean_ecg):
        """Extracts a small feature vector: mean alpha PSD and ECG variance."""
        # Use a reasonable nperseg, don't exceed the signal length
        nperseg = min(256, max(8, len(clean_eeg)))
        freqs, psd = welch(clean_eeg, fs=self.fs, nperseg=nperseg)
        alpha_idx = np.where((freqs >= 8) & (freqs <= 12))[0]
        alpha_power = float(np.mean(psd[alpha_idx])) if alpha_idx.size > 0 else 0.0

        ecg_variance = float(np.var(clean_ecg))

        return [alpha_power, ecg_variance]


class MLClassification:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.classes_map = {0: "Relaxed / Focused", 1: "Stressed / Alert"}
        self.trained = False

    def mock_train_model(self, seed=42):
        """Train a deterministic mock model for testing only."""
        logger.info("[CORE 4] Training machine learning model components (mock)...")
        rng = np.random.RandomState(seed)

        relaxed_features = rng.normal(loc=[25.0, 5.0], scale=[3.0, 1.0], size=(50, 2))
        stressed_features = rng.normal(loc=[5.0, 35.0], scale=[2.0, 5.0], size=(50, 2))

        X_train = np.vstack((relaxed_features, stressed_features))
        y_train = np.array([0] * 50 + [1] * 50)

        self.model.fit(X_train, y_train)
        self.trained = True
        logger.info("[CORE 4] Random Forest mock training complete.")

    def classify_state(self, feature_vector):
        if not self.trained:
            raise RuntimeError("Model is not trained. Call mock_train_model() or load a trained model first.")

        input_data = np.array(feature_vector).reshape(1, -1)
        if input_data.shape[1] != 2:
            raise ValueError("Expected feature vector of length 2: [alpha_power, ecg_variance]")

        predicted_class_id = int(self.model.predict(input_data)[0])
        probabilities = self.model.predict_proba(input_data)[0]
        confidence = float(probabilities[predicted_class_id])
        return self.classes_map[predicted_class_id], confidence


class FeedbackSystem:
    def trigger_action(self, state, confidence):
        logger.info("[CORE 5 FEEDBACK] State output -> %s (%.1f%% Confidence)", state, confidence * 100)
        if state == "Stressed / Alert" and confidence > 0.70:
            logger.warning("AUTOMATED TRIGGER: Activating biofeedback music playlist.")


def create_dummy_mat_file(filename):
    """Create a simulated dataset in the given filename (writes to a temp dir by default)."""
    logger.info("Creating a simulated dataset file: %s", filename)
    total_samples = 250 * 30  # 30 seconds at 250Hz

    mock_eeg = np.sin(2 * np.pi * 10 * np.linspace(0, 30, total_samples)) * 10 + np.random.normal(0, 5, total_samples)
    mock_ecg = np.sin(2 * np.pi * 1.2 * np.linspace(0, 30, total_samples)) * 20 + np.random.normal(0, 2, total_samples)

    data_matrix = np.vstack((mock_eeg, mock_ecg))
    savemat(filename, {"eeg_data": data_matrix})


def main():
    tmpdir = tempfile.gettempdir()
    filename = os.path.join(tmpdir, "sample_dataset.mat")
    if not os.path.exists(filename):
        create_dummy_mat_file(filename)

    logger.info("Initializing Advanced Multimodal ML Pipeline...")

    core1 = FileStreamAcquisition(file_path=filename, sampling_rate=250)
    core2 = Preprocessing(sampling_rate=250)
    core3 = FeatureExtraction(sampling_rate=250)
    core4 = MLClassification()
    core5 = FeedbackSystem()

    core4.mock_train_model(seed=42)

    try:
        for loop in range(1, 5):
            logger.info("Cycle %d: Stream chunk parsing window...", loop)
            raw_eeg, raw_ecg = core1.get_live_chunk(chunk_duration_sec=2.0)
            clean_eeg, clean_ecg = core2.clean_signals(raw_eeg, raw_ecg)
            features = core3.extract_features(clean_eeg, clean_ecg)
            state, confidence = core4.classify_state(features)
            core5.trigger_action(state, confidence)
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Execution terminated by user.")


if __name__ == "__main__":
    main()
