import os
import time
import numpy as np
from scipy.io import loadmat, savemat
from scipy.signal import butter, lfilter, welch
from sklearn.ensemble import RandomForestClassifier

# =======================================================================
# CORE 1: SIGNAL ACQUISITION (Real .mat File Reader & Real-Time Emulator)
# =======================================================================
class FileStreamAcquisition:
    def __init__(self, file_path, sampling_rate=250):
        self.file_path = file_path
        self.sampling_rate = sampling_rate
        self.current_sample_idx = 0
        self.raw_data = None
        self.load_dataset_file()

    def load_dataset_file(self):
        """Loads a standard .mat structural file used in SEED/DEAP datasets."""
        print(f"[CORE 1] Extracting data from: {self.file_path}")
        mat_contents = loadmat(self.file_path)
        
        # Open datasets typically store data arrays under keys like 'eeg_data'
        # Shape structure: (channels, total_samples)
        self.raw_data = mat_contents['eeg_data'] 
        print(f"[CORE 1] Loaded matrix shape: {self.raw_data.shape} (Channels x Samples)")

    def get_live_chunk(self, chunk_duration_sec=2.0):
        """Simulates an LSL hardware hardware stream pulling chunks from the file."""
        chunk_size = int(self.sampling_rate * chunk_duration_sec)
        start = self.current_sample_idx
        end = start + chunk_size

        # Wrap around to the beginning if we run out of data
        if end > self.raw_data.shape[1]:
            print("[CORE 1] Dataset reached the end. Resetting stream loop to beginning.")
            start, end = 0, chunk_size
            self.current_sample_idx = 0

        # Slice out a 2-second block of raw multichannel data
        chunk = self.raw_data[:, start:end]
        self.current_sample_idx = end
        
        # For our architecture, let's treat row 0 as EEG and row 1 as ECG
        eeg_channel = chunk[0, :]
        ecg_channel = chunk[1, :]
        return eeg_channel, ecg_channel

# ================================================
# CORE 2: PREPROCESSING (Signal Cleansing Filters)
# ================================================
class Preprocessing:
    def __init__(self, sampling_rate=250):
        self.fs = sampling_rate

    def _butter_bandpass(self, lowcut, highcut, order=4):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def clean_signals(self, eeg_data, ecg_data):
        """Cleans raw time series data via frequency domain filtering."""
        b_eeg, a_eeg = self._butter_bandpass(1.0, 45.0)
        cleaned_eeg = lfilter(b_eeg, a_eeg, eeg_data)
        
        b_ecg, a_ecg = self._butter_bandpass(5.0, 35.0)
        cleaned_ecg = lfilter(b_ecg, a_ecg, ecg_data)
        return cleaned_eeg, cleaned_ecg

# ================================================
# CORE 3: FEATURE EXTRACTION (Vector Construction)
# ================================================
class FeatureExtraction:
    def __init__(self, sampling_rate=250):
        self.fs = sampling_rate

    def extract_features(self, clean_eeg, clean_ecg):
        """Transforms continuous waves into static model feature inputs."""
        # EEG: Alpha Power Spectral Density (8-12 Hz)
        freqs, psd = welch(clean_eeg, fs=self.fs, nperseg=len(clean_eeg))
        alpha_idx = np.where((freqs >= 8) & (freqs <= 12))[0]
        alpha_power = np.mean(psd[alpha_idx]) if len(alpha_idx) > 0 else 0.0
        
        # ECG: Variance as an alternative metric for Heart Rate Variability changes
        ecg_variance = np.var(clean_ecg)
        
        # Combine everything into an ordered numerical list (a feature vector)
        return [alpha_power, ecg_variance]

# =============================================================
# CORE 4: CLASSIFICATION (Scikit-Learn Machine Learning Engine)
# =============================================================
class MLClassification:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.classes_map = {0: "Relaxed / Focused", 1: "Stressed / Alert"}

    def mock_train_model(self):
        """Trains the actual Random Forest using mock feature matrices."""
        print("[CORE 4] Training machine learning model components...")
        
        # Generate 100 mock examples: [Alpha Power, Heart Variance]
        relaxed_features = np.random.normal(loc=[25.0, 5.0], scale=[3.0, 1.0], size=(50, 2))
        stressed_features = np.random.normal(loc=[5.0, 35.0], scale=[2.0, 5.0], size=(50, 2))
        
        X_train = np.vstack((relaxed_features, stressed_features))
        y_train = np.array([0] * 50 + [1] * 50)  # Labels: 0 for Relaxed, 1 for Stressed
        
        self.model.fit(X_train, y_train)
        print("[CORE 4] Random Forest training complete. Ready for real-time inference.")

    def classify_state(self, feature_vector):
        """Runs live model predictions on incoming dynamic vectors."""
        # Reshape to a 2D array structure expected by Scikit-Learn: [[feat1, feat2]]
        input_data = np.array(feature_vector).reshape(1, -1)
        
        predicted_class_id = self.model.predict(input_data)[0]
        probabilities = self.model.predict_proba(input_data)[0]
        confidence = probabilities[predicted_class_id]
        
        return self.classes_map[predicted_class_id], confidence

# =========================================
# CORE 5: FEEDBACK LOOP (Action Dispatcher)
# =========================================
class FeedbackSystem:
    def trigger_action(self, state, confidence):
        print(f"[CORE 5 FEEDBACK] State output -> **{state}** ({confidence*100:.1f}% Confidence)")
        if state == "Stressed / Alert" and confidence > 0.70:
            print("  ⚠️ AUTOMATED TRIGGER: Activating biofeedback music playlist.")
        print("-" * 65)

# ====================================
# ENVIRONMENT SETUP & RUNNER EXECUTION
# ====================================
def create_dummy_mat_file(filename):
    """Helper method to construct a valid structured dataset file for testing."""
    print(f"Creating a simulated open dataset file: {filename}")
    total_samples = 250 * 30  # 30 seconds of data at 250Hz
    
    # 2 Channels: Channel 0 is EEG, Channel 1 is ECG
    mock_eeg = np.sin(2 * np.pi * 10 * np.linspace(0, 30, total_samples)) * 10 + np.random.normal(0, 5, total_samples)
    mock_ecg = np.sin(2 * np.pi * 1.2 * np.linspace(0, 30, total_samples)) * 20 + np.random.normal(0, 2, total_samples)
    
    data_matrix = np.vstack((mock_eeg, mock_ecg))
    savemat(filename, {'eeg_data': data_matrix})

def main():
    filename = "sample_dataset.mat"
    if not os.path.exists(filename):
        create_dummy_mat_file(filename)

    print("\nInitializing Advanced Multimodal ML Pipeline...\n" + "="*65)
    
    # Instantiate architecture modules
    core1 = FileStreamAcquisition(file_path=filename, sampling_rate=250)
    core2 = Preprocessing(sampling_rate=250)
    core3 = FeatureExtraction(sampling_rate=250)
    core4 = MLClassification()
    core5 = FeedbackSystem()
    
    # Train the classification model before launching the loop
    core4.mock_train_model()
    print("\nStarting Live Acquisition Execution Cycle Loop...\n" + "-"*65)
    
    try:
        for loop in range(1, 5):
            print(f"Cycle {loop}: Stream chunk parsing window...")
            
            # Core 1: File stream ingest
            raw_eeg, raw_ecg = core1.get_live_chunk(chunk_duration_sec=2.0)
            
            # Core 2: Dynamic filter processing
            clean_eeg, clean_ecg = core2.clean_signals(raw_eeg, raw_ecg)
            
            # Core 3: Matrix feature extraction
            features = core3.extract_features(clean_eeg, clean_ecg)
            
            # Core 4: Machine learning inference
            state, confidence = core4.classify_state(features)
            
            # Core 5: Interface feedback actuation
            core5.trigger_action(state, confidence)
            
            time.sleep(1)  # Simulates system processing latency spacing
            
    except KeyboardInterrupt:
        print("\nExecution terminated smoothly.")

if __name__ == "__main__":
    main()
