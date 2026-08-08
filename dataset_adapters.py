"""
dataset_adapters.py

The single place that knows about *dataset-specific* formats. Everything
downstream (features.py, pipeline.py, train.py) only ever sees the
normalized shape below — that's what makes "upload a different dataset and
still get a result" possible, within real limits (see README's honesty
note about what this can and can't do).

NORMALIZED FORMAT every adapter must return, a dict with:
    eeg:               ndarray (n_trials, n_channels, n_samples)
    eeg_channel_names: list[str], len == n_channels
    eeg_fs:            int, EEG sampling rate in Hz
    cardiac:           ndarray (n_trials, n_samples_cardiac)
    cardiac_fs:        int, cardiac signal sampling rate in Hz
                       (can differ from eeg_fs — features.py handles that)
    labels:            dict with "valence" and "arousal", each shape
                       (n_trials,), binarized to 0 (Low) / 1 (High)

Currently registered adapters: "deap", "dreamer", "synthetic".
To add a new dataset: write a loader that returns the shape above, then
add it to ADAPTERS at the bottom. That's the whole extension point.
"""

import pickle
import numpy as np


# =============================================================================
# DEAP
# =============================================================================
DEAP_EEG_CHANNEL_NAMES = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1",
    "P3", "P7", "PO3", "O1", "Oz", "Pz", "Fp2", "AF4", "Fz", "F4",
    "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8",
    "PO4", "O2",
]
DEAP_PLETH_CHANNEL_IDX = 38
DEAP_FS = 128


def load_deap_dat_bytes(file_bytes_list):
    """
    file_bytes_list: list of raw bytes, one per s01.dat/s02.dat/... file
    (DEAP files are pickled dicts with 'data' (40,40,8064) and 'labels' (40,4))
    """
    import io
    all_eeg, all_pleth, all_val, all_aro = [], [], [], []
    for fb in file_bytes_list:
        subject_dict = pickle.load(io.BytesIO(fb), encoding="latin1")
        data = subject_dict["data"]
        labels = subject_dict["labels"]
        all_eeg.append(data[:, 0:32, :])
        all_pleth.append(data[:, DEAP_PLETH_CHANNEL_IDX, :])
        all_val.append((labels[:, 0] >= 5).astype(int))
        all_aro.append((labels[:, 1] >= 5).astype(int))

    return {
        "eeg": np.concatenate(all_eeg, axis=0),
        "eeg_channel_names": DEAP_EEG_CHANNEL_NAMES,
        "eeg_fs": DEAP_FS,
        "cardiac": np.concatenate(all_pleth, axis=0),
        "cardiac_fs": DEAP_FS,  # plethysmograph shares DEAP's 128Hz preprocessing
        "labels": {
            "valence": np.concatenate(all_val, axis=0),
            "arousal": np.concatenate(all_aro, axis=0),
        },
    }


def load_deap_dat_paths(paths):
    file_bytes_list = []
    for p in paths:
        with open(p, "rb") as f:
            file_bytes_list.append(f.read())
    return load_deap_dat_bytes(file_bytes_list)


# =============================================================================
# DREAMER
# =============================================================================
# Reference structure (Katsigiannis & Ramzan, 2017): a single DREAMER.mat file
# containing one struct with a `Data` field, a 1x23 array of per-subject
# structs. Each subject struct has EEG.stimuli (18x1 cell, each cell
# n_samples x 14, 128Hz), ECG.stimuli (18x1 cell, each cell n_samples x 2,
# 256Hz), and ScoreValence / ScoreArousal (18x1, 1-5 scale).
#
# HONESTY NOTE — read before your first real run:
# this loader is written to that published spec, but has NOT been run
# against a real DREAMER.mat file (none was available to test with here).
# scipy.io.loadmat's handling of nested MATLAB structs is notoriously
# fiddly and can differ by MATLAB version the file was saved with. The
# first time you run this on the real file, add a `print(mat.keys())` and
# `print(mat['DREAMER'].dtype)` before trusting the parsed output, and
# adjust the field-access lines below if the structure doesn't match.
DREAMER_EEG_CHANNEL_NAMES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
    "P8", "T8", "FC6", "F4", "F8", "AF4",
]
DREAMER_EEG_FS = 128
DREAMER_ECG_FS = 256


def load_dreamer_mat_bytes(file_bytes):
    import io
    from scipy.io import loadmat

    mat = loadmat(io.BytesIO(file_bytes), struct_as_record=False, squeeze_me=True)
    root = mat["DREAMER"].Data  # 1x23 array of per-subject structs

    all_eeg, all_ecg, all_val, all_aro = [], [], [], []
    for subject in root:
        eeg_stimuli = subject.EEG.stimuli   # 18-element array of (n_samples, 14) arrays
        ecg_stimuli = subject.ECG.stimuli   # 18-element array of (n_samples, 2) arrays
        valence = np.atleast_1d(subject.ScoreValence)
        arousal = np.atleast_1d(subject.ScoreArousal)

        n_trials = len(eeg_stimuli)
        min_eeg_len = min(trial.shape[0] for trial in eeg_stimuli)
        min_ecg_len = min(trial.shape[0] for trial in ecg_stimuli)

        for i in range(n_trials):
            eeg_trial = eeg_stimuli[i][:min_eeg_len, :].T  # -> (14, n_samples)
            ecg_trial = ecg_stimuli[i][:min_ecg_len, 0]    # use ECG channel 1
            all_eeg.append(eeg_trial)
            all_ecg.append(ecg_trial)

        # DREAMER ratings are 1-5; binarize at the midpoint (3), same convention as DEAP's 1-9/5
        all_val.append((valence >= 3).astype(int))
        all_aro.append((arousal >= 3).astype(int))

    return {
        "eeg": np.stack(all_eeg, axis=0),
        "eeg_channel_names": DREAMER_EEG_CHANNEL_NAMES,
        "eeg_fs": DREAMER_EEG_FS,
        "cardiac": np.stack(all_ecg, axis=0),
        "cardiac_fs": DREAMER_ECG_FS,
        "labels": {
            "valence": np.concatenate(all_val, axis=0),
            "arousal": np.concatenate(all_aro, axis=0),
        },
    }


def load_dreamer_mat_path(path):
    with open(path, "rb") as f:
        return load_dreamer_mat_bytes(f.read())


# =============================================================================
# Synthetic (dry-run / pipeline validation only — see generate_synthetic_deap.py)
# =============================================================================
def load_synthetic(n_subjects=3, seed_offset=1):
    from generate_synthetic_deap import generate_subject_data

    all_eeg, all_pleth, all_val, all_aro = [], [], [], []
    for i in range(seed_offset, seed_offset + n_subjects):
        d = generate_subject_data(seed=i)
        data = d["data"]
        labels = d["labels"]
        all_eeg.append(data[:, 0:32, :])
        all_pleth.append(data[:, 38, :])
        all_val.append((labels[:, 0] >= 5).astype(int))
        all_aro.append((labels[:, 1] >= 5).astype(int))

    return {
        "eeg": np.concatenate(all_eeg, axis=0),
        "eeg_channel_names": DEAP_EEG_CHANNEL_NAMES,
        "eeg_fs": DEAP_FS,
        "cardiac": np.concatenate(all_pleth, axis=0),
        "cardiac_fs": DEAP_FS,
        "labels": {
            "valence": np.concatenate(all_val, axis=0),
            "arousal": np.concatenate(all_aro, axis=0),
        },
    }


ADAPTERS = {
    "deap": {
        "label": "DEAP (.dat files, one or more)",
        "load_bytes": load_deap_dat_bytes,   # takes list[bytes]
    },
    "dreamer": {
        "label": "DREAMER (single .mat file) — untested loader, verify field names first",
        "load_bytes": lambda file_bytes_list: load_dreamer_mat_bytes(file_bytes_list[0]),
    },
}
