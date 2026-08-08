"""
deap_loader.py

Loads DEAP dataset files (the "preprocessed Python" version).

IMPORTANT — you have to get the data yourself:
DEAP requires signing an End User License Agreement before download
(it's free for academic use). Request access here:
http://www.eecs.qmul.ac.uk/mmv/datasets/deap/

Once approved, download the "data_preprocessed_python" folder. It contains
32 files, one per subject: s01.dat, s02.dat, ... s32.dat

WHAT'S INSIDE EACH FILE (this is the part worth understanding, not memorizing):
Each .dat file is a Python pickle containing a dict with two keys:

  'data':   shape (40, 40, 8064)
            -> 40 trials (one per music video watched)
            -> 40 channels total:
                 channels 0-31  = EEG (international 10-20 system, 32 electrodes)
                 channels 32-39 = peripheral physiological signals:
                     32 = hEOG (horizontal eye movement)
                     33 = vEOG (vertical eye movement)
                     34 = zEMG (zygomaticus major, cheek muscle)
                     35 = tEMG (trapezius, shoulder muscle)
                     36 = GSR (skin conductance)
                     37 = Respiration belt
                     38 = Plethysmograph (blood volume pulse — this is what
                          we'll use for heart-rate-style features. It is NOT
                          a literal ECG electrode signal, but it reflects the
                          same underlying cardiac cycle, so RR-interval-style
                          analysis still applies.)
                     39 = Temperature
            -> 8064 samples per trial = 63 seconds at 128 Hz (after DEAP's
               own preprocessing, which already downsampled from 512 Hz)

  'labels': shape (40, 4)
            -> one row per trial, four self-reported ratings (1-9 scale):
               [valence, arousal, dominance, liking]

We'll turn valence/arousal into binary classes (High/Low) by splitting at
the middle of the 1-9 scale, which is the standard approach in DEAP papers.
"""

import pickle
import numpy as np

# DEAP's 32 EEG channels are in a fixed order defined by the dataset creators.
# This is the standard 10-20 system channel order used in DEAP's own docs.
EEG_CHANNEL_NAMES = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1",
    "P3", "P7", "PO3", "O1", "Oz", "Pz", "Fp2", "AF4", "Fz", "F4",
    "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8",
    "PO4", "O2",
]

PLETHYSMOGRAPH_CHANNEL_IDX = 38  # peripheral channel used for cardiac features
SAMPLING_RATE = 128  # Hz, after DEAP's own preprocessing


def load_subject(dat_file_path):
    """
    Load one subject's .dat file.

    Returns:
        eeg:        ndarray, shape (40 trials, 32 channels, 8064 samples)
        plethysmo:  ndarray, shape (40 trials, 8064 samples)
        labels:     dict with 'valence' and 'arousal', each shape (40,),
                    already binarized to 0 (Low) / 1 (High)
    """
    # DEAP files were pickled under Python 2, so we need latin1 encoding
    # to unpickle them correctly in Python 3 — this trips up almost everyone
    # the first time, so it's worth knowing why, not just copying the line.
    with open(dat_file_path, "rb") as f:
        subject_dict = pickle.load(f, encoding="latin1")

    data = subject_dict["data"]      # (40, 40, 8064)
    labels = subject_dict["labels"]  # (40, 4)

    eeg = data[:, 0:32, :]
    plethysmo = data[:, PLETHYSMOGRAPH_CHANNEL_IDX, :]

    valence_raw = labels[:, 0]
    arousal_raw = labels[:, 1]

    # Standard DEAP convention: split the 1-9 rating scale at 5
    valence_binary = (valence_raw >= 5).astype(int)
    arousal_binary = (arousal_raw >= 5).astype(int)

    return eeg, plethysmo, {"valence": valence_binary, "arousal": arousal_binary}


def load_all_subjects(data_dir, n_subjects=32):
    """
    Loads s01.dat through sNN.dat and stacks everything together.
    Returns the same shapes as load_subject, but concatenated across subjects
    along the trial axis.
    """
    import os

    all_eeg, all_pleth = [], []
    all_valence, all_arousal = [], []

    for i in range(1, n_subjects + 1):
        fname = os.path.join(data_dir, f"s{i:02d}.dat")
        eeg, pleth, labels = load_subject(fname)
        all_eeg.append(eeg)
        all_pleth.append(pleth)
        all_valence.append(labels["valence"])
        all_arousal.append(labels["arousal"])

    return (
        np.concatenate(all_eeg, axis=0),
        np.concatenate(all_pleth, axis=0),
        {
            "valence": np.concatenate(all_valence, axis=0),
            "arousal": np.concatenate(all_arousal, axis=0),
        },
    )


