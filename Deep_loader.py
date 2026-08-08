uscle)
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
the
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

