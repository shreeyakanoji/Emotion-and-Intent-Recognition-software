uscle)
                     3

    labels = subject_dict["labels"]  # (40, 4)

    eeg = data[:, 0:32, :]
    plethysmo = data[:, PLETHYSMOGRAPH_CHANNEL_IDX, :]

    valence_raw = labels[:, 0]
    arousal_raw = labels[:, 1]


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

