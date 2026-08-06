"""
generate_synthetic_deap.py

Creates fake .dat files in EXACTLY the same format as real DEAP subject
files, so that i can dry-run deap_loader.py / features.py / train.py end to
end before your real DEAP access is approved.

This is NOT the same as my original v1 mock training — that trained on
synthetic Gaussian blobs disconnected from any real signal shape. This
generates synthetic RAW SIGNALS (sine waves + noise, DEAP's exact array
shapes) and pushes them through your REAL feature extraction pipeline, so
I'm testing the actual pipeline logic, not just the classifier.

We deliberately bake in a small label-dependent difference (higher alpha
power for "Low" valence trials) so the classifier has *something* real to
learn — otherwise we'd just be testing that the code runs, not that a
real signal-to-label relationship gets picked up correctly.
"""

import os
import pickle
import numpy as np

N_TRIALS = 40
N_EEG_CHANNELS = 32
N_PERIPHERAL_CHANNELS = 8
N_SAMPLES = 8064  # 63 sec at 128Hz, matches real DEAP
FS = 128


def generate_subject_data(seed=0):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, N_SAMPLES / FS, N_SAMPLES)

    data = np.zeros((N_TRIALS, N_EEG_CHANNELS + N_PERIPHERAL_CHANNELS, N_SAMPLES))
    labels = np.zeros((N_TRIALS, 4))  # valence, arousal, dominance, liking

    for trial in range(N_TRIALS):
        # Randomly assign this trial a "true" valence class (for our synthetic ground truth)
        low_valence = rng.rand() < 0.5
        valence_rating = rng.uniform(1, 4.5) if low_valence else rng.uniform(5.5, 9)
        arousal_rating = rng.uniform(1, 9)

        labels[trial] = [valence_rating, arousal_rating, rng.uniform(1, 9), rng.uniform(1, 9)]

        # EEG channels: baseline 10Hz alpha rhythm + noise.
        # Low valence trials get boosted alpha power (synthetic signal, not a real
        # neuroscience claim — this is just so the classifier has a pattern to find).
        alpha_amplitude = 15.0 if low_valence else 6.0
        for ch in range(N_EEG_CHANNELS):
            alpha_wave = alpha_amplitude * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
            noise = rng.normal(0, 5, N_SAMPLES)
            data[trial, ch, :] = alpha_wave + noise

        # Peripheral channels 32-39. We only care about index 38 (plethysmograph)
        # for this project, but fill the others with plausible noise so the
        # array shape matches real DEAP exactly.
        for ch in range(N_EEG_CHANNELS, N_EEG_CHANNELS + N_PERIPHERAL_CHANNELS):
            data[trial, ch, :] = rng.normal(0, 1, N_SAMPLES)

        # Plethysmograph (index 38): simulate a pulse wave around ~70bpm,
        # with slightly faster/more variable rate on high-arousal trials.
        bpm = 70 + (20 if arousal_rating > 5 else 0) + rng.uniform(-5, 5)
        pulse_freq = bpm / 60.0
        pulse_wave = np.sin(2 * np.pi * pulse_freq * t) + 0.3 * np.sin(4 * np.pi * pulse_freq * t)
        data[trial, 38, :] = pulse_wave * 50 + rng.normal(0, 3, N_SAMPLES)

    return {"data": data, "labels": labels}


def main(out_dir, n_subjects=3):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(1, n_subjects + 1):
        subject_dict = generate_subject_data(seed=i)
        out_path = os.path.join(out_dir, f"s{i:02d}.dat")
        with open(out_path, "wb") as f:
            pickle.dump(subject_dict, f)
        print(f"Wrote synthetic subject file: {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", nargs="?", default="./synthetic_deap_data")
    parser.add_argument("--subjects", type=int, default=3)
    args = parser.parse_args()
    main(args.out_dir, n_subjects=args.subjects)
    print("\nNow dry-run the real pipeline with:")
    print(f"  python train.py {args.out_dir} --target valence --subjects {args.subjects}")

