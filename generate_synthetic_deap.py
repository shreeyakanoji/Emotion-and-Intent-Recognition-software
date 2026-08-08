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

We deliberately bake in a *small, noisy* label-dependent tendency (slightly
higher alpha power on a subset of channels for "Low" valence trials) so the
classifier has something real but hard to learn — not a clean, noise-free
signal. A model trained on this should land somewhere in a realistic
60-85% accuracy range, similar to what's typical on real DEAP data (see
README), NOT 95-100%. If you're seeing near-perfect accuracy, that's a sign
something's leaking information (e.g. testing on data the model already
saw), not that the model is genuinely great.
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

        # EEG channels: baseline alpha rhythm + noise. Only a random subset of
        # channels carries any label-related effect at all (real EEG effects
        # aren't uniform across all 32 electrodes), the effect size itself is
        # small and noisy trial-to-trial (not a fixed constant), and the
        # background noise is large relative to that effect — this is what
        # keeps the task genuinely hard instead of trivially separable.
        trial_effect_size = rng.normal(2.0, 1.5) if low_valence else rng.normal(0.0, 1.5)
        for ch in range(N_EEG_CHANNELS):
            channel_carries_effect = rng.rand() < 0.35
            base_amplitude = rng.uniform(6.0, 10.0)
            amplitude = base_amplitude + (trial_effect_size if channel_carries_effect else 0.0)
            amplitude = max(amplitude, 0.5)
            alpha_wave = amplitude * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
            noise = rng.normal(0, 9, N_SAMPLES)
            data[trial, ch, :] = alpha_wave + noise

        # Peripheral channels 32-39. We only care about index 38 (plethysmograph)
        # for this project, but fill the others with plausible noise so the
        # array shape matches real DEAP exactly.
        for ch in range(N_EEG_CHANNELS, N_EEG_CHANNELS + N_PERIPHERAL_CHANNELS):
            data[trial, ch, :] = rng.normal(0, 1, N_SAMPLES)

        # Plethysmograph (index 38): simulate a pulse wave around ~70bpm, with
        # a small, noisy arousal-linked rate shift rather than a clean jump.
        bpm = 70 + rng.normal(6 if arousal_rating > 5 else 0, 6) + rng.uniform(-5, 5)
        pulse_freq = max(bpm, 40) / 60.0
        pulse_wave = np.sin(2 * np.pi * pulse_freq * t) + 0.3 * np.sin(4 * np.pi * pulse_freq * t)
        data[trial, 38, :] = pulse_wave * 50 + rng.normal(0, 6, N_SAMPLES)

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

