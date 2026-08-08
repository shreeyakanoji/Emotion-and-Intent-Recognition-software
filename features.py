"""
features.py

Turns raw EEG + cardiac signals into a feature vector per trial.

CHANGED FROM v2: band-power extraction now takes an explicit `channel_names`
list instead of importing DEAP's hardcoded 32-channel list. This is what
lets the same feature code run on DEAP (32 ch), DREAMER (14 ch), or any
other montage — see dataset_adapters.py for where channel_names comes from.

Three feature families, same as before:

  1. Band power per EEG channel (theta, alpha, beta, gamma)
     -> WHY: different frequency bands reflect different brain states.
        Alpha (8-12Hz) drops when you're alert/engaged. Beta (12-30Hz) rises
        with active thinking/anxiety. This is the most standard EEG feature
        in the field, so it's also what lets you compare your results
        against published numbers later.

  2. Frontal alpha asymmetry
     -> WHY: this is a specific, well-established finding in affective
        neuroscience (Davidson et al.) — relatively more LEFT frontal alpha
        activity is associated with negative affect/withdrawal, and more
        RIGHT frontal alpha with positive affect/approach. Computed as
        ln(right_alpha) - ln(left_alpha) using an F4/F3 electrode pair.
        NOTE: if the dataset's montage doesn't include an F3/F4 pair (or
        an equivalent), this feature is skipped rather than faked — see
        `frontal_pair` below.

  3. Heart-rate variability (HRV) style features from the cardiac signal
     -> WHY: cardiac rhythm variability reflects autonomic nervous system
        state (sympathetic vs. parasympathetic balance), which correlates
        with arousal. We detect pulse/QRS peaks, compute inter-beat
        intervals (IBIs), then compute two standard HRV metrics:
          RMSSD = root mean square of successive IBI differences
                  (short-term variability, parasympathetic-linked)
          SDNN  = standard deviation of all IBIs
                  (overall variability)
     Works the same whether the cardiac signal is a literal ECG or a
     plethysmograph/PPG proxy — both are periodic with the cardiac cycle,
     which is all peak-detection-based HRV needs.
"""

import numpy as np
from scipy.signal import welch, find_peaks

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 45),
}

# Common frontal asymmetry pairs, checked in order of preference against
# whatever channel_names the current dataset actually has.
FRONTAL_PAIR_CANDIDATES = [("F3", "F4"), ("AF3", "AF4"), ("F7", "F8")]


def band_power(signal, fs, band, nperseg=256):
    """Mean power spectral density in a given frequency band, via Welch's method."""
    nperseg = min(nperseg, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(band_mask):
        return 0.0
    return float(np.mean(psd[band_mask]))


def _find_frontal_pair(channel_names):
    """Return (left_idx, right_idx) for the first available frontal pair, or None."""
    for left, right in FRONTAL_PAIR_CANDIDATES:
        if left in channel_names and right in channel_names:
            return channel_names.index(left), channel_names.index(right)
    return None


def eeg_band_features(eeg_trial, channel_names, fs):
    """
    eeg_trial: shape (n_channels, n_samples)
    channel_names: list of str, len == n_channels — dataset-specific montage
    Returns a flat feature vector: n_channels x 4 bands, plus 1 frontal
    asymmetry feature IF a known frontal pair is present in channel_names
    (silently omitted otherwise — no fake asymmetry value is invented).
    """
    features = []
    alpha_by_channel = {}

    for ch_idx, ch_name in enumerate(channel_names):
        signal = eeg_trial[ch_idx, :]
        for band_name, band_range in BANDS.items():
            power = band_power(signal, fs, band_range)
            if band_name == "alpha":
                alpha_by_channel[ch_name] = power
            features.append(power)

    pair = _find_frontal_pair(channel_names)
    if pair is not None:
        left_name, right_name = channel_names[pair[0]], channel_names[pair[1]]
        eps = 1e-10
        asymmetry = np.log(alpha_by_channel[right_name] + eps) - np.log(alpha_by_channel[left_name] + eps)
        features.append(asymmetry)

    return np.array(features)


def hrv_features(cardiac_trial, fs):
    """
    cardiac_trial: shape (n_samples,) — ECG or plethysmograph/PPG, either works.
    Detects pulse/QRS peaks and computes RMSSD + SDNN from inter-beat intervals.
    """
    min_distance_samples = int(fs * 60 / 180)  # assume heart rate won't exceed ~180 bpm
    peaks, _ = find_peaks(cardiac_trial, distance=min_distance_samples)

    if len(peaks) < 3:
        return np.array([0.0, 0.0])

    ibis = np.diff(peaks) / fs * 1000.0  # inter-beat intervals in milliseconds

    rmssd = float(np.sqrt(np.mean(np.diff(ibis) ** 2)))
    sdnn = float(np.std(ibis))

    return np.array([rmssd, sdnn])


def extract_trial_features(eeg_trial, cardiac_trial, channel_names, eeg_fs, cardiac_fs):
    """Combines EEG band-power/asymmetry features with HRV features."""
    eeg_feats = eeg_band_features(eeg_trial, channel_names, eeg_fs)
    hrv_feats = hrv_features(cardiac_trial, cardiac_fs)
    return np.concatenate([eeg_feats, hrv_feats])
