"""
features.py

will turn raw EEG + plethysmograph signals into a feature vector per trial.

This replaces my original 2-feature version (alpha power + ECG variance)
with three feature families that are actually used in published EEG emotion
recognition work:

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
        RIGHT frontal alpha with positive affect/approach. Because alpha
        power is inversely related to activation, the asymmetry score is
        usually computed as ln(right_alpha) - ln(left_alpha) using an
        F4/F3 electrode pair. This is a much more theoretically-grounded
        feature than raw band power alone — worth highlighting in any
        write-up, since it shows you understand *why* a feature works,
        not just that scipy can compute it.

  3. Heart-rate variability (HRV) style features from the plethysmograph
     -> WHY: cardiac rhythm variability reflects autonomic nervous system
        state (sympathetic vs. parasympathetic balance), which correlates
        with arousal. We detect pulse peaks, compute inter-beat intervals
        (IBIs), then compute two standard HRV metrics:
          RMSSD = root mean square of successive IBI differences
                  (short-term variability, parasympathetic-linked)
          SDNN  = standard deviation of all IBIs
                  (overall variability)
"""

import numpy as np
from scipy.signal import welch, find_peaks

from deap_loader import EEG_CHANNEL_NAMES, SAMPLING_RATE

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (12, 30),
    "gamma": (30, 45),
}

# Standard frontal asymmetry pair used in the affective-neuroscience literature
LEFT_FRONTAL = "F3"
RIGHT_FRONTAL = "F4"


def band_power(signal, fs, band, nperseg=256):
    """Mean power spectral density in a given frequency band, via Welch's method."""
    nperseg = min(nperseg, len(signal))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(band_mask):
        return 0.0
    return float(np.mean(psd[band_mask]))


def eeg_band_features(eeg_trial, fs=SAMPLING_RATE, channel_names=None):
    """
    eeg_trial: shape (n_channels, n_samples) — n_channels no longer has to
    be 32/DEAP-specific; this works with any EEG device's channel count.

    channel_names: optional list of channel names matching eeg_trial's rows.
    Defaults to DEAP's 32 standard names (for backward compatibility with
    DEAP data). If frontal channels F3/F4 aren't present in channel_names
    (e.g. your own headset uses different electrode positions), frontal
    asymmetry falls back to an approximate "first half vs second half of
    channels" split as a rough left/right hemisphere proxy — worth flagging
    in any write-up as an approximation, not the literature-standard F3/F4
    measure, if your channel names don't include real 10-20 system labels.

    Returns a flat feature vector: n_channels x 4 bands, plus 1 frontal
    asymmetry feature at the end.
    """
    n_channels = eeg_trial.shape[0]
    if channel_names is None:
        if n_channels <= len(EEG_CHANNEL_NAMES):
            channel_names = EEG_CHANNEL_NAMES[:n_channels]
        else:
            channel_names = [f"ch{i}" for i in range(n_channels)]

    features = []
    channel_alpha_power = {}  # keep alpha power per channel for asymmetry calc

    for ch_idx in range(n_channels):
        ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"ch{ch_idx}"
        signal = eeg_trial[ch_idx, :]
        for band_name, band_range in BANDS.items():
            power = band_power(signal, fs, band_range)
            if band_name == "alpha":
                channel_alpha_power[ch_name] = power
            features.append(power)

    # Frontal alpha asymmetry: ln(right alpha) - ln(left alpha).
    eps = 1e-10
    left_name = next((n for n in channel_names if n.upper() == LEFT_FRONTAL), None)
    right_name = next((n for n in channel_names if n.upper() == RIGHT_FRONTAL), None)

    if left_name is not None and right_name is not None:
        right_alpha = channel_alpha_power[right_name]
        left_alpha = channel_alpha_power[left_name]
    else:
        # Generic fallback for non-DEAP channel layouts: approximate left/right
        # hemisphere as the first half vs. second half of channels. This is a
        # rough proxy, not the real F3/F4 measure — fine for getting a working
        # pipeline on arbitrary hardware, but call this out explicitly if used
        # in any publication.
        half = max(n_channels // 2, 1)
        left_vals = [channel_alpha_power[n] for n in channel_names[:half]]
        right_vals = [channel_alpha_power[n] for n in channel_names[half:]] or left_vals
        left_alpha = float(np.mean(left_vals))
        right_alpha = float(np.mean(right_vals))

    asymmetry = np.log(right_alpha + eps) - np.log(left_alpha + eps)
    features.append(asymmetry)

    return np.array(features)


def hrv_features(plethysmo_trial, fs=SAMPLING_RATE):
    """
    plethysmo_trial: shape (n_samples,)
    Detects pulse peaks and computes RMSSD + SDNN from inter-beat intervals.
    """
    # min distance between peaks: assume heart rate won't exceed ~180 bpm
    min_distance_samples = int(fs * 60 / 180)
    peaks, _ = find_peaks(plethysmo_trial, distance=min_distance_samples)

    if len(peaks) < 3:
        # Not enough peaks to compute meaningful HRV — return zeros rather
        # than crashing, but this is a real data-quality signal worth logging.
        return np.array([0.0, 0.0])

    ibis = np.diff(peaks) / fs * 1000.0  # inter-beat intervals in milliseconds

    rmssd = float(np.sqrt(np.mean(np.diff(ibis) ** 2)))
    sdnn = float(np.std(ibis))

    return np.array([rmssd, sdnn])


def extract_trial_features(eeg_trial, plethysmo_trial, fs=SAMPLING_RATE, channel_names=None):
    """Combines EEG band-power/asymmetry features with HRV features."""
    eeg_feats = eeg_band_features(eeg_trial, fs, channel_names=channel_names)
    hrv_feats = hrv_features(plethysmo_trial, fs)
    return np.concatenate([eeg_feats, hrv_feats])

