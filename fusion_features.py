"""
fusion_features.py

Combines EEG band-power features with cardiac (HRV) features into
interaction terms — the "combinations of waves at different levels" 

TWO APPROACHES ARE GIVEN, DELIBERATELY:

1. full_pairwise_interactions() — : every
   feature multiplied by every other feature (degree-2 interaction terms,
   via sklearn's PolynomialFeatures). With ~131 base features this produces
   ~131*130/2 = 8,515 new interaction terms.

   THE PROBLEM: DEAP gives me at most 32 subjects x 40 trials = 1,280
   total trials. Having ~8,500 features for ~1,280 samples means my
   model has more "knobs to turn" than data points — this is the textbook
   curse-of-dimensionality setup for overfitting. A model can find a
   spurious interaction term that perfectly separates my training data
   by pure chance, then fail completely on new data. This is a real risk,
   not a theoretical one — worth understanding before I lean on it. (future note to myself while revising my code)

2. curated_cross_modal_interactions() — a much smaller, theory-driven set
   of EEG x HRV interaction terms, chosen because there's an actual
   physiological reason to expect them to matter (e.g. alpha power is
   linked to relaxation, RMSSD is linked to parasympathetic/relaxed
   states — their PRODUCT is a reasonable "relaxation index" hypothesis,
   not a blind combination).

RECOMMENDATION to myself: start with the curated version. If I want to explore the
full combinatorial space anyway (which is a legitimate thing to try), pair
it with either PCA (to compress 8,500 features down to something like 20-50
components) or a Lasso/L1-regularized classifier (which can automatically
zero out most of the interaction terms and keep only the useful few) —
never feed 8,500 raw features straight into a RandomForest on 1,280 samples.

NOTE ON BLOOD PRESSURE:
DEAP does not include a blood pressure channel — it has EEG, EOG, EMG, GSR,
respiration, plethysmograph, and temperature, but no BP signal. The
plethysmograph gives you PULSE TIMING (useful for HRV) but not blood
pressure itself. If I want to genuinely fuse EEG + heart rate + blood
pressure, I have two real options:
  (a) find/collect a dataset that includes continuous BP (rare in public
      affective-computing datasets — most use PPG or ECG, not BP directly)
  (b) collect my own data with a wearable continuous BP sensor alongside
      an EEG headset — a much bigger undertaking, worth considering only
      once the DEAP-based version is working and validated.

-------------------------------------------------------------------------
-------------------------------------------------------------------------
**This module is written so a BP feature array can be added later without
restructuring anything — see `bp_features` parameter below.** 
-------------------------------------------------------------------------
-------------------------------------------------------------------------


"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures


def full_pairwise_interactions(feature_vector):
    """
    Literal 'permutations and combinations' of every feature pair.
    Input: 1D array of base features (e.g. the ~131 from features.py)
    Output: 1D array including original features + all pairwise products.

    WARNING: use PCA or L1 regularization downstream — see module docstring.
    """
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    # PolynomialFeatures expects a 2D array (n_samples, n_features)
    expanded = poly.fit_transform(feature_vector.reshape(1, -1))
    return expanded.flatten()


def curated_cross_modal_interactions(eeg_band_powers, hrv_feats, bp_features=None):
    """
    A small, theory-driven set of EEG x cardiac interaction terms.

    eeg_band_powers: dict like {"alpha": value, "beta": value, "theta": value, "gamma": value}
                      (use average across channels, or pick specific channels
                      like frontal ones depending on your hypothesis)
    hrv_feats: dict like {"rmssd": value, "sdnn": value}
    bp_features: optional dict like {"systolic": value, "diastolic": value} —
                 not available from DEAP, included so this function doesn't
                 need to change if you add a BP-capable dataset later.

    Returns a dict of named interaction features (named, not just indexed —
    this makes your later write-up much easier, since "alpha_x_rmssd" is
    interpretable and "feature[57]" is not).
    """
    eps = 1e-10
    interactions = {}

    # Alpha (relaxation-linked) x RMSSD (parasympathetic/relaxation-linked):
    # hypothesis -- both should be elevated together in genuinely relaxed states,
    # so their product should be a more specific "relaxation index" than either alone.
    interactions["alpha_x_rmssd"] = eeg_band_powers.get("alpha", 0) * hrv_feats.get("rmssd", 0)

    # Beta (active/anxious-linked) x SDNN (overall autonomic variability):
    interactions["beta_x_sdnn"] = eeg_band_powers.get("beta", 0) * hrv_feats.get("sdnn", 0)

    # Beta / RMSSD ratio: high beta with LOW relaxation-linked HRV is a
    # plausible "stress index" -- cortical activation without autonomic calm.
    interactions["beta_over_rmssd"] = eeg_band_powers.get("beta", 0) / (hrv_feats.get("rmssd", 0) + eps)

    # Gamma x SDNN: exploratory -- gamma is linked to active cognitive
    # processing, less established in emotion work, included as a genuine
    # open question rather than a confident claim.
    interactions["gamma_x_sdnn"] = eeg_band_powers.get("gamma", 0) * hrv_feats.get("sdnn", 0)

    if bp_features is not None:
        # Placeholder structure for when/if you add a BP-capable dataset.
        interactions["alpha_x_systolic"] = eeg_band_powers.get("alpha", 0) * bp_features.get("systolic", 0)
        interactions["beta_x_diastolic"] = eeg_band_powers.get("beta", 0) * bp_features.get("diastolic", 0)

    return interactions

