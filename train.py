"""
train.py

Ties everything together on REAL data:
  1. Load DEAP subjects
  2. Extract features (band power + frontal asymmetry + HRV) per trial
  3. Train a classifier with a proper train/test split AND cross-validation
  4. Report real accuracy/F1 — not a mock number

USAGE:
    python train.py /path/to/data_preprocessed_python --target valence --subjects 5

Start with a small number of subjects (e.g. 5) first — 32 subjects x 40 trials
x 128 features is small enough to run fast, but starting small lets you sanity
check the whole pipeline before committing to a full run.
"""

import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from deap_loader import load_all_subjects
from features import extract_trial_features


def build_feature_matrix(eeg, plethysmo, channel_names=None, fs=None):
    """
    eeg:       (n_trials, n_channels, n_samples) — n_channels can be any
               value, not just DEAP's 32, as long as it's consistent across
               all trials in this matrix.
    plethysmo: (n_trials, n_samples)
    channel_names: optional list of channel names for eeg's channel axis.
    fs: sampling rate in Hz of this data. Defaults to DEAP's 128Hz if not
        given — IMPORTANT to set correctly for non-DEAP hardware, since
        frequency-band features (theta/alpha/beta/gamma) are computed
        relative to the sampling rate.
    Returns X: (n_trials, n_features)
    """
    from deap_loader import SAMPLING_RATE as DEAP_FS
    fs = fs or DEAP_FS
    n_trials = eeg.shape[0]
    feature_rows = []
    for i in range(n_trials):
        feats = extract_trial_features(eeg[i], plethysmo[i], fs=fs, channel_names=channel_names)
        feature_rows.append(feats)
    return np.vstack(feature_rows)


def train_model(eeg, plethysmo, y, target, n_estimators=200, progress_cb=None,
                 channel_names=None, fs=None):
    """
    Reusable training routine — same logic the CLI below uses, but callable
    directly (e.g. from a Streamlit app) without going through argparse or
    touching the filesystem.

    eeg:       (n_trials, n_channels, n_samples) — any channel count, as
               long as it matches what you'll feed EmotionPipeline later.
    plethysmo: (n_trials, n_samples)
    y:         (n_trials,) binary labels
    target:    a short label name (e.g. "valence", "arousal", "stress") —
               just stored in the bundle for the pipeline to label
               predictions with later.
    channel_names: optional list of channel names. Stored in the bundle so
               EmotionPipeline can reuse the exact same channel semantics
               (matters for the frontal-asymmetry feature) at inference time.
    fs: sampling rate in Hz of this data (defaults to DEAP's 128Hz). Stored
               in the bundle so EmotionPipeline automatically uses the
               correct rate at inference time instead of assuming DEAP's.
    progress_cb: optional callable(str) for status messages (e.g. st.write)

    Returns (bundle, report) where:
      bundle = {"model": clf, "scaler": scaler, "target": target,
                "channel_names": channel_names, "n_channels": eeg.shape[1],
                "fs": fs}
               (the exact dict shape trained_model.joblib stores)
      report = dict of metrics: accuracy, f1, confusion_matrix (list),
               cv_mean, cv_std, feature_importances (top 10 as list of
               (index, importance))
    """
    from deap_loader import SAMPLING_RATE as DEAP_FS
    fs = fs or DEAP_FS

    def log(msg):
        if progress_cb:
            progress_cb(msg)

    n_channels = eeg.shape[1]
    log(f"Loaded {eeg.shape[0]} trials ({n_channels} channels, {fs}Hz). Extracting features ...")
    X = build_feature_matrix(eeg, plethysmo, channel_names=channel_names, fs=fs)
    log(f"Feature matrix shape: {X.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    log("Running 5-fold cross-validation ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")

    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    top_features = [(int(idx), float(importances[idx])) for idx in top_idx]

    bundle = {
        "model": clf,
        "scaler": scaler,
        "target": target,
        "channel_names": channel_names,
        "n_channels": n_channels,
        "fs": fs,
    }
    report = {
        "n_trials": int(eeg.shape[0]),
        "n_features": int(X.shape[1]),
        "accuracy": float(acc),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "top_features": top_features,
    }
    return bundle, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to DEAP's data_preprocessed_python folder")
    parser.add_argument("--target", choices=["valence", "arousal"], default="valence")
    parser.add_argument("--subjects", type=int, default=5, help="Number of subjects to load (max 32)")
    args = parser.parse_args()

    print(f"Loading {args.subjects} subject(s) from {args.data_dir} ...")
    eeg, plethysmo, labels = load_all_subjects(args.data_dir, n_subjects=args.subjects)
    y = labels[args.target]

    bundle, report = train_model(eeg, plethysmo, y, args.target, progress_cb=print)

    import joblib
    joblib.dump(bundle, "trained_model.joblib")
    print("\nSaved trained model + scaler to trained_model.joblib")

    print("\n--- Held-out test set results ---")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(f"F1 score: {report['f1']:.3f}")
    print("Confusion matrix:")
    print(np.array(report["confusion_matrix"]))

    print(f"\n5-fold CV accuracy: {report['cv_mean']:.3f} +/- {report['cv_std']:.3f}")

    print("\nTop 10 most important features (by index):")
    for idx, importance in report["top_features"]:
        print(f"  feature[{idx}]: importance={importance:.4f}")


if __name__ == "__main__":
    main()

