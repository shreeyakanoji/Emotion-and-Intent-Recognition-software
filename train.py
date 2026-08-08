"""
train.py

Ties everything together on real (or synthetic dry-run) data:
  1. Load a dataset via dataset_adapters.py (DEAP, DREAMER, or synthetic)
  2. Extract features (band power + frontal asymmetry + HRV) per trial
  3. Train a classifier with a proper train/test split AND cross-validation
  4. Report real accuracy/F1 — not a mock number

CHANGED FROM v2: the trained bundle now also stores eeg_channel_names,
eeg_fs, and cardiac_fs alongside the model/scaler. This is what lets
pipeline.py extract features correctly at inference time regardless of
which dataset the model was trained on — inference must use the exact
same montage/sampling-rate assumptions as training, or features silently
won't line up.

USAGE:
    python train.py deap /path/to/data_preprocessed_python --target valence --subjects 5
    python train.py synthetic --target valence --subjects 3
"""

import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from features import extract_trial_features
import dataset_adapters


def build_feature_matrix(eeg, cardiac, channel_names, eeg_fs, cardiac_fs):
    """
    eeg:     (n_trials, n_channels, n_samples)
    cardiac: (n_trials, n_samples_cardiac)
    Returns X: (n_trials, n_features)
    """
    n_trials = eeg.shape[0]
    feature_rows = []
    for i in range(n_trials):
        feats = extract_trial_features(eeg[i], cardiac[i], channel_names, eeg_fs, cardiac_fs)
        feature_rows.append(feats)
    return np.vstack(feature_rows)


def train_model(eeg, cardiac, channel_names, eeg_fs, cardiac_fs, y, target,
                 n_estimators=200, progress_cb=None):
    """
    Reusable training routine — callable directly (e.g. from unified_app.py)
    without going through argparse or touching the filesystem.

    Returns (bundle, report) where:
      bundle = {"model", "scaler", "target", "eeg_channel_names", "eeg_fs",
                "cardiac_fs"} — everything pipeline.py needs to reproduce
                the exact same features at inference time.
      report = dict of metrics for display.
    """
    def log(msg):
        if progress_cb:
            progress_cb(msg)

    log(f"Loaded {eeg.shape[0]} trials. Extracting features ...")
    X = build_feature_matrix(eeg, cardiac, channel_names, eeg_fs, cardiac_fs)
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
        "eeg_channel_names": channel_names,
        "eeg_fs": eeg_fs,
        "cardiac_fs": cardiac_fs,
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
    parser.add_argument("dataset", choices=["deap", "synthetic"],
                         help="'deap' needs data_dir; 'synthetic' needs no real data")
    parser.add_argument("data_dir", nargs="?", default=None,
                         help="Path to DEAP's data_preprocessed_python folder (deap only)")
    parser.add_argument("--target", choices=["valence", "arousal"], default="valence")
    parser.add_argument("--subjects", type=int, default=5, help="Number of subjects to load")
    args = parser.parse_args()

    print(f"Loading data ({args.dataset}) ...")
    if args.dataset == "deap":
        if not args.data_dir:
            raise SystemExit("deap requires a data_dir argument")
        import os
        paths = [os.path.join(args.data_dir, f"s{i:02d}.dat") for i in range(1, args.subjects + 1)]
        bundle_data = dataset_adapters.load_deap_dat_paths(paths)
    else:
        bundle_data = dataset_adapters.load_synthetic(n_subjects=args.subjects)

    y = bundle_data["labels"][args.target]

    bundle, report = train_model(
        bundle_data["eeg"], bundle_data["cardiac"],
        bundle_data["eeg_channel_names"], bundle_data["eeg_fs"], bundle_data["cardiac_fs"],
        y, args.target, progress_cb=print,
    )

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
