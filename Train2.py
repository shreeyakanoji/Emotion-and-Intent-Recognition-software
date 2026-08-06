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


def build_feature_matrix(eeg, plethysmo):
    """
    eeg:       (n_trials, 32, n_samples)
    plethysmo: (n_trials, n_samples)
    Returns X: (n_trials, n_features)
    """
    n_trials = eeg.shape[0]
    feature_rows = []
    for i in range(n_trials):
        feats = extract_trial_features(eeg[i], plethysmo[i])
        feature_rows.append(feats)
    return np.vstack(feature_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to DEAP's data_preprocessed_python folder")
    parser.add_argument("--target", choices=["valence", "arousal"], default="valence")
    parser.add_argument("--subjects", type=int, default=5, help="Number of subjects to load (max 32)")
    args = parser.parse_args()

    print(f"Loading {args.subjects} subject(s) from {args.data_dir} ...")
    eeg, plethysmo, labels = load_all_subjects(args.data_dir, n_subjects=args.subjects)
    y = labels[args.target]

    print(f"Loaded {eeg.shape[0]} trials. Extracting features ...")
    X = build_feature_matrix(eeg, plethysmo)
    print(f"Feature matrix shape: {X.shape}")  # (n_trials, 129) — 128 EEG + 1 asymmetry + 2 HRV = 131

    # Standardizing features matters here: band power and HRV features are on
    # very different numeric scales (microvolts^2 vs. milliseconds), and
    # RandomForest is fairly robust to this, but if you swap in an SVM or
    # logistic regression later, unscaled features will hurt you badly.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # Persist the trained model + scaler so other programs (like the live
    # dashboard app) can load a ready-to-use classifier instead of retraining
    # from scratch every time they start.
    import joblib
    joblib.dump({"model": clf, "scaler": scaler, "target": args.target}, "trained_model.joblib")
    print("\nSaved trained model + scaler to trained_model.joblib")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--- Held-out test set results ---")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 score: {f1:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Cross-validation gives a more honest estimate than a single train/test
    # split, especially with a small number of subjects — a single split can
    # get lucky or unlucky.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
    print(f"\n5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")

    # Feature importance — worth checking which features actually matter.
    # If frontal asymmetry or HRV rank highly, that's a genuinely interesting
    # thing to mention in a write-up.
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("\nTop 10 most important features (by index):")
    for idx in top_idx:
        print(f"  feature[{idx}]: importance={importances[idx]:.4f}")


if __name__ == "__main__":
    main()

