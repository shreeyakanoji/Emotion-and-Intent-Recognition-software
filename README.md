# EEG/Cardiac Emotion Recognition — v2

Upgraded from the original mock-trained prototype to a pipeline trained on
real, labeled data (DEAP dataset).

## Setup

1. Request DEAP dataset access (free, academic use, requires signing a EULA):
   http://www.eecs.qmul.ac.uk/mmv/datasets/deap/
2. Download the **"data_preprocessed_python"** folder — this gives you
   `s01.dat` through `s32.dat`.
3. Install dependencies:
   ```
   pip install numpy scipy scikit-learn --break-system-packages
   ```
4. Run:
   ```
   python train.py /path/to/data_preprocessed_python --target valence --subjects 5
   ```
   Start with `--subjects 5` to sanity-check the pipeline runs before doing
   a full 32-subject run (which takes longer and uses more memory).

## What changed from v1

| | v1 (mock) | v2 (this version) |
|---|---|---|
| Data | Synthetic sine waves | Real DEAP EEG + plethysmograph |
| Training | Fake Gaussian blobs, seeded | Real labels (valence/arousal) |
| Features | 2 (alpha power, ECG variance) | ~131 (band power x 32 channels x 4 bands, frontal asymmetry, HRV) |
| Evaluation | None | Held-out test set + 5-fold cross-validation |
| Channels | Hardcoded row 0/1 | Named 10-20 system channels |

## Honest expectations for results

Binary valence/arousal classification on DEAP is a hard, noisy problem —
self-reported emotion labels are subjective, and simple feature-based
classifiers (like this RandomForest) typically land somewhere in the
**55-70% accuracy** range in published work, not 90%+. If you get something
in that range, that's a legitimate result, not a failure. If you get 95%+,
be suspicious — that usually means a data leak (e.g. trials from the same
subject/video ending up in both train and test) rather than a genuinely
strong model.

## Known limitations, worth stating explicitly rather than hiding

- The plethysmograph channel is a proxy for cardiac activity, not literal ECG.
- DEAP's labels are self-reported after the fact, which is a real source of
  label noise in the field, not something specific to this code.
- Only using the plethysmograph for HRV; GSR (channel 36) is a widely-used
  arousal-correlated signal that isn't used here yet — a natural next
  extension if you want to expand the project.
# Emotion-and-Intent-Recognition-software
