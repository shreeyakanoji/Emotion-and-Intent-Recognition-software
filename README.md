# EEG/Cardiac Emotion Recognition — v2

Upgraded from the original mock-trained prototype to a pipeline trained on
real, labeled data (DEAP dataset).

## Setup

```
pip install -r requirements.txt --break-system-packages
```

### Option A — one-page app (recommended): get data, train, and test in the browser

```
streamlit run unified_app.py
```

This single page lets you:
1. **Get data** — generate synthetic DEAP-shaped data (no DEAP access
   needed) or upload your own DEAP-format `.dat` file(s).
2. **Train** — pick valence or arousal, train a model on the loaded data,
   see accuracy/F1/cross-validation/feature importances, and download the
   resulting `trained_model.joblib`.
3. **Test** — run the trained model on a trial from the loaded dataset, or
   upload a *different* `.dat` file and test on that, and see the
   prediction, confidence, and every intermediate feature the pipeline
   computed.

### Option B — CLI + separate live-replay demo

```
python generate_synthetic_deap.py ./synthetic_deap_data --subjects 3
python train.py ./synthetic_deap_data --target valence --subjects 3
streamlit run app.py
```

`app.py` is the original live-replay dashboard: it loads an existing
`trained_model.joblib` and streams a synthetic subject's data through it
chunk by chunk, for a "live sensor feed" style demo. It expects
`trained_model.joblib` to already exist in the working directory (produced
by either `train.py` or the unified app's download button).

### Using real DEAP data

Request access (free, academic use, requires signing a EULA):
http://www.eecs.qmul.ac.uk/mmv/datasets/deap/
Download the **"data_preprocessed_python"** folder — this gives you
`s01.dat` through `s32.dat`. Either upload those files in `unified_app.py`,
or point the CLI at the folder:
```
python train.py /path/to/data_preprocessed_python --target valence --subjects 5
```
Start with a handful of subjects to sanity-check the pipeline runs before
committing to a full 32-subject run (which takes longer and uses more
memory).

## Project layout

```
deap_loader.py             loads real or synthetic DEAP .dat files
features.py                band power, frontal asymmetry, HRV features
fusion_features.py         EEG x cardiac interaction/fusion features
pipeline.py                EmotionPipeline: filter -> features -> fuse -> classify
train.py                   trains + saves trained_model.joblib
generate_synthetic_deap.py makes fake-but-realistically-shaped DEAP files
generic_loader.py          loads your own device's CSV/TSV data (any channel count, any rate)
theme.py                   shared pastel-neon visual theme for both apps
unified_app.py             one-page app: get data -> train -> test, in the browser
app.py                     Streamlit live-replay dashboard for an existing trained model
legacy/                    old v1 mock-data prototype, kept for history only —
                            not imported by anything above, not required to run
```

## Fixed since last version

- File/import mismatch: source files had been saved with capitalized names
  (`Deep_loader.py`, `Features.py`, `Pipeline.py`, ...) while every module
  imported the lowercase names (`deap_loader`, `features`, `pipeline`, ...).
  On a case-sensitive filesystem (Linux) none of these resolved. Files are
  now saved under the exact lowercase names their own imports expect.
  `Deep_loader.py` was also a plain typo for `deap_loader.py` (the DEAP
  dataset loader) — fixed.
- `Train.py` had a syntax error (its module docstring was never closed) and
  was missing the `joblib.dump(...)` call needed to produce
  `trained_model.joblib`, which `app.py` requires to start. It's been
  removed. `Train2.py`, the working version with the save step, is now the
  single `train.py`.
- The built-in synthetic data generator baked in an unrealistically clean,
  noise-free signal difference between classes, so models trained on it hit
  95-100% accuracy — misleadingly high compared to the 55-70% you'd expect
  on real DEAP data. It's now genuinely noisy (small, randomized,
  partial-channel effects against a large noise floor), landing models in a
  believable 60-80% range instead. The app also now flags a warning if
  accuracy comes back ≥95%, since that's a sign of data leakage, not a
  great model.
- The pipeline was hard-locked to DEAP's exact 32-channel, 128Hz layout.
  Feature extraction, training, and inference are now generalized to work
  with any channel count and any sampling rate — see "Using your own
  hardware's data" below. Channel count and sampling rate are tracked
  through the whole train → save → load → predict cycle and validated at
  inference time, so a channel-count mismatch fails with a clear error
  instead of a cryptic crash or (worse) a silently wrong prediction.

## Using your own hardware's data (not just DEAP)

`unified_app.py`'s "Upload your own EEG/cardiac CSV files (any device)"
option accepts a CSV/TSV per trial/recording: each column is a channel, one
column is your cardiac signal (PPG/ECG/pulse — auto-detected by name, e.g.
a column called `ppg` or `ecg`, or falls back to the last numeric column),
and everything else is treated as an EEG channel. You set the sampling rate
and label each uploaded file (Low/High) yourself, then train exactly like
the DEAP path.

Two things worth knowing:
- **Channel count and sampling rate must match between training and
  testing.** A model trained on 8 channels at 256Hz can only be tested on
  8-channel, 256Hz data — this is validated automatically with a clear
  error if it doesn't match.
- **Frontal asymmetry is approximate on non-DEAP layouts.** The literature-
  standard measure uses real F3/F4 electrode positions. If your channel
  names don't include those (most consumer headsets won't), the pipeline
  falls back to an approximate first-half-vs-second-half channel split as a
  rough left/right hemisphere proxy — worth flagging as an approximation,
  not the standard measure, in any write-up.

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
