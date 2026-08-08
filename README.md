# EEG/Cardiac Emotion Recognition — v3 (multi-dataset)

## What changed from v2 in this pass

**Two things that would have crashed on launch, fixed:**
- `requirements.txt` was referenced in the old README but never included — added.
- `unified_app.py` imported `from theme import apply_theme`, but `theme.py` didn't
  exist anywhere in the repo — added a minimal placeholder; swap in real styling
  whenever you want.

**Made dataset-agnostic**, per your ask for "upload whatever dataset and it still
works": `features.py` and `pipeline.py` no longer hardcode DEAP's 32-channel montage —
they take `channel_names`/`fs` explicitly, and `train.py` now stores those in the
trained model bundle so inference automatically matches whatever it was trained on.
All dataset-specific parsing now lives in one place: `dataset_adapters.py`.

**Honesty note, read before you trust this:** this is NOT a universal "any dataset"
parser — that isn't realistically possible, because raw EEG/physiological file
formats genuinely differ (channel counts, sampling rates, file structure, pickle
vs. .mat vs. something else). What you actually have is a clean *extension point*:
adding a new dataset means writing one loader function in `dataset_adapters.py`
that returns the normalized shape documented at the top of that file, then
registering it. DEAP is fully tested (the synthetic generator mimics its exact
shape byte-for-byte). **DREAMER's adapter is written to the published spec but has
not been run against a real DREAMER.mat file** — verify the parsed shapes on your
first real run before trusting results; scipy's nested MATLAB-struct handling can
be finicky and may need small adjustments.

## Setup

```
pip install -r requirements.txt --break-system-packages
```

### Option A — one-page app (recommended)

```
streamlit run unified_app.py
```

1. **Get data** — synthetic (no real data needed), DEAP `.dat` file(s), or
   DREAMER `.mat` (see honesty note above on DREAMER).
2. **Train** — pick valence or arousal, see accuracy/F1/CV/feature importances,
   download `trained_model.joblib`.
3. **Test** — run the model on a loaded trial and see prediction + every
   intermediate feature.

### Option B — CLI

```
python train.py synthetic --target valence --subjects 3
python train.py deap /path/to/data_preprocessed_python --target valence --subjects 5
```

Note: `app.py` (the separate live-replay demo mentioned in earlier versions) wasn't
in this batch of files, so it isn't in this zip — `unified_app.py` covers the same
ground (generate/upload data, train, test) in one page. If you want the standalone
live-replay version back, upload it and I'll wire it into the new adapter layer too.

## Getting real data — fast options while DEAP is pending

DEAP access (http://www.eecs.qmul.ac.uk/mmv/datasets/deap/) requires an EULA
review that can take days to weeks. If you want real EEG+cardiac results sooner:

- **DREAMER** — 23 subjects, 14-ch EEG + 2-ch ECG, valence/arousal/dominance
  ratings. Original source is a single `DREAMER.mat` download (much lighter
  request process than DEAP's manual review). Adapter included — verify shapes
  on first run (see honesty note above).

Both give you real multimodal EEG+cardiac data your pipeline is actually built
for — unlike WESAD/CASE, which have no EEG channel at all and would need an
entirely different (non-EEG) feature set to use.

## Project layout

```
dataset_adapters.py    all dataset-specific parsing lives here (DEAP, DREAMER, synthetic)
features.py            band power / frontal asymmetry / HRV — dataset-agnostic
fusion_features.py     curated + full-pairwise cross-modal interaction terms
pipeline.py            filter -> features -> fuse -> classify, reads montage from the trained bundle
train.py               CLI + train_model() used by the Streamlit app
unified_app.py          one-page get-data/train/test app
generate_synthetic_deap.py   dry-run data generator — DO NOT report results from this as real findings
theme.py                minimal Streamlit styling (placeholder, added this pass)
draft_1.py, realtime_pipeline.py   earlier v1 mock-data prototype — kept for history, not wired in
```

## Honest expectations for results (on REAL data)

Binary valence/arousal classification is a hard, noisy problem — self-reported
emotion labels are subjective, and simple feature-based classifiers typically
land in the **55–70% accuracy** range in published work, not 90%+. If you get
95%+ on real data, be suspicious — that usually means a data leak, not a
genuinely strong model. (On the synthetic generator, near-100% is *expected*
and means nothing — see the in-app warning.)
