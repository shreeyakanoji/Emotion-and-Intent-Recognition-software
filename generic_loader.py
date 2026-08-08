"""
generic_loader.py

Loads EEG/cardiac data that ISN'T DEAP's format — your own headset export,
your own ECG/PPG device, any CSV/TSV file. This is what lets the pipeline
work with any hardware, not just the DEAP dataset.

EXPECTED FILE FORMAT (per trial/recording):
A CSV or TSV file where each ROW is one time sample and each COLUMN is one
channel. Any non-numeric columns (like a timestamp column) are dropped
automatically. One numeric column should be your cardiac signal (PPG/ECG/
pulse) — auto-detected by column name if it contains a recognizable
keyword (ppg, ecg, ekg, cardiac, pulse, plethysmo, heart, hr); otherwise
you specify it explicitly, or it falls back to assuming the last column.
Every other numeric column is treated as an EEG channel.

This is intentionally a simple, permissive format precisely because "any
device, any lab" can't be expected to already match a specific dataset's
layout — the tradeoff is that channel *identity* (e.g. true 10-20 system
electrode positions) isn't guaranteed, which affects how meaningful the
frontal-asymmetry feature is (see features.py's fallback behavior for
channel layouts that don't include real F3/F4 names).
"""

import io

import numpy as np
import pandas as pd

CARDIAC_KEYWORDS = ["ppg", "ecg", "ekg", "cardiac", "pulse", "plethysmo", "heart", "hr"]


def _sniff_delimiter(text_sample):
    first_line = text_sample.splitlines()[0] if text_sample else ""
    if "\t" in first_line:
        return "\t"
    if ";" in first_line and "," not in first_line:
        return ";"
    return ","


def load_trial_csv(file_bytes, cardiac_column=None):
    """
    Parses one uploaded trial file's raw bytes into arrays.

    Returns (eeg_array, cardiac_array, eeg_channel_names, cardiac_column_used)
      eeg_array: (n_channels, n_samples)
      cardiac_array: (n_samples,)
    Raises ValueError with a clear message if the file doesn't have enough
    usable numeric columns.
    """
    text = file_bytes.decode("utf-8", errors="ignore")
    delim = _sniff_delimiter(text)
    df = pd.read_csv(io.StringIO(text), sep=delim)

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        raise ValueError(
            f"Found only {numeric_df.shape[1]} numeric column(s) — need at "
            "least 2 (one or more EEG channels + one cardiac channel). "
            "Check that your file has a header row and numeric data columns."
        )

    if cardiac_column is None or cardiac_column not in numeric_df.columns:
        cardiac_column = next(
            (c for c in numeric_df.columns if any(k in str(c).lower() for k in CARDIAC_KEYWORDS)),
            numeric_df.columns[-1],  # fallback: assume last numeric column
        )

    eeg_cols = [c for c in numeric_df.columns if c != cardiac_column]
    eeg_array = numeric_df[eeg_cols].to_numpy(dtype=float).T  # (n_channels, n_samples)
    cardiac_array = numeric_df[cardiac_column].to_numpy(dtype=float)

    return eeg_array, cardiac_array, [str(c) for c in eeg_cols], str(cardiac_column)


def sniff_columns(file_bytes):
    """Quick peek at a file's numeric column names, for a 'pick your cardiac
    column' dropdown in the UI without fully parsing everything twice."""
    text = file_bytes.decode("utf-8", errors="ignore")
    delim = _sniff_delimiter(text)
    df = pd.read_csv(io.StringIO(text), sep=delim, nrows=5)
    numeric_df = df.select_dtypes(include=[np.number])
    return [str(c) for c in numeric_df.columns]
