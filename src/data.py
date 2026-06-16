"""Data loading and cleaning.

The raw IBM Telco file ships with a handful of well-known data-quality issues.
This module fixes each of them deterministically and returns a tidy DataFrame.
"""

from __future__ import annotations

import pandas as pd

from . import config


def load_raw(path=None) -> pd.DataFrame:
    """Read the raw CSV exactly as distributed (no transformations yet)."""
    path = path or config.RAW_DATA_PATH
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw Telco DataFrame.

    Fixes applied, each motivated by something actually wrong in the file:

    1. `TotalCharges` is stored as text and contains 11 blank strings. Those
       11 rows are all brand-new customers (tenure == 0), so their true total
       spend is 0. We coerce the column to numeric and fill those blanks with 0
       rather than dropping them, since they are legitimate customers.
    2. `SeniorCitizen` is encoded as 0/1 while every other yes/no field uses
       the strings "Yes"/"No". We re-map it to strings so categoricals are
       handled uniformly downstream.
    3. The target `Churn` is mapped to a 0/1 integer for modelling.
    4. `customerID` carries no predictive signal and is removed.
    """
    df = df.copy()

    # (1) TotalCharges: blank strings -> NaN -> numeric -> fill new customers with 0
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].replace(" ", pd.NA), errors="coerce"
    )
    # Every NaN here corresponds to tenure == 0 (verified on the raw data),
    # meaning the customer has not been billed a full cycle yet.
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # (2) SeniorCitizen 0/1 -> "No"/"Yes" for consistent categorical handling
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # (3) Target -> integer label
    df[config.TARGET] = (df[config.TARGET] == "Yes").astype(int)

    # (4) Drop the identifier
    df = df.drop(columns=[config.ID_COLUMN])

    return df


def load_clean(path=None) -> pd.DataFrame:
    """Convenience wrapper: load the raw file and clean it in one call."""
    return clean(load_raw(path))
