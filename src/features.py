"""Feature engineering + the preprocessing transformer.

Two responsibilities:
  * `engineer_features` adds a few domain-motivated columns.
  * `build_preprocessor` builds a leak-free sklearn ColumnTransformer that
    scales numeric features and one-hot-encodes categorical ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns that encode churn intuition the raw data hides.

    - `num_services`: how many of the 8 add-on services the customer actually
      uses. Counting only literal "Yes" values means "No internet service" and
      "No phone service" correctly count as zero. Customers with a deeper
      product footprint tend to be stickier.
    - `tenure_group`: tenure bucketed into business-friendly bands. Churn is
      highly non-linear in tenure (most churn happens early), so a binned
      version helps linear models and aids interpretation.
    - `avg_charges_per_month`: lifetime spend normalised by tenure. For brand
      new customers (tenure == 0) we fall back to MonthlyCharges to avoid
      dividing by zero.
    """
    df = df.copy()

    df["num_services"] = (df[config.SERVICE_COLUMNS] == "Yes").sum(axis=1)

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, np.inf],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5yr+"],
    ).astype(str)

    df["avg_charges_per_month"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"].replace(0, np.nan),
        df["MonthlyCharges"],
    )

    return df


def split_columns(X: pd.DataFrame):
    """Return (numeric_cols, categorical_cols) inferred from dtypes."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build the preprocessing step.

    Wrapping scaling + encoding in a ColumnTransformer (and later inside a
    Pipeline) is the key to avoiding data leakage: the scaler's mean/std and
    the encoder's category list are learned on training folds only, then
    applied to validation/test data.
    """
    numeric_cols, categorical_cols = split_columns(X)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )
