"""Candidate models and full (preprocess + estimator) pipelines.

We compare three families that trade off interpretability vs. flexibility:
  * Logistic Regression  - linear, fast, highly interpretable baseline.
  * Random Forest        - bagged trees, captures interactions, robust.
  * HistGradientBoosting - sklearn's gradient-boosted trees (XGBoost-style),
                           usually the strongest tabular performer.

Every model is wrapped in a Pipeline together with the preprocessor so that
cross-validation re-fits preprocessing on each fold (no leakage) and so the
saved artifact accepts raw, human-readable input at inference time.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config
from .features import build_preprocessor


def candidate_models() -> dict:
    """Return the dict of model name -> unfitted estimator.

    `class_weight="balanced"` tells the linear / forest models to up-weight the
    minority churn class (~27%), which matters because we care far more about
    catching churners than about raw accuracy.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=config.RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
        # HistGradientBoosting has no class_weight param; we pass per-sample
        # weights at fit time instead (see pipeline.py).
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=400,
            max_depth=None,
            l2_regularization=1.0,
            random_state=config.RANDOM_STATE,
        ),
    }


def build_pipeline(estimator, X: pd.DataFrame) -> Pipeline:
    """Glue the preprocessor and an estimator into one Pipeline."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X)),
            ("model", estimator),
        ]
    )
