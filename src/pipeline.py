"""End-to-end training pipeline (CLI).

Run with:  python -m src.pipeline

Steps: load -> clean -> engineer -> split -> cross-validate 3 models ->
select best by CV ROC-AUC -> fit on train -> tune threshold -> evaluate on the
held-out test set -> save model, metrics, and figures.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from . import config, data, evaluate, features, model as model_mod


def _prepare():
    """Load, clean, engineer features, and split into train/test."""
    df = data.load_clean()
    df = features.engineer_features(df)

    # Persist the analysis-ready table for transparency / reuse.
    config.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.PROCESSED_DATA_PATH, index=False)

    y = df[config.TARGET]
    X = df.drop(columns=[config.TARGET])

    # Stratify so train and test keep the same ~27% churn rate.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test


def _cross_validate(X_train, y_train) -> dict:
    """5-fold stratified CV (ROC-AUC) for every candidate model."""
    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=config.RANDOM_STATE
    )
    results = {}
    for name, estimator in model_mod.candidate_models().items():
        pipe = model_mod.build_pipeline(estimator, X_train)
        scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        results[name] = {"cv_auc_mean": float(scores.mean()),
                         "cv_auc_std": float(scores.std())}
        print(f"  {name:24s} ROC-AUC = {scores.mean():.4f} +/- {scores.std():.4f}")
    return results


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("1) Preparing data ...")
    X_train, X_test, y_train, y_test = _prepare()
    print(f"   train={X_train.shape}  test={X_test.shape}  "
          f"churn_rate={y_train.mean():.3f}")

    print("2) Cross-validating candidate models ...")
    cv_results = _cross_validate(X_train, y_train)

    best_name = max(cv_results, key=lambda k: cv_results[k]["cv_auc_mean"])
    print(f"   -> best by CV ROC-AUC: {best_name}")

    print("3) Fitting best model on full training set ...")
    best_estimator = model_mod.candidate_models()[best_name]
    pipe = model_mod.build_pipeline(best_estimator, X_train)
    pipe.fit(X_train, y_train)

    print("4) Evaluating on held-out test set ...")
    proba = pipe.predict_proba(X_test)[:, 1]

    # Default 0.5 threshold vs. an F1-optimised threshold.
    pred_default = (proba >= 0.5).astype(int)
    threshold = evaluate.best_threshold_by_f1(y_train, pipe.predict_proba(X_train)[:, 1])
    pred_tuned = (proba >= threshold).astype(int)

    metrics = {
        "best_model": best_name,
        "cv_results": cv_results,
        "decision_threshold": threshold,
        "test_default_threshold": evaluate.compute_metrics(
            y_test.values, pred_default, proba
        ),
        "test_tuned_threshold": evaluate.compute_metrics(
            y_test.values, pred_tuned, proba
        ),
    }

    print("5) Saving figures ...")
    evaluate.save_confusion_matrix(
        y_test.values, pred_tuned,
        f"Confusion matrix ({best_name}, thr={threshold:.2f})",
        "confusion_matrix.png",
    )
    evaluate.save_roc_curve(
        y_test.values, proba, f"ROC curve ({best_name})", "roc_curve.png"
    )
    evaluate.save_pr_curve(
        y_test.values, proba, f"Precision-Recall ({best_name})", "pr_curve.png"
    )

    print("6) Permutation importance (top drivers of churn) ...")
    perm = permutation_importance(
        pipe, X_test, y_test, n_repeats=10,
        random_state=config.RANDOM_STATE, scoring="roc_auc", n_jobs=-1,
    )
    importances = (
        pd.Series(perm.importances_mean, index=X_test.columns)
        .sort_values(ascending=False)
    )
    metrics["top_features"] = importances.head(10).round(4).to_dict()
    print(importances.head(10).to_string())

    print("7) Persisting model + metrics ...")
    joblib.dump(pipe, config.MODEL_PATH)
    config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    t = metrics["test_tuned_threshold"]
    print("\nDONE. Test-set performance (tuned threshold):")
    print(f"  ROC-AUC={t['roc_auc']:.3f}  recall={t['recall']:.3f}  "
          f"precision={t['precision']:.3f}  f1={t['f1']:.3f}")


if __name__ == "__main__":
    main()
