"""Evaluation metrics and diagnostic plots.

All plotting helpers save a PNG to reports/figures/ and return the path, so the
same functions are reusable from both the notebook and the CLI pipeline.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # non-interactive backend so scripts run headless
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Headline classification metrics for the positive (churn) class."""
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def best_threshold_by_f1(y_true, y_proba) -> float:
    """Pick the probability cutoff that maximises F1.

    The default 0.5 cutoff is rarely optimal on imbalanced data; tuning the
    threshold lets us trade precision for the recall the business actually
    wants (catching would-be churners).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision/recall have one more element than thresholds; align them.
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def save_confusion_matrix(y_true, y_pred, title, filename) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Stay", "Churn"], cmap="Blues", ax=ax
    )
    ax.set_title(title)
    fig.tight_layout()
    path = config.FIGURES_DIR / filename
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def save_roc_curve(y_true, y_proba, title, filename) -> str:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = config.FIGURES_DIR / filename
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def save_pr_curve(y_true, y_proba, title, filename) -> str:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    path = config.FIGURES_DIR / filename
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
