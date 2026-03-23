"""Evaluation metrics and backtesting for model comparison."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
)


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute classification metrics.

    Returns dict with all key metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # AUC-ROC (needs both classes present)
    if len(np.unique(y_true)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics["auc_roc"] = 0.0

    return metrics


def precision_at_top_n(y_true: np.ndarray, y_proba: np.ndarray, n: int = 10) -> float:
    """Precision among the top-N highest-probability predictions."""
    if len(y_true) < n:
        n = len(y_true)
    top_indices = np.argsort(y_proba)[-n:]
    top_true = y_true[top_indices]
    return float(np.mean(top_true))


def backtest_roi(
    test_df: pd.DataFrame,
    y_proba: np.ndarray,
    top_n: int = 10,
) -> dict:
    """Simple backtest: pick top-N predictions, compute average actual return.

    Args:
        test_df: Test DataFrame with 'future_return' column.
        y_proba: Predicted buy probabilities aligned with test_df.
        top_n: Number of top picks to evaluate.

    Returns:
        Dict with backtest metrics.
    """
    if len(test_df) < top_n:
        top_n = len(test_df)

    indices = np.argsort(y_proba)[-top_n:]

    actual_returns = test_df.iloc[indices]["future_return"].values
    avg_return = float(np.nanmean(actual_returns))
    positive_pct = float(np.mean(actual_returns > 0)) * 100
    max_return = float(np.nanmax(actual_returns))
    min_return = float(np.nanmin(actual_returns))

    # Sharpe-like ratio (mean return / std of returns)
    std_return = float(np.nanstd(actual_returns))
    sharpe = avg_return / std_return if std_return > 0 else 0.0

    return {
        "avg_return_pct": round(avg_return, 3),
        "positive_rate_pct": round(positive_pct, 1),
        "max_return_pct": round(max_return, 3),
        "min_return_pct": round(min_return, 3),
        "sharpe_ratio": round(sharpe, 3),
        "top_n": top_n,
    }


def full_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    test_df: pd.DataFrame,
    model_name: str,
) -> dict:
    """Run all evaluations and return a summary dict."""
    clf_metrics = evaluate_classifier(y_true, y_pred, y_proba)
    p_at_10 = precision_at_top_n(y_true, y_proba, n=10)
    p_at_20 = precision_at_top_n(y_true, y_proba, n=20)
    bt = backtest_roi(test_df, y_proba, top_n=20)

    return {
        "model_name": model_name,
        **clf_metrics,
        "precision_at_10": round(p_at_10, 3),
        "precision_at_20": round(p_at_20, 3),
        "backtest": bt,
    }
