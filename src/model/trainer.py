"""Model training — train, compare, and select the best model."""

import json
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data.loader import load_all_data
from src.features.pipeline import engineer_features, get_feature_columns
from src.model.targets import add_targets_all_symbols
from src.model.evaluator import full_evaluation
from src.utils.config import (
    BEST_MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    SCALER_PATH,
    TRAIN_FRACTION,
    DEFAULT_PERIOD_DAYS,
    DEFAULT_PROFIT_THRESHOLD,
)


def get_candidate_models() -> dict:
    """Return dict of model name -> untrained model instance."""
    return {
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=1.5,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=1.5,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight={0: 1, 1: 1.5},
            random_state=42,
            n_jobs=-1,
        ),
    }


def prepare_training_data(
    period_days: int = DEFAULT_PERIOD_DAYS,
    profit_threshold: float = DEFAULT_PROFIT_THRESHOLD,
) -> tuple[pd.DataFrame, list[str]]:
    """Load data, engineer features, build targets.

    Returns:
        (full_df, feature_columns)
    """
    print("Loading data from all categories...")
    raw = load_all_data()
    print(f"  Loaded {len(raw)} rows, {raw['Symbol'].nunique()} symbols")

    print("Engineering features...")
    featured = engineer_features(raw)
    print(f"  Feature-engineered: {len(featured)} rows, {len(featured.columns)} columns")

    print("Building target variables...")
    with_targets = add_targets_all_symbols(featured, period_days, profit_threshold)

    # Drop rows with NaN targets or NaN features
    feature_cols = get_feature_columns(with_targets)
    all_needed = feature_cols + ["target_buy", "future_return", "optimal_hold_days"]
    clean = with_targets.dropna(subset=all_needed).copy()
    clean.reset_index(drop=True, inplace=True)

    print(f"  Clean dataset: {len(clean)} rows, {clean['Symbol'].nunique()} symbols")
    buy_pct = clean["target_buy"].mean() * 100
    print(f"  Target distribution: {buy_pct:.1f}% BUY, {100-buy_pct:.1f}% HOLD")

    return clean, feature_cols


def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split per-symbol: first 80% of each symbol's rows for train, last 20% for test."""
    train_frames = []
    test_frames = []

    for symbol in df["Symbol"].unique():
        sym_df = df[df["Symbol"] == symbol].sort_values("DateTime")
        n = len(sym_df)
        split_idx = int(n * TRAIN_FRACTION)
        train_frames.append(sym_df.iloc[:split_idx])
        test_frames.append(sym_df.iloc[split_idx:])

    train = pd.concat(train_frames, ignore_index=True)
    test = pd.concat(test_frames, ignore_index=True)
    return train, test


def train_and_evaluate(
    period_days: int = DEFAULT_PERIOD_DAYS,
    profit_threshold: float = DEFAULT_PROFIT_THRESHOLD,
) -> dict:
    """Train all candidate models, evaluate, and save the best.

    Returns:
        Summary dict with results for all models.
    """
    clean_df, feature_cols = prepare_training_data(period_days, profit_threshold)
    train_df, test_df = time_based_split(clean_df)

    print(f"\nTrain: {len(train_df)} rows | Test: {len(test_df)} rows")

    X_train = train_df[feature_cols].values
    y_train = train_df["target_buy"].values.astype(int)
    X_test = test_df[feature_cols].values
    y_test = test_df["target_buy"].values.astype(int)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = get_candidate_models()
    results = []

    for name, model in candidates.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        start = time.time()

        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start
        print(f"  Training time: {train_time:.1f}s")

        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        evaluation = full_evaluation(y_test, y_pred, y_proba, test_df, name)
        evaluation["train_time_sec"] = round(train_time, 1)

        results.append(evaluation)

        print(f"  AUC-ROC:          {evaluation['auc_roc']:.4f}")
        print(f"  Accuracy:         {evaluation['accuracy']:.4f}")
        print(f"  Precision:        {evaluation['precision']:.4f}")
        print(f"  Recall:           {evaluation['recall']:.4f}")
        print(f"  F1:               {evaluation['f1']:.4f}")
        print(f"  Precision@10:     {evaluation['precision_at_10']:.3f}")
        print(f"  Precision@20:     {evaluation['precision_at_20']:.3f}")
        print(f"  Backtest avg ROI: {evaluation['backtest']['avg_return_pct']:.2f}%")
        print(f"  Backtest Sharpe:  {evaluation['backtest']['sharpe_ratio']:.3f}")

    # Select best model by AUC-ROC
    results.sort(key=lambda r: r["auc_roc"], reverse=True)
    best = results[0]
    best_name = best["model_name"]
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name} (AUC-ROC = {best['auc_roc']:.4f})")
    print(f"{'='*60}")

    # Retrain best on full data (train + test) for production
    print(f"\nRetraining {best_name} on full dataset for production...")
    best_model = get_candidate_models()[best_name]
    X_full = scaler.fit_transform(clean_df[feature_cols].values)
    y_full = clean_df["target_buy"].values.astype(int)
    best_model.fit(X_full, y_full)

    # Save artifacts
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    metadata = {
        "model_type": best_name,
        "model_version": f"{best_name}_v1_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "period_days": period_days,
        "profit_threshold": profit_threshold,
        "train_rows": len(clean_df),
        "num_symbols": int(clean_df["Symbol"].nunique()),
        "num_features": len(feature_cols),
        "best_metrics": best,
        "all_results": results,
    }
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nSaved model to {BEST_MODEL_PATH}")
    print(f"Saved metadata to {MODEL_METADATA_PATH}")

    return metadata
