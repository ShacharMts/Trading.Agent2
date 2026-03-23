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


def walk_forward_evaluate(
    clean_df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    n_folds: int = 3,
) -> dict:
    """Walk-forward cross-validation: train on expanding window, test on next fold.

    Returns averaged metrics across folds.
    """
    from src.model.evaluator import full_evaluation

    # Sort by DateTime globally
    clean_df = clean_df.sort_values("DateTime").reset_index(drop=True)
    total = len(clean_df)
    fold_size = total // (n_folds + 1)  # reserve 1 fold for initial training

    all_auc = []
    all_precision = []
    all_backtest_roi = []
    all_sharpe = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)  # expanding window
        test_start = train_end
        test_end = min(test_start + fold_size, total)

        if test_end <= test_start:
            break

        print(f"    [{model_name}] WF fold {fold+1}/{n_folds}: train={train_end} rows, test={test_end-test_start} rows")
        train_data = clean_df.iloc[:train_end]
        test_data = clean_df.iloc[test_start:test_end]

        X_train = train_data[feature_cols].values
        y_train = train_data["target_buy"].values.astype(int)
        X_test = test_data[feature_cols].values
        y_test = test_data["target_buy"].values.astype(int)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = get_candidate_models()[model_name]
        model.fit(X_train_s, y_train)

        y_proba = model.predict_proba(X_test_s)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        ev = full_evaluation(y_test, y_pred, y_proba, test_data, model_name)
        all_auc.append(ev["auc_roc"])
        all_precision.append(ev["precision"])
        all_backtest_roi.append(ev["backtest"]["avg_return_pct"])
        all_sharpe.append(ev["backtest"]["sharpe_ratio"])

    return {
        "wf_avg_auc": np.mean(all_auc) if all_auc else 0,
        "wf_avg_precision": np.mean(all_precision) if all_precision else 0,
        "wf_avg_roi": np.mean(all_backtest_roi) if all_backtest_roi else 0,
        "wf_avg_sharpe": np.mean(all_sharpe) if all_sharpe else 0,
        "wf_folds": len(all_auc),
    }


def _train_single_model(
    name: str,
    X_train_scaled: np.ndarray,
    y_train: np.ndarray,
    X_test_scaled: np.ndarray,
    y_test: np.ndarray,
    test_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Train and evaluate a single model (designed to run in a subprocess)."""
    model = get_candidate_models()[name]

    print(f"  [{name}] Step 1/3: Training model...")
    start = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start
    print(f"  [{name}] Step 1/3: Training done ({train_time:.1f}s)")

    print(f"  [{name}] Step 2/3: Evaluating on test set...")
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    evaluation = full_evaluation(y_test, y_pred, y_proba, test_df, name)
    evaluation["train_time_sec"] = round(train_time, 1)
    print(f"  [{name}] Step 2/3: Evaluation done (AUC={evaluation['auc_roc']:.4f})")

    print(f"  [{name}] Step 3/3: Walk-forward CV (3 folds)...")
    wf = walk_forward_evaluate(clean_df, feature_cols, name, n_folds=3)
    evaluation["walk_forward"] = wf
    print(f"  [{name}] Step 3/3: Walk-forward done (Sharpe={wf['wf_avg_sharpe']:.3f})")

    return evaluation


def _fit_model(name: str, X: np.ndarray, y: np.ndarray):
    """Fit a single model on the full dataset (designed to run in a subprocess)."""
    model = get_candidate_models()[name]
    model.fit(X, y)
    return model


def train_and_evaluate(
    period_days: int = DEFAULT_PERIOD_DAYS,
    profit_threshold: float = DEFAULT_PROFIT_THRESHOLD,
) -> dict:
    """Train all candidate models, evaluate with walk-forward CV, and save the best.

    Returns:
        Summary dict with results for all models.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

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

    # Train and evaluate all models in parallel
    candidates = list(get_candidate_models().keys())
    print(f"\n{'='*60}")
    print(f"Training {len(candidates)} models in parallel: {', '.join(candidates)}")
    print(f"Each model: train -> evaluate -> walk-forward CV (3 folds)")
    print(f"{'='*60}")
    overall_start = time.time()

    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                _train_single_model,
                name, X_train_scaled, y_train, X_test_scaled, y_test,
                test_df, clean_df, feature_cols,
            ): name
            for name in candidates
        }
        results = []
        for future in as_completed(futures):
            name = futures[future]
            evaluation = future.result()
            results.append(evaluation)
            elapsed = time.time() - overall_start
            print(f"\n{'='*60}")
            print(f"  [{len(results)}/{len(candidates)}] {name} completed ({elapsed:.0f}s elapsed):")
            print(f"  Training time:    {evaluation['train_time_sec']}s")
            print(f"  WF Avg AUC:       {evaluation['walk_forward']['wf_avg_auc']:.4f}")
            print(f"  WF Avg ROI:       {evaluation['walk_forward']['wf_avg_roi']:.2f}%")
            print(f"  WF Avg Sharpe:    {evaluation['walk_forward']['wf_avg_sharpe']:.3f}")
            print(f"  AUC-ROC:          {evaluation['auc_roc']:.4f}")
            print(f"  Precision:        {evaluation['precision']:.4f}")
            print(f"  Precision@10:     {evaluation['precision_at_10']:.3f}")
            print(f"  Precision@20:     {evaluation['precision_at_20']:.3f}")
            print(f"  Backtest avg ROI: {evaluation['backtest']['avg_return_pct']:.2f}%")
            print(f"  Backtest Sharpe:  {evaluation['backtest']['sharpe_ratio']:.3f}")

    # Select best model by walk-forward Sharpe (practical profitability, not just AUC)
    results.sort(key=lambda r: r["walk_forward"]["wf_avg_sharpe"], reverse=True)
    best = results[0]
    best_name = best["model_name"]
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name} (WF Avg Sharpe = {best['walk_forward']['wf_avg_sharpe']:.3f})")
    print(f"{'='*60}")

    # Train ALL models on full data for ensemble (parallel)
    print(f"\nPhase 2: Training ensemble (all {len(candidates)} models) on full dataset...")
    ensemble_start = time.time()
    X_full = scaler.fit_transform(clean_df[feature_cols].values)
    y_full = clean_df["target_buy"].values.astype(int)

    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_fit_model, name, X_full, y_full): name
            for name in candidates
        }
        ensemble_models = {}
        done_count = 0
        for future in as_completed(futures):
            name = futures[future]
            ensemble_models[name] = future.result()
            done_count += 1
            print(f"  [{done_count}/{len(candidates)}] {name} trained on full data")
    print(f"  Ensemble training done ({time.time() - ensemble_start:.0f}s)")

    # Save artifacts — save ensemble dict instead of single model
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(ensemble_models, BEST_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(feature_cols, f, indent=2)

    metadata = {
        "model_type": "ensemble",
        "ensemble_models": list(ensemble_models.keys()),
        "best_single_model": best_name,
        "model_version": f"ensemble_v3_{datetime.now(timezone.utc).strftime('%Y%m%d')}",
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

    print(f"\nSaved ensemble to {BEST_MODEL_PATH}")
    print(f"Saved metadata to {MODEL_METADATA_PATH}")

    return metadata
