"""Target variable construction for supervised learning."""

import pandas as pd
import numpy as np
from src.utils.config import HOURLY_BARS_PER_DAY


def add_targets(df: pd.DataFrame, period_days: int = 10, profit_threshold: float = 3.0) -> pd.DataFrame:
    """Add forward-looking target variables per symbol.

    Must be called per-symbol (sorted by DateTime).

    Args:
        df: Feature-engineered DataFrame for a single symbol.
        period_days: Forward horizon in trading days.
        profit_threshold: Minimum % return to classify as BUY.

    Returns:
        DataFrame with target columns added. Rows where targets can't be
        computed (near the end) will have NaN targets.
    """
    df = df.copy()
    df.sort_values("DateTime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    horizon_bars = period_days * HOURLY_BARS_PER_DAY
    n = len(df)
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values

    future_max_close = np.full(n, np.nan)
    future_min_low = np.full(n, np.nan)
    optimal_hold = np.full(n, np.nan)

    for i in range(n):
        end = min(i + horizon_bars + 1, n)
        if i + 1 >= end:
            continue
        window_close = close[i + 1: end]
        window_low = low[i + 1: end]

        if len(window_close) == 0:
            continue

        max_idx = np.argmax(window_close)
        future_max_close[i] = window_close[max_idx]
        future_min_low[i] = np.min(window_low)
        # optimal hold = bars to peak / bars per day
        optimal_hold[i] = (max_idx + 1) / HOURLY_BARS_PER_DAY

    df["future_max_close"] = future_max_close
    df["future_min_low"] = future_min_low

    # Future return (%)
    df["future_return"] = (df["future_max_close"] - df["Close"]) / df["Close"] * 100

    # Classification target
    df["target_buy"] = (df["future_return"] >= profit_threshold).astype(float)
    df.loc[df["future_return"].isna(), "target_buy"] = np.nan

    # Optimal hold days
    df["optimal_hold_days"] = optimal_hold

    # Target price (the max close in the window)
    df["target_price"] = future_max_close

    # Stop-loss: volatility-adaptive
    # High-ATR stocks get wider stops, low-ATR stocks get tighter stops
    if "atr_14" in df.columns:
        # Dynamic multiplier: 2.5x for high-vol, 1.5x for low-vol
        atr = df["atr_14"]
        median_atr_pct = (atr / df["Close"]).median()
        atr_pct = atr / df["Close"]
        # Normalize: stocks with ATR% above median get wider stops (up to 3x),
        # those below get tighter stops (down to 1.5x)
        multiplier = np.where(
            atr_pct > median_atr_pct,
            np.clip(2.0 + (atr_pct - median_atr_pct) / median_atr_pct, 2.0, 3.0),
            np.clip(2.0 - (median_atr_pct - atr_pct) / median_atr_pct * 0.5, 1.5, 2.0),
        )
        atr_stop = df["Close"] - multiplier * atr
        min_low_stop = future_min_low
        df["stop_loss"] = np.where(
            pd.notna(df["atr_14"]),
            np.maximum(atr_stop, min_low_stop * 0.99),
            min_low_stop * 0.99,
        )
    else:
        df["stop_loss"] = future_min_low * 0.99

    return df


def add_targets_all_symbols(df: pd.DataFrame, period_days: int = 10, profit_threshold: float = 3.0) -> pd.DataFrame:
    """Add targets for all symbols in the dataset."""
    symbols = df["Symbol"].unique()
    frames = []
    for symbol in symbols:
        symbol_data = df[df["Symbol"] == symbol]
        with_targets = add_targets(symbol_data, period_days, profit_threshold)
        frames.append(with_targets)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
