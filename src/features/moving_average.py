"""Moving average features — SMA, EMA, crossovers, slopes."""

import pandas as pd
import numpy as np


def add_moving_average_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add MA/EMA features. Must be called per-symbol (sorted by DateTime)."""
    close = df["Close"]

    # Simple Moving Averages
    for window in [7, 20, 50, 100]:
        df[f"sma_{window}"] = close.rolling(window=window, min_periods=window).mean()

    # Exponential Moving Averages
    for span in [7, 20, 50]:
        df[f"ema_{span}"] = close.ewm(span=span, adjust=False).mean()

    # Crossover signals (1 = golden cross, -1 = death cross, 0 = no change)
    if "sma_7" in df.columns and "sma_20" in df.columns:
        above_7_20 = (df["sma_7"] > df["sma_20"]).astype(int)
        df["sma_crossover_7_20"] = above_7_20.diff().fillna(0).astype(int)

    if "sma_20" in df.columns and "sma_50" in df.columns:
        above_20_50 = (df["sma_20"] > df["sma_50"]).astype(int)
        df["sma_crossover_20_50"] = above_20_50.diff().fillna(0).astype(int)

    # MA slope (rate of change over 5 bars)
    if "sma_20" in df.columns:
        df["ma_slope_20"] = df["sma_20"].diff(5) / 5

    # Price deviation from MAs (%)
    for window in [20, 50]:
        col = f"sma_{window}"
        if col in df.columns:
            df[f"price_vs_sma_{window}"] = (close - df[col]) / df[col] * 100

    return df
