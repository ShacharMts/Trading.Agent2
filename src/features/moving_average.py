"""Moving average features — SMA, EMA, slopes, deviations."""

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

    # MA slope (rate of change over 5 bars)
    if "sma_20" in df.columns:
        df["ma_slope_20"] = df["sma_20"].diff(5) / 5

    # Price deviation from MAs (%)
    for window in [20, 50]:
        col = f"sma_{window}"
        if col in df.columns:
            df[f"price_vs_sma_{window}"] = (close - df[col]) / df[col] * 100

    return df
