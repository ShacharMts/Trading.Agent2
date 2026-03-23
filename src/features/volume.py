"""Volume-based features."""

import pandas as pd
import numpy as np


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume features. Must be called per-symbol (sorted by DateTime)."""
    vol = df["Volume"].astype(float)

    # Volume moving average
    df["volume_sma_20"] = vol.rolling(window=20, min_periods=20).mean()

    # Volume ratio (current vs average)
    df["volume_ratio"] = vol / df["volume_sma_20"].replace(0, np.nan)

    # Volume trend (slope of volume SMA over 5 bars)
    df["volume_trend"] = df["volume_sma_20"].diff(5) / 5

    return df
