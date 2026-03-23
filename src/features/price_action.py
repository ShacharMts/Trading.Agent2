"""Price action features — returns, volatility, RSI, ATR."""

import pandas as pd
import numpy as np


def add_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price-action derived features. Must be called per-symbol (sorted by DateTime)."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Returns over different windows
    for period in [1, 7, 20]:
        df[f"returns_{period}h"] = close.pct_change(periods=period) * 100

    # Volatility: 20-bar rolling std of 1-bar returns
    df["volatility_20"] = df["returns_1h"].rolling(window=20, min_periods=20).std()

    # Average True Range (14-bar)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_14"] = true_range.rolling(window=14, min_periods=14).mean()

    # RSI (14-bar)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # High-low range as percentage of close
    df["high_low_range"] = (high - low) / close * 100

    # Body to range ratio (candle quality)
    candle_range = high - low
    df["body_to_range"] = df["Body"] / candle_range.replace(0, np.nan)

    return df
