"""Momentum indicators — MACD, Bollinger Bands, Stochastic Oscillator."""

import pandas as pd
import numpy as np


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum-based technical indicators. Must be called per-symbol (sorted by DateTime)."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # ── MACD (12, 26, 9) ──
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd_line"] = ema_12 - ema_26
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd_line"] - df["macd_signal"]

    # ── Bollinger Bands (20, 2σ) ──
    sma_20 = close.rolling(window=20, min_periods=20).mean()
    std_20 = close.rolling(window=20, min_periods=20).std()
    df["bb_upper"] = sma_20 + 2 * std_20
    df["bb_lower"] = sma_20 - 2 * std_20
    bb_width = df["bb_upper"] - df["bb_lower"]
    df["bb_width_pct"] = bb_width / sma_20 * 100  # width as % of price
    df["bb_position"] = (close - df["bb_lower"]) / bb_width.replace(0, np.nan)  # 0=lower, 1=upper

    # ── Stochastic Oscillator (14, 3) ──
    low_14 = low.rolling(window=14, min_periods=14).min()
    high_14 = high.rolling(window=14, min_periods=14).max()
    denom = (high_14 - low_14).replace(0, np.nan)
    df["stoch_k"] = (close - low_14) / denom * 100
    df["stoch_d"] = df["stoch_k"].rolling(window=3, min_periods=3).mean()

    # ── Multi-day returns (using HOURLY_BARS_PER_DAY=7) ──
    df["returns_3d"] = close.pct_change(periods=21) * 100   # 3 days * 7 bars
    df["returns_5d"] = close.pct_change(periods=35) * 100   # 5 days * 7 bars

    return df
