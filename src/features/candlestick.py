"""Candlestick pattern detection — only patterns with proven feature importance.

Keeps: three_white_soldiers, three_black_crows (non-zero LightGBM importance).
Adds: composite bullish/bearish scores aggregating multiple patterns.
"""

import pandas as pd
import numpy as np


def add_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add candlestick pattern columns to the DataFrame.

    Expects columns: Open, High, Low, Close, Body, UpperShadow, LowerShadow, Direction.
    Must be called per-symbol (sorted by DateTime).
    """
    h = df["High"].values
    lo = df["Low"].values
    o = df["Open"].values
    c = df["Close"].values
    body = df["Body"].values
    upper = df["UpperShadow"].values
    lower = df["LowerShadow"].values
    direction = df["Direction"].values
    rng = h - lo
    safe_rng = np.where(rng == 0, 1e-9, rng)
    body_ratio = body / safe_rng

    n = len(df)

    # Previous bar values
    prev_c = np.roll(c, 1)
    prev_dir = np.roll(direction, 1)
    prev2_dir = np.roll(direction, 2)
    prev2_c = np.roll(c, 2)

    # Three White Soldiers: three consecutive bullish with increasing closes
    mask = (
        (direction == "BULLISH")
        & (prev_dir == "BULLISH")
        & (prev2_dir == "BULLISH")
        & (c > prev_c)
        & (prev_c > prev2_c)
    )
    mask[:2] = False
    df["pat_three_white_soldiers"] = mask.astype(int)

    # Three Black Crows: three consecutive bearish with decreasing closes
    mask = (
        (direction == "BEARISH")
        & (prev_dir == "BEARISH")
        & (prev2_dir == "BEARISH")
        & (c < prev_c)
        & (prev_c < prev2_c)
    )
    mask[:2] = False
    df["pat_three_black_crows"] = mask.astype(int)

    # Composite bullish score: count of bullish signals in current bar
    bullish_signals = np.zeros(n, dtype=float)
    # Hammer-like
    bullish_signals += ((lower >= 2 * body) & (upper < body) & (direction == "BULLISH")).astype(float)
    # Bullish engulfing
    prev_body = np.roll(body, 1)
    prev_o = np.roll(o, 1)
    eng_mask = ((direction == "BULLISH") & (prev_dir == "BEARISH") & (o < prev_c) & (c > prev_o) & (body > prev_body))
    eng_mask[0] = False
    bullish_signals += eng_mask.astype(float)
    # Marubozu bullish
    bullish_signals += ((body_ratio > 0.9) & (direction == "BULLISH")).astype(float)
    # Three white soldiers
    bullish_signals += df["pat_three_white_soldiers"].values.astype(float)
    df["bullish_score"] = bullish_signals

    # Composite bearish score
    bearish_signals = np.zeros(n, dtype=float)
    bearish_signals += ((upper >= 2 * body) & (lower < body) & (direction == "BEARISH")).astype(float)
    eng_mask2 = ((direction == "BEARISH") & (prev_dir == "BULLISH") & (o > prev_c) & (c < prev_o) & (body > prev_body))
    eng_mask2[0] = False
    bearish_signals += eng_mask2.astype(float)
    bearish_signals += ((body_ratio > 0.9) & (direction == "BEARISH")).astype(float)
    bearish_signals += df["pat_three_black_crows"].values.astype(float)
    df["bearish_score"] = bearish_signals

    return df
