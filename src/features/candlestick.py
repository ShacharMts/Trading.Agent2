"""Candlestick pattern detection — binary features per candle.

Based on The Candlestick Trading Bible patterns.
Uses the pre-computed Body, UpperShadow, LowerShadow, Direction columns.
"""

import pandas as pd
import numpy as np


def _range(row):
    """Total candle range (High - Low)."""
    return row["High"] - row["Low"]


def add_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary candlestick pattern columns to the DataFrame.

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
    rng = h - lo  # candle range

    n = len(df)

    # Avoid division by zero
    safe_rng = np.where(rng == 0, 1e-9, rng)
    safe_body = np.where(body == 0, 1e-9, body)

    body_ratio = body / safe_rng  # body as fraction of range

    # ── Single-candle patterns ────────────────────────────────────────

    # Doji: very small body relative to range
    df["pat_doji"] = (body_ratio < 0.1).astype(int)

    # Dragonfly Doji: doji with long lower shadow, no upper shadow
    df["pat_dragonfly_doji"] = (
        (body_ratio < 0.1) & (lower / safe_rng > 0.6) & (upper / safe_rng < 0.1)
    ).astype(int)

    # Gravestone Doji: doji with long upper shadow, no lower shadow
    df["pat_gravestone_doji"] = (
        (body_ratio < 0.1) & (upper / safe_rng > 0.6) & (lower / safe_rng < 0.1)
    ).astype(int)

    # Hammer: small body at top, long lower shadow (>=2x body), bullish context
    df["pat_hammer"] = (
        (lower >= 2 * body)
        & (upper < body)
        & (body_ratio < 0.35)
        & (direction == "BULLISH")
    ).astype(int)

    # Inverted Hammer: small body at bottom, long upper shadow
    df["pat_inverted_hammer"] = (
        (upper >= 2 * body)
        & (lower < body)
        & (body_ratio < 0.35)
        & (direction == "BULLISH")
    ).astype(int)

    # Hanging Man: hammer shape but bearish
    df["pat_hanging_man"] = (
        (lower >= 2 * body) & (upper < body) & (body_ratio < 0.35) & (direction == "BEARISH")
    ).astype(int)

    # Shooting Star: inverted hammer shape but bearish
    df["pat_shooting_star"] = (
        (upper >= 2 * body) & (lower < body) & (body_ratio < 0.35) & (direction == "BEARISH")
    ).astype(int)

    # Marubozu: body is almost entire range (no shadows)
    df["pat_marubozu"] = (body_ratio > 0.9).astype(int)

    # ── Dual-candle patterns (use shifted values) ─────────────────────

    prev_o = np.roll(o, 1)
    prev_c = np.roll(c, 1)
    prev_body = np.roll(body, 1)
    prev_dir = np.roll(direction, 1)
    prev_h = np.roll(h, 1)
    prev_lo = np.roll(lo, 1)

    # Bullish Engulfing
    df["pat_bullish_engulfing"] = np.zeros(n, dtype=int)
    mask = (
        (direction == "BULLISH")
        & (prev_dir == "BEARISH")
        & (o < prev_c)  # open below prev close
        & (c > prev_o)  # close above prev open
        & (body > prev_body)
    )
    mask[0] = False
    df["pat_bullish_engulfing"] = mask.astype(int)

    # Bearish Engulfing
    mask = (
        (direction == "BEARISH")
        & (prev_dir == "BULLISH")
        & (o > prev_c)
        & (c < prev_o)
        & (body > prev_body)
    )
    mask[0] = False
    df["pat_bearish_engulfing"] = mask.astype(int)

    # Piercing Line: bearish followed by bullish opening below prev low, closing >50% of prev body
    mask = (
        (direction == "BULLISH")
        & (prev_dir == "BEARISH")
        & (o < prev_lo)
        & (c > (prev_o + prev_c) / 2)
        & (c < prev_o)
    )
    mask[0] = False
    df["pat_piercing_line"] = mask.astype(int)

    # Dark Cloud Cover: bullish followed by bearish opening above prev high, closing <50% of prev body
    mask = (
        (direction == "BEARISH")
        & (prev_dir == "BULLISH")
        & (o > prev_h)
        & (c < (prev_o + prev_c) / 2)
        & (c > prev_o)
    )
    mask[0] = False
    df["pat_dark_cloud_cover"] = mask.astype(int)

    # Tweezer Bottom: two candles with same low
    mask = (np.abs(lo - prev_lo) < 0.01 * safe_rng) & (direction == "BULLISH") & (prev_dir == "BEARISH")
    mask[0] = False
    df["pat_tweezer_bottom"] = mask.astype(int)

    # Tweezer Top: two candles with same high
    mask = (np.abs(h - prev_h) < 0.01 * safe_rng) & (direction == "BEARISH") & (prev_dir == "BULLISH")
    mask[0] = False
    df["pat_tweezer_top"] = mask.astype(int)

    # ── Triple-candle patterns ────────────────────────────────────────

    prev2_dir = np.roll(direction, 2)
    prev2_body = np.roll(body, 2)
    prev2_c = np.roll(c, 2)
    prev2_o = np.roll(o, 2)

    # Morning Star: bearish, small body, bullish (reversal up)
    mask = (
        (prev2_dir == "BEARISH")
        & (np.roll(body_ratio, 1) < 0.2)  # middle candle small
        & (direction == "BULLISH")
        & (c > (prev2_o + prev2_c) / 2)
    )
    mask[:2] = False
    df["pat_morning_star"] = mask.astype(int)

    # Evening Star: bullish, small body, bearish (reversal down)
    mask = (
        (prev2_dir == "BULLISH")
        & (np.roll(body_ratio, 1) < 0.2)
        & (direction == "BEARISH")
        & (c < (prev2_o + prev2_c) / 2)
    )
    mask[:2] = False
    df["pat_evening_star"] = mask.astype(int)

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

    # Three Inside Up: bearish, bullish engulfed inside, bullish close > first open
    mask = (
        (prev2_dir == "BEARISH")
        & (prev_dir == "BULLISH")
        & (prev_o > prev2_c)  # prev open > prev2 close (inside)
        & (prev_c < prev2_o)  # prev close < prev2 open (inside)
        & (direction == "BULLISH")
        & (c > prev2_o)
    )
    mask[:2] = False
    df["pat_three_inside_up"] = mask.astype(int)

    # Three Inside Down
    mask = (
        (prev2_dir == "BULLISH")
        & (prev_dir == "BEARISH")
        & (prev_o < prev2_c)
        & (prev_c > prev2_o)
        & (direction == "BEARISH")
        & (c < prev2_o)
    )
    mask[:2] = False
    df["pat_three_inside_down"] = mask.astype(int)

    return df
