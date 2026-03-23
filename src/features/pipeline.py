"""Orchestrate full feature engineering for all symbols."""

import pandas as pd
import numpy as np
from src.features.candlestick import add_candlestick_features
from src.features.moving_average import add_moving_average_features
from src.features.volume import add_volume_features
from src.features.price_action import add_price_action_features


# Category encoding
CATEGORY_MAP = {"snp500": 0, "snp100": 1, "etfs": 2, "merchandise": 3}


def engineer_features_for_symbol(symbol_df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering to a single symbol's data."""
    df = symbol_df.copy()
    df.sort_values("DateTime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = add_candlestick_features(df)
    df = add_moving_average_features(df)
    df = add_volume_features(df)
    df = add_price_action_features(df)

    # Encode direction as numeric
    df["direction_num"] = (df["Direction"] == "BULLISH").astype(int)

    # Encode category
    df["category_num"] = df["category"].map(CATEGORY_MAP).fillna(-1).astype(int)

    return df


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering to all symbols in the dataset.

    Args:
        data: Combined DataFrame with all symbols (must have 'Symbol' column).

    Returns:
        DataFrame with all features added, NaN rows from rolling windows dropped.
    """
    symbols = data["Symbol"].unique()
    frames = []

    for symbol in symbols:
        symbol_data = data[data["Symbol"] == symbol].copy()
        if len(symbol_data) < 50:  # skip symbols with too little data
            continue
        featured = engineer_features_for_symbol(symbol_data)
        frames.append(featured)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    return result


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns (excludes metadata and targets)."""
    exclude = {
        "Symbol", "DateTime", "Open", "High", "Low", "Close", "Volume",
        "Body", "UpperShadow", "LowerShadow", "Direction", "category",
        # Target columns (added later)
        "future_return", "target_buy", "optimal_hold_days",
        "target_price", "stop_loss", "future_max_close", "future_min_low",
    }
    return [col for col in df.columns if col not in exclude]
