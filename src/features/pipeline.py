"""Orchestrate full feature engineering for all symbols."""

import pandas as pd
import numpy as np
from src.features.candlestick import add_candlestick_features
from src.features.moving_average import add_moving_average_features
from src.features.volume import add_volume_features
from src.features.price_action import add_price_action_features
from src.features.momentum import add_momentum_features
from src.utils.config import HOURLY_BARS_PER_DAY


# Category encoding
CATEGORY_MAP = {"snp500": 0, "snp100": 1, "etfs": 2, "merchandise": 3}


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week and week-of-month features."""
    dt = df["DateTime"]
    df["day_of_week"] = dt.dt.dayofweek  # 0=Mon, 4=Fri
    df["week_of_month"] = (dt.dt.day - 1) // 7  # 0-4
    return df


def _add_relative_strength(df: pd.DataFrame, market_returns: pd.Series, market_regime: pd.Series, sector_returns: pd.Series | None = None) -> pd.DataFrame:
    """Add stock return relative to market (VOO) return and market regime."""
    # Per-bar return of the stock
    stock_return = df["Close"].pct_change() * 100
    # Align market return by DateTime
    df["market_return"] = df["DateTime"].map(market_returns).fillna(0)
    df["relative_strength"] = stock_return - df["market_return"]
    # Rolling 20-bar relative strength
    df["relative_strength_20"] = df["relative_strength"].rolling(window=20, min_periods=5).mean()
    # Market regime: rolling 20-bar return of VOO (positive = uptrend, negative = downtrend)
    df["market_regime"] = df["DateTime"].map(market_regime).fillna(0)
    # Sector relative strength (vs category average)
    if sector_returns is not None:
        df["sector_return"] = df["DateTime"].map(sector_returns).fillna(0)
        df["vs_sector"] = stock_return - df["sector_return"]
        df["vs_sector_20"] = df["vs_sector"].rolling(window=20, min_periods=5).mean()
    else:
        df["sector_return"] = 0.0
        df["vs_sector"] = 0.0
        df["vs_sector_20"] = 0.0
    return df


def _compute_market_returns(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute per-bar returns and regime of VOO as a market proxy.

    Returns (returns_series, regime_series) both indexed by DateTime.
    """
    voo = data[data["Symbol"] == "VOO"].sort_values("DateTime")
    if voo.empty:
        data_sorted = data.sort_values("DateTime")
        all_returns = data_sorted.groupby("DateTime")["Close"].apply(
            lambda x: x.pct_change().iloc[-1] if len(x) > 1 else 0
        )
        return all_returns * 100, pd.Series(0, index=all_returns.index)

    voo_close = voo.set_index("DateTime")["Close"]
    voo_returns = voo_close.pct_change() * 100
    # Market regime: rolling 20-bar cumulative return (trend direction)
    voo_regime = voo_close.pct_change(periods=20) * 100
    return voo_returns, voo_regime


def _compute_sector_returns(data: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute per-bar average returns by category (sector proxy).

    Returns dict of category -> returns Series indexed by DateTime.
    """
    sector_returns = {}
    for cat in data["category"].unique():
        cat_data = data[data["category"] == cat].copy()
        cat_data = cat_data.sort_values("DateTime")
        # Average close per DateTime across all symbols in this category
        avg_close = cat_data.groupby("DateTime")["Close"].mean().sort_index()
        returns = avg_close.pct_change() * 100
        sector_returns[cat] = returns
    return sector_returns


def engineer_features_for_symbol(
    symbol_df: pd.DataFrame,
    market_returns: pd.Series,
    market_regime: pd.Series,
    sector_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Apply all feature engineering to a single symbol's data."""
    df = symbol_df.copy()
    df.sort_values("DateTime", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = add_candlestick_features(df)
    df = add_moving_average_features(df)
    df = add_volume_features(df)
    df = add_price_action_features(df)
    df = add_momentum_features(df)
    df = _add_calendar_features(df)
    df = _add_relative_strength(df, market_returns, market_regime, sector_returns)

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
        DataFrame with all features added.
    """
    # Pre-compute market returns and regime once for relative strength
    market_returns, market_regime = _compute_market_returns(data)
    sector_returns_map = _compute_sector_returns(data)

    symbols = data["Symbol"].unique()
    frames = []

    for symbol in symbols:
        symbol_data = data[data["Symbol"] == symbol].copy()
        if len(symbol_data) < 50:
            continue
        cat = symbol_data["category"].iloc[0] if "category" in symbol_data.columns else None
        sector_ret = sector_returns_map.get(cat) if cat else None
        featured = engineer_features_for_symbol(symbol_data, market_returns, market_regime, sector_ret)
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
        "market_return",  # intermediate col, not a direct feature
        "sector_return",  # intermediate col, not a direct feature
        # Target columns (added later)
        "future_return", "target_buy", "optimal_hold_days",
        "target_price", "stop_loss", "future_max_close", "future_min_low",
    }
    return [col for col in df.columns if col not in exclude]
