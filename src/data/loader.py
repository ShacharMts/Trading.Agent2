"""Load pipe-delimited candle data files into DataFrames."""

import pandas as pd
from pathlib import Path
from src.utils.config import DATA_CATEGORIES, FILE_DELIMITER, FILE_SUFFIX
from src.data.ticker_registry import get_tickers_for_category


def load_ticker_data(filepath: Path) -> pd.DataFrame:
    """Load a single ticker file into a DataFrame."""
    df = pd.read_csv(
        filepath,
        delimiter=FILE_DELIMITER,
        parse_dates=["DateTime"],
    )
    df.sort_values("DateTime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_category(category: str) -> pd.DataFrame:
    """Load all tickers for a category into a single DataFrame."""
    folder = DATA_CATEGORIES[category]
    tickers = get_tickers_for_category(category)
    frames = []
    for symbol in tickers:
        filepath = folder / f"{symbol}{FILE_SUFFIX}"
        try:
            df = load_ticker_data(filepath)
            df["category"] = category
            frames.append(df)
        except Exception as e:
            print(f"Warning: failed to load {symbol}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_all_data() -> pd.DataFrame:
    """Load data from all categories into a single DataFrame."""
    frames = []
    for category in DATA_CATEGORIES:
        df = load_category(category)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    # De-duplicate: a ticker in snp100 is also in snp500 — keep snp100 category
    # Keep first occurrence (snp100 is loaded before snp500 alphabetically by key order)
    combined.drop_duplicates(subset=["Symbol", "DateTime"], keep="first", inplace=True)
    combined.sort_values(["Symbol", "DateTime"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined
