"""Scan Data/ folders to build the ticker registry."""

from pathlib import Path
from src.utils.config import DATA_CATEGORIES, FILE_SUFFIX


def get_tickers_for_category(category: str) -> list[str]:
    """Return sorted list of ticker symbols for a data category."""
    folder = DATA_CATEGORIES.get(category)
    if folder is None or not folder.exists():
        return []
    tickers = []
    for f in folder.iterdir():
        if f.name.endswith(FILE_SUFFIX) and not f.name.startswith("."):
            symbol = f.name.replace(FILE_SUFFIX, "")
            tickers.append(symbol)
    return sorted(tickers)


def get_all_tickers() -> dict[str, list[str]]:
    """Return {category: [tickers]} for all categories."""
    return {cat: get_tickers_for_category(cat) for cat in DATA_CATEGORIES}


def get_ticker_file(symbol: str, category: str) -> Path:
    """Return the file path for a given symbol and category."""
    return DATA_CATEGORIES[category] / f"{symbol}{FILE_SUFFIX}"


def find_ticker_category(symbol: str) -> str | None:
    """Find which category a ticker belongs to (first match)."""
    for cat, tickers in get_all_tickers().items():
        if symbol in tickers:
            return cat
    return None
