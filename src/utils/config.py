"""Application configuration — paths, defaults, constants."""

import os
from pathlib import Path

# Root of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "Data"
DATA_CATEGORIES = {
    "snp500": DATA_DIR / "snp500_hourly",
    "snp100": DATA_DIR / "snp100_hourly",
    "etfs": DATA_DIR / "etfs_hourly",
    "merchandise": DATA_DIR / "merchandise_hourly",
}

# Model artifacts
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Data file format
FILE_DELIMITER = "|"
FILE_SUFFIX = "_hourly_candles.txt"

# Feature engineering defaults
HOURLY_BARS_PER_DAY = 7  # ~7 trading hours per day

# Training defaults
TRAIN_FRACTION = 0.8
DEFAULT_PROFIT_THRESHOLD = 3.0  # percent
DEFAULT_PERIOD_DAYS = 10
DEFAULT_NUM_STOCKS = 5

# Ensure models directory exists
MODELS_DIR.mkdir(exist_ok=True)
