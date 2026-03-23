# Trading Agent — Architecture Document

## 1. Overview

A Python-based ML trading recommendation system that analyzes hourly candlestick data for ~517 tickers (S&P 500, S&P 100, sector ETFs, and commodity/crypto ETFs). It applies candlestick pattern recognition and moving average strategies to recommend stocks to buy, with risk assessment, holding period, target price, and stop-loss levels.

The system runs locally on macOS and deploys to the cloud via Docker.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client (curl / app)                      │
└────────────────────────────┬────────────────────────────────────┘
                             │  HTTP REST
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Application                         │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐  │
│  │ /recommend   │  │ /refresh     │  │ /health               │  │
│  │  endpoint    │  │  endpoint    │  │  endpoint             │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────────────────┘  │
│         │                 │                                     │
│         ▼                 ▼                                     │
│  ┌─────────────┐  ┌─────────────┐                               │
│  │ Prediction  │  │ Data        │                               │
│  │ Engine      │  │ Refresher   │                               │
│  └──────┬──────┘  └──────┬──────┘                               │
│         │                │                                      │
│         ▼                ▼                                      │
│  ┌─────────────┐  ┌──────────────┐                              │
│  │ Trained     │  │ Yahoo Finance│                              │
│  │ Model (.pkl)│  │ Client       │                              │
│  └─────────────┘  └──────────────┘                              │
│         │                │                                      │
│         ▼                ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Data Layer (local flat files)               │   │
│  │   Data/snp500_hourly/  snp100_hourly/  etfs_hourly/     │   │
│  │   merchandise_hourly/                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Layer

### 3.1 Source Data

| Folder | Tickers | Description |
|---|---|---|
| `Data/snp500_hourly/` | ~396 | S&P 500 constituents |
| `Data/snp100_hourly/` | ~101 | S&P 100 large-caps |
| `Data/etfs_hourly/` | 15 | Sector & thematic ETFs (XLK, XLF, VOO, MAGS, etc.) |
| `Data/merchandise_hourly/` | 5 | Commodities & crypto ETFs (GLD, USO, IBIT, ETHA, SLVR) |

### 3.2 File Format

Pipe-delimited text files: `{SYMBOL}_hourly_candles.txt`

| Column | Type | Description |
|---|---|---|
| Symbol | string | Ticker symbol |
| DateTime | datetime | Candle timestamp (hourly) |
| Open | float | Opening price |
| High | float | High price |
| Low | float | Low price |
| Close | float | Closing price |
| Volume | int | Trade volume |
| Body | float | Candle body size (abs(Close - Open)) |
| UpperShadow | float | Upper wick length |
| LowerShadow | float | Lower wick length |
| Direction | string | BULLISH / BEARISH |

### 3.3 Data Storage Strategy

- **Current phase**: Local flat files (pipe-delimited `.txt`), loaded into pandas DataFrames at runtime.
- **Model artifacts**: Serialized to `models/` directory as `.pkl` files.
- **No external database** — all data lives on the filesystem.

---

## 4. Feature Engineering Pipeline

### 4.1 Candlestick Pattern Features

Based on **The Candlestick Trading Bible**, the following patterns are detected and encoded as binary features:

| Category | Patterns |
|---|---|
| **Single-candle reversal** | Hammer, Inverted Hammer, Hanging Man, Shooting Star, Doji, Dragonfly Doji, Gravestone Doji, Marubozu |
| **Dual-candle** | Bullish/Bearish Engulfing, Piercing Line, Dark Cloud Cover, Tweezer Top/Bottom |
| **Triple-candle** | Morning Star, Evening Star, Three White Soldiers, Three Black Crows, Three Inside Up/Down |

### 4.2 Moving Average Features

Based on **Moving Average Trading**, computed over hourly candles:

| Feature | Description |
|---|---|
| SMA_7, SMA_20, SMA_50, SMA_200 | Simple Moving Averages (in hourly bars: ~1d, ~3d, ~1w, ~1mo) |
| EMA_7, EMA_20, EMA_50 | Exponential Moving Averages |
| SMA_crossover_7_20 | Golden/death cross signal (short vs. medium) |
| SMA_crossover_20_50 | Medium-term trend cross |
| MA_slope_20 | Slope of SMA-20 (trend direction & strength) |
| price_vs_SMA_20 | Deviation of price from SMA-20 (%) |
| price_vs_SMA_50 | Deviation of price from SMA-50 (%) |

### 4.3 Volume Features

| Feature | Description |
|---|---|
| volume_sma_20 | 20-bar volume moving average |
| volume_ratio | Current volume / volume_sma_20 |
| volume_trend | Slope of volume SMA |

### 4.4 Price Action Features

| Feature | Description |
|---|---|
| returns_1h, returns_7h, returns_20h | Percentage returns over windows |
| volatility_20 | 20-bar rolling standard deviation of returns |
| atr_14 | Average True Range (14-bar) |
| rsi_14 | Relative Strength Index |
| high_low_range | (High - Low) / Close |
| body_to_range | Body / (High - Low) — candle "quality" |

### 4.5 Target Variable Construction

For supervised learning, the target is derived from **forward-looking returns**:

```
future_return_Nd = (max_close_in_next_N_days - current_close) / current_close
```

- **Classification target**: `1` (BUY) if `future_return_Nd >= expected_profit_threshold`, else `0`.
- **Regression targets**: `optimal_hold_days`, `target_price`, `stop_loss_price`.
- Stop-loss: lowest low in the next N days, or a fixed ATR-based trailing stop.

---

## 5. Model Architecture

### 5.1 Model Selection — Comparative Evaluation

Three model families will be trained, evaluated, and compared. The best performer is selected automatically.

| Model | Library | Rationale |
|---|---|---|
| **XGBoost** | `xgboost` | Strong on tabular data, handles feature interactions well |
| **LightGBM** | `lightgbm` | Fast training, good with high-cardinality features |
| **Random Forest** | `scikit-learn` | Robust baseline, less prone to overfitting |

> **Optional stretch**: An LSTM model via PyTorch for sequence-aware predictions, evaluated against the tabular models.

### 5.2 Training Strategy

- **Single unified model** across all ~517 tickers.
- Ticker category encoded as a categorical feature (`snp500`, `snp100`, `etf`, `merchandise`).
- **Train/test split**: Time-based (e.g., first 80% of data for training, last 20% for testing). No random split — preserves temporal ordering.
- **Cross-validation**: Walk-forward validation with expanding window.
- **Hyperparameter tuning**: Optuna for Bayesian optimization.

### 5.3 Model Outputs

For each ticker at prediction time, the model produces:

| Output | Type | Description |
|---|---|---|
| `buy_probability` | float 0–1 | Probability that the stock meets the profit target |
| `score` | int 1–100 | Normalized confidence score (from buy_probability) |
| `predicted_return` | float | Expected % return over the holding period |
| `optimal_hold_days` | int | Recommended number of days to hold |
| `target_price` | float | Predicted price at optimal exit |
| `stop_loss` | float | Recommended stop-loss price (ATR-based) |

### 5.4 Evaluation Metrics

| Metric | Purpose |
|---|---|
| Precision @ top-N | Of the top N recommendations, how many were actually profitable? |
| ROI simulation | Backtest: simulated return using model picks |
| AUC-ROC | Classification discrimination ability |
| Sharpe Ratio | Risk-adjusted return of model's portfolio vs. baseline |
| MAE on hold days | Accuracy of holding period prediction |

### 5.5 Model Persistence

```
models/
├── best_model.pkl            # Serialized best model
├── model_metadata.json       # Training date, metrics, hyperparams, model type
├── feature_columns.json      # Ordered feature list for inference consistency
└── scaler.pkl                # Feature scaler (if applicable)
```

---

## 6. API Design (FastAPI)

### 6.1 Endpoints

#### `POST /recommend`

Returns top-N stock recommendations.

**Request body:**

```json
{
  "num_stocks": 5,
  "expected_profit_pct": 3.0,
  "period_days": 10
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `num_stocks` | int | Yes | Number of stocks to recommend (top-N) |
| `expected_profit_pct` | float | Yes | Desired profit % (model filters for ±1% of this target) |
| `period_days` | int | Yes | Investment horizon in trading days |

**Response:**

```json
{
  "generated_at": "2026-03-23T14:00:00Z",
  "model_version": "xgboost_v1_20260320",
  "parameters": {
    "num_stocks": 5,
    "expected_profit_pct": 3.0,
    "period_days": 10
  },
  "recommendations": [
    {
      "symbol": "NVDA",
      "full_name": "NVIDIA Corporation",
      "current_price": 142.50,
      "ytd_pct": 12.3,
      "expected_profit_pct": 3.2,
      "period_days": 8,
      "score": 92,
      "target_price": 147.06,
      "stop_loss": 138.25
    }
  ]
}
```

#### `POST /refresh`

Pulls latest hourly candle data from Yahoo Finance for all tickers in the data folders.

**Request body:**

```json
{
  "categories": ["snp100", "snp500", "etfs", "merchandise"]
}
```

- If `categories` is omitted, all categories are refreshed.
- The endpoint reads existing ticker files to determine which symbols to fetch.
- New data is appended to existing files (deduped by DateTime).

**Response:**

```json
{
  "status": "completed",
  "tickers_updated": 517,
  "new_candles_added": 3420,
  "errors": []
}
```

#### `GET /health`

Health check returning model status and data freshness.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "xgboost_v1_20260320",
  "data_last_updated": "2026-03-23T12:00:00Z",
  "total_tickers": 517
}
```

#### `POST /train`

Triggers model retraining. Runs all candidate models, compares results, selects the best.

**Response:**

```json
{
  "status": "completed",
  "best_model": "xgboost",
  "metrics": {
    "auc_roc": 0.78,
    "precision_at_10": 0.72,
    "backtest_roi_pct": 14.5
  },
  "models_evaluated": [
    { "name": "xgboost", "auc_roc": 0.78 },
    { "name": "lightgbm", "auc_roc": 0.76 },
    { "name": "random_forest", "auc_roc": 0.71 }
  ]
}
```

### 6.2 Error Handling

Standard HTTP status codes with JSON error bodies:

```json
{
  "detail": "Model not trained yet. Call POST /train first.",
  "error_code": "MODEL_NOT_FOUND"
}
```

---

## 7. Project Structure

```
Trading.Agent2/
├── spec.md
├── architecture.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
│
├── Data/
│   ├── snp500_hourly/          # ~396 ticker files
│   ├── snp100_hourly/          # ~101 ticker files
│   ├── etfs_hourly/            # 15 ticker files
│   └── merchandise_hourly/     # 5 ticker files
│
├── Spec/
│   ├── THE CANDLESTICK TRADING BIBLE.pdf
│   └── Moving Average Trading.pdf
│
├── models/                     # Trained model artifacts
│   ├── best_model.pkl
│   ├── model_metadata.json
│   ├── feature_columns.json
│   └── scaler.pkl
│
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry point
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py           # Endpoint definitions
│   │   └── schemas.py          # Pydantic request/response models
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Read pipe-delimited files into DataFrames
│   │   ├── refresher.py        # Yahoo Finance data fetcher (yfinance)
│   │   └── ticker_registry.py  # Scan Data/ folders, resolve ticker lists
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── candlestick.py      # Candlestick pattern detection
│   │   ├── moving_average.py   # MA/EMA computation & crossovers
│   │   ├── volume.py           # Volume-based features
│   │   ├── price_action.py     # RSI, ATR, returns, volatility
│   │   └── pipeline.py         # Orchestrates full feature engineering
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Train, evaluate, compare models
│   │   ├── predictor.py        # Load model, generate predictions
│   │   ├── targets.py          # Target variable construction
│   │   └── evaluator.py        # Metrics, backtesting, model comparison
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # App configuration (paths, defaults)
│       └── ticker_info.py      # Symbol → full name mapping (yfinance)
│
└── tests/
    ├── test_features.py
    ├── test_model.py
    ├── test_api.py
    └── test_data_loader.py
```

---

## 8. Technology Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.11+ |
| API Framework | FastAPI | 0.110+ |
| ASGI Server | Uvicorn | 0.29+ |
| ML - Gradient Boosting | XGBoost | 2.0+ |
| ML - Gradient Boosting | LightGBM | 4.0+ |
| ML - Ensemble | scikit-learn | 1.4+ |
| Hyperparameter Tuning | Optuna | 3.6+ |
| Data Manipulation | pandas, numpy | latest |
| Technical Analysis | ta-lib or pandas-ta | latest |
| Market Data | yfinance | 0.2+ |
| Serialization | joblib | latest |
| Validation | Pydantic | 2.0+ |
| Containerization | Docker | latest |
| Testing | pytest | 8.0+ |

---

## 9. Deployment

### 9.1 Local (macOS)

```bash
pip install -r requirements.txt
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 9.2 Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

The `Data/` and `models/` directories are mounted as Docker volumes so data persists across container restarts:

```yaml
# docker-compose.yml
services:
  trading-agent:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./Data:/app/Data
      - ./models:/app/models
```

---

## 10. Data Flow

```
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │ POST /refresh│────▶│ Yahoo Finance│────▶│ Update .txt  │
 │              │     │ (yfinance)   │     │ files in Data│
 └──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │ POST /train  │────▶│ Feature Eng. │────▶│ Train Models │
 │              │     │ Pipeline     │     │ (XGB/LGB/RF) │
 └──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                           Compare & select
                                                  │
                                                  ▼
                                           ┌──────────────┐
                                           │ Save best to │
                                           │ models/*.pkl │
                                           └──────────────┘
                                                  │
                                                  ▼
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │POST /recommend────▶│ Load model + │────▶│ Return top-N │
 │              │     │ latest data  │     │ ranked picks │
 └──────────────┘     └──────────────┘     └──────────────┘
```

### Workflow:

1. **Refresh** → `POST /refresh` pulls latest hourly data from Yahoo Finance, appends to existing files
2. **Train** → `POST /train` runs the full pipeline: load data → engineer features → build targets → train 3 models → evaluate → save best
3. **Recommend** → `POST /recommend` loads the trained model, applies features to latest data, scores all tickers, returns top-N filtered by profit target and period

---

## 11. Key Design Decisions

| Decision | Rationale |
|---|---|
| **Single unified model** | One model learns cross-asset patterns; ticker category is a feature, not a model boundary |
| **Flat files over DB** | Simplicity for current scale (~517 files); avoids infrastructure overhead |
| **FastAPI** | Async-ready, auto-generated OpenAPI docs, Pydantic validation |
| **Time-based train/test split** | Prevents look-ahead bias inherent in random splits on time-series data |
| **Model comparison built-in** | Automatically selects the best model — no manual tuning required |
| **ATR-based stop-loss** | Adapts to each stock's volatility rather than using a fixed percentage |
| **No auth** | Internal/local use; add API key middleware later if needed |

---

## 12. Constraints & Assumptions

- Yahoo Finance rate limits may affect refresh speed for 500+ tickers; the refresher batches requests with delays.
- The model assumes sufficient historical data (at least 2–3 months of hourly candles per ticker).
- Predictions are informational — not financial advice.
- Market hours data only (no extended/pre-market).
- The `full_name` for each ticker is resolved via `yfinance` ticker info and cached locally.
