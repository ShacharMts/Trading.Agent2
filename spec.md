# Trading Agent — Product Requirements Specification

**Version:** v7 (March 2026)
**Status:** Implemented
**Repository:** [ShacharMts/AlertsAnalyzer2](https://github.com/ShacharMts/Trading.Agent2)

---

## 1. Product Summary

An ML-powered stock recommendation engine that analyzes hourly candlestick data to generate daily buy recommendations with confidence scores, price targets, and stop-loss levels.

**Domain expertise** is derived from two reference works:
- *The Candlestick Trading Bible* — candlestick pattern recognition
- *Moving Average Trading* — trend detection via moving average strategies

---

## 2. Objectives

| Objective | Description | Status |
|---|---|---|
| Analyze hourly trading data | Process OHLCV candle data from ~421 tickers across S&P 500, S&P 100, sector ETFs, and commodity/crypto ETFs | ✅ Implemented |
| Predict buy opportunities | Score stocks by volume, moving averages, candlestick patterns, and technical indicators | ✅ Implemented |
| Recommend buy + sell targets | Provide entry signal, target price, holding period, and stop-loss for each pick | ✅ Implemented |
| Multi-model comparison | Train multiple ML models, evaluate, and select the best automatically | ✅ Implemented (3-model ensemble) |
| REST API | Expose recommendations, training, data refresh, and health via HTTP endpoints | 🔲 Planned (FastAPI) |
| Run locally + cloud | macOS local execution; Docker container for cloud deployment | 🔲 Planned (Docker) |
| Yahoo Finance refresh | API endpoint to pull latest data for all existing tickers | 🔲 Planned |
| No authentication | API endpoints are unauthenticated for internal use | Designed |

---

## 3. Data Requirements

### 3.1 Data Sources

| Category | Folder | Tickers | Examples |
|---|---|---|---|
| S&P 500 | `Data/snp500_hourly/` | ~500 | AAPL, MSFT, NVDA, AMZN, GOOGL, META, ... |
| S&P 100 | `Data/snp100_hourly/` | ~100 | Largest S&P 500 constituents (deduplicated) |
| Sector ETFs | `Data/etfs_hourly/` | 15 | XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLU, XLRE, XLC, XAR, VOO, MAGS, HACK, IGV |
| Merchandise | `Data/merchandise_hourly/` | 5 | GLD (gold), USO (oil), IBIT (bitcoin), ETHA (ethereum), SLVR (silver) |

**Total unique symbols after deduplication:** 421
**Data range:** September 2, 2025 — March 23, 2026
**Total candles:** ~404,544 hourly bars

### 3.2 File Format

- **Format:** Pipe-delimited (`|`) text files
- **Naming:** `{SYMBOL}_hourly_candles.txt`
- **Columns:** `DateTime | Open | High | Low | Close | Volume`
- **Frequency:** Hourly, market hours only (9:30 AM – 4:00 PM ET, 7 bars/day)

### 3.3 Storage

- **Current:** Local flat files in `Data/` directory, loaded into pandas DataFrames
- **Model artifacts:** Serialized to `models/` directory (`.pkl`, `.json`)
- **No external database** — all state on filesystem

---

## 4. Feature Engineering

**49 features** across 8 categories, computed per-symbol:

### 4.1 Candlestick Patterns (4 features)

Derived from *The Candlestick Trading Bible*. Only patterns with non-zero model importance are retained:

- `pat_three_white_soldiers` — 3 consecutive bullish candles with rising closes
- `pat_three_black_crows` — 3 consecutive bearish candles with falling closes
- `bullish_score` — Composite count of bullish signals (hammer, bullish engulfing, marubozu, 3WS)
- `bearish_score` — Composite count of bearish signals (inv. hammer, bearish engulfing, marubozu, 3BC)

### 4.2 Moving Averages (10 features)

Derived from *Moving Average Trading*:

- **SMAs:** `sma_7`, `sma_20`, `sma_50`, `sma_100`
- **EMAs:** `ema_7`, `ema_20`, `ema_50`
- **Derivatives:** `ma_slope_20` (trend direction), `price_vs_sma_20` (% deviation), `price_vs_sma_50` (% deviation)

### 4.3 Volume (3 features)

- `volume_sma_20` — 20-bar rolling average volume
- `volume_ratio` — Current volume / average (spike detection)
- `volume_trend` — Slope of volume SMA over 5 bars

### 4.4 Price Action (7 features)

- **Returns:** `returns_1h`, `returns_7h`, `returns_20h`
- **Volatility:** `volatility_20` (20-bar rolling std of returns)
- **Indicators:** `atr_14` (Average True Range), `rsi_14` (Relative Strength Index)
- **Candle quality:** `high_low_range`, `body_to_range`

### 4.5 Technical Indicators (12 features)

- **MACD (12, 26, 9):** `macd_line`, `macd_signal`, `macd_histogram`
- **Bollinger Bands (20, 2σ):** `bb_upper`, `bb_lower`, `bb_width_pct`, `bb_position`
- **Stochastic (14, 3):** `stoch_k`, `stoch_d`
- **Multi-day returns:** `returns_3d`, `returns_5d`, `returns_10d`
- **Momentum gates:** `overextension`, `momentum_decel`, `gap_pct`

### 4.6 Market & Sector Context (7 features)

- **Calendar:** `day_of_week`, `week_of_month`
- **Market relative (VOO proxy):** `relative_strength`, `relative_strength_20`, `market_regime`
- **Sector relative:** `vs_sector`, `vs_sector_20`

### 4.7 Categorical (2 features)

- `direction_num` — 1 (bullish) / 0 (bearish)
- `category_num` — 0 (snp500), 1 (snp100), 2 (etfs), 3 (merchandise)

---

## 5. Model Requirements

### 5.1 Model Type

**3-model ensemble** (averaged probabilities):

| Model | Library | Config |
|---|---|---|
| XGBoost | `xgboost` | 500 trees, depth 7, lr 0.03, subsample 0.8 |
| LightGBM | `lightgbm` | 500 trees, depth 7, lr 0.03, subsample 0.8 |
| Random Forest | `scikit-learn` | 500 trees, depth 15, min_samples_leaf 5 |

All models use `scale_pos_weight=1.5` to handle class imbalance (71.6% HOLD / 28.4% BUY).

### 5.2 Training Strategy

- **Single unified model** across all 421 tickers — ticker category is a feature, not a model boundary
- **Train/test split:** Time-based per-symbol — first 80% train, last 20% test (no random shuffle)
- **Cross-validation:** Walk-forward with 3 expanding folds
- **Model selection:** Best walk-forward average **Sharpe ratio** (risk-adjusted profitability)
- **Training parallelization:** `ProcessPoolExecutor(max_workers=3)` — all 3 models train simultaneously
- **Training data:** 362,406 clean rows (after NaN removal), 49 features, StandardScaler normalization

### 5.3 Target Construction

- **Classification target:** `target_buy = 1` if stock achieves ≥ `profit_threshold` % gain within `period_days` trading days
- **Forward return:** Maximum close price in next N×7 bars vs current close
- **Adaptive stop-loss:** ATR-based volatility multiplier — 1.5× ATR for low-vol stocks, up to 3.0× ATR for high-vol stocks
- **Default parameters:** 3% profit threshold, 10-day horizon

### 5.4 Current Model Performance

| Model | AUC | P@10 | P@20 | WF Sharpe | WF ROI |
|---|---|---|---|---|---|
| **LightGBM** (best) | 0.606 | 0.80 | 0.70 | **2.845** | +9.06% |
| XGBoost | 0.608 | 1.00 | 0.70 | 1.935 | +8.48% |
| Random Forest | 0.600 | 0.60 | 0.65 | 1.454 | +6.35% |

---

## 6. Recommendation Engine

### 6.1 Scoring Formula

Each stock is scored using a multi-factor approach:

```
score = (25% × model_confidence + 40% × historical_feasibility + 35% × gain_ratio)
        × overextension_penalty × momentum_decel_penalty
        × mean_reversion_factor × profit_cap_penalty × regime_penalty
```

- **Model confidence (25%)** — Averaged probability from 3 ensemble models
- **Historical feasibility (40%)** — How often the stock achieved the target in the last 20 trading days
- **Gain ratio (35%)** — Is the target realistic given the stock's recent price movements
- **Overextension penalty (0.5–1.0×)** — Reduces score if stock already moved >target% in 5 days
- **Momentum deceleration penalty (0.7–1.0×)** — Reduces score if 3-day momentum is fading vs 5-day
- **Mean-reversion factor (0.85–1.15×)** — Bonus for below-SMA-20 with momentum; penalty for >5% above SMA-20
- **Profit cap penalty (0.5–1.0×)** — Penalizes unrealistic expected profits (>2× target)
- **Regime penalty (0.7–1.0×)** — Higher bar when market (VOO) declining >2%

### 6.2 Pre-Scoring Filters

| Filter | Rule | Rationale |
|---|---|---|
| **Hard blacklist** | Block INTC, MU, LRCX, NEM | Chronic 0–28% hit rate despite high model scores |
| **Min feasibility** | Reject if hist_pct < max(15%, 35% − target×2) | Scales with target difficulty |
| **ATR stop-out blocker** | Reject if ATR/price > 80% of target | Too-volatile for the profit target |

### 6.3 Diversity Rules

To prevent sector concentration:
- **Category penalty:** `0.70 ^ count` — each repeat from same sector is multiplied by 0.70
- **Hard cap (stocks):** Maximum 3 recommendations per snp100/snp500 category
- **Hard cap (ETFs/merch):** Maximum 2 recommendations per merchandise/ETFs category
- **Acceptance threshold:** Adjusted score must be ≥ 60% of the previous pick's score

### 6.4 Adaptive Stop-Loss

Stop-loss adjusts per-stock based on ATR volatility:
- **Low-vol stocks:** Tighter stop (1.5× – 2.0× ATR below entry)
- **High-vol stocks:** Wider stop (2.0× – 3.0× ATR below entry)
- **Hard floor:** 99% of the future minimum low

---

## 7. API Specification

### 7.1 `POST /recommend` — Get Buy Recommendations

**Request:**
```json
{
  "num_stocks": 5,
  "expected_profit_pct": 3.0,
  "period_days": 10
}
```

| Field | Type | Description |
|---|---|---|
| `num_stocks` | int | Number of stocks to recommend (top-N) |
| `expected_profit_pct` | float | Expected profit % (model targets ±1% of this value) |
| `period_days` | int | Holding period in trading days |

**Response:**
```json
{
  "generated_at": "2026-03-23T14:00:00Z",
  "model_version": "ensemble_v3_20260323",
  "parameters": {
    "num_stocks": 5,
    "expected_profit_pct": 3.0,
    "period_days": 10
  },
  "recommendations": [
    {
      "symbol": "NVDA",
      "current_price": 142.50,
      "expected_profit_pct": 3.2,
      "period_days": 8,
      "score": 92,
      "target_price": 147.06,
      "stop_loss": 138.25,
      "sma_20": 141.80,
      "vs_sma_20": "Above",
      "ytd_pct": 12.3,
      "month_pct": 4.1
    }
  ]
}
```

| Response Field | Type | Description |
|---|---|---|
| `symbol` | string | Ticker symbol |
| `current_price` | float | Latest closing price |
| `expected_profit_pct` | float | Estimated return (%) |
| `period_days` | int | Recommended holding period |
| `score` | int 1–100 | Multi-factor confidence score |
| `target_price` | float | Exit price target |
| `stop_loss` | float | Adaptive ATR-based stop-loss price |
| `sma_20` | float | Current 20-bar moving average |
| `vs_sma_20` | string | "Above" or "Below" SMA-20 |
| `ytd_pct` | float | Year-to-date performance (%) |
| `month_pct` | float | Month-to-date performance (%) |

### 7.2 `POST /refresh` — Update Market Data

Pulls latest hourly candle data from Yahoo Finance for all tickers already in the data folder.

**Request:**
```json
{
  "categories": ["snp100", "snp500", "etfs", "merchandise"]
}
```

**Response:**
```json
{
  "status": "completed",
  "tickers_updated": 421,
  "new_candles_added": 3420,
  "errors": []
}
```

### 7.3 `POST /train` — Retrain Model

Triggers full retraining pipeline: load data → engineer features → train 3 models → walk-forward CV → select best → save ensemble.

**Response:**
```json
{
  "status": "completed",
  "best_model": "lightgbm",
  "metrics": {
    "auc_roc": 0.606,
    "precision_at_10": 0.80,
    "wf_sharpe": 2.845
  },
  "models_evaluated": [
    { "name": "lightgbm", "wf_sharpe": 2.845 },
    { "name": "xgboost", "wf_sharpe": 1.935 },
    { "name": "random_forest", "wf_sharpe": 1.454 }
  ]
}
```

### 7.4 `GET /health` — Health Check

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "ensemble_v3_20260323",
  "data_last_updated": "2026-03-23T12:00:00Z",
  "total_tickers": 421
}
```

### 7.5 API Configuration

- **Framework:** FastAPI (auto-generated OpenAPI docs)
- **Authentication:** None (internal/local use)
- **Error format:** `{"detail": "...", "error_code": "..."}`

---

## 8. Backtest Results

### 8.1 February 2026 Backtest

19 trading days × 4 profit target scenarios × 10 stocks/day = 760 total recommendations.

| Scenario | Hit Rate | Avg 10d P&L |
|---|---|---|
| **3% / 10d** | **159/190 (83.7%)** | +3.06% |
| **5% / 10d** | **154/190 (81.1%)** | +3.87% |
| **8% / 10d** | **140/190 (73.7%)** | +3.54% |
| **10% / 10d** | **140/190 (73.7%)** | +3.69% |

### 8.2 Version Progression

| Version | Changes | 3%/10d | 5%/10d | 8%/10d | 10%/10d |
|---|---|---|---|---|---|
| v1 | Single XGBoost, 43 features | — | — | — | — |
| v2 | Dropped dead features, added MACD/BB/Stochastic | — | — | — | — |
| v3 | 3-model ensemble, walk-forward CV, Sharpe selection | 71.6% | 63.7% | 52.6% | 48.4% |
| v4 | Parallel training + backtest | 71.6% | 65.8% | 57.4% | 53.2% |
| v5 | Adaptive stops, overextension filter, momentum decel, sector diversity | 71.6% | 66.8% | 61.1% | 62.1% |
| v6 | Scoring recalibration (25/40/35), mean-reversion, regime gate, feasibility threshold | 78.9% | 74.7% | 66.3% | 58.4% |
| **v7** | **Hard blacklist, ATR stop-out blocker, Above-SMA gate, tighter profit cap** | **83.7%** | **81.1%** | **73.7%** | **73.7%** |

---

## 9. Deployment Requirements

### 9.1 Local (macOS)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py                  # Train model
python -m src.model.predictor            # Generate recommendations
uvicorn src.main:app --port 8000         # Start API server (planned)
```

### 9.2 Docker (Planned)

```yaml
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

## 10. Technology Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.14 |
| ML — Gradient Boosting | XGBoost, LightGBM | 2.0+, 4.0+ |
| ML — Ensemble & Utilities | scikit-learn | 1.4+ |
| Data Processing | pandas, numpy | latest |
| Model Serialization | joblib | latest |
| API Framework | FastAPI + Uvicorn | 0.110+ (planned) |
| Market Data | yfinance | 0.2+ (planned) |
| Containerization | Docker | latest (planned) |
| Testing | pytest | 8.0+ |

---

## 11. Project Structure

```
Trading.Agent2/
├── architecture.md              # Detailed technical architecture
├── spec.md                      # This document
├── requirements.txt             # Python dependencies
├── backtest_feb_report.md       # Auto-generated backtest results
├── backtest_feb_results.csv     # Raw backtest data (760 rows)
│
├── Data/                        # Market data (pipe-delimited .txt files)
│   ├── snp500_hourly/           # ~500 S&P 500 tickers
│   ├── snp100_hourly/           # ~100 S&P 100 tickers
│   ├── etfs_hourly/             # 15 sector/thematic ETFs
│   └── merchandise_hourly/      # 5 commodity/crypto ETFs
│
├── Spec/                        # Reference documents
│   ├── THE CANDLESTICK TRADING BIBLE.pdf
│   └── Moving Average Trading.pdf
│
├── models/                      # Trained model artifacts
│   ├── best_model.pkl           # 3-model ensemble
│   ├── model_metadata.json      # Metrics, version, training info
│   ├── feature_columns.json     # 49 ordered feature names
│   └── scaler.pkl               # StandardScaler
│
├── src/
│   ├── data/                    # Data loading & ticker discovery
│   │   ├── loader.py
│   │   └── ticker_registry.py
│   ├── features/                # Feature engineering (49 features)
│   │   ├── candlestick.py       # Pattern detection
│   │   ├── moving_average.py    # SMA/EMA features
│   │   ├── volume.py            # Volume indicators
│   │   ├── price_action.py      # Returns, volatility, RSI, ATR
│   │   ├── momentum.py          # MACD, Bollinger, Stochastic
│   │   └── pipeline.py          # Orchestration, sector/market context
│   ├── model/                   # ML pipeline
│   │   ├── targets.py           # Forward targets + adaptive stops
│   │   ├── trainer.py           # Train, evaluate, ensemble
│   │   ├── predictor.py         # Inference, scoring, diversity
│   │   └── evaluator.py         # Classification + trading metrics
│   └── utils/
│       └── config.py            # Paths, constants, defaults
│
├── scripts/
│   ├── train.py                 # Training entry point
│   └── backtest_feb.py          # Feb 2026 backtesting framework
│
└── tests/
    └── __init__.py
```

---

## 12. Constraints & Assumptions

- **Data minimum:** ≥ 50 hourly bars per symbol (~7 trading days) for feature computation
- **Market hours only:** 7 bars/day (9:30 AM – 4:00 PM ET), no pre/after-market
- **Yahoo Finance rate limits:** Refresh batches requests with delays for 421+ tickers
- **Predictions are informational** — not financial advice
- **No authentication** — designed for internal/local use
- **Reproducibility:** All models use `random_state=42`
- **Memory:** Full training pipeline requires ~5 GB RAM

---

## 13. Original Requirements (Raw)

> *Finance trader expert and ML developer expert:*
>
> Develop a ML model to analyze hourly trading stocks data from the last few months based on 2 documents: *THE CANDLESTICK TRADING BIBLE* and *Moving Average Trading*. Train and test the model based on previous data. Predict what stocks are recommended to buy based on volume, moving averages, and candlestick patterns. Recommend buy and when to sell.
>
> - Risk as expected profit percentage ±1%
> - Score: 1 to 100
> - API: number of stocks, expected profit %, period → Symbol, price, YTD, profit %, period, score, target price, stop-loss
> - One model for all tickers
> - Try multiple models, recommend best
> - No authentication
> - Database: local files
> - Yahoo Finance refresh via API call
> - Scope: refresh for tickers already in data folder
> - Run locally on macOS and on Docker in the cloud

