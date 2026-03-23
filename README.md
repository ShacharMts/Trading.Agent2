# Trading Agent

An ML-powered stock recommendation engine that analyzes hourly candlestick data across **421 tickers** (S&P 500, S&P 100, sector ETFs, commodity/crypto ETFs) and generates daily buy recommendations with confidence scores, price targets, and adaptive stop-losses.

## Performance

Backtested on **February 2026** (19 trading days × 10 stocks/day = 190 recommendations per scenario):

| Profit Target | Hit Rate | Avg 10-Day P&L |
|---------------|----------|-----------------|
| 3% in 10 days | **71.6%** | +3.86% |
| 5% in 10 days | **66.8%** | +2.95% |
| 8% in 10 days | **61.1%** | +2.32% |
| 10% in 10 days | **62.1%** | +2.52% |

## How It Works

1. **49 engineered features** — candlestick patterns, moving averages, volume, RSI, MACD, Bollinger Bands, Stochastic, market regime, sector-relative strength
2. **3-model ensemble** — XGBoost + LightGBM + RandomForest (averaged probabilities)
3. **Walk-forward validation** — 3-fold expanding window, model selected by Sharpe ratio
4. **Multi-factor scoring** — 40% model confidence + 30% historical feasibility + 30% gain ratio, with overextension and momentum deceleration penalties
5. **Adaptive stop-loss** — ATR-based volatility multiplier (1.5×–3.0× ATR)
6. **Sector diversity** — Hard cap of 3 picks per category, 0.80× penalty per repeat

## Quick Start

```bash
# Clone
git clone https://github.com/ShacharMts/Trading.Agent2.git
cd Trading.Agent2

# Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train the model
python scripts/train.py

# Run backtest
python scripts/backtest_feb.py
```

## Project Structure

```
Trading.Agent2/
├── Data/                        # Hourly OHLCV candle data (pipe-delimited .txt)
│   ├── snp500_hourly/           # ~500 S&P 500 tickers
│   ├── snp100_hourly/           # ~100 S&P 100 tickers
│   ├── etfs_hourly/             # 15 sector/thematic ETFs
│   └── merchandise_hourly/      # 5 commodity/crypto ETFs (GLD, USO, IBIT, ETHA, SLVR)
│
├── src/
│   ├── data/                    # Data loading & ticker discovery
│   │   ├── loader.py            # Load pipe-delimited files → DataFrames
│   │   └── ticker_registry.py   # Scan Data/ folders, discover tickers
│   ├── features/                # Feature engineering (49 features)
│   │   ├── candlestick.py       # Pattern detection (Three White Soldiers, etc.)
│   │   ├── moving_average.py    # SMA/EMA features & slopes
│   │   ├── volume.py            # Volume ratio, trend
│   │   ├── price_action.py      # Returns, volatility, RSI, ATR
│   │   ├── momentum.py          # MACD, Bollinger Bands, Stochastic
│   │   └── pipeline.py          # Feature orchestration, market/sector context
│   ├── model/
│   │   ├── targets.py           # Forward-looking targets & adaptive stop-loss
│   │   ├── trainer.py           # Train, evaluate, compare, save ensemble
│   │   ├── predictor.py         # Inference, scoring, diversity filtering
│   │   └── evaluator.py         # Classification & trading-specific metrics
│   └── utils/
│       └── config.py            # Paths, constants, defaults
│
├── scripts/
│   ├── train.py                 # Training entry point
│   └── backtest_feb.py          # February 2026 backtesting framework
│
├── models/                      # Trained model artifacts
│   ├── best_model.pkl           # 3-model ensemble
│   ├── model_metadata.json      # Training metrics & version info
│   ├── feature_columns.json     # 49 ordered feature names
│   └── scaler.pkl               # StandardScaler
│
├── Spec/                        # Reference documents
│   ├── THE CANDLESTICK TRADING BIBLE.pdf
│   └── Moving Average Trading.pdf
│
├── architecture.md              # Detailed technical architecture
├── spec.md                      # Product requirements specification
├── backtest_feb_report.md       # Latest backtest results
└── requirements.txt             # Python dependencies
```

## Model Details

### Ensemble Architecture

| Model | Trees | Depth | Learning Rate | WF Sharpe |
|-------|-------|-------|---------------|-----------|
| XGBoost | 500 | 7 | 0.03 | 1.935 |
| **LightGBM** | 500 | 7 | 0.03 | **2.845** |
| Random Forest | 500 | 15 | — | 1.454 |

All models use `scale_pos_weight=1.5` for class imbalance (71.6% HOLD / 28.4% BUY). LightGBM selected as best single model by walk-forward Sharpe ratio. Predictions are averaged probabilities across all three.

### Features (49 total)

| Category | Count | Examples |
|----------|-------|---------|
| Candlestick Patterns | 4 | Three White Soldiers, bullish/bearish composite scores |
| Moving Averages | 10 | SMA(7,20,50,100), EMA(7,20,50), slope, price deviation |
| Volume | 3 | Volume ratio, trend |
| Price Action | 7 | Returns (1h,7h,20h), volatility, ATR, RSI |
| Technical Indicators | 12 | MACD, Bollinger Bands, Stochastic, multi-day returns, momentum gates |
| Market & Sector | 5 | Relative strength vs VOO, market regime, sector-relative returns |
| Calendar & Categorical | 4 | Day of week, week of month, direction, category |
| **Encoding** | **4** | direction_num, category_num, day_of_week, week_of_month |

### Training Pipeline

```
Load 421 symbols → Engineer 49 features → Build targets (10d, 3%) → 
Drop NaN → 362,406 rows → Time-based 80/20 split → StandardScaler →
Train 3 models in parallel (ProcessPoolExecutor) →
Walk-forward CV (3 folds) → Select by Sharpe → Save ensemble
```

## Recommendation Output

Each recommendation includes:

| Field | Description |
|-------|-------------|
| `symbol` | Ticker symbol |
| `current_price` | Latest closing price |
| `score` | Confidence score (1–100) |
| `expected_profit_pct` | Estimated return (%) |
| `period_days` | Recommended holding period |
| `target_price` | Exit price target |
| `stop_loss` | Adaptive ATR-based stop-loss |
| `sma_20` | Current 20-bar moving average |
| `vs_sma_20` | Position relative to SMA-20 |
| `ytd_pct` | Year-to-date return (%) |

## Data Format

Pipe-delimited (`|`) text files: `{SYMBOL}_hourly_candles.txt`

```
DateTime|Open|High|Low|Close|Volume
2025-09-02 09:30:00|150.00|151.20|149.80|150.95|1234567
```

- **Frequency:** Hourly (7 bars/day, 9:30 AM – 4:00 PM ET)
- **Range:** September 2025 – March 2026 (~404K total candles)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.14 |
| ML | XGBoost, LightGBM, scikit-learn |
| Data | pandas, numpy |
| Serialization | joblib |
| API (planned) | FastAPI + Uvicorn |
| Market Data (planned) | yfinance |
| Containerization (planned) | Docker |

## Documentation

- [architecture.md](architecture.md) — Full technical architecture (917 lines)
- [spec.md](spec.md) — Product requirements specification
- [backtest_feb_report.md](backtest_feb_report.md) — Detailed backtest results by date and scenario

## Version History

| Version | Key Changes | 10%/10d Hit Rate |
|---------|-------------|------------------|
| v1 | Single XGBoost, 43 features | — |
| v2 | Added MACD, Bollinger, Stochastic | — |
| v3 | 3-model ensemble, walk-forward CV | 48.4% |
| v4 | Parallel training & backtest | 53.2% |
| **v5** | **Adaptive stops, overextension filter, momentum gates, sector diversity** | **62.1%** |

## Disclaimer

This project is for educational and research purposes only. Predictions are informational and do not constitute financial advice.
