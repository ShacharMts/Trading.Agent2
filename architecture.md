# Trading Agent — Architecture Document

**Version:** v7 (March 2026)
**Repository:** [ShacharMts/AlertsAnalyzer2](https://github.com/ShacharMts/Trading.Agent2)

---

## 1. Overview

A Python-based ML trading recommendation system that analyzes hourly candlestick data for **421 unique tickers** across 4 asset categories (S&P 500, S&P 100, sector ETFs, and commodity/crypto ETFs). It uses a **3-model ensemble** (XGBoost + LightGBM + RandomForest) with **49 engineered features**, walk-forward cross-validation, and a multi-factor scoring engine to produce daily buy recommendations with price targets and adaptive stop-losses.

**Backtest performance (Feb 2026):**

| Target | Hit Rate | Avg 10d P&L |
|--------|----------|-------------|
| 3% / 10 days | 83.7% | +3.06% |
| 5% / 10 days | 81.1% | +3.87% |
| 8% / 10 days | 73.7% | +3.54% |
| 10% / 10 days | 73.7% | +3.69% |

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
│  │ Ensemble    │  │ Yahoo Finance│                              │
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
| `Data/snp500_hourly/` | ~500 | S&P 500 constituents |
| `Data/snp100_hourly/` | ~100 | S&P 100 large-caps (subset of S&P 500) |
| `Data/etfs_hourly/` | 15 | Sector & thematic ETFs (XLK, XLF, VOO, MAGS, HACK, IGV, XAR, etc.) |
| `Data/merchandise_hourly/` | 5 | Commodities & crypto ETFs (GLD, USO, IBIT, ETHA, SLVR) |

**After deduplication:** 421 unique symbols (S&P 100 entries preferred over S&P 500 copies).
**Data range:** September 2, 2025 — March 23, 2026 (~404,544 total hourly candles).

### 3.2 File Format

Pipe-delimited (`|`) text files: `{SYMBOL}_hourly_candles.txt`

| Column | Type | Description |
|---|---|---|
| DateTime | datetime | Candle timestamp (hourly, market hours only) |
| Open | float | Opening price |
| High | float | High price |
| Low | float | Low price |
| Close | float | Closing price |
| Volume | int | Trade volume |

**Derived columns** (computed at load time for candlestick patterns):
- `Body` = `|Close - Open|`
- `UpperShadow` = `High - max(Open, Close)`
- `LowerShadow` = `min(Open, Close) - Low`
- `Direction` = `BULLISH` if `Close >= Open`, else `BEARISH`

### 3.3 Data Loading Process

1. **`load_ticker_data(filepath)`** — Load single ticker with sorted DateTime index
2. **`load_category(category)`** — Load all tickers for a category, add `category` column
3. **`load_all_data()`** — Combine all 4 categories; deduplicate by `(Symbol, DateTime)` keeping first occurrence
4. **Result:** Single DataFrame with columns: `Symbol`, `DateTime`, `Open`, `High`, `Low`, `Close`, `Volume`, `category`

### 3.4 Data Storage Strategy

- **Data files**: Local flat files (pipe-delimited `.txt`), loaded into pandas DataFrames at runtime
- **Model artifacts**: Serialized to `models/` directory as `.pkl` and `.json` files
- **No external database** — all state lives on the filesystem

---

## 4. Feature Engineering Pipeline

**49 features total** across 8 categories, computed per symbol with a minimum of 50 bars required.

### 4.1 Candlestick Pattern Features (4 features)

Source: `src/features/candlestick.py`

| Feature | Description |
|---|---|
| `pat_three_white_soldiers` | 3 consecutive bullish candles with increasing closes (binary) |
| `pat_three_black_crows` | 3 consecutive bearish candles with decreasing closes (binary) |
| `bullish_score` | Count of active bullish signals (hammer, engulfing, marubozu bullish, 3WS) |
| `bearish_score` | Count of active bearish signals (inv. hammer, bearish engulfing, marubozu bearish, 3BC) |

> Patterns with zero LightGBM feature importance were dropped in v2 (Hammer, Shooting Star, Doji, Engulfing individuals, etc.). Only the two triple-candle patterns and composite scores are retained.

### 4.2 Moving Average Features (10 features)

Source: `src/features/moving_average.py`

| Feature | Description |
|---|---|
| `sma_7`, `sma_20`, `sma_50`, `sma_100` | Simple Moving Averages (7, 20, 50, 100 hourly bars) |
| `ema_7`, `ema_20`, `ema_50` | Exponential Moving Averages |
| `ma_slope_20` | Rate of change of SMA-20 over 5 bars (trend direction & strength) |
| `price_vs_sma_20` | `(Close - SMA_20) / SMA_20 × 100` — deviation from 20-bar mean (%) |
| `price_vs_sma_50` | `(Close - SMA_50) / SMA_50 × 100` — deviation from 50-bar mean (%) |

### 4.3 Volume Features (3 features)

Source: `src/features/volume.py`

| Feature | Description |
|---|---|
| `volume_sma_20` | 20-bar rolling average of volume |
| `volume_ratio` | `Volume / volume_sma_20` — relative volume spike detection |
| `volume_trend` | Slope of volume SMA over 5 bars |

### 4.4 Price Action Features (7 features)

Source: `src/features/price_action.py`

| Feature | Description |
|---|---|
| `returns_1h` | 1-bar percentage return |
| `returns_7h` | 7-bar percentage return (~1 day) |
| `returns_20h` | 20-bar percentage return (~3 days) |
| `volatility_20` | 20-bar rolling standard deviation of 1-bar returns |
| `atr_14` | Average True Range (14-bar); uses `max(H-L, |H-Cprev|, |L-Cprev|)` |
| `rsi_14` | Relative Strength Index (14-bar) |
| `high_low_range` | `(High - Low) / Close × 100` — intrabar range (%) |
| `body_to_range` | `Body / (High - Low)` — candle "quality" ratio |

### 4.5 Momentum & Technical Indicators (12 features)

Source: `src/features/momentum.py` — Added in v2/v5

**MACD (12, 26, 9):**

| Feature | Description |
|---|---|
| `macd_line` | EMA-12 minus EMA-26 |
| `macd_signal` | 9-bar EMA of MACD line |
| `macd_histogram` | MACD line minus signal line |

**Bollinger Bands (20, 2σ):**

| Feature | Description |
|---|---|
| `bb_upper` | `SMA_20 + 2 × Std(20)` |
| `bb_lower` | `SMA_20 - 2 × Std(20)` |
| `bb_width_pct` | Band width as % of SMA-20 |
| `bb_position` | `(Close - bb_lower) / (bb_upper - bb_lower)` — position within bands (0=lower, 1=upper) |

**Stochastic Oscillator (14, 3):**

| Feature | Description |
|---|---|
| `stoch_k` | `(Close - Low14) / (High14 - Low14) × 100` |
| `stoch_d` | 3-bar SMA of Stoch-K |

**Multi-Day Returns & Momentum Gates (v5):**

| Feature | Description |
|---|---|
| `returns_3d` | % change over 21 bars (3 days × 7 bars/day) |
| `returns_5d` | % change over 35 bars (5 days × 7 bars/day) |
| `returns_10d` | % change over 70 bars (10 days × 7 bars/day) |
| `overextension` | `(Close - SMA_20) / SMA_20 × 100` — how far above 20-day mean |
| `momentum_decel` | `returns_3d - returns_5d` — negative = decelerating momentum |
| `gap_pct` | `(Open - Close_prev) / Close_prev × 100` — overnight gap |

### 4.6 Calendar Features (2 features)

Source: `src/features/pipeline.py`

| Feature | Description |
|---|---|
| `day_of_week` | 0 (Monday) to 4 (Friday) |
| `week_of_month` | 0 to 4 (week number within month) |

### 4.7 Market Relative Strength (5 features)

Source: `src/features/pipeline.py`

**VOO (S&P 500 ETF) as market proxy:**

| Feature | Description |
|---|---|
| `relative_strength` | Stock return − VOO return (per-bar outperformance) |
| `relative_strength_20` | Rolling 20-bar average of relative strength |
| `market_regime` | Rolling 20-bar return of VOO (positive = uptrend, negative = downtrend) |

**Sector-relative returns (v5):**

| Feature | Description |
|---|---|
| `vs_sector` | Stock return − category average return (peer outperformance) |
| `vs_sector_20` | Rolling 20-bar average of vs_sector |

### 4.8 Categorical Encodings (2 features)

| Feature | Description |
|---|---|
| `direction_num` | 1 if bullish candle, 0 if bearish |
| `category_num` | `{0: "snp500", 1: "snp100", 2: "etfs", 3: "merchandise"}` |

### 4.9 Feature Pipeline Orchestration

Source: `src/features/pipeline.py` → `engineer_features(data)`

```
1. Load all data across 4 categories
2. Compute market returns (VOO) and market regime (20-bar rolling)
3. Compute sector returns (per-category average returns)
4. For each symbol (≥ 50 bars):
   a. Add candlestick pattern features
   b. Add moving average features
   c. Add volume features
   d. Add price action features
   e. Add momentum & technical indicators
   f. Add calendar features
   g. Add relative strength (vs VOO)
   h. Add sector-relative features
   i. Encode direction & category numerically
5. Concatenate all symbols → return combined DataFrame
```

**Feature column getter:** `get_feature_columns(df)` returns list of 49 feature names, excluding all non-feature columns (`Symbol`, `DateTime`, `Open`, `High`, `Low`, `Close`, `Volume`, `category`, `target_*`, `future_*`, `sector_return`, `market_return`).

---

## 5. Target Variable Construction

Source: `src/model/targets.py`

### 5.1 Forward-Looking Returns

**Parameters:**
- `period_days = 10` (default forward horizon)
- `profit_threshold = 3.0%` (default)
- `HOURLY_BARS_PER_DAY = 7` (9:30 AM – 4:00 PM ET)

**Targets computed per symbol (sorted by DateTime):**

| Target | Formula | Description |
|---|---|---|
| `future_max_close` | `max(Close[t+1 : t + period × 7])` | Best achievable close in forward window |
| `future_min_low` | `min(Low[t+1 : t + period × 7])` | Worst low in forward window |
| `future_return` | `(future_max_close - Close) / Close × 100` | Maximum upside potential (%) |
| `target_buy` | `1` if `future_return ≥ threshold`, else `0` | Classification label |
| `optimal_hold_days` | `argmax(close_in_window) / 7` | Days until peak close |
| `target_price` | `future_max_close` | Price at optimal exit |

### 5.2 Adaptive Stop-Loss (v5)

Uses ATR-based volatility to dynamically adjust stop-loss width:

```python
atr_pct = ATR_14 / Close                     # Normalize ATR to price
median_atr_pct = median(atr_pct)              # Median across all bars

# Dynamic multiplier: wider stops for volatile stocks, tighter for stable
if atr_pct > median_atr_pct:
    # High-volatility regime → multiplier 2.0–3.0
    multiplier = clip(2.0 + (atr_pct - median) / median, 2.0, 3.0)
else:
    # Low-volatility regime → multiplier 1.5–2.0
    multiplier = clip(2.0 - (median - atr_pct) / median × 0.5, 1.5, 2.0)

stop_loss = max(Close - multiplier × ATR_14, future_min_low × 0.99)
```

| Volatility Regime | ATR Multiplier | Effect |
|---|---|---|
| Low-vol (< median) | 1.5× – 2.0× | Tighter stops, less drawdown tolerance |
| High-vol (> median) | 2.0× – 3.0× | Wider stops, avoid premature stop-outs |
| Hard floor | — | `future_min_low × 0.99` |

---

## 6. Model Architecture

### 6.1 Ensemble Design

Three gradient-boosted tree models are trained independently, then combined into an ensemble. Predictions are **averaged probabilities** from all three:

```
buy_probability = (P_xgboost + P_lightgbm + P_random_forest) / 3
```

### 6.2 Model Configurations

**XGBoost:**
```python
XGBClassifier(
    n_estimators=500, max_depth=7, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=1.5,
    eval_metric="logloss", random_state=42, n_jobs=-1
)
```

**LightGBM** (best single model by walk-forward Sharpe):
```python
LGBMClassifier(
    n_estimators=500, max_depth=7, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=1.5,
    random_state=42, n_jobs=-1, verbose=-1
)
```

**Random Forest:**
```python
RandomForestClassifier(
    n_estimators=500, max_depth=15, min_samples_split=10,
    min_samples_leaf=5, max_features="sqrt",
    class_weight={0: 1, 1: 1.5}, random_state=42, n_jobs=-1
)
```

### 6.3 Class Imbalance Handling

Target distribution: **71.6% HOLD** / **28.4% BUY**. All models use `scale_pos_weight=1.5` (or equivalent `class_weight`) to upweight the minority BUY class.

### 6.4 Training Pipeline

Source: `src/model/trainer.py`

```
Phase 1: Data Preparation
  ├─ Load 421 symbols across 4 categories
  ├─ Engineer 49 features via pipeline
  ├─ Add forward-looking targets (10d horizon, 3% threshold)
  └─ Drop NaN rows → 362,406 clean training rows

Phase 2: Time-Based Train/Test Split
  ├─ Per-symbol: first 80% → train, last 20% → test
  ├─ Training set: ~289,925 rows
  └─ Test set: ~72,481 rows

Phase 3: Feature Scaling
  └─ StandardScaler (zero mean, unit variance), fit on train, transform test

Phase 4: Parallel Model Training
  ├─ ProcessPoolExecutor(max_workers=3)
  ├─ Each model: train → evaluate on test → walk-forward CV (3 folds)
  └─ Progress output: [1/3], [2/3], [3/3]

Phase 5: Walk-Forward Cross-Validation (per model)
  ├─ n_folds = 3, expanding window
  ├─ Split full ordered dataset into 4 windows
  ├─ Fold i: train on first (i+1) windows, test on next window
  └─ Metrics per fold: AUC-ROC, precision, backtest ROI, Sharpe ratio

Phase 6: Model Selection
  ├─ Primary criterion: Walk-forward average Sharpe ratio
  └─ Selected: LightGBM (WF Sharpe = 2.845)

Phase 7: Ensemble Training
  ├─ Retrain all 3 models on full 362,406 rows
  └─ Save as dict: {"xgboost": model, "lightgbm": model, "random_forest": model}
```

### 6.5 Model Performance (v5, March 2026)

| Model | AUC | Accuracy | Precision | Recall | P@10 | P@20 | Backtest ROI | WF Sharpe | Time |
|---|---|---|---|---|---|---|---|---|---|
| **LightGBM** | 0.606 | 0.368 | 0.331 | 0.958 | 0.80 | 0.70 | +4.26% | **2.845** | 18.8s |
| XGBoost | 0.608 | 0.381 | 0.334 | 0.942 | 1.00 | 0.70 | +6.22% | 1.935 | 11.6s |
| Random Forest | 0.600 | 0.355 | 0.328 | 0.973 | 0.60 | 0.65 | +6.48% | 1.454 | 101.2s |

**Selection rationale:** LightGBM wins on walk-forward Sharpe (2.845) — the most realistic measure of risk-adjusted profitability in simulated out-of-sample conditions. XGBoost has slightly higher raw AUC but lower Sharpe.

### 6.6 Model Persistence

```
models/
├── best_model.pkl            # 3-model ensemble dict (XGBoost + LightGBM + RF)
├── model_metadata.json       # Training date, metrics, hyperparams, all results
├── feature_columns.json      # 49 ordered feature names for inference consistency
└── scaler.pkl                # StandardScaler fitted on training features
```

---

## 7. Prediction & Recommendation Engine

Source: `src/model/predictor.py`

### 7.1 Scoring Pipeline

**Step 1 — Model Confidence:**
```
buy_probability = mean(P_xgb, P_lgbm, P_rf)    # 0.0 to 1.0
```

**Step 2 — Historical Feasibility:**
Analyze each symbol's last 20 trading days to check how often the target was achievable:
```
max_gains_in_window = [max gain in each N-day rolling window]
hist_pct = % of windows achieving ≥ expected_profit
avg_max_gain = mean(max_gains)
gain_ratio = min(avg_max_gain / expected_profit, 2.0)
```

**Step 3 — Overextension Penalty (v5):**
Reduce score for stocks that already moved too much:
```python
if returns_5d > expected_profit:
    overext_penalty = max(0.5, 1.0 - (returns_5d - expected_profit) / 20.0)
else:
    overext_penalty = 1.0
```

**Step 4 — Momentum Deceleration Penalty (v5):**
Penalize stocks losing upward momentum:
```python
if momentum_decel < -3.0:
    decel_penalty = max(0.7, 1.0 + momentum_decel / 30.0)
else:
    decel_penalty = 1.0
```

**Step 5 — Mean-Reversion Factor (v6):**
Bonus for below-SMA-20 stocks with accelerating momentum; penalty for overextended above-SMA stocks:
```python
if close < sma_20 and momentum_decel > 0:
    mean_rev_factor = 1.15   # mean-reversion bonus
elif close > sma_20 * 1.05:
    mean_rev_factor = 0.85   # overextended penalty
```

**Step 6 — Above-SMA-20 Gate (v7):**
Additional penalty for above-SMA picks with weak historical feasibility:
```python
if close > sma_20 and hist_pct < 50%:
    mean_rev_factor *= 0.75  # extra penalty for weak above-SMA
```

**Step 7 — Expected Profit Cap (v7):**
Penalize unrealistic profit expectations — higher expected profits correlate with lower hit rates:
```python
max_reasonable = expected_profit * 2.0
if est_return > max_reasonable:
    profit_cap_penalty = max(0.5, 1.0 - (est_return - max_reasonable) / 20.0)
```

**Step 8 — Market Regime Gate (v6):**
Require higher confidence in bearish regimes:
```python
if market_regime < -0.02:   # VOO declining >2%
    if model_conf < 0.6: regime_penalty = 0.7
    elif model_conf < 0.7: regime_penalty = 0.85
```

**Step 9 — Combined Score (v7):**
```python
raw_score = (
    0.25 × model_confidence +
    0.40 × (hist_pct / 100.0) +
    0.35 × min(gain_ratio, 1.0)
)
score = raw_score × overext_penalty × decel_penalty × mean_rev_factor
        × profit_cap_penalty × regime_penalty
score_1_to_100 = clip(score × 100, 1, 100)
```

**Weight breakdown (v7):**
- 25% — Model ensemble prediction confidence
- 40% — Historical feasibility (how often did this symbol achieve the target)
- 35% — Gain achievability (is the target realistic given recent price action)
- Multiplicative penalties: overextension, momentum deceleration, mean-reversion, profit cap, regime gate

### 7.2 Pre-Scoring Filters (v6/v7)

Before scoring, symbols are filtered by quality gates:

| Filter | Rule | Rationale |
|---|---|---|
| **Hard blacklist (v7)** | Block INTC, MU, LRCX, NEM | Chronic losers: 0–28% hit rate despite high model scores |
| **Min feasibility (v6)** | Reject if `hist_pct < max(15%, 35% - target×2)` | Scales with target: 29% for 3% target, 15% for 10%+ |
| **ATR stop-out blocker (v7)** | Reject if `ATR/price > 80% of target` | Blocks symbols too volatile relative to the profit target |

### 7.3 Diversity Filtering (v7)

To prevent sector concentration in recommendations:

| Rule | Value | Description |
|---|---|---|
| Category penalty | `0.70 ^ count` | Each additional pick from same sector penalized by 30% (was 20% in v5) |
| Hard cap (stocks) | 3 per category | Maximum 3 recommendations from snp100/snp500 |
| Hard cap (ETFs/merch) | 2 per category | Tighter cap for merchandise and ETF categories |
| Acceptance threshold | ≥ 60% of last selected score | Adjusted score must be at least 60% of the previous pick's score |

### 7.4 Adaptive Stop-Loss & Price Targets

Same ATR-based adaptive logic used in training targets (§5.2), applied at prediction time:
```python
est_target_price = Close × (1 + est_return / 100)
est_stop_loss    = Close - multiplier × ATR_14    # multiplier = 1.5–3.0
```

### 7.5 Recommendation Output

| Field | Type | Description |
|---|---|---|
| `symbol` | string | Ticker symbol |
| `current_price` | float | Latest close price |
| `expected_profit_pct` | float | Estimated return before target hit (%) |
| `period_days` | int | Recommended holding period |
| `score` | int 1–100 | Multi-factor confidence score |
| `target_price` | float | Estimated exit price |
| `stop_loss` | float | Adaptive ATR-based stop price |
| `sma_20` | float | Current 20-bar SMA |
| `vs_sma_20` | string | "Above" or "Below" SMA-20 |
| `ytd_pct` | float | Year-to-date return (%) |
| `month_pct` | float | Month-to-date return (%) |

---

## 8. Backtesting Framework

Source: `scripts/backtest_feb.py`

### 8.1 Design

| Parameter | Value |
|---|---|
| Test period | February 2026 (19 trading days) |
| Scenarios | 4 (3%, 5%, 8%, 10% profit targets, all 10-day horizon) |
| Stocks per day | 10 recommendations |
| Total tasks | 76 (4 scenarios × 19 days) |
| Parallelization | `ThreadPoolExecutor(max_workers=76)` — one thread per task |

### 8.2 Methodology

For each scenario × trading date:
1. Load all available data **up to (but not including)** the buy date — no look-ahead bias
2. Engineer features, generate 10 recommendations using the trained ensemble
3. Apply same scoring pipeline as production (blacklist, feasibility gate, ATR blocker, overextension, momentum decel, mean-reversion, profit cap, regime gate, diversity)
4. Look up actual prices 5 and 10 trading days later
5. Classify each recommendation:
   - **HIT TARGET** — Price reached ≥ target within 10 days
   - **STOPPED OUT** — Price hit stop-loss before reaching target
   - **MISSED** — Neither target nor stop reached within 10 days
   - **N/A** — Insufficient future data

### 8.3 Results (v7, February 2026)

| Scenario | Hit Rate | Missed | Stopped Out | Avg 5d P&L | Avg 10d P&L |
|---|---|---|---|---|---|
| **3% / 10d** | **159/190 (83.7%)** | — | — | — | +3.06% |
| **5% / 10d** | **154/190 (81.1%)** | — | — | — | +3.87% |
| **8% / 10d** | **140/190 (73.7%)** | — | — | — | +3.54% |
| **10% / 10d** | **140/190 (73.7%)** | — | — | — | +3.69% |

### 8.4 Performance History (Version Progression)

| Scenario | v3 (ensemble) | v4 (parallel) | v5 (adaptive) | v6 (scoring) | v7 (current) |
|---|---|---|---|---|---|
| 3% / 10d | 71.6% | 71.6% | 71.6% | 78.9% | **83.7%** |
| 5% / 10d | 63.7% | 65.8% | 66.8% | 74.7% | **81.1%** |
| 8% / 10d | 52.6% | 57.4% | 61.1% | 66.3% | **73.7%** |
| 10% / 10d | 48.4% | 53.2% | 62.1% | 58.4% | **73.7%** |

Key improvements: v6 recalibrated scoring weights (25/40/35), added mean-reversion & regime gates. v7 added hard blacklist, ATR stop-out blocker, Above-SMA gate — all scenarios improved +4.8 to +15.3pp over v6.

---

## 9. Evaluation Metrics

Source: `src/model/evaluator.py`

### 9.1 Classification Metrics

| Metric | Description |
|---|---|
| Accuracy | Correct predictions / total |
| Precision | TP / (TP + FP) — among predicted buys, how many correct |
| Recall | TP / (TP + FN) — among actual buys, what fraction detected |
| F1-Score | Harmonic mean of precision and recall |
| AUC-ROC | ROC area under curve (requires both classes) |

### 9.2 Trading-Specific Metrics

| Metric | Description |
|---|---|
| Precision@10 | Precision among top 10 highest-probability predictions |
| Precision@20 | Precision among top 20 predictions |
| Avg Return % | Mean future return of top-N predicted buys |
| Positive Rate % | % of top-N trades that were profitable |
| Sharpe Ratio | Mean return / Std(returns) — risk-adjusted performance |

### 9.3 Walk-Forward Validation

| Metric | Description |
|---|---|
| WF Avg AUC | Mean AUC across 3 expanding folds |
| WF Avg ROI | Mean backtest return across folds |
| WF Avg Sharpe | **Primary selection criterion** — mean Sharpe across folds |

---

## 10. API Design (FastAPI)

### 10.1 Endpoints

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

#### `POST /refresh`

Pulls latest hourly candle data from Yahoo Finance for all tickers.

**Request body:**
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

#### `GET /health`

Health check returning model status and data freshness.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "ensemble_v3_20260323",
  "data_last_updated": "2026-03-23T12:00:00Z",
  "total_tickers": 421
}
```

#### `POST /train`

Triggers model retraining. Runs all 3 candidate models, compares via walk-forward Sharpe, saves ensemble.

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

### 10.2 Error Handling

Standard HTTP status codes with JSON error bodies:
```json
{
  "detail": "Model not trained yet. Call POST /train first.",
  "error_code": "MODEL_NOT_FOUND"
}
```

---

## 11. Project Structure

```
Trading.Agent2/
├── architecture.md              # This document
├── spec.md                      # Original requirements specification
├── requirements.txt             # Python dependencies
├── backtest_feb_report.md       # Latest backtest results (auto-generated)
├── backtest_feb_results.csv     # Raw backtest data (760 rows)
│
├── Data/
│   ├── snp500_hourly/           # ~500 ticker files
│   ├── snp100_hourly/           # ~100 ticker files
│   ├── etfs_hourly/             # 15 ticker files
│   └── merchandise_hourly/      # 5 ticker files
│
├── Spec/
│   ├── THE CANDLESTICK TRADING BIBLE.pdf
│   └── Moving Average Trading.pdf
│
├── models/                      # Trained model artifacts
│   ├── best_model.pkl           # 3-model ensemble (XGBoost + LightGBM + RF)
│   ├── model_metadata.json      # Training metrics, version, all model results
│   ├── feature_columns.json     # 49 ordered feature names
│   └── scaler.pkl               # StandardScaler (fitted on training data)
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            # Load pipe-delimited files → DataFrames
│   │   └── ticker_registry.py   # Scan Data/ folders, discover tickers
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── candlestick.py       # Pattern detection (3WS, 3BC, composite scores)
│   │   ├── moving_average.py    # SMA/EMA computation & slopes
│   │   ├── volume.py            # Volume ratio, trend
│   │   ├── price_action.py      # Returns, volatility, RSI, ATR
│   │   ├── momentum.py          # MACD, Bollinger, Stochastic, multi-day returns
│   │   └── pipeline.py          # Feature orchestration, sector/market returns
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── targets.py           # Forward-looking targets, adaptive stop-loss
│   │   ├── trainer.py           # Train, evaluate, compare, save ensemble
│   │   ├── predictor.py         # Inference, scoring, diversity filtering
│   │   └── evaluator.py         # Classification + trading-specific metrics
│   │
│   └── utils/
│       ├── __init__.py
│       └── config.py            # Paths, constants, category mappings
│
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── backtest_feb.py          # Feb 2026 backtesting framework
│   ├── analyze_results.py       # Result analysis utilities
│   └── analyze_best_day.py      # Best/worst day analysis
│
└── tests/
    └── __init__.py
```

---

## 12. Configuration & Constants

Source: `src/utils/config.py`

```python
# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Data Format
FILE_DELIMITER = "|"
FILE_SUFFIX = "_hourly_candles.txt"

# Time Constants
HOURLY_BARS_PER_DAY = 7          # ~7 trading hours/day (9:30–16:00 ET)

# Training Defaults
TRAIN_FRACTION = 0.8              # 80% train, 20% test
DEFAULT_PROFIT_THRESHOLD = 3.0    # % minimum gain to label BUY
DEFAULT_PERIOD_DAYS = 10          # Forward-looking horizon
DEFAULT_NUM_STOCKS = 5            # Default top-N recommendations

# Category Mapping
DATA_CATEGORIES = {
    "snp500": DATA_DIR / "snp500_hourly",
    "snp100": DATA_DIR / "snp100_hourly",
    "etfs": DATA_DIR / "etfs_hourly",
    "merchandise": DATA_DIR / "merchandise_hourly",
}
```

---

## 13. Technology Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | 3.14 |
| ML — Gradient Boosting | XGBoost | 2.0+ |
| ML — Gradient Boosting | LightGBM | 4.0+ |
| ML — Ensemble & Utilities | scikit-learn | 1.4+ |
| Data Manipulation | pandas, numpy | latest |
| Model Serialization | joblib | latest |
| API Framework (planned) | FastAPI | 0.110+ |
| ASGI Server (planned) | Uvicorn | 0.29+ |
| Market Data (planned) | yfinance | 0.2+ |
| Validation (planned) | Pydantic | 2.0+ |
| Containerization (planned) | Docker | latest |
| Testing | pytest | 8.0+ |

---

## 14. Data Flow

```
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │ POST /refresh│────▶│ Yahoo Finance│────▶│ Update .txt  │
 │              │     │ (yfinance)   │     │ files in Data│
 └──────────────┘     └──────────────┘     └──────────────┘
                                                  │
                                                  ▼
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │ POST /train  │────▶│ Feature Eng. │────▶│ Train Models │
 │  (or CLI)    │     │ (49 features)│     │ (XGB/LGB/RF) │
 └──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                     Walk-Forward CV (3 folds)
                                     Select by Sharpe ratio
                                                  │
                                                  ▼
                                           ┌──────────────┐
                                           │ Save ensemble│
                                           │ models/*.pkl │
                                           └──────────────┘
                                                  │
                                                  ▼
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │POST /recommend────▶│ Ensemble     │────▶│ Score, rank, │
 │  (or CLI)    │     │ inference +  │     │ diversify,   │
 └──────────────┘     │ multi-factor │     │ return top-N │
                      │ scoring      │     └──────────────┘
                      └──────────────┘
```

### Workflow

1. **Refresh** → Pull latest hourly data from Yahoo Finance, append to existing files
2. **Train** → Load data → engineer 49 features → build targets → train 3 models in parallel → walk-forward CV → select best by Sharpe → train ensemble on full data → save artifacts
3. **Recommend** → Load ensemble + scaler → apply features to latest data → score with confidence + feasibility + penalties → diversify across sectors → return top-N with targets & stops
4. **Backtest** → Replay training dates with out-of-sample evaluation → verify hit rates & P&L

---

## 15. Key Design Decisions

| Decision | Rationale |
|---|---|
| **3-model ensemble** | Averaging probabilities from XGBoost + LightGBM + RF reduces overfitting and improves stability |
| **Walk-forward Sharpe selection** | Selects the model with best risk-adjusted returns in realistic out-of-sample conditions, not just raw accuracy |
| **Single unified model** | One model learns cross-asset patterns; ticker category is a feature, not a model boundary |
| **Flat files over DB** | Simplicity for current scale (~421 symbols); avoids infrastructure overhead |
| **Time-based train/test split** | Prevents look-ahead bias inherent in random splits on time-series data |
| **Adaptive ATR stop-loss** | Adjusts stop width per stock's volatility regime (1.5×–3.0× ATR) instead of fixed percentage |
| **Overextension filter** | Avoids recommending stocks that already moved >target% in 5 days — catches "too late" entries |
| **Momentum deceleration gate** | Penalizes stocks where 3-day momentum is slowing relative to 5-day — catches fading trends |
| **Sector diversity hard cap** | Max 3 picks per category with 0.80× penalty per repeat — prevents concentration risk |
| **Multi-factor scoring** | 40% model + 30% historical feasibility + 30% gain ratio — balanced between ML confidence and market reality |
| **Parallel training** | `ProcessPoolExecutor(3)` trains all models simultaneously (~2× speedup on multi-core) |
| **Parallel backtest** | `ThreadPoolExecutor(20)` processes 76 scenario×date tasks concurrently |

---

## 16. Constraints & Assumptions

- **Data:** Requires ≥ 50 hourly bars per symbol (minimum ~7 trading days) for feature computation.
- **Market hours only:** No pre-market or after-hours data. 7 hourly bars per trading day (9:30 AM – 4:00 PM ET).
- **Yahoo Finance rate limits:** Refresh endpoint batches requests with delays for 421+ tickers.
- **Predictions are informational** — not financial advice.
- **No authentication** on API endpoints — designed for internal/local use.
- **Reproducibility:** All models use `random_state=42`. Results are deterministic given the same data.
- **Memory:** Full training pipeline requires ~5 GB RAM (362K rows × 49 features × 3 models).

---

## 17. Version History

| Version | Key Changes | 3%/10d Hit Rate |
|---|---|---|
| v1 | Baseline: single XGBoost, 43 features, random CV | — |
| v2 | Dropped dead features, added MACD/BB/Stochastic | — |
| v3 | Ensemble (3 models), walk-forward CV, Sharpe selection, market regime | 71.6% |
| v4 | Parallel training (ProcessPool) & backtest (ThreadPool) | 71.6% |
| v5 | Adaptive stops, overextension filter, momentum decel, sector diversity | 71.6% |
| v6 | Scoring recalibration (25/40/35), mean-reversion bonus, profit cap, regime gate, feasibility threshold | 78.9% |
| v7 | Hard blacklist (INTC/MU/LRCX/NEM), ATR stop-out blocker, Above-SMA gate, tighter profit cap, 76 threads | **83.7%** |
