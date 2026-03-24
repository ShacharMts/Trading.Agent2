# Trading Recommendation Portal — Architecture

## System Overview

The portal is a single-page web application that generates ML-driven stock trading recommendations with configurable filters, interactive charts, and recommendation history management. It runs locally and is served entirely by a FastAPI backend.

```
┌──────────────────────────────────────────────────────┐
│                     Browser                          │
│  ┌────────────────────────────────────────────────┐  │
│  │  index.html + style.css + app.js + Chart.js    │  │
│  └──────────────────┬─────────────────────────────┘  │
│                     │  fetch() calls                 │
└─────────────────────┼────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI Server  (src/api/main.py)            port 8000     │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  GET  /                  → serves index.html          │  │
│  │  GET  /static/*          → serves CSS, JS             │  │
│  │  GET  /api/recommend     → generate recommendations   │  │
│  │  GET  /api/chart/{sym}   → OHLCV candle data          │  │
│  │  GET  /api/dates         → available trading dates     │  │
│  │  POST /api/save          → persist recs to JSON        │  │
│  │  GET  /api/saved         → list saved rec files        │  │
│  │  GET  /api/saved/{file}  → read a saved rec file       │  │
│  └───────────────────┬───────────────────────────────────┘  │
│                      │                                      │
│  ┌───────────────────▼───────────────────────────────────┐  │
│  │  RecommendEngine  (src/api/recommend_engine.py)       │  │
│  │  • Loads ensemble model (XGBoost+LightGBM+RF)         │  │
│  │  • Loads scaler + feature columns                     │  │
│  │  • Caches raw market data on first request             │  │
│  │  • Applies v7 scoring, SMA/volatility filters          │  │
│  └───┬───────────────┬───────────────┬───────────────────┘  │
│      │               │               │                      │
│      ▼               ▼               ▼                      │
│  ┌────────┐   ┌───────────┐   ┌────────────┐               │
│  │ loader │   │ pipeline  │   │  config    │               │
│  │.py     │   │.py        │   │.py         │               │
│  └────┬───┘   └───────────┘   └────────────┘               │
│       │                                                     │
└───────┼─────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────┐    ┌──────────────────────────┐
│  Data/                     │    │  models/                 │
│  ├── snp100_hourly/        │    │  ├── best_model.pkl      │
│  ├── snp500_hourly/        │    │  ├── scaler.pkl          │
│  ├── etfs_hourly/          │    │  ├── feature_columns.json│
│  └── merchandise_hourly/   │    │  ├── model_metadata.json │
│      (421 symbols, .txt)   │    │  └── Recumendations/     │
└────────────────────────────┘    │      └── rec_*.json      │
                                  └──────────────────────────┘
```

## Component Details

### 1. Frontend (`src/portal/`)

| File | Purpose |
|------|---------|
| `index.html` | Single-page layout: header, filter bar, results table, chart canvas, history section |
| `style.css` | Dark theme, responsive design (mobile: stacks filters, horizontal-scroll table) |
| `app.js` | All client logic: form handling, API calls via `fetch()`, table rendering, Chart.js integration, save/load history |

**External dependencies** (loaded via CDN):
- Chart.js 4.4.7 — interactive line charts
- chartjs-adapter-date-fns 3.0.0 — time-scale axis

**No build step.** Files are served directly by FastAPI's `StaticFiles` mount.

### 2. Backend (`src/api/`)

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app — defines all API routes, serves static frontend, handles JSON save/load |
| `recommend_engine.py` | Core ML recommendation logic — loads model once, caches data, scores symbols |

**Singleton pattern:** `RecommendEngine` is instantiated once at module level. Model artifacts and raw market data are lazy-loaded on the first `/api/recommend` call and cached for subsequent requests.

### 3. ML Pipeline (existing, reused)

| Module | Role |
|--------|------|
| `src/data/loader.py` | Loads all 421 symbols from pipe-delimited hourly candle files, deduplicates S&P 100/500 overlap |
| `src/features/pipeline.py` | Engineers 49 features (SMAs, ATR, momentum, volume, market regime via VOO) |
| `src/utils/config.py` | Central config: paths, constants (`HOURLY_BARS_PER_DAY=7`, delimiters, model paths) |
| `src/model/predictor.py` | Original Predictor class (used for CLI scripts, not directly by portal) |

### 4. Data Flow

```
User clicks [Generate]
       │
       ▼
   app.js builds query params
   (num, profit, hold, sma, vol, cutoff)
       │
       ▼
   GET /api/recommend?num=10&profit=5&hold=14&sma=100&vol=Low
       │
       ▼
   RecommendEngine.recommend()
       │
       ├── 1. Load raw data → apply cutoff filter
       ├── 2. engineer_features() → 49 features per bar
       ├── 3. Get latest bar per symbol
       ├── 4. Ensemble predict_proba → buy probability
       ├── 5. v7 composite scoring with penalties
       │      (overextension, deceleration, mean-rev, regime, ATR blocker)
       ├── 6. Apply SMA filter (close > SMA-N)
       ├── 7. Apply volatility filter (composite = ATR%×0.5 + DailyVol%×0.5)
       ├── 8. Diverse category selection (max 3 per sector, 2 per ETF/merch)
       └── 9. Return top-N ranked results
       │
       ▼
   app.js renders table → user clicks [View] on a symbol
       │
       ▼
   GET /api/chart/AAPL?period=1m
       │
       ▼
   RecommendEngine.get_chart_data() → OHLCV array
       │
       ▼
   Chart.js renders interactive price chart
```

### 5. Scoring Algorithm (v7)

```
score = (0.25 × model_confidence
       + 0.40 × historical_pct / 100
       + 0.35 × min(gain_ratio, 1.0))
       × overextension_penalty
       × deceleration_penalty
       × mean_reversion_factor
       × profit_cap_penalty
       × regime_penalty
```

**Filters applied in order:**
1. Blacklist exclusion: {INTC, MU, LRCX, NEM}
2. Minimum feasibility threshold (historical achievement rate)
3. ATR blocker (reject if ATR% > 80% of profit target)
4. SMA gate (close must be above selected SMA)
5. Volatility classification (composite < 1.8 = Low, < 3.5 = Medium, ≥ 3.5 = High)
6. Category diversity caps

### 6. Persistence

Saved recommendations are stored as JSON files in `models/Recumendations/`:

```
Filename format: rec_YYYY-MM-DD_<count>_<seq>.json

{
  "created": "2026-03-24T10:30:00",
  "filters": { "num": "10", "profit": "5", "hold": "14", "sma": "100", "vol": "Low" },
  "recommendations": [
    { "symbol": "AAPL", "current_price": 218.50, "score": 85, ... }
  ]
}
```

### 7. Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.14, FastAPI, Uvicorn |
| ML Models | XGBoost, LightGBM, RandomForest (ensemble) |
| Data Processing | pandas, NumPy, scikit-learn, joblib |
| Frontend | HTML5, vanilla JavaScript (ES6+), CSS3 |
| Charts | Chart.js 4.x + date-fns adapter |
| Deployment | localhost (Uvicorn), planned remote deployment |

### 8. Directory Structure (Portal-related)

```
Trading.Agent2/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              ← FastAPI app + routes
│   │   └── recommend_engine.py  ← ML recommendation engine
│   ├── portal/
│   │   ├── index.html           ← Single-page dashboard
│   │   ├── style.css            ← Dark theme, responsive
│   │   └── app.js               ← Client-side logic + charts
│   ├── data/
│   │   ├── loader.py            ← Data loading utilities
│   │   └── ticker_registry.py   ← Symbol/category lookup
│   ├── features/
│   │   └── pipeline.py          ← Feature engineering (49 features)
│   ├── model/
│   │   └── predictor.py         ← Predictor class (CLI usage)
│   └── utils/
│       └── config.py            ← Paths, constants, configuration
├── Data/                        ← 421 symbols, hourly candles (.txt)
├── models/
│   ├── best_model.pkl           ← Trained ensemble model
│   ├── scaler.pkl               ← Feature scaler
│   ├── feature_columns.json     ← Feature column names
│   └── Recumendations/          ← Saved recommendation JSONs
└── requirements.txt
```
