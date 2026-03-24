# Trading Recommendation Portal

A web-based dashboard for generating ML-driven stock trading recommendations. Uses an ensemble model (XGBoost + LightGBM + RandomForest) trained on 421 symbols of hourly candle data to score and rank trading opportunities with configurable filters.

## Quick Start

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install dependencies (if not already installed)
pip install -r requirements.txt

# 3. Start the server
python -m uvicorn src.api.main:app --port 8000

# 4. Open in browser
open http://localhost:8000
```

## Features

- **Configurable filters** — Number of symbols (1–10), hold period (1–30 days), profit target (1–30%), SMA filter (None/20/50/100), volatility filter (None/Low/Medium/High)
- **Date cutoff** — Calendar picker to generate recommendations using historical data as of any past date
- **Results table** — Ranked recommendations showing: Symbol, Current Price, Score, Target Price, Stop-Loss, Expected Profit, YTD %, Last Month %, Volatility
- **Interactive charts** — Chart.js price charts with Close/High/Low lines, switchable between 1W, 1M, and YTD timeframes
- **Save & history** — Save recommendation sets as JSON files, browse and reload previous recommendations

## Default Settings

| Parameter | Default |
|-----------|---------|
| Symbols | 10 |
| Hold period | 14 days |
| Profit target | 5% |
| Moving average | SMA-100 |
| Volatility | Low |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the portal dashboard |
| `GET` | `/api/recommend` | Generate recommendations (params: `num`, `profit`, `hold`, `sma`, `vol`, `cutoff`) |
| `GET` | `/api/chart/{symbol}` | Get OHLCV candle data (params: `period` = 1w/1m/ytd, `cutoff`) |
| `GET` | `/api/dates` | List available trading dates in the data |
| `POST` | `/api/save` | Save current recommendations to JSON |
| `GET` | `/api/saved` | List all saved recommendation files |
| `GET` | `/api/saved/{filename}` | Read a specific saved recommendation |

## Example API Usage

```bash
# Generate 5 recommendations with 8% profit target, SMA-50, Medium volatility
curl "http://localhost:8000/api/recommend?num=5&profit=8&hold=14&sma=50&vol=Medium"

# Get 1-month chart data for AAPL
curl "http://localhost:8000/api/chart/AAPL?period=1m"

# Generate using historical data as of March 20
curl "http://localhost:8000/api/recommend?num=10&profit=5&hold=14&sma=100&vol=Low&cutoff=2026-03-20%2019:30:00"
```

## Project Structure

```
src/
├── api/
│   ├── main.py              # FastAPI app and route definitions
│   └── recommend_engine.py  # ML recommendation engine (scoring, filtering)
├── portal/
│   ├── index.html           # Dashboard UI
│   ├── style.css            # Dark theme, responsive layout
│   └── app.js               # Client logic, Chart.js integration
├── data/                    # Data loading utilities
├── features/                # Feature engineering pipeline (49 features)
├── model/                   # Predictor class
└── utils/                   # Configuration and constants
```

## Technology Stack

- **Backend:** Python 3.14, FastAPI, Uvicorn
- **ML:** XGBoost, LightGBM, RandomForest ensemble with v7 composite scoring
- **Frontend:** HTML, vanilla JS, CSS (no build step)
- **Charts:** Chart.js 4.x with time-scale axis
- **Data:** 421 symbols across S&P 100, S&P 500, ETFs, and commodities — hourly candles (pipe-delimited .txt)

## Saved Recommendations

Recommendations are saved as JSON files in `models/Recumendations/` with the format:

```
rec_YYYY-MM-DD_<symbol_count>_<sequence>.json
```

Each file contains the filters used, timestamp, and the full recommendation list with scores and price targets.
