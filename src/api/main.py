"""FastAPI backend for the Trading Recommendation Portal."""

import json
import os
import glob
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import requests as http_requests

from src.api.recommend_engine import RecommendEngine
from src.utils.config import PROJECT_ROOT

app = FastAPI(title="Trading Recommendation Portal")

# Singleton engine — loaded once on first request
engine = RecommendEngine()

RECOMMENDATIONS_DIR = PROJECT_ROOT / "models" / "Recumendations"
RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)

PORTAL_DIR = Path(__file__).parent.parent.parent / "src" / "portal"


# --- API Endpoints ---


@app.get("/api/recommend")
def recommend(
    num: int = Query(10, ge=1, le=10),
    profit: float = Query(5.0, ge=1, le=30),
    hold: int = Query(14, ge=1, le=30),
    sma: int = Query(100, ge=0),
    vol: str = Query("Low"),
    cutoff: str = Query(None),
):
    """Generate recommendations with filters."""
    sma_filter = sma if sma > 0 else None
    vol_filter = vol if vol and vol != "None" else None

    recs = engine.recommend(
        num_stocks=num,
        expected_profit_pct=profit,
        period_days=hold,
        sma_filter=sma_filter,
        vol_filter=vol_filter,
        cutoff=cutoff,
    )
    return {"recommendations": recs, "count": len(recs)}


@app.get("/api/chart/{symbol}")
def chart_data(
    symbol: str,
    period: str = Query("1m"),
    cutoff: str = Query(None),
    sma: int = Query(0, ge=0),
):
    """Return OHLCV candle data for charting."""
    if period not in ("ytd", "1m", "1w"):
        raise HTTPException(400, "period must be ytd, 1m, or 1w")
    sma_val = sma if sma > 0 else None
    result = engine.get_chart_data(symbol, period, cutoff, sma=sma_val)
    if not result["candles"]:
        raise HTTPException(404, f"No data for {symbol}")
    return {"symbol": symbol, "name": result["name"], "period": period, "candles": result["candles"]}


@app.get("/api/dates")
def available_dates():
    """Return list of available dates in data."""
    if not engine.is_loaded:
        engine.load()
    dates = engine.get_available_dates()
    return {"dates": dates}


@app.get("/api/quote/{symbol}")
def live_quote(symbol: str):
    """Fetch latest price and daily change from Yahoo Finance."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=2d&interval=1d"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = http_requests.get(url, headers=headers, timeout=5)
        r.raise_for_status()
        meta = r.json()["chart"]["result"][0]["meta"]
        last_price = meta["regularMarketPrice"]
        prev_close = meta["chartPreviousClose"]
        change = last_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close else 0
        return {
            "symbol": symbol,
            "price": round(last_price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "prev_close": round(prev_close, 2),
        }
    except Exception:
        raise HTTPException(404, f"Quote not available for {symbol}")


@app.get("/api/info/{symbol}")
def symbol_info(symbol: str):
    """Fetch comprehensive symbol data from Yahoo Finance using crumb auth."""
    import re
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if not re.match(r'^[A-Za-z0-9.\-]{1,10}$', symbol):
        raise HTTPException(400, "Invalid symbol")
    try:
        # Create session with crumb authentication
        session = http_requests.Session()
        session.headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        session.verify = False

        # Get cookies from Yahoo
        session.get('https://fc.yahoo.com', timeout=8, allow_redirects=True)

        # Get crumb
        crumb_resp = session.get('https://query2.finance.yahoo.com/v1/test/getcrumb', timeout=8)
        crumb = crumb_resp.text.strip()

        # Fetch quoteSummary with all modules
        url = (
            f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
            f"?modules=assetProfile,defaultKeyStatistics,financialData,"
            f"summaryDetail,price,calendarEvents"
            f"&crumb={crumb}"
        )
        r = session.get(url, timeout=10)
        r.raise_for_status()
        result_json = r.json()
        qr = result_json.get("quoteSummary", {}).get("result")
        if not qr:
            raise HTTPException(404, f"No data for {symbol}")
        modules = {}
        for mod in qr:
            modules.update(mod)

        def val(d, key, fallback=None):
            """Extract value from Yahoo's nested {raw, fmt} format."""
            if d is None:
                return fallback
            v = d.get(key)
            if v is None:
                return fallback
            if isinstance(v, dict):
                return v.get("raw", v.get("fmt", fallback))
            return v

        profile = modules.get("assetProfile", {})
        stats = modules.get("defaultKeyStatistics", {})
        fin = modules.get("financialData", {})
        summary = modules.get("summaryDetail", {})
        price_mod = modules.get("price", {})

        result = {
            "symbol": symbol,
            "profile": {
                "name": val(price_mod, "longName") or val(price_mod, "shortName", symbol),
                "sector": profile.get("sector"),
                "industry": profile.get("industry"),
                "description": profile.get("longBusinessSummary"),
                "website": profile.get("website"),
                "employees": profile.get("fullTimeEmployees"),
                "country": profile.get("country"),
                "city": profile.get("city"),
            },
            "price": {
                "current": val(price_mod, "regularMarketPrice"),
                "previous_close": val(price_mod, "regularMarketPreviousClose") or val(summary, "previousClose"),
                "open": val(price_mod, "regularMarketOpen") or val(summary, "open"),
                "day_low": val(price_mod, "regularMarketDayLow") or val(summary, "dayLow"),
                "day_high": val(price_mod, "regularMarketDayHigh") or val(summary, "dayHigh"),
                "52w_low": val(summary, "fiftyTwoWeekLow"),
                "52w_high": val(summary, "fiftyTwoWeekHigh"),
                "50d_avg": val(summary, "fiftyDayAverage"),
                "200d_avg": val(summary, "twoHundredDayAverage"),
            },
            "market": {
                "market_cap": val(price_mod, "marketCap") or val(summary, "marketCap"),
                "enterprise_value": val(stats, "enterpriseValue"),
                "volume": val(price_mod, "regularMarketVolume") or val(summary, "volume"),
                "avg_volume": val(summary, "averageVolume"),
                "avg_volume_10d": val(summary, "averageDailyVolume10Day"),
                "shares_outstanding": val(stats, "sharesOutstanding"),
                "float_shares": val(stats, "floatShares"),
                "beta": val(stats, "beta") or val(summary, "beta"),
            },
            "valuation": {
                "pe_trailing": val(summary, "trailingPE"),
                "pe_forward": val(summary, "forwardPE") or val(stats, "forwardPE"),
                "peg_ratio": val(stats, "pegRatio"),
                "price_to_book": val(stats, "priceToBook"),
                "price_to_sales": val(stats, "priceToSalesTrailing12Months"),
                "ev_to_ebitda": val(stats, "enterpriseToEbitda"),
                "ev_to_revenue": val(stats, "enterpriseToRevenue"),
            },
            "financials": {
                "revenue": val(fin, "totalRevenue"),
                "revenue_per_share": val(fin, "revenuePerShare"),
                "gross_profit": val(fin, "grossProfits"),
                "ebitda": val(fin, "ebitda"),
                "eps_trailing": val(stats, "trailingEps"),
                "eps_forward": val(stats, "forwardEps") or val(fin, "forwardEps"),
                "profit_margin": val(fin, "profitMargins"),
                "operating_margin": val(fin, "operatingMargins"),
                "gross_margin": val(fin, "grossMargins"),
                "return_on_equity": val(fin, "returnOnEquity"),
                "return_on_assets": val(fin, "returnOnAssets"),
                "debt_to_equity": val(fin, "debtToEquity"),
                "current_ratio": val(fin, "currentRatio"),
                "free_cash_flow": val(fin, "freeCashflow"),
                "operating_cash_flow": val(fin, "operatingCashflow"),
            },
            "dividends": {
                "dividend_rate": val(summary, "dividendRate"),
                "dividend_yield": val(summary, "dividendYield"),
                "payout_ratio": val(summary, "payoutRatio"),
                "ex_dividend_date": val(summary, "exDividendDate"),
            },
            "analyst": {
                "target_mean": val(fin, "targetMeanPrice"),
                "target_low": val(fin, "targetLowPrice"),
                "target_high": val(fin, "targetHighPrice"),
                "target_median": val(fin, "targetMedianPrice"),
                "recommendation": val(fin, "recommendationKey"),
                "num_analysts": val(fin, "numberOfAnalystOpinions"),
            },
        }
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(404, f"Info not available for {symbol}: {str(e)}")


class SaveRequest(BaseModel):
    recommendations: list[dict]
    filters: dict


@app.post("/api/save")
def save_recommendation(req: SaveRequest):
    """Save current recommendations to JSON file."""
    today = datetime.now().strftime("%Y-%m-%d")
    count = len(req.recommendations)

    # Find next sequence number
    pattern = str(RECOMMENDATIONS_DIR / f"rec_{today}_{count}_*.json")
    existing = glob.glob(pattern)
    seq = len(existing) + 1

    filename = f"rec_{today}_{count}_{seq:03d}.json"
    filepath = RECOMMENDATIONS_DIR / filename

    payload = {
        "created": datetime.now().isoformat(),
        "filters": req.filters,
        "recommendations": req.recommendations,
    }
    with open(filepath, "w") as f:
        json.dump(payload, f, indent=2)

    return {"filename": filename, "path": str(filepath)}


@app.get("/api/saved")
def list_saved():
    """List all saved recommendation files."""
    files = sorted(RECOMMENDATIONS_DIR.glob("rec_*.json"), reverse=True)
    result = []
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            result.append(
                {
                    "filename": f.name,
                    "created": data.get("created", ""),
                    "filters": data.get("filters", {}),
                    "count": len(data.get("recommendations", [])),
                }
            )
        except (json.JSONDecodeError, KeyError):
            result.append({"filename": f.name, "created": "", "filters": {}, "count": 0})
    return {"saved": result}


@app.get("/api/saved/{filename}")
def get_saved(filename: str):
    """View a specific saved recommendation file."""
    # Prevent path traversal
    safe_name = Path(filename).name
    filepath = RECOMMENDATIONS_DIR / safe_name
    if not filepath.exists() or not filepath.suffix == ".json":
        raise HTTPException(404, "File not found")
    with open(filepath) as f:
        return json.load(f)


# --- Static files & SPA ---

@app.get("/")
def serve_index():
    return FileResponse(PORTAL_DIR / "index.html")


# Mount static files after specific routes
app.mount("/static", StaticFiles(directory=str(PORTAL_DIR)), name="static")
