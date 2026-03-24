"""Generate recommendations as-of 2026-03-23 19:30 for buying 2026-03-23.

Filters:
  - Data cutoff: 2026-03-23 19:30
  - All picks must be above their SMA-100 (100 hourly bars ≈ 14 trading days)
  - 5% and 8% targets: LOW volatility/velocity only
  - 10% target: MEDIUM volatility/velocity only
"""
import sys, os, json, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import joblib

from src.data.loader import load_all_data
from src.features.pipeline import engineer_features
from src.utils.config import (
    BEST_MODEL_PATH, FEATURE_COLUMNS_PATH, SCALER_PATH,
)

BLACKLIST = {"INTC", "MU", "LRCX", "NEM"}
CUTOFF = pd.Timestamp("2026-03-20 19:30:00")


def classify_volatility(atr_pct: float, daily_vol: float) -> str:
    """Classify a symbol as Low / Medium / High volatility/velocity.

    atr_pct: ATR-14 as % of price
    daily_vol: std of daily returns (%)
    """
    # Combine ATR% and daily vol into a single composite score
    composite = atr_pct * 0.5 + daily_vol * 0.5
    if composite < 1.8:
        return "Low"
    elif composite < 3.5:
        return "Medium"
    else:
        return "High"


def generate_recommendations(
    raw: pd.DataFrame,
    model, scaler, feature_columns: list,
    num_stocks: int,
    expected_profit_pct: float,
    period_days: int,
    vol_filter: str,  # "Low", "Medium", "High", or None
):
    """Generate ranked recommendations with vol/velocity classification."""
    # Filter data to cutoff
    data = raw[raw["DateTime"] <= CUTOFF].copy()
    if data.empty:
        return []

    featured = engineer_features(data)
    if featured.empty:
        return []

    # Per-symbol volatility stats
    symbol_vol = {}
    symbol_vol_class = {}
    for symbol in data["Symbol"].unique():
        sym_raw = data[data["Symbol"] == symbol].sort_values("DateTime")
        if len(sym_raw) < 20:
            continue
        sym_raw = sym_raw.copy()
        sym_raw["date"] = sym_raw["DateTime"].dt.date
        daily = sym_raw.groupby("date")["Close"].last()
        daily_returns = daily.pct_change().dropna() * 100
        if len(daily_returns) < 5:
            continue

        max_gains = []
        closes = daily.values
        window = min(period_days, len(closes) - 1)
        for i in range(len(closes) - window):
            future_max = np.max(closes[i + 1: i + window + 1])
            gain = (future_max - closes[i]) / closes[i] * 100
            max_gains.append(gain)

        current_close = sym_raw.iloc[-1]["Close"]
        atr_series = sym_raw["Close"].diff().abs().rolling(14).mean()
        atr_val = atr_series.iloc[-1] if len(atr_series) > 14 else 0
        atr_pct = (atr_val / current_close * 100) if current_close > 0 else 0
        daily_vol = daily_returns.std() if len(daily_returns) > 1 else 0

        vol_label = classify_volatility(atr_pct, daily_vol)
        symbol_vol_class[symbol] = {"label": vol_label, "atr_pct": atr_pct, "daily_vol": daily_vol}

        if max_gains:
            symbol_vol[symbol] = {
                "avg_max_gain": np.mean(max_gains),
                "pct_achieving": np.mean([g >= expected_profit_pct for g in max_gains]) * 100,
            }

    # Per-symbol stats for YTD, 1-month
    as_of_date = CUTOFF.date()
    symbol_stats = {}
    for symbol in data["Symbol"].unique():
        sym_raw = data[data["Symbol"] == symbol].sort_values("DateTime")
        if sym_raw.empty:
            continue
        current_close = sym_raw.iloc[-1]["Close"]
        year_data = sym_raw[sym_raw["DateTime"].dt.year == as_of_date.year]
        first_close = year_data.iloc[0]["Close"] if not year_data.empty else current_close
        ytd_pct = (current_close - first_close) / first_close * 100
        daily_closes = sym_raw.groupby(sym_raw["DateTime"].dt.date)["Close"].last()
        month_ago_close = daily_closes.iloc[-20] if len(daily_closes) >= 20 else daily_closes.iloc[0]
        month_pct = (current_close - month_ago_close) / month_ago_close * 100
        symbol_stats[symbol] = {"ytd_pct": ytd_pct, "month_pct": month_pct}

    # Latest bar per symbol
    latest_rows = []
    for symbol in featured["Symbol"].unique():
        sym_data = featured[featured["Symbol"] == symbol].sort_values("DateTime")
        last_row = sym_data.iloc[-1:]
        if last_row[feature_columns].isna().any(axis=1).iloc[0]:
            if len(sym_data) > 1:
                last_row = sym_data.iloc[-2:-1]
        latest_rows.append(last_row)

    if not latest_rows:
        return []

    latest_df = pd.concat(latest_rows, ignore_index=True)
    latest_df[feature_columns] = latest_df[feature_columns].fillna(0)
    if latest_df.empty:
        return []

    X = latest_df[feature_columns].values
    X_scaled = scaler.transform(X)

    if isinstance(model, dict):
        probas = [m.predict_proba(X_scaled)[:, 1] for m in model.values()]
        proba = np.mean(probas, axis=0)
    else:
        proba = model.predict_proba(X_scaled)[:, 1]
    latest_df["buy_probability"] = proba

    market_regime_val = 0.0
    if "market_regime" in latest_df.columns:
        market_regime_val = latest_df["market_regime"].median()

    feasibility_scores = []
    for _, row in latest_df.iterrows():
        symbol = row["Symbol"]
        if symbol in BLACKLIST:
            feasibility_scores.append(0.0)
            continue

        vol_data = symbol_vol.get(symbol, {})
        model_conf = proba[row.name]
        hist_pct = vol_data.get("pct_achieving", 0.0)
        avg_gain = vol_data.get("avg_max_gain", 0.0)

        min_feasibility = max(15.0, 35.0 - expected_profit_pct * 2.0)
        if hist_pct < min_feasibility:
            feasibility_scores.append(0.0)
            continue

        atr_val = row.get("atr_14", None)
        close = row["Close"]
        if atr_val is not None and pd.notna(atr_val) and close > 0:
            atr_pct_of_price = atr_val / close * 100
            if atr_pct_of_price > expected_profit_pct * 0.8:
                feasibility_scores.append(0.0)
                continue

        gain_ratio = min(avg_gain / expected_profit_pct, 2.0) if expected_profit_pct > 0 else 1.0

        returns_5d = row.get("returns_5d", 0.0)
        overext_penalty = max(0.5, 1.0 - (returns_5d - expected_profit_pct) / 20.0) if (pd.notna(returns_5d) and returns_5d > expected_profit_pct) else 1.0

        momentum_decel = row.get("momentum_decel", 0.0)
        decel_penalty = max(0.7, 1.0 + momentum_decel / 30.0) if (pd.notna(momentum_decel) and momentum_decel < -3.0) else 1.0

        sma_20 = row.get("sma_20", None)
        mean_rev_factor = 1.0
        if sma_20 is not None and pd.notna(sma_20) and sma_20 > 0:
            if close < sma_20 and pd.notna(momentum_decel) and momentum_decel > 0:
                mean_rev_factor = 1.15
            elif close > sma_20 * 1.05:
                mean_rev_factor = 0.85
        if sma_20 is not None and pd.notna(sma_20) and close > sma_20:
            if hist_pct < 50.0:
                mean_rev_factor *= 0.75

        est_ret = avg_gain * model_conf
        max_reasonable = expected_profit_pct * 2.0
        profit_cap_penalty = max(0.5, 1.0 - (est_ret - max_reasonable) / 20.0) if est_ret > max_reasonable else 1.0

        regime_penalty = 1.0
        if market_regime_val < -0.02:
            if model_conf < 0.6:
                regime_penalty = 0.7
            elif model_conf < 0.7:
                regime_penalty = 0.85

        combined = (
            0.25 * model_conf
            + 0.40 * (hist_pct / 100.0)
            + 0.35 * min(gain_ratio, 1.0)
        ) * overext_penalty * decel_penalty * mean_rev_factor * profit_cap_penalty * regime_penalty
        feasibility_scores.append(combined)

    latest_df["combined_score"] = feasibility_scores
    latest_df["score"] = np.clip((np.array(feasibility_scores) * 100).astype(int), 1, 100)
    latest_df["est_return"] = latest_df.apply(
        lambda r: symbol_vol.get(r["Symbol"], {}).get("avg_max_gain", 0.0) * proba[r.name], axis=1
    )
    latest_df["est_target_price"] = latest_df["Close"] * (1 + latest_df["est_return"] / 100)

    if "atr_14" in latest_df.columns:
        atr = latest_df["atr_14"]
        atr_pct = atr / latest_df["Close"]
        median_atr_pct = atr_pct.median()
        multiplier = np.where(
            atr_pct > median_atr_pct,
            np.clip(2.0 + (atr_pct - median_atr_pct) / median_atr_pct, 2.0, 3.0),
            np.clip(2.0 - (median_atr_pct - atr_pct) / median_atr_pct * 0.5, 1.5, 2.0),
        )
        latest_df["est_stop_loss"] = latest_df["Close"] - multiplier * atr
    else:
        latest_df["est_stop_loss"] = latest_df["Close"] * 0.97

    # Sort and apply filters: above SMA-20 + volatility class
    latest_df = latest_df.sort_values("combined_score", ascending=False).reset_index(drop=True)

    # Filter: above SMA-100 only
    above_sma_mask = latest_df.apply(
        lambda r: r.get("sma_100") is not None and pd.notna(r.get("sma_100")) and r["Close"] > r["sma_100"],
        axis=1
    )
    # Filter: volatility class
    vol_mask = latest_df["Symbol"].apply(
        lambda s: symbol_vol_class.get(s, {}).get("label", "Unknown") == vol_filter
    ) if vol_filter else pd.Series(True, index=latest_df.index)

    filtered = latest_df[above_sma_mask & vol_mask].reset_index(drop=True)

    # Diverse selection
    selected_indices = []
    category_counts = {}
    for idx, row in filtered.iterrows():
        if len(selected_indices) >= num_stocks:
            break
        cat = row.get("category", "unknown")
        count = category_counts.get(cat, 0)
        max_per_cat = 2 if cat in ("merchandise", "etfs") else 3
        if count >= max_per_cat:
            continue
        penalty = 0.70 ** count
        adj_score = row["combined_score"] * penalty
        if len(selected_indices) == 0 or adj_score >= filtered.loc[selected_indices[-1], "combined_score"] * 0.5:
            selected_indices.append(idx)
            category_counts[cat] = count + 1

    if len(selected_indices) < num_stocks:
        remaining = filtered[~filtered.index.isin(selected_indices)]
        needed = num_stocks - len(selected_indices)
        selected_indices.extend(remaining.head(needed).index.tolist())

    top = filtered.loc[selected_indices] if selected_indices else filtered.head(num_stocks)

    results = []
    for _, row in top.iterrows():
        symbol = row["Symbol"]
        close = row["Close"]
        sma20 = row.get("sma_20", None)
        stats = symbol_stats.get(symbol, {})
        vc = symbol_vol_class.get(symbol, {})
        results.append({
            "symbol": symbol,
            "current_price": round(float(close), 2),
            "score": int(row["score"]),
            "target_price": round(float(row["est_target_price"]), 2),
            "stop_loss": round(float(row["est_stop_loss"]), 2),
            "expected_profit": round(float(row["est_return"]), 1),
            "ytd_pct": round(stats.get("ytd_pct", 0.0), 1),
            "month_pct": round(stats.get("month_pct", 0.0), 1),
            "vol_label": vc.get("label", "?"),
            "atr_pct": round(vc.get("atr_pct", 0), 2),
            "daily_vol": round(vc.get("daily_vol", 0), 2),
        })
    return results


def print_table(title: str, recs: list):
    print(f"\n### {title}\n")
    print("| Rank | Symbol | Current Price | Score | Target Price | Stop-Loss | Expected Profit | YTD % | Last Month % | Volatility/Velocity |")
    print("|------|--------|--------------|-------|-------------|-----------|----------------|-------|-------------|-------------------|")
    for i, r in enumerate(recs, 1):
        print(
            f"| {i} | **{r['symbol']}** | ${r['current_price']:.2f} | {r['score']} | "
            f"${r['target_price']:.2f} | ${r['stop_loss']:.2f} | {r['expected_profit']:.1f}% | "
            f"{r['ytd_pct']:+.1f}% | {r['month_pct']:+.1f}% | {r['vol_label']} ({r['atr_pct']:.1f}%/{r['daily_vol']:.1f}%) |"
        )
    if not recs:
        print("| — | No qualifying stocks found | — | — | — | — | — | — | — | — |")


if __name__ == "__main__":
    print("Loading model artifacts...")
    model = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_COLUMNS_PATH) as f:
        feature_columns = json.load(f)

    print(f"Loading all data (cutoff: {CUTOFF})...")
    raw = load_all_data()
    raw["DateTime"] = pd.to_datetime(raw["DateTime"])

    print("Generating recommendations...\n")
    print("=" * 100)
    print("RECOMMENDATIONS FOR MARCH 23, 2026 — Data as of March 20, 2026 19:30")
    print("All picks: Above SMA-100 (100 hourly bars) | Hold: 4–9 days")
    print("=" * 100)

    # 5% target — Low volatility
    recs_5 = generate_recommendations(raw, model, scaler, feature_columns,
                                       num_stocks=10, expected_profit_pct=5.0,
                                       period_days=7, vol_filter="Low")
    print_table("5% Profit Target — Low Volatility/Velocity", recs_5)

    # 8% target — Low volatility
    recs_8 = generate_recommendations(raw, model, scaler, feature_columns,
                                       num_stocks=10, expected_profit_pct=8.0,
                                       period_days=7, vol_filter="Low")
    print_table("8% Profit Target — Low Volatility/Velocity", recs_8)

    # 10% target — Medium volatility
    recs_10 = generate_recommendations(raw, model, scaler, feature_columns,
                                        num_stocks=10, expected_profit_pct=10.0,
                                        period_days=7, vol_filter="Medium")
    print_table("10% Profit Target — Medium Volatility/Velocity", recs_10)
