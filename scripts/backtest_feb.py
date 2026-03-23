"""Backtest predictions for every February 2026 trading day.

For each trading day in Feb, generates recommendations using data
available up to that date, then checks actual performance over the
next 10 trading days.
"""

import json
import datetime
import sys
import os
import time
import numpy as np
import pandas as pd
import joblib

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import load_all_data
from src.features.pipeline import engineer_features, get_feature_columns
from src.utils.config import (
    BEST_MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    SCALER_PATH,
    HOURLY_BARS_PER_DAY,
)


def load_model_artifacts():
    model = joblib.load(BEST_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURE_COLUMNS_PATH) as f:
        feature_columns = json.load(f)
    return model, scaler, feature_columns


def get_trading_days(raw: pd.DataFrame) -> list:
    """Return sorted list of unique trading dates."""
    return sorted(raw["DateTime"].dt.date.unique())


def predict_as_of(
    raw: pd.DataFrame,
    model,
    scaler,
    feature_columns: list,
    as_of_date: datetime.date,
    num_stocks: int,
    expected_profit_pct: float,
    period_days: int,
) -> pd.DataFrame:
    """Generate recommendations using only data up to as_of_date (inclusive).

    Returns a DataFrame with columns:
        Symbol, Price, Score, Target, StopLoss, ExpProfit, SMA20, vsSMA20, YTD, Month
    """
    # Filter data: only bars on or before as_of_date
    cutoff = pd.Timestamp(as_of_date) + pd.Timedelta(hours=23, minutes=59)
    data = raw[raw["DateTime"] <= cutoff].copy()

    if data.empty:
        return pd.DataFrame()

    featured = engineer_features(data)
    if featured.empty:
        return pd.DataFrame()

    # Per-symbol volatility stats for feasibility scoring
    symbol_vol = {}
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
            future_max = np.max(closes[i + 1 : i + window + 1])
            gain = (future_max - closes[i]) / closes[i] * 100
            max_gains.append(gain)

        if max_gains:
            symbol_vol[symbol] = {
                "avg_max_gain": np.mean(max_gains),
                "pct_achieving": np.mean([g >= expected_profit_pct for g in max_gains]) * 100,
            }

    # Per-symbol stats for YTD, 1-month, SMA-20
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

    # Take the latest bar per symbol
    latest_rows = []
    for symbol in featured["Symbol"].unique():
        sym_data = featured[featured["Symbol"] == symbol].sort_values("DateTime")
        last_row = sym_data.iloc[-1:]
        if last_row[feature_columns].isna().any(axis=1).iloc[0]:
            if len(sym_data) > 1:
                last_row = sym_data.iloc[-2:-1]
        latest_rows.append(last_row)

    if not latest_rows:
        return pd.DataFrame()

    latest_df = pd.concat(latest_rows, ignore_index=True)

    # Fill NaN features with 0 — early dates lack long lookback windows (e.g. SMA-200)
    # so dropping NaN rows would eliminate all symbols
    latest_df[feature_columns] = latest_df[feature_columns].fillna(0)

    if latest_df.empty:
        return pd.DataFrame()

    X = latest_df[feature_columns].values
    X_scaled = scaler.transform(X)

    # Ensemble prediction: average probabilities from all models
    if isinstance(model, dict):
        probas = []
        for name, m in model.items():
            probas.append(m.predict_proba(X_scaled)[:, 1])
        proba = np.mean(probas, axis=0)
    else:
        proba = model.predict_proba(X_scaled)[:, 1]
    latest_df["buy_probability"] = proba

    # Feasibility-adjusted scoring
    feasibility_scores = []
    for _, row in latest_df.iterrows():
        symbol = row["Symbol"]
        vol_data = symbol_vol.get(symbol, {})
        model_conf = proba[row.name]
        hist_pct = vol_data.get("pct_achieving", 0.0)
        avg_gain = vol_data.get("avg_max_gain", 0.0)

        if expected_profit_pct > 0:
            gain_ratio = min(avg_gain / expected_profit_pct, 2.0)
        else:
            gain_ratio = 1.0

        combined = (
            0.40 * model_conf
            + 0.30 * (hist_pct / 100.0)
            + 0.30 * min(gain_ratio, 1.0)
        )
        feasibility_scores.append(combined)

    latest_df["combined_score"] = feasibility_scores
    latest_df["score"] = np.clip((np.array(feasibility_scores) * 100).astype(int), 1, 100)

    latest_df["est_return"] = latest_df.apply(
        lambda r: symbol_vol.get(r["Symbol"], {}).get("avg_max_gain", 0.0) * proba[r.name],
        axis=1,
    )
    latest_df["est_target_price"] = latest_df["Close"] * (1 + latest_df["est_return"] / 100)

    if "atr_14" in latest_df.columns:
        latest_df["est_stop_loss"] = latest_df["Close"] - 2 * latest_df["atr_14"]
    else:
        latest_df["est_stop_loss"] = latest_df["Close"] * 0.97

    top = latest_df.nlargest(num_stocks, "combined_score")

    results = []
    for _, row in top.iterrows():
        symbol = row["Symbol"]
        close = row["Close"]
        sma20 = row.get("sma_20", None)
        above_sma20 = "Above" if (sma20 is not None and pd.notna(sma20) and close > sma20) else "Under"
        stats = symbol_stats.get(symbol, {})

        results.append({
            "Symbol": symbol,
            "Price": round(float(close), 2),
            "Score": int(row["score"]),
            "Target": round(float(row["est_target_price"]), 2),
            "StopLoss": round(float(row["est_stop_loss"]), 2),
            "ExpProfit": round(float(row["est_return"]), 1),
            "SMA20": round(float(sma20), 2) if sma20 is not None and pd.notna(sma20) else None,
            "vsSMA20": above_sma20,
            "YTD": round(stats.get("ytd_pct", 0.0), 1),
            "Month": round(stats.get("month_pct", 0.0), 1),
        })

    return pd.DataFrame(results)


def lookup_future_prices(
    raw: pd.DataFrame,
    symbol: str,
    buy_date: datetime.date,
    target_price: float,
    stop_loss: float,
) -> dict:
    """Look up actual prices 5 and 10 trading days after buy_date.

    Returns dict with: price_5d, price_10d, max_price_10d, status
    """
    sym_data = raw[raw["Symbol"] == symbol].copy()
    sym_data["date"] = sym_data["DateTime"].dt.date
    daily = sym_data.groupby("date").agg(
        Close=("Close", "last"),
        High=("High", "max"),
        Low=("Low", "min"),
    )
    daily = daily.sort_index()

    future_days = daily.loc[daily.index > buy_date]

    price_5d = None
    price_10d = None
    max_price_10d = None
    status = "N/A"

    if len(future_days) >= 5:
        price_5d = round(float(future_days.iloc[4]["Close"]), 2)

    if len(future_days) >= 10:
        price_10d = round(float(future_days.iloc[9]["Close"]), 2)
        max_price_10d = round(float(future_days.iloc[:10]["High"].max()), 2)
    elif len(future_days) > 0:
        # Use whatever days are available
        price_10d = round(float(future_days.iloc[-1]["Close"]), 2)
        max_price_10d = round(float(future_days["High"].max()), 2)

    if max_price_10d is not None:
        if max_price_10d >= target_price:
            status = "HIT TARGET"
        elif price_10d is not None and price_10d < stop_loss:
            status = "STOPPED OUT"
        else:
            status = "MISSED"

    return {
        "price_5d": price_5d,
        "price_10d": price_10d,
        "max_price_10d": max_price_10d,
        "status": status,
    }


def _process_single_day(scenario_key, pct, period, n, raw, model, scaler, feature_columns, buy_date, progress_counter=None):
    """Process a single trading day for a scenario (designed to run in a thread)."""
    recs = predict_as_of(
        raw, model, scaler, feature_columns,
        as_of_date=buy_date,
        num_stocks=n,
        expected_profit_pct=pct,
        period_days=period,
    )

    if recs.empty:
        return []

    results = []
    for _, rec in recs.iterrows():
        future = lookup_future_prices(
            raw, rec["Symbol"], buy_date, rec["Target"], rec["StopLoss"]
        )

        pnl_5d = None
        pnl_10d = None
        if future["price_5d"] is not None:
            pnl_5d = round((future["price_5d"] - rec["Price"]) / rec["Price"] * 100, 2)
        if future["price_10d"] is not None:
            pnl_10d = round((future["price_10d"] - rec["Price"]) / rec["Price"] * 100, 2)

        results.append({
            "Scenario": scenario_key,
            "TradingDate": buy_date,
            "Symbol": rec["Symbol"],
            "Price": rec["Price"],
            "Score": rec["Score"],
            "Target": rec["Target"],
            "StopLoss": rec["StopLoss"],
            "ExpProfit": rec["ExpProfit"],
            "SMA20": rec["SMA20"],
            "vsSMA20": rec["vsSMA20"],
            "YTD": rec["YTD"],
            "LastMonth": rec["Month"],
            "Price5d": future["price_5d"],
            "PnL5d": pnl_5d,
            "Price10d": future["price_10d"],
            "PnL10d": pnl_10d,
            "MaxPrice10d": future["max_price_10d"],
            "Status": future["status"],
        })

    day_hits = sum(1 for r in results if r["Status"] == "HIT TARGET")
    if progress_counter is not None:
        with progress_counter["lock"]:
            progress_counter["done"] += 1
            done = progress_counter["done"]
            total = progress_counter["total"]
            elapsed = time.time() - progress_counter["start"]
            pct_done = done / total * 100
            print(f"  [{done}/{total} ({pct_done:.0f}%) {elapsed:.0f}s] {scenario_key} | {buy_date}: {day_hits}/{len(results)} hit target")
    else:
        print(f"  [{scenario_key}] {buy_date}: {day_hits}/{len(results)} hit target")
    return results


def run_backtest():
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("Loading model artifacts...")
    model, scaler, feature_columns = load_model_artifacts()

    print("Loading all data...")
    raw = load_all_data()
    all_trading_days = get_trading_days(raw)

    # February 2026 trading days
    feb_days = [d for d in all_trading_days if d.year == 2026 and d.month == 2]
    print(f"February 2026 trading days: {len(feb_days)}")

    # Define test scenarios
    scenarios = [
        {"profit": 3, "period": 10, "num_stocks": 10},
        {"profit": 5, "period": 10, "num_stocks": 10},
        {"profit": 8, "period": 10, "num_stocks": 10},
        {"profit": 10, "period": 10, "num_stocks": 10},
    ]

    # Run all 4 scenarios × days = tasks, 20 at a time (5 days per scenario)
    total_tasks = len(scenarios) * len(feb_days)
    max_workers = len(scenarios) * 5  # 4 scenarios × 5 days = 20 threads
    print(f"\n{'='*60}")
    print(f"Running {total_tasks} tasks with {max_workers} parallel workers")
    print(f"Scenarios: {', '.join(f'{s['profit']}%/{s['period']}d' for s in scenarios)}")
    print(f"{'='*60}")

    progress = {
        "done": 0,
        "total": total_tasks,
        "start": time.time(),
        "lock": threading.Lock(),
    }
    all_results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for scenario in scenarios:
            pct = scenario["profit"]
            period = scenario["period"]
            n = scenario["num_stocks"]
            scenario_key = f"{pct}%/{period}d"
            for buy_date in feb_days:
                future = executor.submit(
                    _process_single_day,
                    scenario_key, pct, period, n,
                    raw, model, scaler, feature_columns, buy_date, progress,
                )
                futures[future] = (scenario_key, buy_date)

        for future in as_completed(futures):
            day_results = future.result()
            all_results.extend(day_results)

    total_elapsed = time.time() - progress["start"]
    print(f"\n{'='*60}")
    print(f"All {total_tasks} tasks completed in {total_elapsed:.0f}s")
    print(f"{'='*60}")

    # Print scenario summaries
    results_df = pd.DataFrame(all_results)
    for scenario in scenarios:
        pct = scenario["profit"]
        period = scenario["period"]
        key = f"{pct}%/{period}d"
        sdf = results_df[results_df["Scenario"] == key] if not results_df.empty else pd.DataFrame()
        hits = (sdf["Status"] == "HIT TARGET").sum() if not sdf.empty else 0
        total = len(sdf)
        rate = hits / total * 100 if total > 0 else 0
        avg_pnl = sdf["PnL10d"].dropna().mean() if not sdf.empty else 0
        print(f"  {key}:  {hits}/{total} hit target ({rate:.1f}%)  Avg 10d P&L: {avg_pnl:+.2f}%")

    return results_df


def generate_report(df: pd.DataFrame):
    """Generate markdown report from backtest results."""

    lines = [
        "# February 2026 Backtest Report",
        "",
        f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary",
        "",
    ]

    # Summary statistics per scenario
    for scenario in df["Scenario"].unique():
        sdf = df[df["Scenario"] == scenario]
        total = len(sdf)
        hits = (sdf["Status"] == "HIT TARGET").sum()
        misses = (sdf["Status"] == "MISSED").sum()
        stops = (sdf["Status"] == "STOPPED OUT").sum()
        na = (sdf["Status"] == "N/A").sum()
        hit_rate = hits / total * 100 if total > 0 else 0

        avg_pnl_5d = sdf["PnL5d"].dropna().mean()
        avg_pnl_10d = sdf["PnL10d"].dropna().mean()

        lines.append(f"### Scenario: {scenario}")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total Recommendations | {total} |")
        lines.append(f"| Hit Target | {hits} ({hit_rate:.1f}%) |")
        lines.append(f"| Missed | {misses} |")
        lines.append(f"| Stopped Out | {stops} |")
        lines.append(f"| Insufficient Data | {na} |")
        lines.append(f"| Avg P&L after 5 days | {avg_pnl_5d:+.2f}% |" if pd.notna(avg_pnl_5d) else f"| Avg P&L after 5 days | N/A |")
        lines.append(f"| Avg P&L after 10 days | {avg_pnl_10d:+.2f}% |" if pd.notna(avg_pnl_10d) else f"| Avg P&L after 10 days | N/A |")
        lines.append("")

    # Summary by trading date (for each scenario)
    lines.append("## Hit Rate by Trading Date")
    lines.append("")

    for scenario in df["Scenario"].unique():
        sdf = df[df["Scenario"] == scenario]
        lines.append(f"### {scenario}")
        lines.append("")
        lines.append("| Trading Date | Hits | Total | Hit Rate | Avg 5d P&L | Avg 10d P&L |")
        lines.append("|-------------|------|-------|----------|------------|-------------|")

        for date in sorted(sdf["TradingDate"].unique()):
            ddf = sdf[sdf["TradingDate"] == date]
            total = len(ddf)
            hits = (ddf["Status"] == "HIT TARGET").sum()
            rate = hits / total * 100 if total > 0 else 0
            avg5 = ddf["PnL5d"].dropna().mean()
            avg10 = ddf["PnL10d"].dropna().mean()
            avg5_str = f"{avg5:+.2f}%" if pd.notna(avg5) else "N/A"
            avg10_str = f"{avg10:+.2f}%" if pd.notna(avg10) else "N/A"
            lines.append(f"| {date} | {hits} | {total} | {rate:.0f}% | {avg5_str} | {avg10_str} |")

        lines.append("")

    # Detailed results per scenario
    lines.append("## Detailed Results")
    lines.append("")

    for scenario in df["Scenario"].unique():
        sdf = df[df["Scenario"] == scenario]
        lines.append(f"### {scenario}")
        lines.append("")
        lines.append("| Trading Date | Symbol | Price | Score | Target | Stop-Loss | Exp. Profit | SMA-20 | vs MA20 | YTD | Last-Month | Price 5d | P&L 5d | Price 10d | P&L 10d | Status |")
        lines.append("|-------------|--------|-------|-------|--------|-----------|-------------|--------|---------|-----|------------|----------|--------|-----------|---------|--------|")

        for _, row in sdf.iterrows():
            sma_str = f"${row['SMA20']:.2f}" if row["SMA20"] is not None and pd.notna(row.get("SMA20")) else "N/A"
            p5_str = f"${row['Price5d']:.2f}" if row["Price5d"] is not None and pd.notna(row.get("Price5d")) else "N/A"
            p10_str = f"${row['Price10d']:.2f}" if row["Price10d"] is not None and pd.notna(row.get("Price10d")) else "N/A"
            pnl5_str = f"{row['PnL5d']:+.2f}%" if row["PnL5d"] is not None and pd.notna(row.get("PnL5d")) else "N/A"
            pnl10_str = f"{row['PnL10d']:+.2f}%" if row["PnL10d"] is not None and pd.notna(row.get("PnL10d")) else "N/A"

            status_emoji = "✅" if row["Status"] == "HIT TARGET" else ("❌" if row["Status"] == "STOPPED OUT" else ("⚠️" if row["Status"] == "MISSED" else "—"))

            lines.append(
                f"| {row['TradingDate']} | {row['Symbol']} | ${row['Price']:.2f} | {row['Score']} | "
                f"${row['Target']:.2f} | ${row['StopLoss']:.2f} | {row['ExpProfit']:.1f}% | "
                f"{sma_str} | {row['vsSMA20']} | {row['YTD']:+.1f}% | {row['LastMonth']:+.1f}% | "
                f"{p5_str} | {pnl5_str} | {p10_str} | {pnl10_str} | {status_emoji} {row['Status']} |"
            )

        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    results_df = run_backtest()

    # Save raw CSV
    results_df.to_csv("backtest_feb_results.csv", index=False)
    print(f"\nSaved {len(results_df)} results to backtest_feb_results.csv")

    # Generate markdown report
    report = generate_report(results_df)
    with open("backtest_feb_report.md", "w") as f:
        f.write(report)
    print("Saved report to backtest_feb_report.md")

    # Print summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    for scenario in results_df["Scenario"].unique():
        sdf = results_df[results_df["Scenario"] == scenario]
        total = len(sdf)
        hits = (sdf["Status"] == "HIT TARGET").sum()
        rate = hits / total * 100 if total > 0 else 0
        avg_pnl = sdf["PnL10d"].dropna().mean()
        avg_str = f"{avg_pnl:+.2f}%" if pd.notna(avg_pnl) else "N/A"
        print(f"  {scenario:>10s}:  {hits:>3d}/{total:>3d} hit target ({rate:>5.1f}%)  Avg 10d P&L: {avg_str}")
