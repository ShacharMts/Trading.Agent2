"""Analyze which day of the week is the best for buying stocks."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.data.loader import load_all_data


def main():
    print("Loading data...")
    df = load_all_data()
    print(f"Loaded {len(df)} rows, {df['Symbol'].nunique()} symbols\n")

    df_sorted = df.sort_values(["Symbol", "DateTime"])

    results = []
    for symbol in df_sorted["Symbol"].unique():
        sym = df_sorted[df_sorted["Symbol"] == symbol].copy()
        sym["date"] = sym["DateTime"].dt.date
        daily = sym.groupby("date").last().reset_index()
        daily["fwd_return_1d"] = daily["Close"].shift(-1) / daily["Close"] * 100 - 100
        daily["fwd_return_3d"] = daily["Close"].shift(-3) / daily["Close"] * 100 - 100
        daily["fwd_return_5d"] = daily["Close"].shift(-5) / daily["Close"] * 100 - 100
        daily["day_of_week"] = pd.to_datetime(daily["date"]).dt.day_name()
        daily["symbol"] = symbol
        daily["category"] = sym["category"].iloc[0]
        results.append(daily)

    all_daily = pd.concat(results, ignore_index=True)
    print(f"Daily close bars: {len(all_daily)}\n")

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    print("=" * 70)
    print("ANALYSIS: BEST DAY OF THE WEEK TO BUY")
    print("=" * 70)

    # --- Average Forward Return ---
    print("\n--- Average Forward Return (%) by Buy Day ---")
    print(f"{'Day':<12} {'1-Day Fwd':>10} {'3-Day Fwd':>10} {'5-Day Fwd':>10} {'Samples':>8}")
    print("-" * 55)
    for day in day_order:
        mask = all_daily["day_of_week"] == day
        r1 = all_daily.loc[mask, "fwd_return_1d"].mean()
        r3 = all_daily.loc[mask, "fwd_return_3d"].mean()
        r5 = all_daily.loc[mask, "fwd_return_5d"].mean()
        n = mask.sum()
        print(f"{day:<12} {r1:>10.4f} {r3:>10.4f} {r5:>10.4f} {n:>8}")

    # --- Win Rate ---
    print("\n--- Win Rate (% of positive returns) by Buy Day ---")
    print(f"{'Day':<12} {'1-Day Win%':>10} {'3-Day Win%':>10} {'5-Day Win%':>10}")
    print("-" * 45)
    for day in day_order:
        mask = all_daily["day_of_week"] == day
        w1 = (all_daily.loc[mask, "fwd_return_1d"] > 0).mean() * 100
        w3 = (all_daily.loc[mask, "fwd_return_3d"] > 0).mean() * 100
        w5 = (all_daily.loc[mask, "fwd_return_5d"] > 0).mean() * 100
        print(f"{day:<12} {w1:>10.1f} {w3:>10.1f} {w5:>10.1f}")

    # --- Median Forward Return ---
    print("\n--- Median Forward Return (%) by Buy Day ---")
    print(f"{'Day':<12} {'1-Day Fwd':>10} {'3-Day Fwd':>10} {'5-Day Fwd':>10}")
    print("-" * 45)
    for day in day_order:
        mask = all_daily["day_of_week"] == day
        r1 = all_daily.loc[mask, "fwd_return_1d"].median()
        r3 = all_daily.loc[mask, "fwd_return_3d"].median()
        r5 = all_daily.loc[mask, "fwd_return_5d"].median()
        print(f"{day:<12} {r1:>10.4f} {r3:>10.4f} {r5:>10.4f}")

    # --- Risk (Std Dev) ---
    print("\n--- Risk (Std Dev of Returns %) by Buy Day ---")
    print(f"{'Day':<12} {'1-Day Std':>10} {'3-Day Std':>10} {'5-Day Std':>10}")
    print("-" * 45)
    for day in day_order:
        mask = all_daily["day_of_week"] == day
        s1 = all_daily.loc[mask, "fwd_return_1d"].std()
        s3 = all_daily.loc[mask, "fwd_return_3d"].std()
        s5 = all_daily.loc[mask, "fwd_return_5d"].std()
        print(f"{day:<12} {s1:>10.4f} {s3:>10.4f} {s5:>10.4f}")

    # --- Risk-Adjusted Return (Sharpe-like) ---
    print("\n--- Risk-Adjusted Return (Mean/Std) by Buy Day ---")
    print(f"{'Day':<12} {'1-Day':>10} {'3-Day':>10} {'5-Day':>10}")
    print("-" * 45)
    best_day = {}
    for horizon in ["1", "3", "5"]:
        col = f"fwd_return_{horizon}d"
        best_score = -999
        for day in day_order:
            mask = all_daily["day_of_week"] == day
            m = all_daily.loc[mask, col].mean()
            s = all_daily.loc[mask, col].std()
            ratio = m / s if s > 0 else 0
            if ratio > best_score:
                best_score = ratio
                best_day[horizon] = day

    for day in day_order:
        vals = []
        for horizon in ["1", "3", "5"]:
            col = f"fwd_return_{horizon}d"
            mask = all_daily["day_of_week"] == day
            m = all_daily.loc[mask, col].mean()
            s = all_daily.loc[mask, col].std()
            ratio = m / s if s > 0 else 0
            vals.append(f"{ratio:>10.4f}")
        print(f"{day:<12} {vals[0]} {vals[1]} {vals[2]}")

    # --- Best Day Summary ---
    print()
    print("=" * 70)
    print("BEST DAY TO BUY:")
    for h in ["1", "3", "5"]:
        d = best_day[h]
        mask = all_daily["day_of_week"] == d
        col = f"fwd_return_{h}d"
        avg = all_daily.loc[mask, col].mean()
        win = (all_daily.loc[mask, col] > 0).mean() * 100
        med = all_daily.loc[mask, col].median()
        print(f"  {h}-day horizon: {d} (avg {avg:+.4f}%, median {med:+.4f}%, win rate {win:.1f}%)")
    print("=" * 70)

    # --- By Category ---
    print("\n\n--- Best Buy Day by Asset Category ---")
    for cat in ["snp100", "snp500", "etfs", "merchandise"]:
        cat_daily = all_daily[all_daily["category"] == cat]
        if cat_daily.empty:
            continue
        print(f"\n  [{cat.upper()}]")
        print(f"  {'Day':<12} {'1d Avg%':>8} {'1d Win%':>8} {'5d Avg%':>8} {'5d Win%':>8}")
        print(f"  {'-'*40}")
        for day in day_order:
            mask = cat_daily["day_of_week"] == day
            if mask.sum() == 0:
                continue
            r1 = cat_daily.loc[mask, "fwd_return_1d"].mean()
            w1 = (cat_daily.loc[mask, "fwd_return_1d"] > 0).mean() * 100
            r5 = cat_daily.loc[mask, "fwd_return_5d"].mean()
            w5 = (cat_daily.loc[mask, "fwd_return_5d"] > 0).mean() * 100
            print(f"  {day:<12} {r1:>8.4f} {w1:>7.1f}% {r5:>8.4f} {w5:>7.1f}%")


if __name__ == "__main__":
    main()
