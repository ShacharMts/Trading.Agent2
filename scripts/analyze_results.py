"""Analyze backtest results to find improvement opportunities."""

import pandas as pd
import numpy as np
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

df = pd.read_csv("backtest_feb_results.csv")

# 1. Hit rate by score bucket
print("=== HIT RATE BY SCORE BUCKET ===")
for scenario in df["Scenario"].unique():
    sdf = df[df["Scenario"] == scenario].copy()
    sdf["score_bucket"] = pd.cut(
        sdf["Score"], bins=[0, 50, 60, 70, 80, 90, 100],
        labels=["<50", "50-60", "60-70", "70-80", "80-90", "90-100"],
    )
    print(f"  {scenario}:")
    for bucket in ["<50", "50-60", "60-70", "70-80", "80-90", "90-100"]:
        bdf = sdf[sdf["score_bucket"] == bucket]
        if len(bdf) > 0:
            hits = (bdf["Status"] == "HIT TARGET").sum()
            print(f"    Score {bucket}: {hits}/{len(bdf)} = {hits/len(bdf)*100:.0f}%")
    print()

# 2. Most recommended symbols
print("=== MOST RECOMMENDED SYMBOLS (3%/10d) ===")
s3 = df[df["Scenario"] == "3%/10d"]
top_syms = s3["Symbol"].value_counts().head(20)
for sym, cnt in top_syms.items():
    sym_df = s3[s3["Symbol"] == sym]
    hits = (sym_df["Status"] == "HIT TARGET").sum()
    avg_pnl = sym_df["PnL10d"].dropna().mean()
    pnl_str = f"{avg_pnl:+.2f}%" if pd.notna(avg_pnl) else "N/A"
    print(f"  {sym:>6}: {cnt}x, hit {hits}x ({hits/cnt*100:.0f}%), avg 10d P&L: {pnl_str}")

# 3. Always-hit and never-hit symbols
print("\n=== ALWAYS HIT (3%/10d, appeared 3+) ===")
for sym in s3["Symbol"].unique():
    sym_df = s3[s3["Symbol"] == sym]
    if len(sym_df) >= 3:
        hits = (sym_df["Status"] == "HIT TARGET").sum()
        if hits == len(sym_df):
            avg = sym_df["PnL10d"].dropna().mean()
            print(f"  {sym}: {hits}/{len(sym_df)}, avg P&L: {avg:+.2f}%")

print("\n=== NEVER HIT (3%/10d, appeared 3+) ===")
for sym in s3["Symbol"].unique():
    sym_df = s3[s3["Symbol"] == sym]
    if len(sym_df) >= 3:
        hits = (sym_df["Status"] == "HIT TARGET").sum()
        if hits == 0:
            avg = sym_df["PnL10d"].dropna().mean()
            pnl_str = f"{avg:+.2f}%" if pd.notna(avg) else "N/A"
            print(f"  {sym}: {hits}/{len(sym_df)}, avg P&L: {pnl_str}")

# 4. Week-of-month pattern
print("\n=== HIT RATE BY WEEK (3%/10d) ===")
s3c = s3.copy()
s3c["date"] = pd.to_datetime(s3c["TradingDate"])
s3c["week"] = s3c["date"].dt.isocalendar().week.astype(int)
for w in sorted(s3c["week"].unique()):
    wdf = s3c[s3c["week"] == w]
    hits = (wdf["Status"] == "HIT TARGET").sum()
    avg = wdf["PnL10d"].dropna().mean()
    print(f"  Week {w}: {hits}/{len(wdf)} = {hits/len(wdf)*100:.0f}%, avg P&L: {avg:+.2f}%")

# 5. Current features
print("\n=== CURRENT FEATURES ===")
with open("models/feature_columns.json") as f:
    features = json.load(f)
print(f"  {len(features)} features: {features}")

# 6. Score vs actual performance correlation
print("\n=== SCORE-PERFORMANCE CORRELATION ===")
for scenario in df["Scenario"].unique():
    sdf = df[df["Scenario"] == scenario]
    valid = sdf.dropna(subset=["PnL10d"])
    if len(valid) > 5:
        corr = valid["Score"].corr(valid["PnL10d"])
        print(f"  {scenario}: Score-PnL10d correlation = {corr:.3f}")

# 7. vsSMA20 pattern
print("\n=== HIT RATE BY vs SMA-20 (3%/10d) ===")
for pos in ["Above", "Under"]:
    pdf = s3[s3["vsSMA20"] == pos]
    if len(pdf) > 0:
        hits = (pdf["Status"] == "HIT TARGET").sum()
        avg = pdf["PnL10d"].dropna().mean()
        print(f"  {pos}: {hits}/{len(pdf)} = {hits/len(pdf)*100:.0f}%, avg P&L: {avg:+.2f}%")

# 8. Category analysis - check which types of stocks perform best
print("\n=== UNIQUE SYMBOLS PER SCENARIO ===")
for scenario in df["Scenario"].unique():
    sdf = df[df["Scenario"] == scenario]
    print(f"  {scenario}: {sdf['Symbol'].nunique()} unique symbols out of {len(sdf)} recs")
