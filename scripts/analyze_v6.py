"""Analyze v6 backtest results to find improvement opportunities."""
import pandas as pd
import numpy as np

df = pd.read_csv("backtest_feb_results.csv")

# 1. Worst symbols
print("=== WORST SYMBOLS (≥10 recs) ===")
sym = df.groupby("Symbol").agg(
    total=("Status", "count"),
    hits=("Status", lambda x: (x == "HIT TARGET").sum()),
    stops=("Status", lambda x: (x == "STOPPED OUT").sum()),
)
sym["hit_rate"] = (sym["hits"] / sym["total"] * 100).round(1)
worst = sym[sym["total"] >= 10].sort_values("hit_rate")
print(worst.head(15).to_string())

# 2. Best symbols
print("\n=== BEST SYMBOLS ===")
print(worst.sort_values("hit_rate", ascending=False).head(10).to_string())

# 3. 10%/10d weekly breakdown
print("\n=== 10%/10d WEEKLY BREAKDOWN ===")
s10 = df[df["Scenario"] == "10%/10d"].copy()
s10["day"] = pd.to_datetime(s10["TradingDate"]).dt.day
s10["period"] = s10["day"].apply(lambda d: "Week1(2-6)" if d <= 6 else ("Week2(9-13)" if d <= 13 else ("Week3(17-20)" if d <= 20 else "Week4(23-27)")))
for p in ["Week1(2-6)", "Week2(9-13)", "Week3(17-20)", "Week4(23-27)"]:
    sp = s10[s10["period"] == p]
    hits = (sp["Status"] == "HIT TARGET").sum()
    print(f"  {p}: {hits}/{len(sp)} = {hits/len(sp)*100:.1f}%")

# 4. Score vs hit rate
print("\n=== SCORE vs HIT RATE ===")
for lo, hi in [(0, 40), (41, 60), (61, 80), (81, 100)]:
    subset = df[(df["Score"] >= lo) & (df["Score"] <= hi)]
    if len(subset) > 0:
        hits = (subset["Status"] == "HIT TARGET").sum()
        print(f"  Score {lo}-{hi}: {hits}/{len(subset)} = {hits/len(subset)*100:.1f}%")

# 5. SMA20 position
print("\n=== SMA20 vs HIT RATE ===")
for pos in ["Under", "Above"]:
    subset = df[df["vsSMA20"] == pos]
    hits = (subset["Status"] == "HIT TARGET").sum()
    stops = (subset["Status"] == "STOPPED OUT").sum()
    print(f"  {pos}: {hits}/{len(subset)} = {hits/len(subset)*100:.1f}%  (stopped: {stops})")

# 6. Unique symbols
print("\n=== UNIQUE SYMBOLS PER SCENARIO ===")
for s in sorted(df["Scenario"].unique()):
    print(f"  {s}: {df[df['Scenario']==s]['Symbol'].nunique()}")

# 7. Stop-out analysis
print("\n=== TOP STOPPED-OUT SYMBOLS ===")
stops = df[df["Status"] == "STOPPED OUT"]
print(f"Total: {len(stops)}/{len(df)} ({len(stops)/len(df)*100:.1f}%)")
print(stops.groupby("Symbol").size().sort_values(ascending=False).head(10).to_string())

# 8. ExpProfit vs outcome
print("\n=== EXPECTED PROFIT vs OUTCOME ===")
for outcome in ["HIT TARGET", "MISSED", "STOPPED OUT"]:
    subset = df[df["Status"] == outcome]
    print(f"  {outcome}: avg ExpProfit={subset['ExpProfit'].mean():.1f}%  avg Score={subset['Score'].mean():.1f}")
