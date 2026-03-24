"""Analyze VOO (S&P 500 proxy) trend during Feb and March 2026."""
import pandas as pd

voo = pd.read_csv("Data/etfs_hourly/VOO_hourly_candles.txt", sep="|")
voo["DateTime"] = pd.to_datetime(voo["DateTime"])
voo = voo.sort_values("DateTime")

# Get daily closes (last bar of each day)
voo["Date"] = voo["DateTime"].dt.date
daily = voo.groupby("Date").last().reset_index()

# Filter Feb-Mar 2026
mask = (daily["Date"] >= pd.Timestamp("2026-02-01").date()) & (
    daily["Date"] <= pd.Timestamp("2026-03-23").date()
)
d = daily[mask][["Date", "Close"]].copy()
d["pct_1d"] = d["Close"].pct_change() * 100
d["pct_5d"] = d["Close"].pct_change(5) * 100
d["sma20"] = d["Close"].rolling(20).mean()
d["vs_sma20"] = ((d["Close"] / d["sma20"]) - 1) * 100

# The regime feature uses hourly bars (pct_change(20) on hourly = ~3 trading days)
# Let's also compute an hourly-based regime
voo_hourly = voo.sort_values("DateTime")
voo_hourly["regime_20bar"] = voo_hourly["Close"].pct_change(periods=20) * 100

print("=" * 80)
print("VOO Daily Close in Feb-Mar 2026:")
print("=" * 80)
for _, row in d.iterrows():
    sma_str = f"{row['vs_sma20']:+.2f}%" if pd.notna(row["vs_sma20"]) else "N/A"
    pct5_str = f"{row['pct_5d']:+.2f}%" if pd.notna(row["pct_5d"]) else "N/A"
    print(
        f"  {row['Date']}  Close: ${row['Close']:.2f}  "
        f"1d: {row['pct_1d']:+.2f}%  5d: {pct5_str}  vs SMA20: {sma_str}"
        if pd.notna(row["pct_1d"])
        else f"  {row['Date']}  Close: ${row['Close']:.2f}  (first day)"
    )

# For March test days, show the hourly regime value at market open
print("\n" + "=" * 80)
print("Hourly market_regime (20-bar pct_change) at start of each March test day:")
print("=" * 80)
mar_days = [
    "2026-03-02",
    "2026-03-03",
    "2026-03-04",
    "2026-03-05",
    "2026-03-06",
    "2026-03-09",
    "2026-03-10",
    "2026-03-11",
    "2026-03-12",
    "2026-03-13",
    "2026-03-16",
]
for day_str in mar_days:
    day = pd.Timestamp(day_str).date()
    day_data = voo_hourly[voo_hourly["DateTime"].dt.date == day]
    if not day_data.empty:
        first_regime = day_data.iloc[0]["regime_20bar"]
        last_regime = day_data.iloc[-1]["regime_20bar"]
        first_close = day_data.iloc[0]["Close"]
        last_close = day_data.iloc[-1]["Close"]
        print(
            f"  {day_str}  Open: ${first_close:.2f}  Close: ${last_close:.2f}  "
            f"Regime(open): {first_regime:+.2f}%  Regime(close): {last_regime:+.2f}%"
        )

# Also show what the backtest results were per day
print("\n" + "=" * 80)
print("March backtest results vs VOO regime:")
print("=" * 80)
try:
    results = pd.read_csv("backtest_mar_results.csv")
    results["hit_target"] = results["Status"].str.contains("HIT", case=False, na=False).astype(int)
    results["profit_target"] = results["Scenario"].str.extract(r"(\d+)%").astype(int)
    for day_str in mar_days:
        day_data = voo_hourly[voo_hourly["DateTime"].dt.date == pd.Timestamp(day_str).date()]
        regime = day_data.iloc[0]["regime_20bar"] if not day_data.empty else 0
        day_results = results[results["TradingDate"] == day_str]
        day3 = day_results[day_results["profit_target"] == 3]
        day5 = day_results[day_results["profit_target"] == 5]
        day8 = day_results[day_results["profit_target"] == 8]
        day10 = day_results[day_results["profit_target"] == 10]
        h3 = day3["hit_target"].sum()
        h5 = day5["hit_target"].sum()
        h8 = day8["hit_target"].sum()
        h10 = day10["hit_target"].sum()
        print(
            f"  {day_str}  VOO regime: {regime:+.2f}%  |  "
            f"3%={h3}/10  5%={h5}/10  8%={h8}/10  10%={h10}/10"
        )
except Exception as e:
    print(f"  Could not load results: {e}")

# Summary: would a simple filter have helped?
print("\n" + "=" * 80)
print("ANALYSIS: Would regime suppression help?")
print("=" * 80)
try:
    results = pd.read_csv("backtest_mar_results.csv")
    results["hit_target"] = results["Status"].str.contains("HIT", case=False, na=False).astype(int)
    results["profit_target"] = results["Scenario"].str.extract(r"(\d+)%").astype(int)
    for threshold in [-1, -2, -3, -4, -5]:
        suppressed_days = []
        for day_str in mar_days:
            day_data = voo_hourly[
                voo_hourly["DateTime"].dt.date == pd.Timestamp(day_str).date()
            ]
            regime = day_data.iloc[0]["regime_20bar"] if not day_data.empty else 0
            if regime < threshold:
                suppressed_days.append(day_str)

        kept = results[~results["TradingDate"].isin(suppressed_days)]
        suppressed = results[results["TradingDate"].isin(suppressed_days)]

        for profit_target in [3, 5, 8, 10]:
            k = kept[kept["profit_target"] == profit_target]
            s = suppressed[suppressed["profit_target"] == profit_target]
            k_hits = k["hit_target"].sum()
            k_total = len(k)
            k_rate = k_hits / k_total * 100 if k_total > 0 else 0
            s_hits = s["hit_target"].sum()
            s_total = len(s)
            s_rate = s_hits / s_total * 100 if s_total > 0 else 0
            if profit_target == 3:
                print(
                    f"\n  Threshold: VOO 20-bar < {threshold}%  "
                    f"(suppresses {len(suppressed_days)} days: {suppressed_days})"
                )
            print(
                f"    {profit_target}%/10d: Kept {k_hits}/{k_total} ({k_rate:.1f}%)  "
                f"| Suppressed would have been {s_hits}/{s_total} ({s_rate:.1f}%)"
            )

    # Also test: would Feb results be harmed?
    print("\n" + "=" * 80)
    print("CROSS-CHECK: Would this filter harm February results?")
    print("=" * 80)
    feb_results = pd.read_csv("backtest_feb_results.csv")
    feb_results["hit_target"] = feb_results["Status"].str.contains("HIT", case=False, na=False).astype(int)
    feb_results["profit_target"] = feb_results["Scenario"].str.extract(r"(\d+)%").astype(int)
    feb_days_list = sorted(feb_results["TradingDate"].unique())
    for threshold in [-1, -2, -3]:
        suppressed_days = []
        for day_str in feb_days_list:
            day_data = voo_hourly[
                voo_hourly["DateTime"].dt.date == pd.Timestamp(day_str).date()
            ]
            regime = day_data.iloc[0]["regime_20bar"] if not day_data.empty else 0
            if regime < threshold:
                suppressed_days.append(day_str)
        kept = feb_results[~feb_results["TradingDate"].isin(suppressed_days)]
        n_sup = len(suppressed_days)
        for profit_target in [3, 5, 8, 10]:
            k = kept[kept["profit_target"] == profit_target]
            k_hits = k["hit_target"].sum()
            k_total = len(k)
            k_rate = k_hits / k_total * 100 if k_total > 0 else 0
            if profit_target == 3:
                print(
                    f"\n  Threshold: VOO 20-bar < {threshold}%  "
                    f"(suppresses {n_sup} Feb days: {suppressed_days})"
                )
            print(
                f"    {profit_target}%/10d: Kept {k_hits}/{k_total} ({k_rate:.1f}%)"
            )
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  Could not load results: {e}")
