"""Generate recommendations for March 24, 2026 with above-SMA-20 filter."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.predictor import Predictor

predictor = Predictor()
predictor.load()

# Generate more candidates (50) so we can filter down to 10 above-SMA-20
for target in [5, 8, 10]:
    recs = predictor.predict(
        num_stocks=50,
        expected_profit_pct=float(target),
        period_days=7,  # ~4-9 day hold period centered on 7
    )

    # Filter: only above SMA-20
    above_sma = [r for r in recs if r["vs_sma_20"] == "Above"]
    top10 = above_sma[:10]

    print(f"\n### {target}% Target / 4-9 Day Hold (Above SMA-20 Only)\n")
    print("| Rank | Symbol | Current Price | Score | Target Price | Stop-Loss | Expected Profit | YTD % | Last Month % |")
    print("|------|--------|--------------|-------|-------------|-----------|----------------|-------|-------------|")
    for i, r in enumerate(top10, 1):
        print(
            f"| {i} | **{r['symbol']}** | ${r['current_price']:.2f} | {r['score']} | "
            f"${r['target_price']:.2f} | ${r['stop_loss']:.2f} | {r['expected_profit_pct']:.1f}% | "
            f"{r['ytd_pct']:+.1f}% | {r['month_pct']:+.1f}% |"
        )
    print(f"\n_({len(above_sma)} stocks above SMA-20 available, showing top 10)_")
