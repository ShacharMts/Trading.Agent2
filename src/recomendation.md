# v6 Improvement Plan

**Based on:** v5 February 2026 backtest analysis (760 recommendations across 4 scenarios)

---

## Current Performance (v5 Baseline)

| Scenario | Hit Rate | Stopped Out | Avg 10d P&L |
|----------|----------|-------------|-------------|
| 3%/10d | 71.6% | 10.5% | +3.86% |
| 5%/10d | 66.8% | 13.7% | +2.95% |
| 8%/10d | 61.1% | 17.4% | +2.32% |
| 10%/10d | 62.1% | 18.4% | +2.52% |

---

## Key Findings

### 1. Toxic Symbols — The Model Loves Losers

The most critical problem: the model keeps recommending symbols that **never or rarely hit target**, even giving them high scores:

| Symbol | Total Recs | Hit Rate | Avg Score | Problem |
|--------|-----------|----------|-----------|---------|
| **INTC** | 56 | **0%** | 79 | 0 hits out of 56 recommendations |
| **MU** | 57 | **26%** | 84 | Gets highest scores (89-92) but only 26% hit rate |
| **KLAC** | 11 | **18%** | 87 | High scores, very low hit rate |
| **CAT** | 24 | **33%** | — | Stopped out 13x across scenarios |
| **LRCX** | 44 | **43%** | 88 | Score 87-89 but only 43% success |
| **NEM** | 36 | **39%** | 86 | Stopped out 15x, high scores (86-88) |

Meanwhile near-perfect: USO (100%), XLE (100%), XAR (89%), MRNA (90%), IBIT (92%).

**Root cause:** Model overfits on volatility/momentum — volatile semiconductors and miners trigger high momentum signals but are mean-reverting.

### 2. Score Does NOT Predict Success

| Score Range | Hit Rate (3%/10d) |
|------------|-------------------|
| 80-100 | 68% |
| 60-79 | 76% |
| 40-59 | 100% |

Higher scores actually perform WORSE. The model's confidence is anti-correlated with success at the top end.

### 3. Expected Profit is Inversely Correlated with Success

| Outcome | Avg Expected Profit |
|---------|-------------------|
| Hit target | 5.0% |
| Stopped out | 5.6% |
| Missed | **7.8%** |

Stocks the model expects to gain the most are the most likely to fail.

### 4. Below-SMA-20 Outperforms Above

| Position vs SMA-20 | Hit Rate (3%/10d) |
|--------------------|-------------------|
| **Below SMA-20** | **77%** |
| Above SMA-20 | 65% |

12pp advantage for mean-reversion setups.

### 5. Late-Month Deterioration (Scenario-Dependent)

| Period | 3%/10d | 8%/10d |
|--------|--------|--------|
| Week 1 (Feb 2-6) | 70% | 62% |
| Week 4 (Feb 23-27) | 70% | **44%** |

Higher targets suffer more during corrective regimes (-18pp at 8%/10d).

### 6. Symbol Concentration

| Scenario | Unique Symbols |
|----------|---------------|
| 3%/10d | 45 |
| 5%/10d | 32 |
| 8%/10d | 26 |
| 10%/10d | 31 |

---

## Improvement Plan

### Priority 1: Symbol Blacklist / Cool-Down (HIGH IMPACT, expected +3-5pp)

- Track each symbol's rolling hit rate over last 10 recommendations
- If hit rate < 40%, apply 0.3x penalty or block
- Cool-down: if stopped out in last 5 trading days, skip for 3 days
- If stopped out ≥ 3 times in last 20 days, block symbol

### Priority 2: Recalibrate Scoring Weights (HIGH IMPACT, expected +2-4pp)

```
Current:  0.40 × model + 0.30 × hist_pct + 0.30 × gain_ratio
Proposed: 0.25 × model + 0.40 × hist_pct + 0.35 × gain_ratio
```

Add minimum feasibility threshold: reject if hist_pct < 30%.

### Priority 3: Mean-Reversion Bonus (MEDIUM-HIGH, expected +2-3pp)

- Bonus 1.15x for stocks below SMA-20 with accelerating momentum
- Penalty 0.85x for stocks >5% above SMA-20

### Priority 4: Expected Profit Cap (MEDIUM, expected +1-2pp)

- Cap expected profit at 2.5× target
- Prefer moderate returns (1.5-2x target) over extreme outliers

### Priority 5: Broader Symbol Coverage (MEDIUM, expected +1-2pp)

- Per-symbol frequency cap: max 3 appearances per 5-day window
- Stronger diversity penalty: 0.70^count for recent repeats
- Reduce hard cap from 3 to 2 for merchandise/etfs

### Priority 6: Market Regime Gate (MEDIUM, expected +2-3pp)

- In bearish regime (VOO -2% over 20 bars): reduce target or raise quality bar
- Require higher confidence during downtrends

### Priority 7: New Features (LOWER, expected +1-2pp)

- consecutive_down_days, distance_to_52w_high, sector_momentum_rank

---

## v6 Targets

| Scenario | v5 | v6 Target |
|----------|-----|-----------|
| 3%/10d | 71.6% | ≥ 78% |
| 5%/10d | 66.8% | ≥ 73% |
| 8%/10d | 61.1% | ≥ 67% |
| 10%/10d | 62.1% | ≥ 68% |

Stop-out rate < 8%. Score ≥80 must outperform score 60-79. Min 40 unique symbols per scenario.

---

## Historical Notes (v1-v5 Evolution)

- v1-v2: Dropped 18 zero-importance candlestick patterns, added MACD/BB/Stochastic
- v3: Ensemble (XGBoost + LightGBM + RF), walk-forward CV, Sharpe-based selection, market regime
- v4: Parallelized training (ProcessPoolExecutor) and backtest (ThreadPoolExecutor)
- v5: Adaptive stop-loss (1.5-3.0x ATR), overextension filter, momentum deceleration, sector diversity (hard cap 3, penalty 0.80^count)