how can we impruve the model ?

Analysis Findings
What's working:

Score calibration is decent: Score 90-100 → 89% hit rate, 80-90 → 69%, 70-80 → 53-56%
"Under SMA-20" picks outperform: 78% hit rate, +6.75% avg vs "Above" at 64%, +2.68%
Weeks 6-8 strong (74-80%), Week 9 collapsed to 50% (market downturn)
Key problems identified:

Problem	Evidence
20 candlestick patterns are useless	18/20 have zero importance. Dead weight
Low symbol diversity	Only 12-26 unique symbols across 190 recs. Same stocks picked repeatedly
Score-PnL correlation is weak	Only 0.20-0.27 — model barely predicts magnitude
No momentum/trend features	Missing MACD, Bollinger Bands, multi-day returns, day-of-week
MU always recommended, rarely hits	16x recommended, only 25% hit rate
SMA crossovers unused	Importance = 0 for both crossover features
Recommended Improvements
Drop dead features — remove 18 zero-importance candlestick patterns and both crossover signals
Add momentum features — MACD, Bollinger Bands, multi-timeframe returns (3d, 5d), Stochastic oscillator
Add calendar features — day-of-week, week-of-month (Week 9 collapse pattern)
Add relative strength — stock return vs SPY/VOO return (sector-relative momentum)
Improve diversity in scoring — penalize symbols that are over-concentrated in recommendations
Use walk-forward training — retrain the model on a rolling window instead of training on all data up-front, prevents lookahead bias
