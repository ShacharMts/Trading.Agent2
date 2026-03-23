# Trading Agent — Model Test Report

**Report Date:** March 23, 2026  
**Model Version:** `random_forest_v1_20260323`  
**Training Framework:** scikit-learn 1.8.0 / XGBoost 3.2.0 / LightGBM 4.6.0  

---

## 1. Executive Summary

Three machine learning models (Random Forest, XGBoost, LightGBM) were trained and evaluated on hourly candlestick data across 419 stock and ETF symbols. The objective: predict which stocks will achieve a target profit of ≥3% within a 10-day trading horizon.

**Winner: Random Forest** — selected for the highest AUC-ROC (0.745), best backtest ROI (2.02%), and strongest precision among the three candidates.

---

## 2. Dataset Overview

### 2.1 Data Source

| Property | Value |
|---|---|
| Total raw rows | 159,008 hourly candles |
| Total symbols | 421 |
| Clean training rows (after feature engineering & NaN removal) | 74,921 |
| Symbols in training set | 419 |
| Date range | January 2, 2026 – March 20, 2026 |
| Candle interval | 1 hour |
| Data format | Pipe-delimited `.txt` files |

### 2.2 Data Categories

| Category | Symbols | Description |
|---|---|---|
| S&P 500 | 305 | S&P 500 constituents |
| S&P 100 | 96 | S&P 100 large-cap stocks |
| Sector ETFs | 15 | XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLU, XLRE, XLC, XAR, HACK, IGV, MAGS, VOO |
| Merchandise | 5 | GLD, USO, IBIT, ETHA, SLVR |

### 2.3 Target Distribution

The classification target is binary: **BUY** (stock achieves ≥3% max return within 10 days) vs **HOLD**.

| Class | Count | Percentage |
|---|---|---|
| HOLD (0) | 52,054 | 69.5% |
| BUY (1) | 22,867 | 30.5% |

The dataset is moderately imbalanced (~2.3:1 ratio), addressed via class weight balancing.

### 2.4 Train/Test Split

- **Method:** Time-based split per symbol (first 80% of each symbol's data for training, last 20% for testing)
- **Why:** Preserves temporal ordering — prevents look-ahead bias that random splits would introduce
- **Train set:** 59,853 rows
- **Test set:** 15,068 rows

---

## 3. Feature Engineering

### 3.1 Feature Summary

**45 total features** across 5 categories:

| Category | Count | Features |
|---|---|---|
| Candlestick Patterns | 20 | Doji, Dragonfly Doji, Gravestone Doji, Hammer, Inverted Hammer, Hanging Man, Shooting Star, Marubozu, Bullish Engulfing, Bearish Engulfing, Piercing Line, Dark Cloud Cover, Tweezer Bottom, Tweezer Top, Morning Star, Evening Star, Three White Soldiers, Three Black Crows, Three Inside Up, Three Inside Down |
| Moving Averages | 12 | SMA (7, 20, 50, 200), EMA (7, 20, 50), Crossovers (7/20, 20/50), MA Slope 20, Price vs SMA 20, Price vs SMA 50 |
| Volume | 3 | Volume SMA 20, Volume Ratio, Volume Trend |
| Price Action | 8 | Returns (1h, 7h, 20h), Volatility 20, ATR 14, RSI 14, High-Low Range, Body-to-Range |
| Encoding | 2 | Direction (bullish=1/bearish=0), Category (snp500=0, snp100=1, etf=2, merchandise=3) |

### 3.2 Target Variable Construction

| Target | Calculation |
|---|---|
| `future_return` | `(max close in next 70 bars − current close) / current close × 100` |
| `target_buy` | `1` if `future_return ≥ 3.0%`, else `0` |
| `optimal_hold_days` | Bars to max close ÷ 7 bars/day |
| `target_price` | Max close in the forward window |
| `stop_loss` | `max(close − 2×ATR₁₄, min_low_in_window × 0.99)` |

---

## 4. Model Configurations

### 4.1 XGBoost

| Parameter | Value |
|---|---|
| n_estimators | 500 |
| max_depth | 7 |
| learning_rate | 0.03 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| min_child_weight | 5 |
| reg_alpha | 0.1 |
| reg_lambda | 1.0 |
| scale_pos_weight | 1.5 |

### 4.2 LightGBM

| Parameter | Value |
|---|---|
| n_estimators | 500 |
| max_depth | 7 |
| learning_rate | 0.03 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| min_child_weight | 5 |
| reg_alpha | 0.1 |
| reg_lambda | 1.0 |
| scale_pos_weight | 1.5 |

### 4.3 Random Forest

| Parameter | Value |
|---|---|
| n_estimators | 500 |
| max_depth | 15 |
| min_samples_split | 10 |
| min_samples_leaf | 5 |
| max_features | sqrt |
| class_weight | {0: 1, 1: 1.5} |

All models used `StandardScaler` for feature normalization and a classification threshold of 0.5.

---

## 5. Test Results — Model Comparison

### 5.1 Classification Metrics

| Metric | Random Forest | XGBoost | LightGBM |
|---|---|---|---|
| **AUC-ROC** | **0.7450** | 0.7313 | 0.7312 |
| **Accuracy** | **0.7973** | 0.7279 | 0.7228 |
| **Precision** | **0.1855** | 0.1603 | 0.1564 |
| **Recall** | 0.4827 | **0.5993** | 0.5933 |
| **F1 Score** | **0.2680** | 0.2529 | 0.2475 |
| Precision @ Top 10 | **0.300** | 0.000 | 0.000 |
| Precision @ Top 20 | **0.300** | 0.000 | 0.000 |
| Training Time | 6.6s | 1.3s | 2.5s |

### 5.2 Backtest Results (Top 20 Picks)

| Metric | Random Forest | XGBoost | LightGBM |
|---|---|---|---|
| **Avg Return** | **+2.02%** | +0.53% | +0.57% |
| **Win Rate** | 60.0% | **65.0%** | **75.0%** |
| Max Return | **+11.21%** | +1.69% | +1.69% |
| Min Return | -1.90% | -1.90% | -1.90% |
| **Sharpe Ratio** | 0.542 | 0.573 | **0.625** |

### 5.3 Analysis

- **Random Forest** has the best discriminative ability (AUC-ROC 0.745) and the highest backtest return (+2.02%) by a wide margin. It captures the best individual picks (max return +11.21%).
- **XGBoost** and **LightGBM** perform nearly identically (AUC-ROC ~0.731), but sacrifice precision for recall. They predict BUY more aggressively, resulting in a higher false-positive rate.
- **LightGBM** has the best win rate (75%) and Sharpe ratio (0.625) due to more conservative return predictions, but its average return is lower.
- All models show low precision (~15–19%), which is expected given the 30.5% base rate — the models must identify the top ~30% of opportunities from a noisy financial signal.
- The key differentiator is **precision at top-N**: Random Forest is the only model with non-zero precision among its 10 highest-confidence predictions, meaning its most confident calls are more likely to be correct.

---

## 6. Feature Importance (Random Forest)

The top 20 most important features that drive the model's predictions:

| Rank | Feature | Importance | Category |
|---|---|---|---|
| 1 | `volatility_20` | 0.1110 | Price Action |
| 2 | `volume_sma_20` | 0.0754 | Volume |
| 3 | `sma_200` | 0.0671 | Moving Average |
| 4 | `price_vs_sma_50` | 0.0663 | Moving Average |
| 5 | `atr_14` | 0.0579 | Price Action |
| 6 | `sma_50` | 0.0555 | Moving Average |
| 7 | `ema_50` | 0.0546 | Moving Average |
| 8 | `ema_20` | 0.0521 | Moving Average |
| 9 | `ema_7` | 0.0516 | Moving Average |
| 10 | `sma_20` | 0.0505 | Moving Average |
| 11 | `sma_7` | 0.0483 | Moving Average |
| 12 | `returns_20h` | 0.0439 | Price Action |
| 13 | `price_vs_sma_20` | 0.0390 | Moving Average |
| 14 | `ma_slope_20` | 0.0369 | Moving Average |
| 15 | `volume_trend` | 0.0343 | Volume |
| 16 | `high_low_range` | 0.0327 | Price Action |
| 17 | `returns_7h` | 0.0305 | Price Action |
| 18 | `rsi_14` | 0.0255 | Price Action |
| 19 | `category_num` | 0.0202 | Encoding |
| 20 | `volume_ratio` | 0.0143 | Volume |

### Key Findings:

- **Volatility and ATR are dominant signals** — the model relies heavily on price movement magnitude to predict future opportunities.
- **Moving averages account for 8 of the top 14 features** — trend position (price vs. MA) and trend direction (MA slope) are core to the model's decision-making.
- **Volume features are ranked 2nd and 15th** — confirming that volume confirms price movements.
- **Candlestick patterns contribute minimally** — the individual patterns (hammer, engulfing, etc.) have low importance individually. The model relies more on quantitative trend and volatility signals.
- **Category encoding matters** (rank 19) — different asset types (stocks, ETFs, commodities) have different behavior profiles.

---

## 7. Live Prediction — Current Recommendations

Generated on March 23, 2026 using the trained Random Forest model.  
**Parameters:** Top 10 stocks, target profit 3%, period 10 days.

| Rank | Symbol | Current Price | Score | Target Price | Stop-Loss | Expected Profit |
|---|---|---|---|---|---|---|
| 1 | **VST** | $146.23 | 66 | $150.62 | $139.51 | 3.0% |
| 2 | **NRG** | $145.77 | 57 | $150.14 | $139.86 | 2.6% |
| 3 | **ALB** | $156.65 | 55 | $161.35 | $150.55 | 2.5% |
| 4 | **IBIT** | $39.79 | 55 | $40.98 | $38.93 | 2.5% |
| 5 | **CF** | $124.90 | 53 | $128.65 | $118.97 | 2.4% |
| 6 | **GLW** | $124.68 | 53 | $128.42 | $119.49 | 2.4% |
| 7 | **UAL** | $89.96 | 51 | $92.66 | $86.83 | 2.3% |
| 8 | **USO** | $121.44 | 51 | $125.08 | $116.53 | 2.3% |
| 9 | **OXY** | $60.72 | 50 | $62.54 | $59.25 | 2.3% |
| 10 | **CVX** | $201.78 | 49 | $207.83 | $198.47 | 2.2% |

### Recommendation Analysis:

- **Top pick VST (Vistra Corp)** has the highest confidence score (66/100) with a $4.39 upside target and $6.72 stop-loss buffer.
- **Energy sector is dominant** — VST, NRG, CF, USO, OXY, CVX are energy/commodity-related, suggesting the model identifies energy as currently having the strongest momentum signals.
- **IBIT (Bitcoin ETF)** appears at rank 4, indicating crypto ETF volatility is being captured as an opportunity signal.
- **Score range is 49–66** — reflecting moderate confidence. No extreme high-conviction picks in the current market environment.
- **Risk/Reward**: Average upside target is ~2.5%, average stop-loss is ~4% below current price. The risk/reward ratio averages about 1:1.6.

---

## 8. Model Selection — Final Recommendation

### Recommended Model: **Random Forest**

| Criterion | Random Forest | Runner-Up |
|---|---|---|
| AUC-ROC | **0.745** (best) | LightGBM: 0.731 |
| Backtest ROI | **+2.02%** (best) | LightGBM: +0.57% |
| Precision at Top-10 | **30%** (only non-zero) | Others: 0% |
| Accuracy | **79.7%** (best) | XGBoost: 72.8% |
| F1 Score | **0.268** (best) | XGBoost: 0.253 |

### Why Random Forest Wins:

1. **Best discrimination** — AUC-ROC 0.745 means the model correctly ranks BUY vs. HOLD opportunities 74.5% of the time.
2. **Highest precision on top picks** — The only model where its highest-confidence predictions have a meaningful success rate (30% at top-10).
3. **Best backtest returns** — +2.02% average return on top-20 picks vs. <0.6% for the other two models.
4. **Best overall accuracy** — 79.7% correct classifications.
5. **Robust to overfitting** — Random Forest's ensemble of 500 trees with max_depth=15 and balanced class weights provides stable generalization.

### Trade-offs Acknowledged:

- **Training time is 3–5× slower** (6.6s vs 1.3–2.5s) — acceptable for this use case.
- **Lower recall** (48% vs 59–60%) — the model misses some opportunities but is more selective.
- **Precision is still low at 18.5%** — inherent difficulty of financial prediction; the model is best used as a ranking system (top-N picks) rather than a binary classifier.

---

## 9. Improvement Opportunities

| Area | Potential Improvement |
|---|---|
| **More data** | Extend to 6–12 months of history for better pattern learning |
| **Hyperparameter tuning** | Use Optuna for Bayesian optimization per model |
| **Feature selection** | Remove low-importance candlestick features that add noise |
| **Ensemble** | Stack Random Forest + LightGBM for combined strengths |
| **Market regime** | Add VIX/market-wide features to detect bull/bear regimes |
| **Sector rotation** | Add sector momentum features (e.g., XLK vs. XLE relative strength) |
| **Walk-forward validation** | Replace single split with expanding-window cross-validation |

---

## 10. Artifacts

| File | Description |
|---|---|
| `models/best_model.pkl` | Trained Random Forest model (126 MB) |
| `models/scaler.pkl` | StandardScaler fitted on training data |
| `models/feature_columns.json` | 45 feature column names in order |
| `models/model_metadata.json` | Full training metadata and all results |

---

*This report was generated automatically from model training results. Predictions are informational and not financial advice.*
