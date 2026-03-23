"""Load trained model and generate predictions for recommendations."""

import json
import numpy as np
import pandas as pd
import joblib

from src.data.loader import load_all_data
from src.features.pipeline import engineer_features, get_feature_columns
from src.utils.config import (
    BEST_MODEL_PATH,
    FEATURE_COLUMNS_PATH,
    MODEL_METADATA_PATH,
    SCALER_PATH,
    HOURLY_BARS_PER_DAY,
)


class Predictor:
    """Loads a trained model and generates stock recommendations."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.metadata = None

    def load(self):
        """Load model artifacts from disk."""
        self.model = joblib.load(BEST_MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        with open(FEATURE_COLUMNS_PATH, "r") as f:
            self.feature_columns = json.load(f)

        with open(MODEL_METADATA_PATH, "r") as f:
            self.metadata = json.load(f)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(
        self,
        num_stocks: int = 5,
        expected_profit_pct: float = 3.0,
        period_days: int = 10,
    ) -> list[dict]:
        """Generate top-N stock recommendations.

        Args:
            num_stocks: Number of tickers to return.
            expected_profit_pct: Target profit % (filter ±1%).
            period_days: Investment horizon.

        Returns:
            List of recommendation dicts sorted by score (descending).
        """
        if not self.is_loaded:
            self.load()

        # Load latest data and engineer features
        raw = load_all_data()
        featured = engineer_features(raw)

        # Pre-compute per-symbol stats from raw data for YTD, 1-month, SMA-20
        symbol_stats = {}
        for symbol in raw["Symbol"].unique():
            sym_raw = raw[raw["Symbol"] == symbol].sort_values("DateTime")
            if sym_raw.empty:
                continue
            current_close = sym_raw.iloc[-1]["Close"]

            # YTD: first close of the year vs current
            year_start = sym_raw[sym_raw["DateTime"].dt.year == sym_raw["DateTime"].dt.year.max()]
            first_close = year_start.iloc[0]["Close"] if not year_start.empty else current_close
            ytd_pct = (current_close - first_close) / first_close * 100

            # Last month: close ~20 trading days ago vs current
            daily_closes = sym_raw.groupby(sym_raw["DateTime"].dt.date)["Close"].last()
            if len(daily_closes) >= 20:
                month_ago_close = daily_closes.iloc[-20]
            else:
                month_ago_close = daily_closes.iloc[0]
            month_pct = (current_close - month_ago_close) / month_ago_close * 100

            symbol_stats[symbol] = {"ytd_pct": ytd_pct, "month_pct": month_pct}

        # Pre-compute per-symbol volatility stats for feasibility scoring
        symbol_vol = {}
        for symbol in raw["Symbol"].unique():
            sym_raw = raw[raw["Symbol"] == symbol].sort_values("DateTime")
            if len(sym_raw) < 20:
                continue
            # Daily returns from last bar of each day
            sym_raw = sym_raw.copy()
            sym_raw["date"] = sym_raw["DateTime"].dt.date
            daily = sym_raw.groupby("date")["Close"].last()
            daily_returns = daily.pct_change().dropna() * 100

            if len(daily_returns) < 5:
                continue

            # Historical max gain in N-day windows
            max_gains = []
            closes = daily.values
            window = min(period_days, len(closes) - 1)
            for i in range(len(closes) - window):
                future_max = np.max(closes[i + 1: i + window + 1])
                gain = (future_max - closes[i]) / closes[i] * 100
                max_gains.append(gain)

            if max_gains:
                avg_max_gain = np.mean(max_gains)
                pct_achieving = np.mean([g >= expected_profit_pct for g in max_gains]) * 100
                daily_vol = daily_returns.std()
            else:
                avg_max_gain = 0.0
                pct_achieving = 0.0
                daily_vol = 0.0

            symbol_vol[symbol] = {
                "avg_max_gain": avg_max_gain,
                "pct_achieving": pct_achieving,
                "daily_vol": daily_vol,
            }

        # For each symbol, take the latest bar
        latest_rows = []
        for symbol in featured["Symbol"].unique():
            sym_data = featured[featured["Symbol"] == symbol].sort_values("DateTime")
            last_row = sym_data.iloc[-1:]
            if last_row[self.feature_columns].isna().any(axis=1).iloc[0]:
                # Try second-to-last if latest has NaN features
                if len(sym_data) > 1:
                    last_row = sym_data.iloc[-2:-1]
            latest_rows.append(last_row)

        if not latest_rows:
            return []

        latest_df = pd.concat(latest_rows, ignore_index=True)

        # Drop rows with NaN features
        valid_mask = ~latest_df[self.feature_columns].isna().any(axis=1)
        latest_df = latest_df[valid_mask].reset_index(drop=True)

        if latest_df.empty:
            return []

        X = latest_df[self.feature_columns].values
        X_scaled = self.scaler.transform(X)

        # Predict buy probability
        proba = self.model.predict_proba(X_scaled)[:, 1]
        latest_df["buy_probability"] = proba

        # Compute feasibility-adjusted score per symbol
        # Combines: model confidence + historical feasibility of this profit in this period
        feasibility_scores = []
        for _, row in latest_df.iterrows():
            symbol = row["Symbol"]
            vol_data = symbol_vol.get(symbol, {})
            model_conf = proba[row.name]  # 0-1

            # How often has this stock achieved the target in the given window?
            hist_pct = vol_data.get("pct_achieving", 0.0)  # 0-100
            # What's the avg max gain in the window?
            avg_gain = vol_data.get("avg_max_gain", 0.0)

            # Feasibility: how close is avg_max_gain to the requested profit?
            if expected_profit_pct > 0:
                gain_ratio = min(avg_gain / expected_profit_pct, 2.0)  # cap at 2x
            else:
                gain_ratio = 1.0

            # Combined score: 40% model confidence + 30% historical success rate + 30% gain feasibility
            combined = (
                0.40 * model_conf
                + 0.30 * (hist_pct / 100.0)
                + 0.30 * min(gain_ratio, 1.0)
            )
            feasibility_scores.append(combined)

        latest_df["combined_score"] = feasibility_scores
        latest_df["score"] = np.clip((np.array(feasibility_scores) * 100).astype(int), 1, 100)

        # Estimate expected return based on historical avg max gain weighted by model confidence
        latest_df["est_return"] = latest_df.apply(
            lambda r: symbol_vol.get(r["Symbol"], {}).get("avg_max_gain", 0.0)
            * proba[r.name],
            axis=1
        )

        # Estimate target price and stop-loss
        latest_df["est_target_price"] = latest_df["Close"] * (1 + latest_df["est_return"] / 100)

        if "atr_14" in latest_df.columns:
            latest_df["est_stop_loss"] = latest_df["Close"] - 2 * latest_df["atr_14"]
        else:
            latest_df["est_stop_loss"] = latest_df["Close"] * 0.97

        # Sort by combined_score descending, take top-N
        top = latest_df.nlargest(num_stocks, "combined_score")

        recommendations = []
        for _, row in top.iterrows():
            symbol = row["Symbol"]
            sma20 = row.get("sma_20", None)
            close = row["Close"]
            above_sma20 = "Above" if (sma20 is not None and pd.notna(sma20) and close > sma20) else "Under"
            stats = symbol_stats.get(symbol, {})

            recommendations.append({
                "symbol": symbol,
                "current_price": round(float(close), 2),
                "expected_profit_pct": round(float(row["est_return"]), 1),
                "period_days": period_days,
                "score": int(row["score"]),
                "target_price": round(float(row["est_target_price"]), 2),
                "stop_loss": round(float(row["est_stop_loss"]), 2),
                "sma_20": round(float(sma20), 2) if sma20 is not None and pd.notna(sma20) else None,
                "vs_sma_20": above_sma20,
                "ytd_pct": round(stats.get("ytd_pct", 0.0), 2),
                "month_pct": round(stats.get("month_pct", 0.0), 2),
            })

        return recommendations
