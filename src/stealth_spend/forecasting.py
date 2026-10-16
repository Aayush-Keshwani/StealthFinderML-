"""
Cash-flow forecasting module for Stealth-Spend.
Forecasts daily net cash flows and end-of-day balances using chronological holdouts,
compares Naive, Seasonal Naive, and ML regressors, and estimates low-balance risk.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import joblib

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


class CashFlowForecaster:
    """
    Constructs a continuous daily balance time series, builds leakage-free lag features,
    evaluates multiple forecast baselines, and flags low-balance risk horizons.
    """

    def __init__(
        self,
        test_days: int = 30,
        low_balance_threshold: float = 500.0,
        random_seed: int = 42,
    ):
        self.test_days = test_days
        self.low_balance_threshold = low_balance_threshold
        self.random_seed = random_seed
        self.ml_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=random_seed)
        self.feature_cols = []
        self.is_fitted = False

    def build_daily_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates canonical transactions into a complete contiguous daily series.
        """
        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["transaction_date"]).dt.floor("D")

        # Group by day
        daily = df_copy.groupby("date").agg({
            "amount": "sum",
            "balance": "last",
        }).rename(columns={"amount": "net_cash_flow", "balance": "observed_balance"})

        # Re-index to complete continuous calendar range
        full_idx = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="D")
        daily_full = daily.reindex(full_idx)
        daily_full["net_cash_flow"] = daily_full["net_cash_flow"].fillna(0.0)

        # Forward fill balance, or reconstruct if missing
        if daily_full["observed_balance"].isnull().all():
            daily_full["balance"] = 5000.0 + daily_full["net_cash_flow"].cumsum()
        else:
            daily_full["balance"] = daily_full["observed_balance"].ffill().bfill()

        daily_full["day_of_week"] = daily_full.index.dayofweek
        daily_full["day_of_month"] = daily_full.index.day
        daily_full["is_weekend"] = daily_full["day_of_week"].isin([5, 6]).astype(int)
        return daily_full

    def create_lag_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Builds lag and rolling window features strictly from past days.
        """
        df_feat = df_daily.copy()
        for lag in [1, 2, 7, 14, 28]:
            df_feat[f"flow_lag_{lag}"] = df_feat["net_cash_flow"].shift(lag)
            df_feat[f"bal_lag_{lag}"] = df_feat["balance"].shift(lag)

        # Rolling means and stds on lagged flow
        df_feat["flow_roll_mean_7"] = df_feat["net_cash_flow"].shift(1).rolling(7).mean()
        df_feat["flow_roll_std_7"] = df_feat["net_cash_flow"].shift(1).rolling(7).std()
        df_feat["bal_roll_mean_7"] = df_feat["balance"].shift(1).rolling(7).mean()

        return df_feat.dropna()

    def train_and_evaluate(self, df_daily: pd.DataFrame) -> Dict[str, Any]:
        """
        Executes chronological holdout validation:
        Compares:
        1. Naive Baseline (Last Value)
        2. Seasonal Naive (7-day lag)
        3. 7-Day Moving Average
        4. ML Sequential Regressor (Gradient Boosting)
        """
        df_lags = self.create_lag_features(df_daily)
        if len(df_lags) <= self.test_days + 14:
            # Fallback if time series is short: use last 20%
            split_idx = int(len(df_lags) * 0.8)
        else:
            split_idx = len(df_lags) - self.test_days

        train_data = df_lags.iloc[:split_idx].copy()
        test_data = df_lags.iloc[split_idx:].copy()

        self.feature_cols = [c for c in df_lags.columns if c.startswith("flow_lag_") or c.startswith("bal_lag_") or "roll_" in c or c in ["day_of_week", "day_of_month", "is_weekend"]]
        
        X_train = train_data[self.feature_cols].values
        y_train = train_data["balance"].values
        X_test = test_data[self.feature_cols].values
        y_test = test_data["balance"].values

        # 1. Naive baseline: y_pred = bal_lag_1
        pred_naive = test_data["bal_lag_1"].values

        # 2. Seasonal Naive baseline: y_pred = bal_lag_7
        pred_seasonal = test_data["bal_lag_7"].values

        # 3. Moving Average baseline: bal_roll_mean_7
        pred_ma = test_data["bal_roll_mean_7"].values

        # 4. ML Regressor
        self.ml_model.fit(X_train, y_train)
        pred_ml = self.ml_model.predict(X_test)
        self.is_fitted = True

        # Compute Metrics
        def compute_metrics(y_true, y_pred):
            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            return {"mae": round(mae, 2), "rmse": round(rmse, 2)}

        metrics = {
            "naive": compute_metrics(y_test, pred_naive),
            "seasonal_naive": compute_metrics(y_test, pred_seasonal),
            "moving_average_7d": compute_metrics(y_test, pred_ma),
            "gradient_boosting": compute_metrics(y_test, pred_ml),
        }

        # Residuals
        residuals_ml = (y_test - pred_ml).tolist()
        residual_std = float(np.std(residuals_ml))

        # Risk Analysis: check dates where predicted balance < low_balance_threshold
        dates = [str(d.date()) for d in test_data.index]
        low_balance_alerts = []
        for dt, p_bal in zip(dates, pred_ml):
            if p_bal < self.low_balance_threshold:
                low_balance_alerts.append({"date": dt, "projected_balance": round(p_bal, 2)})

        results = {
            "metrics": metrics,
            "dates": dates,
            "actual_balance": [round(float(v), 2) for v in y_test],
            "pred_ml": [round(float(v), 2) for v in pred_ml],
            "pred_naive": [round(float(v), 2) for v in pred_naive],
            "pred_seasonal": [round(float(v), 2) for v in pred_seasonal],
            "upper_bound": [round(float(v + 1.96 * residual_std), 2) for v in pred_ml],
            "lower_bound": [round(float(v - 1.96 * residual_std), 2) for v in pred_ml],
            "residuals": [round(float(r), 2) for r in residuals_ml],
            "low_balance_alerts": low_balance_alerts,
            "threshold": self.low_balance_threshold,
        }
        return results

    def save(self, filepath: str = "models/forecaster.joblib"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str = "models/forecaster.joblib") -> "CashFlowForecaster":
        return joblib.load(filepath)
