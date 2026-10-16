"""
Unit tests for forecasting and lag feature construction.
"""

import pandas as pd
from stealth_spend.forecasting import CashFlowForecaster
from stealth_spend.data_ingestion import SyntheticDataGenerator


def test_forecasting_pipeline():
    gen = SyntheticDataGenerator(seed=42)
    df = gen.generate(days=120)
    forecaster = CashFlowForecaster(test_days=20)
    daily_ts = forecaster.build_daily_time_series(df)
    assert len(daily_ts) >= 120

    res = forecaster.train_and_evaluate(daily_ts)
    assert "metrics" in res
    assert "naive" in res["metrics"]
    assert "gradient_boosting" in res["metrics"]
    assert res["metrics"]["gradient_boosting"]["mae"] >= 0
