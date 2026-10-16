"""
Unit tests for feature engineering.
"""

import pandas as pd
from stealth_spend.feature_engine import FeatureEngine
from stealth_spend.data_ingestion import SyntheticDataGenerator


def test_feature_extraction():
    gen = SyntheticDataGenerator(seed=42)
    df = gen.generate(days=15)
    engine = FeatureEngine(max_tfidf_features=50)
    engine.fit(df)

    feats = engine.transform(df)
    assert "day_of_week" in feats.columns
    assert "log_amount" in feats.columns
    assert "merchant_amount_zscore" in feats.columns
    assert len(feats) == len(df)

    text_mat = engine.transform_text(df)
    assert text_mat.shape[0] == len(df)
