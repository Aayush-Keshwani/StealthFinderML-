"""
Unit tests for anomaly detection and recurring payment intelligence.
"""

import pandas as pd
from stealth_spend.anomaly_detection import AnomalyDetector
from stealth_spend.recurring_payments import DuplicateChargeDetector
from stealth_spend.data_ingestion import SyntheticDataGenerator


def test_duplicate_detector():
    detector = DuplicateChargeDetector(time_window_hours=48.0)
    df = pd.DataFrame({
        "transaction_id": ["tx1", "tx2"],
        "transaction_date": pd.to_datetime(["2024-01-01 12:00:00", "2024-01-01 12:15:00"]),
        "merchant": ["Target", "Target"],
        "amount": [-45.50, -45.50],
        "category": ["Shopping", "Shopping"],
    })
    dups = detector.detect_duplicates(df)
    assert len(dups) == 1
    assert dups["hours_apart"].iloc[0] == 0.25


def test_anomaly_scoring():
    gen = SyntheticDataGenerator(seed=42)
    df = gen.generate(days=60)
    ad = AnomalyDetector(contamination=0.1)
    ad.fit(df)
    scored = ad.score_transactions(df)
    assert "anomaly_score" in scored.columns
    assert "anomaly_reason" in scored.columns
    assert scored["anomaly_score"].between(0.0, 1.0).all()
