"""
Unit tests for data ingestion and canonical mapping.
"""

import pytest
import pandas as pd
from stealth_spend.data_ingestion import SyntheticDataGenerator, normalize_to_canonical, CANONICAL_COLUMNS


def test_synthetic_data_generation():
    gen = SyntheticDataGenerator(seed=123)
    df = gen.generate(days=30)
    assert len(df) > 0
    for col in CANONICAL_COLUMNS:
        assert col in df.columns
    assert (df["amount"] != 0).all()


def test_normalize_to_canonical():
    raw_df = pd.DataFrame({
        "timestamp": ["2024-01-01 10:00:00", "2024-01-02 11:00:00"],
        "amount": [-50.0, 1000.0],
        "label": ["COSTCO STORE", "PAYROLL DEPOSIT"],
        "category": ["groceries", "salary"],
        "currency": ["USD", "USD"],
    })
    canonical = normalize_to_canonical(raw_df, source_filename="test.csv")
    assert len(canonical) == 2
    assert list(canonical.columns) == CANONICAL_COLUMNS
    assert canonical["category"].iloc[0] == "Groceries"
    assert canonical["category"].iloc[1] == "Income"
