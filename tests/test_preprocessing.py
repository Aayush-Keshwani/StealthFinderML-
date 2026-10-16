"""
Unit tests for preprocessing and text sanitization.
"""

import pandas as pd
from stealth_spend.preprocessing import Preprocessor


def test_clean_text():
    prep = Preprocessor()
    assert prep.clean_text("POS DEBIT UBER *TRIP #1234") == "uber trip 1234"
    assert prep.clean_text("ACH DEBIT REALTY APARTMENT RENT") == "realty apartment rent"


def test_harmonize_amounts():
    prep = Preprocessor()
    df = pd.DataFrame({
        "transaction_type": ["debit", "credit"],
        "amount": [50.0, -100.0],
    })
    harmonized = prep.harmonize_amounts(df)
    assert harmonized["amount"].iloc[0] == -50.0
    assert harmonized["amount"].iloc[1] == 100.0
