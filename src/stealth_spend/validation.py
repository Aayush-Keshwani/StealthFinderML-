"""
Data validation and quality assessment for Stealth-Spend.
Audits incoming bank statements for missing fields, inconsistent amounts, and anomalies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


class DataValidator:
    """
    Validates canonical transaction DataFrames, generates data quality audit reports,
    and identifies problematic rows.
    """

    def __init__(self, required_columns: List[str] = None):
        self.required_columns = required_columns or [
            "transaction_id",
            "transaction_date",
            "description_raw",
            "description_clean",
            "amount",
            "transaction_type",
            "merchant",
            "category",
            "currency",
            "balance",
        ]

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs comprehensive data quality checks on the canonical DataFrame.
        """
        total_rows = len(df)
        if total_rows == 0:
            return {"status": "FAILED", "reason": "DataFrame is empty", "total_rows": 0}

        # 1. Missing columns
        missing_cols = [c for c in self.required_columns if c not in df.columns]

        # 2. Null value analysis
        null_counts = df[self.required_columns].isnull().sum().to_dict()
        null_pct = {k: round((v / total_rows) * 100, 2) for k, v in null_counts.items()}

        # 3. Duplicate transaction IDs
        duplicate_ids = int(df["transaction_id"].duplicated().sum()) if "transaction_id" in df.columns else 0

        # 4. Sign consistency
        # In our convention: debit should be negative, credit should be positive
        debit_rows = df[df["transaction_type"].str.lower().str.contains("debit", na=False)]
        positive_debits = int((debit_rows["amount"] > 0).sum()) if len(debit_rows) > 0 else 0

        credit_rows = df[df["transaction_type"].str.lower().str.contains("credit|salary", na=False)]
        negative_credits = int((credit_rows["amount"] < 0).sum()) if len(credit_rows) > 0 else 0

        # 5. Date validation
        invalid_dates = int(df["transaction_date"].isnull().sum())
        date_min = df["transaction_date"].min()
        date_max = df["transaction_date"].max()

        # 6. Currency consistency
        currencies = df["currency"].value_counts().to_dict() if "currency" in df.columns else {}

        # 7. Balance check
        balance_nulls = int(df["balance"].isnull().sum()) if "balance" in df.columns else total_rows

        report = {
            "status": "PASSED" if len(missing_cols) == 0 and duplicate_ids == 0 else "WARNING",
            "total_rows": total_rows,
            "missing_columns": missing_cols,
            "null_counts": null_counts,
            "null_percentages": null_pct,
            "duplicate_ids": duplicate_ids,
            "date_range": {
                "start": str(date_min),
                "end": str(date_max),
                "invalid_dates": invalid_dates,
            },
            "sign_consistency": {
                "positive_debits_flagged": positive_debits,
                "negative_credits_flagged": negative_credits,
            },
            "currencies": currencies,
            "balance_nulls": balance_nulls,
        }
        return report

    def get_quality_summary_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame summary suitable for plotting (fig01: Missing values & quality).
        """
        metrics = []
        for col in self.required_columns:
            if col in df.columns:
                null_c = df[col].isnull().sum()
                unique_c = df[col].nunique()
                valid_pct = round((1.0 - (null_c / len(df))) * 100, 1)
            else:
                null_c = len(df)
                unique_c = 0
                valid_pct = 0.0
            metrics.append({
                "Column": col,
                "Missing_Count": int(null_c),
                "Completeness_Pct": float(valid_pct),
                "Unique_Values": int(unique_c),
            })
        return pd.DataFrame(metrics)
