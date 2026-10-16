"""
Data preprocessing pipeline for Stealth-Spend.
Cleans raw text, harmonizes amounts and types, and prevents data leakage across splits.
"""

import re
import pandas as pd
import numpy as np
from typing import Tuple


class Preprocessor:
    """
    Cleans raw transaction fields, standardizes text tokens,
    and applies time-aware data splits to prevent train/test leakage.
    """

    def __init__(self, amount_convention: str = "negative_debit"):
        self.amount_convention = amount_convention

    def clean_text(self, text: str) -> str:
        """
        Cleans transaction text:
        - Converts to lower case
        - Normalizes whitespace
        - Strips punctuation while keeping meaningful alphanumeric tokens
        """
        if not isinstance(text, str):
            return "unknown"
        # Remove noisy transaction prefixes often seen in statements
        text = re.sub(r"^(pos debit|ach debit|card purchase|check withdrawal|online transfer)\s*", "", text, flags=re.IGNORECASE)
        # Normalize punctuation to spaces
        text = re.sub(r"[^\w\s]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text if text else "unknown"

    def harmonize_amounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures amount sign convention:
        Debits are negative (<0) and Credits are positive (>0).
        """
        df = df.copy()
        is_debit = df["transaction_type"].str.lower().str.contains("debit|purchase|withdrawal|payment", na=False)
        is_credit = df["transaction_type"].str.lower().str.contains("credit|salary|deposit|refund", na=False)

        # Force debit to negative
        df.loc[is_debit, "amount"] = -df.loc[is_debit, "amount"].abs()
        # Force credit to positive
        df.loc[is_credit, "amount"] = df.loc[is_credit, "amount"].abs()

        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs complete preprocessing pipeline on canonical DataFrame.
        """
        cleaned = df.copy()
        cleaned["description_clean"] = cleaned["description_raw"].apply(self.clean_text)
        cleaned = self.harmonize_amounts(cleaned)
        
        # Ensure timestamp sorting
        cleaned["transaction_date"] = pd.to_datetime(cleaned["transaction_date"], errors="coerce")
        cleaned.sort_values(by="transaction_date", inplace=True)
        cleaned.reset_index(drop=True, inplace=True)
        return cleaned

    def train_test_split(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.25,
        time_split: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data cleanly without future leakage.
        """
        df_sorted = df.sort_values(by="transaction_date").reset_index(drop=True)
        if time_split:
            split_idx = int(len(df_sorted) * (1.0 - test_ratio))
            train_df = df_sorted.iloc[:split_idx].copy()
            test_df = df_sorted.iloc[split_idx:].copy()
        else:
            # Random split with fixed seed
            mask = np.random.RandomState(42).rand(len(df_sorted)) >= test_ratio
            train_df = df_sorted[mask].copy()
            test_df = df_sorted[~mask].copy()
            
        train_df["data_split"] = "train"
        test_df["data_split"] = "test"
        return train_df, test_df
