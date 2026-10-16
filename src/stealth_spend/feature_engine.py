"""
Feature engineering pipeline for Stealth-Spend.
Extracts text TF-IDF n-grams, calendar characteristics, trailing merchant statistics,
and recurrence metrics without temporal data leakage.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class FeatureEngine:
    """
    Transforms canonical transactions into model-ready features.
    Ensures vectorizers and scalers are fitted strictly on training subsets.
    """

    def __init__(
        self,
        max_tfidf_features: int = 500,
        ngram_range: Tuple[int, int] = (1, 2),
    ):
        self.max_tfidf_features = max_tfidf_features
        self.ngram_range = ngram_range
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_tfidf_features,
            ngram_range=self.ngram_range,
            stop_words="english",
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.merchant_history: Dict[str, Dict[str, float]] = {}

    def extract_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts temporal and calendar features.
        """
        dates = pd.to_datetime(df["transaction_date"])
        feats = pd.DataFrame(index=df.index)
        feats["day_of_week"] = dates.dt.dayofweek
        feats["day_of_month"] = dates.dt.day
        feats["month"] = dates.dt.month
        feats["is_weekend"] = dates.dt.dayofweek.isin([5, 6]).astype(int)
        return feats

    def extract_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts amount transforms and trailing statistics.
        """
        feats = pd.DataFrame(index=df.index)
        abs_amt = df["amount"].abs().astype(float)
        feats["amount"] = df["amount"].astype(float)
        feats["abs_amount"] = abs_amt
        feats["log_amount"] = np.log1p(abs_amt)
        feats["is_credit"] = (df["amount"] > 0).astype(int)
        return feats

    def compute_trailing_merchant_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes rolling merchant statistics historically to prevent forward leakage.
        """
        feats = pd.DataFrame(index=df.index)
        df_sorted = df.sort_values(by="transaction_date").copy()

        # Track historical amounts per merchant
        merchant_means = []
        merchant_stds = []
        days_since_last = []
        amount_zscores = []

        merchant_tracker: Dict[str, list] = {}
        merchant_last_time: Dict[str, pd.Timestamp] = {}

        for _, row in df_sorted.iterrows():
            m = str(row.get("merchant", "Unknown"))
            amt = abs(float(row.get("amount", 0.0)))
            dt = pd.to_datetime(row.get("transaction_date"))

            history = merchant_tracker.get(m, [])
            last_dt = merchant_last_time.get(m, None)

            if len(history) > 0:
                mean_val = float(np.mean(history))
                std_val = float(np.std(history)) if len(history) > 1 else 1.0
                zscore = abs(amt - mean_val) / (std_val + 1e-5)
            else:
                mean_val = amt
                std_val = 0.0
                zscore = 0.0

            days_diff = (dt - last_dt).total_seconds() / 86400.0 if last_dt is not None else 999.0

            merchant_means.append(mean_val)
            merchant_stds.append(std_val)
            amount_zscores.append(zscore)
            days_since_last.append(min(days_diff, 999.0))

            # Update history
            merchant_tracker.setdefault(m, []).append(amt)
            merchant_last_time[m] = dt

        df_sorted["merchant_hist_mean"] = merchant_means
        df_sorted["merchant_hist_std"] = merchant_stds
        df_sorted["merchant_amount_zscore"] = amount_zscores
        df_sorted["days_since_last_merchant_tx"] = days_since_last

        # Re-align with original DataFrame index
        return df_sorted.loc[df.index, [
            "merchant_hist_mean",
            "merchant_hist_std",
            "merchant_amount_zscore",
            "days_since_last_merchant_tx",
        ]]

    def fit(self, train_df: pd.DataFrame):
        """
        Fits text vectorizer on training transaction descriptions.
        """
        texts = train_df["description_clean"].fillna("").astype(str)
        self.tfidf_vectorizer.fit(texts)
        self.is_fitted = True

    def transform_text(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms transaction text to TF-IDF matrix.
        """
        texts = df["description_clean"].fillna("").astype(str)
        return self.tfidf_vectorizer.transform(texts).toarray()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts all tabular features.
        """
        cal_feats = self.extract_calendar_features(df)
        num_feats = self.extract_numeric_features(df)
        trailing_feats = self.compute_trailing_merchant_stats(df)
        
        combined = pd.concat([cal_feats, num_feats, trailing_feats], axis=1)
        return combined
