"""
Stealth charge anomaly detection module.
Combines Isolation Forest with explainable heuristic signals (near duplicates,
subscription hikes, and robust merchant history deviations).
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from pathlib import Path
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from stealth_spend.recurring_payments import RecurringPaymentTracker, DuplicateChargeDetector


class AnomalyDetector:
    """
    Unsupervised financial anomaly detector combining Isolation Forest outlier detection
    with explainable contextual heuristics.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        zscore_threshold: float = 2.5,
        random_seed: int = 42,
    ):
        self.contamination = contamination
        self.zscore_threshold = zscore_threshold
        self.random_seed = random_seed
        
        self.scaler = RobustScaler()
        self.iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_seed,
            n_estimators=100,
        )
        self.recurring_tracker = RecurringPaymentTracker()
        self.duplicate_detector = DuplicateChargeDetector()
        self.historical_merchants: set = set()
        self.is_fitted = False

    def extract_anomaly_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extracts numeric signals for Isolation Forest:
        - Absolute amount
        - Log amount
        - Historical merchant amount zscore
        - Days since last merchant transaction
        """
        feats = pd.DataFrame(index=df.index)
        abs_amt = df["amount"].abs().values
        feats["abs_amount"] = abs_amt
        feats["log_amount"] = np.log1p(abs_amt)
        feats["merchant_amount_zscore"] = df.get("merchant_amount_zscore", pd.Series(0.0, index=df.index)).fillna(0.0)
        feats["days_since_last"] = df.get("days_since_last_merchant_tx", pd.Series(30.0, index=df.index)).fillna(30.0)
        return feats.values

    def fit(self, df: pd.DataFrame):
        """
        Fits the scaler, Isolation Forest, and records historical merchants.
        """
        X = self.extract_anomaly_features(df)
        X_scaled = self.scaler.fit_transform(X)
        self.iso_forest.fit(X_scaled)
        self.historical_merchants = set(df["merchant"].dropna().unique())
        self.is_fitted = True

    def score_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates raw Isolation Forest scores, combines with explainable rules,
        and generates plain-language reasons for flagged transactions.
        """
        if not self.is_fitted:
            self.fit(df)

        results = df.copy()
        X = self.extract_anomaly_features(results)
        X_scaled = self.scaler.transform(X)

        # Isolation forest scores: lower score = more anomalous
        # decision_function gives negative values for anomalies
        raw_iso_scores = self.iso_forest.decision_function(X_scaled)
        # Normalize into [0, 1] range where 1.0 is most anomalous
        iso_anomaly_norm = 1.0 / (1.0 + np.exp(raw_iso_scores * 5.0))
        results["iso_score"] = iso_anomaly_norm

        # 1. Duplicate check
        dup_df = self.duplicate_detector.detect_duplicates(results)
        dup_ids = set()
        if len(dup_df) > 0:
            dup_ids.update(dup_df["tx1_id"].tolist() + dup_df["tx2_id"].tolist())
        results["is_dup_flag"] = results["transaction_id"].isin(dup_ids)

        # 2. Subscription increase check
        rec_df = self.recurring_tracker.detect_recurring_subscriptions(results)
        price_hike_merchants = set()
        if len(rec_df) > 0:
            price_hike_merchants.update(rec_df[rec_df["is_price_increase"]]["merchant"].tolist())
        results["is_sub_hike"] = results["merchant"].isin(price_hike_merchants)

        # 3. New/rare merchant check
        results["is_new_merchant"] = ~results["merchant"].isin(self.historical_merchants)

        # 4. Extreme amount z-score
        z_scores = results.get("merchant_amount_zscore", pd.Series(0.0, index=results.index)).fillna(0.0)
        results["is_amount_outlier"] = z_scores > self.zscore_threshold

        # Composite Anomaly Score: weighted combination
        composite_score = (
            0.45 * results["iso_score"] +
            0.25 * results["is_dup_flag"].astype(float) +
            0.15 * results["is_sub_hike"].astype(float) +
            0.15 * results["is_amount_outlier"].astype(float)
        )
        results["anomaly_score"] = np.clip(composite_score, 0.0, 1.0)

        # Generate evidence and human-readable reason
        reasons = []
        for _, row in results.iterrows():
            r_list = []
            if row["is_dup_flag"]:
                r_list.append("Potential duplicate charge within short window")
            if row["is_sub_hike"]:
                r_list.append("Subscription amount increased from historical baseline")
            if row["is_amount_outlier"]:
                r_list.append(f"Unusually high amount for merchant (Z-score: {z_scores.loc[row.name]:.1f})")
            if row["iso_score"] > 0.70:
                r_list.append("Multivariate outlier flagged by Isolation Forest")
            if not r_list:
                r_list.append("Routine transaction")
            reasons.append("; ".join(r_list))

        results["anomaly_reason"] = reasons
        results["is_anomaly_review"] = (results["anomaly_score"] >= 0.40) | results["is_dup_flag"] | results["is_sub_hike"] | results["is_amount_outlier"]
        return results

    def get_top_alerts(self, df_scored: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
        """
        Returns top-K highest-ranked review transactions.
        """
        sorted_alerts = df_scored.sort_values(by="anomaly_score", ascending=False)
        return sorted_alerts.head(top_k)

    def save(self, filepath: str = "models/anomaly_detector.joblib"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str = "models/anomaly_detector.joblib") -> "AnomalyDetector":
        return joblib.load(filepath)
