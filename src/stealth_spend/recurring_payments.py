"""
Recurring payment intelligence and duplicate charge detector for Stealth-Spend.
Identifies subscriptions, unexpected price hikes, and duplicate billing candidates.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple


class RecurringPaymentTracker:
    """
    Identifies recurring subscriptions/bills and flags unexpected amount increases.
    """

    def __init__(self, price_increase_threshold: float = 0.10):
        self.price_increase_threshold = price_increase_threshold

    def detect_recurring_subscriptions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Groups transactions by merchant and analyzes interval cadence to detect recurring charges.
        """
        df_sorted = df.sort_values(by="transaction_date").copy()
        candidates = []

        for merchant, group in df_sorted.groupby("merchant"):
            # Only consider debits/expenses
            debits = group[group["amount"] < 0].copy()
            if len(debits) < 2:
                continue

            debits["time_diff_days"] = debits["transaction_date"].diff().dt.total_seconds() / 86400.0
            diffs = debits["time_diff_days"].dropna()

            if len(diffs) == 0:
                continue

            median_interval = float(diffs.median())
            std_interval = float(diffs.std()) if len(diffs) > 1 else 0.0

            # Check if cadence looks monthly (25-35 days) or weekly (6-8 days)
            is_monthly = (25.0 <= median_interval <= 35.0) and (std_interval <= 5.0)
            is_weekly = (6.0 <= median_interval <= 8.0) and (std_interval <= 2.0)

            if is_monthly or is_weekly or debits["is_recurring_candidate"].any():
                cadence = "Monthly" if is_monthly else ("Weekly" if is_weekly else "Periodic")
                amounts = debits["amount"].abs().values
                last_amt = float(amounts[-1])
                prev_amt = float(amounts[-2]) if len(amounts) > 1 else last_amt

                pct_change = (last_amt - prev_amt) / prev_amt if prev_amt > 0 else 0.0
                is_price_hike = pct_change > self.price_increase_threshold

                candidates.append({
                    "merchant": merchant,
                    "category": debits["category"].iloc[-1],
                    "cadence": cadence,
                    "frequency_count": len(debits),
                    "median_interval_days": round(median_interval, 1),
                    "last_charge_date": debits["transaction_date"].iloc[-1],
                    "last_amount": round(last_amt, 2),
                    "previous_amount": round(prev_amt, 2),
                    "amount_change_pct": round(pct_change * 100, 1),
                    "is_price_increase": is_price_hike,
                    "expected_next_charge": debits["transaction_date"].iloc[-1] + pd.Timedelta(days=int(median_interval)),
                    "confidence": round(min(1.0, len(debits) / 5.0), 2),
                })

        return pd.DataFrame(candidates)


class DuplicateChargeDetector:
    """
    Identifies candidate duplicate transactions within short time intervals.
    """

    def __init__(self, time_window_hours: float = 48.0, amount_tolerance: float = 0.01):
        self.time_window_hours = time_window_hours
        self.amount_tolerance = amount_tolerance

    def detect_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scans pairs of transactions for identical amounts and matching merchants
        occurring within the designated time window.
        """
        df_sorted = df.sort_values(by="transaction_date").copy()
        pairs = []

        for merchant, group in df_sorted.groupby("merchant"):
            if len(group) < 2:
                continue

            rows = group.to_dict("records")
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    r1 = rows[i]
                    r2 = rows[j]

                    time_diff = (r2["transaction_date"] - r1["transaction_date"]).total_seconds() / 3600.0
                    if time_diff > self.time_window_hours:
                        break  # Group is sorted by date, no need to check further

                    amt_diff = abs(abs(r1["amount"]) - abs(r2["amount"]))
                    if amt_diff <= self.amount_tolerance and r1["amount"] < 0:
                        pairs.append({
                            "merchant": merchant,
                            "amount": round(abs(r1["amount"]), 2),
                            "date_first": r1["transaction_date"],
                            "date_second": r2["transaction_date"],
                            "hours_apart": round(time_diff, 2),
                            "tx1_id": r1["transaction_id"],
                            "tx2_id": r2["transaction_id"],
                            "category": r1["category"],
                            "risk_reason": f"Identical charge of ${abs(r1['amount']):.2f} repeated within {time_diff:.1f} hours",
                        })

        return pd.DataFrame(pairs)
