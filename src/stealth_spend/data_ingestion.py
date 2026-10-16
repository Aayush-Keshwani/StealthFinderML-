"""
Data ingestion and synthetic data generation for Stealth-Spend.
Normalizes arbitrary bank statements into the canonical transaction schema.
"""

import uuid
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple


CANONICAL_COLUMNS = [
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
    "source_file",
    "is_recurring_candidate",
    "is_duplicate_candidate",
    "anomaly_score",
    "data_split",
]


class SyntheticDataGenerator:
    """
    Generates realistic, deterministic synthetic bank statement datasets
    with ground-truth categories, recurring patterns, and injected anomalies.
    """

    def __init__(self, seed: int = 42, base_balance: float = 5000.0):
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.base_balance = base_balance

    def generate(self, start_date: str = "2024-01-01", days: int = 365) -> pd.DataFrame:
        start_dt = pd.to_datetime(start_date)
        records = []
        current_balance = self.base_balance

        merchants_data = {
            "Groceries": [
                ("Costco Supermarket", -120.0, 30.0),
                ("Whole Foods Organic", -75.0, 20.0),
                ("Trader Joe's", -45.0, 15.0),
                ("Local Bakery & Deli", -15.0, 5.0),
            ],
            "Dining": [
                ("Starbucks Coffee", -6.50, 2.0),
                ("Chipotle Mexican Grill", -16.0, 4.0),
                ("Local Italian Bistro", -65.0, 18.0),
                ("Uber Eats Food Delivery", -32.0, 10.0),
            ],
            "Shopping": [
                ("Amazon.com Marketplace", -55.0, 35.0),
                ("Target Retail Store", -40.0, 20.0),
                ("Best Buy Electronics", -180.0, 80.0),
                ("Nike Sports Store", -95.0, 25.0),
            ],
            "Transport": [
                ("Uber *Trip Rideshare", -24.0, 8.0),
                ("Chevron Gas Station", -48.0, 12.0),
                ("Metropolitan Transit Metro", -3.50, 0.5),
                ("Shell Oil Station", -52.0, 10.0),
            ],
            "Utilities": [
                ("ConEd Electric & Gas", -95.0, 15.0),
                ("City Water Utility", -45.0, 5.0),
                ("Comcast Highspeed Internet", -79.99, 0.0),
            ],
            "Health": [
                ("CVS Health Pharmacy", -28.0, 12.0),
                ("City Dental Care", -150.0, 40.0),
                ("Quest Diagnostics Lab", -85.0, 20.0),
            ],
            "Entertainment": [
                ("AMC Movie Theatres", -34.0, 10.0),
                ("Steam Games Digital", -29.99, 15.0),
                ("Spotify Music Streaming", -11.99, 0.0),
            ],
            "Subscriptions": [
                ("Netflix Streaming Inc", -19.99, 0.0),
                ("Amazon Prime Membership", -14.99, 0.0),
                ("Apple iCloud Storage", -2.99, 0.0),
                ("Gym Fitness Monthly", -49.99, 0.0),
            ],
        }

        # Daily simulation
        date_range = [start_dt + datetime.timedelta(days=i) for i in range(days)]

        for current_day in date_range:
            day_of_month = current_day.day
            day_of_week = current_day.weekday()
            
            # 1. Bi-weekly Salary on 1st and 15th
            if day_of_month in [1, 15]:
                salary_amt = 3250.0 + round(self.rng.normal(0, 50), 2)
                current_balance += salary_amt
                records.append({
                    "transaction_id": str(uuid.uuid4()).replace("-", ""),
                    "transaction_date": current_day + datetime.timedelta(hours=9, minutes=self.rng.randint(0, 59)),
                    "description_raw": "DIRECT DEPOSIT ACME CORP PAYROLL",
                    "description_clean": "acme corp payroll",
                    "amount": round(salary_amt, 2),
                    "transaction_type": "credit",
                    "merchant": "Acme Corp",
                    "category": "Income",
                    "currency": "USD",
                    "balance": round(current_balance, 2),
                    "source_file": "synthetic_generator",
                    "is_recurring_candidate": True,
                    "is_duplicate_candidate": False,
                    "anomaly_score": 0.0,
                    "data_split": "train" if current_day < start_dt + datetime.timedelta(days=int(days * 0.75)) else "test",
                })

            # 2. Monthly Rent / Housing on 1st
            if day_of_month == 1:
                rent_amt = -1650.0
                current_balance += rent_amt
                records.append({
                    "transaction_id": str(uuid.uuid4()).replace("-", ""),
                    "transaction_date": current_day + datetime.timedelta(hours=10),
                    "description_raw": "ACH WITHDRAWAL REALTY APARTMENT RENT",
                    "description_clean": "realty apartment rent",
                    "amount": rent_amt,
                    "transaction_type": "debit",
                    "merchant": "Realty Partners",
                    "category": "Housing",
                    "currency": "USD",
                    "balance": round(current_balance, 2),
                    "source_file": "synthetic_generator",
                    "is_recurring_candidate": True,
                    "is_duplicate_candidate": False,
                    "anomaly_score": 0.0,
                    "data_split": "train" if current_day < start_dt + datetime.timedelta(days=int(days * 0.75)) else "test",
                })

            # 3. Monthly Utilities and Subscriptions around specific days
            if day_of_month == 5:
                # Electric
                amt = -95.0 + round(self.rng.normal(0, 10), 2)
                current_balance += amt
                records.append({
                    "transaction_id": str(uuid.uuid4()).replace("-", ""),
                    "transaction_date": current_day + datetime.timedelta(hours=11),
                    "description_raw": "BILL PAYMENT CONED ELECTRIC GAS",
                    "description_clean": "coned electric gas",
                    "amount": round(amt, 2),
                    "transaction_type": "debit",
                    "merchant": "ConEd Electric & Gas",
                    "category": "Utilities",
                    "currency": "USD",
                    "balance": round(current_balance, 2),
                    "source_file": "synthetic_generator",
                    "is_recurring_candidate": True,
                    "is_duplicate_candidate": False,
                    "anomaly_score": 0.0,
                    "data_split": "train" if current_day < start_dt + datetime.timedelta(days=int(days * 0.75)) else "test",
                })

            if day_of_month == 12:
                # Netflix
                # Injected anomaly: Price increase in month 8
                netflix_amt = -25.99 if current_day.month >= 8 else -19.99
                current_balance += netflix_amt
                is_anomaly = current_day.month == 8
                records.append({
                    "transaction_id": str(uuid.uuid4()).replace("-", ""),
                    "transaction_date": current_day + datetime.timedelta(hours=8),
                    "description_raw": "NETFLIX.COM DIGITAL SUBSCRIPTION",
                    "description_clean": "netflix com digital subscription",
                    "amount": netflix_amt,
                    "transaction_type": "debit",
                    "merchant": "Netflix Streaming Inc",
                    "category": "Subscriptions",
                    "currency": "USD",
                    "balance": round(current_balance, 2),
                    "source_file": "synthetic_generator",
                    "is_recurring_candidate": True,
                    "is_duplicate_candidate": False,
                    "anomaly_score": 0.85 if is_anomaly else 0.05,
                    "data_split": "train" if current_day < start_dt + datetime.timedelta(days=int(days * 0.75)) else "test",
                })

            # 4. Daily discretionary transactions (0 to 4 per day)
            num_tx = self.rng.poisson(lam=1.8)
            cat_keys = list(merchants_data.keys())

            for _ in range(num_tx):
                cat = self.rng.choice(cat_keys)
                options = merchants_data[cat]
                m_info = options[self.rng.randint(len(options))]
                m_name, m_mean, m_std = m_info
                tx_amt = round(self.rng.normal(m_mean, m_std), 2)
                if tx_amt >= 0:
                    tx_amt = -5.00
                current_balance += tx_amt
                
                # Check for rare synthetic anomaly injection
                is_anomaly = False
                raw_desc = f"PURCHASE {m_name.upper()} STORE #{self.rng.randint(100, 999)}"
                if self.rng.rand() < 0.015:
                    # Injected large outlier anomaly
                    tx_amt = round(tx_amt * 8.5, 2)
                    is_anomaly = True
                    raw_desc += " UNUSUAL PEAK"

                records.append({
                    "transaction_id": str(uuid.uuid4()).replace("-", ""),
                    "transaction_date": current_day + datetime.timedelta(hours=self.rng.randint(8, 22), minutes=self.rng.randint(0, 59)),
                    "description_raw": raw_desc,
                    "description_clean": m_name.lower(),
                    "amount": tx_amt,
                    "transaction_type": "debit",
                    "merchant": m_name,
                    "category": cat,
                    "currency": "USD",
                    "balance": round(current_balance, 2),
                    "source_file": "synthetic_generator",
                    "is_recurring_candidate": False,
                    "is_duplicate_candidate": False,
                    "anomaly_score": 0.90 if is_anomaly else 0.02,
                    "data_split": "train" if current_day < start_dt + datetime.timedelta(days=int(days * 0.75)) else "test",
                })

                # Injected duplicate anomaly: same charge repeated 15 minutes later
                if self.rng.rand() < 0.008:
                    dup_balance = round(current_balance + tx_amt, 2)
                    records.append({
                        "transaction_id": str(uuid.uuid4()).replace("-", ""),
                        "transaction_date": current_day + datetime.timedelta(hours=self.rng.randint(8, 22), minutes=self.rng.randint(0, 59)),
                        "description_raw": raw_desc + " DUPLICATE",
                        "description_clean": m_name.lower(),
                        "amount": tx_amt,
                        "transaction_type": "debit",
                        "merchant": m_name,
                        "category": cat,
                        "currency": "USD",
                        "balance": dup_balance,
                        "source_file": "synthetic_generator",
                        "is_recurring_candidate": False,
                        "is_duplicate_candidate": True,
                        "anomaly_score": 0.95,
                        "data_split": "train" if current_day < start_dt + datetime.timedelta(days=int(days * 0.75)) else "test",
                    })

        df = pd.DataFrame(records)
        df.sort_values(by="transaction_date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df


def normalize_to_canonical(
    df: pd.DataFrame,
    source_filename: str = "statement.csv",
    column_mapping: Optional[Dict[str, str]] = None,
    default_currency: str = "USD"
) -> pd.DataFrame:
    """
    Transforms any ingested bank statement DataFrame into the canonical schema.
    Applies column mappings and standardizes fields.
    """
    mapped = pd.DataFrame()
    col_map = column_mapping or {}

    # Handle standard columns from transactions.csv if available
    # Check if this matches transactions.csv schema
    has_amount = "amount" in df.columns or "TransactionAmount (INR)" in df.columns or col_map.get("amount") in df.columns

    # 1. transaction_id
    if "transaction_id" in df.columns:
        mapped["transaction_id"] = df["transaction_id"].astype(str)
    elif "TransactionID" in df.columns:
        mapped["transaction_id"] = df["TransactionID"].astype(str)
    elif "id" in df.columns:
        mapped["transaction_id"] = df["id"].astype(str)
    else:
        mapped["transaction_id"] = [str(uuid.uuid4()).replace("-", "") for _ in range(len(df))]

    # 2. transaction_date
    if "timestamp" in df.columns:
        mapped["transaction_date"] = pd.to_datetime(df["timestamp"], errors="coerce")
    elif "transaction_date" in df.columns:
        mapped["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    elif "TransactionDate" in df.columns:
        # Check if TransactionTime exists as in bank_transactions.csv
        d_parsed = pd.to_datetime(df["TransactionDate"], format="%d/%m/%y", errors="coerce")
        if "TransactionTime" in df.columns:
            t_num = pd.to_numeric(df["TransactionTime"], errors="coerce").fillna(0).astype(int)
            h = t_num // 10000
            m = (t_num % 10000) // 100
            s = t_num % 100
            mapped["transaction_date"] = d_parsed + pd.to_timedelta(h, unit="h") + pd.to_timedelta(m, unit="m") + pd.to_timedelta(s, unit="s")
        else:
            mapped["transaction_date"] = d_parsed
    elif "date" in df.columns:
        mapped["transaction_date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        mapped["transaction_date"] = pd.Timestamp.now()

    # 5. amount (convention: debit negative, credit positive)
    if "amount" in df.columns:
        mapped["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    elif "TransactionAmount (INR)" in df.columns:
        # Standard bank debit amount
        mapped["amount"] = -pd.to_numeric(df["TransactionAmount (INR)"], errors="coerce").fillna(0.0).abs()
    elif "debit" in df.columns and "credit" in df.columns:
        deb = pd.to_numeric(df["debit"], errors="coerce").fillna(0.0)
        cred = pd.to_numeric(df["credit"], errors="coerce").fillna(0.0)
        mapped["amount"] = cred - deb
    else:
        mapped["amount"] = 0.0

    # 6. transaction_type
    if "transaction_type" in df.columns:
        mapped["transaction_type"] = df["transaction_type"].astype(str)
    else:
        mapped["transaction_type"] = np.where(mapped["amount"] >= 0, "credit", "debit")

    # 8. category & merchant & description
    if "category" in df.columns:
        mapped["category"] = df["category"].astype(str).str.title()
    elif "merchant_category" in df.columns:
        mapped["category"] = df["merchant_category"].astype(str).str.title()
    elif "TransactionAmount (INR)" in df.columns:
        abs_amt = mapped["amount"].abs()
        conditions = [
            (abs_amt <= 250),
            (abs_amt > 250) & (abs_amt <= 800),
            (abs_amt > 800) & (abs_amt <= 2500),
            (abs_amt > 2500) & (abs_amt <= 7500),
            (abs_amt > 7500) & (abs_amt <= 25000),
            (abs_amt > 25000),
        ]
        choices = [
            "Transport",
            "Dining",
            "Groceries",
            "Utilities",
            "Shopping",
            "Housing",
        ]
        mapped["category"] = np.select(conditions, choices, default="Shopping")
    else:
        mapped["category"] = "Other"

    # 3. description_raw & 7. merchant
    if "label" in df.columns and "merchant_name" in df.columns:
        mapped["description_raw"] = df["label"].fillna(df["merchant_name"]).astype(str)
        mapped["merchant"] = df["merchant_name"].fillna("Unknown Merchant").astype(str)
    elif "merchant_name" in df.columns:
        mapped["description_raw"] = df["merchant_name"].astype(str)
        mapped["merchant"] = df["merchant_name"].astype(str)
    elif "CustLocation" in df.columns:
        loc = df["CustLocation"].fillna("Central").astype(str).str.title()
        prefix_map = {
            "Transport": "Uber Rideshare ",
            "Dining": "Zomato Food Dining ",
            "Groceries": "Reliance Fresh Supermarket ",
            "Utilities": "Airtel Telecommunications ",
            "Shopping": "Amazon Retail Shopping ",
            "Housing": "Realty Monthly Rental ",
            "Other": "General Retail ",
        }
        prefixes = mapped["category"].map(prefix_map).fillna("Store ")
        mapped["description_raw"] = prefixes + loc + " #" + mapped["transaction_id"].str[-4:]
        mapped["merchant"] = prefixes.str.split().str[0] + " " + loc
    elif "label" in df.columns:
        mapped["description_raw"] = df["label"].astype(str)
        mapped["merchant"] = mapped["description_raw"].str.split().str[:2].str.join(" ").str.title()
    else:
        mapped["description_raw"] = "TRANSACTION"
        mapped["merchant"] = "Unknown Merchant"

    # 4. description_clean
    mapped["description_clean"] = (
        mapped["description_raw"]
        .str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Map synonymous or lowercase categories to canonical taxonomy
    category_map = {
        "Groceries": "Groceries",
        "Shopping": "Shopping",
        "Dining": "Dining",
        "Finance": "Housing",  # loans/mortgages
        "Housing": "Housing",
        "Health": "Health",
        "Travel": "Travel",
        "Telecom": "Utilities",
        "Utilities": "Utilities",
        "Transport": "Transport",
        "Subscriptions": "Subscriptions",
        "Salary": "Income",
        "Income": "Income",
        "Transfer": "Transfers",
        "Transfers": "Transfers",
        "Entertainment": "Entertainment",
        "Education": "Education",
        "Cash Withdrawal": "Cash Withdrawal",
    }
    mapped["category"] = mapped["category"].map(lambda c: category_map.get(c, c))

    # 9. currency
    if "currency" in df.columns:
        mapped["currency"] = df["currency"].fillna(default_currency).astype(str)
    elif "TransactionAmount (INR)" in df.columns:
        mapped["currency"] = "INR"
    else:
        mapped["currency"] = default_currency

    # 10. balance
    if "balance_after_transaction" in df.columns:
        mapped["balance"] = pd.to_numeric(df["balance_after_transaction"], errors="coerce")
    elif "CustAccountBalance" in df.columns:
        mapped["balance"] = pd.to_numeric(df["CustAccountBalance"], errors="coerce")
    elif "balance" in df.columns:
        mapped["balance"] = pd.to_numeric(df["balance"], errors="coerce")
    else:
        # Reconstruct balance cumulatively
        mapped["balance"] = 5000.0 + mapped["amount"].cumsum()

    # 11. source_file
    mapped["source_file"] = source_filename

    # 12. is_recurring_candidate
    if "is_subscription" in df.columns:
        mapped["is_recurring_candidate"] = df["is_subscription"].fillna(False).astype(bool)
    elif "is_recurring_candidate" in df.columns:
        mapped["is_recurring_candidate"] = df["is_recurring_candidate"].astype(bool)
    else:
        mapped["is_recurring_candidate"] = False

    # 13. is_duplicate_candidate
    if "is_duplicate_candidate" in df.columns:
        mapped["is_duplicate_candidate"] = df["is_duplicate_candidate"].astype(bool)
    else:
        mapped["is_duplicate_candidate"] = False

    # 14. anomaly_score
    if "anomaly_score" in df.columns:
        mapped["anomaly_score"] = df["anomaly_score"].astype(float)
    else:
        mapped["anomaly_score"] = 0.0

    # 15. data_split
    if "data_split" in df.columns:
        mapped["data_split"] = df["data_split"].astype(str)
    else:
        # 80/20 time-based split
        mapped.sort_values(by="transaction_date", inplace=True)
        split_idx = int(len(mapped) * 0.8)
        splits = ["train"] * split_idx + ["test"] * (len(mapped) - split_idx)
        mapped["data_split"] = splits

    # Ensure canonical column order
    return mapped[CANONICAL_COLUMNS].copy()
