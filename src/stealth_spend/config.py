"""
Configuration loader for Stealth-Spend.
"""

from pathlib import Path
from typing import Any, Dict
import yaml

DEFAULT_SETTINGS = {
    "project": {"name": "Stealth-Spend", "version": "1.0.0", "random_seed": 42},
    "paths": {
        "raw_data": "data/raw",
        "interim_data": "data/interim",
        "processed_data": "data/processed",
        "synthetic_data": "data/synthetic",
        "models": "models",
        "reports": "reports",
        "figures": "reports/figures",
        "generated_reports": "reports/generated",
    },
    "ingestion": {
        "amount_convention": "negative_debit",
        "date_format": "ISO",
        "default_currency": "USD",
    },
    "categorization": {
        "test_size": 0.25,
        "min_confidence_threshold": 0.40,
        "model_type": "xgboost",
        "max_features_tfidf": 1000,
        "ngram_range": [1, 2],
    },
    "anomaly_detection": {
        "contamination": 0.05,
        "near_duplicate_window_days": 3,
        "amount_deviation_threshold_zscore": 2.5,
        "recurring_increase_pct_threshold": 0.15,
        "top_k_review": 20,
    },
    "clustering": {
        "min_k": 2,
        "max_k": 6,
        "optimal_k": 3,
    },
    "forecasting": {
        "test_days": 30,
        "lag_days": [1, 7, 14, 28],
        "rolling_windows": [7, 14, 30],
        "low_balance_threshold": 500.0,
    },
}

DEFAULT_CATEGORIES = {
    "categories": [
        "Income", "Housing", "Utilities", "Groceries", "Dining",
        "Transport", "Shopping", "Health", "Education", "Entertainment",
        "Travel", "Subscriptions", "Fees", "Transfers", "Cash Withdrawal",
        "Refund", "Other", "Needs Review"
    ],
    "essential_categories": ["Housing", "Utilities", "Groceries", "Health", "Transport"],
    "discretionary_categories": ["Dining", "Shopping", "Entertainment", "Travel", "Subscriptions"],
    "non_expense_categories": ["Income", "Transfers", "Refund"],
}


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Safely loads a YAML file if it exists, otherwise returns empty dict."""
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_config(config_path: str = "config/settings.yml") -> Dict[str, Any]:
    """Loads application settings merged with defaults."""
    path = Path(config_path)
    user_config = load_yaml(path)
    cfg = DEFAULT_SETTINGS.copy()
    for k, v in user_config.items():
        if isinstance(v, dict) and k in cfg:
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def load_categories_config(categories_path: str = "config/categories.yml") -> Dict[str, Any]:
    """Loads categories taxonomy configuration."""
    path = Path(categories_path)
    user_cat = load_yaml(path)
    cfg = DEFAULT_CATEGORIES.copy()
    for k, v in user_cat.items():
        cfg[k] = v
    return cfg
