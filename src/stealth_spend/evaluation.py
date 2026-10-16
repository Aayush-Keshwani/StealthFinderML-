"""
Evaluation tracking and metrics export module for Stealth-Spend.
Computes multi-model evaluation benchmarks and exports structured JSON/CSV metrics.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd


class MetricsTracker:
    """
    Collects, aggregates, and exports experimental metrics across pipeline stages.
    """

    def __init__(self, output_dir: str = "reports/generated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_store: Dict[str, Any] = {}

    def record_stage(self, stage_name: str, metrics: Dict[str, Any]):
        """Records metrics for a specific pipeline stage."""
        self.metrics_store[stage_name] = metrics

    def export_json(self, filename: str = "metrics.json") -> str:
        """Exports all tracked metrics to JSON."""
        target_path = self.output_dir / filename
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics_store, f, indent=2, default=str)
        return str(target_path)

    def export_summary_table(self, filename: str = "model_benchmark_summary.csv") -> str:
        """Exports high-level model benchmark summary to CSV."""
        rows = []

        # Categorizer summary
        cat = self.metrics_store.get("categorization", {})
        if cat:
            rows.append({
                "Model_Type": "Categorizer (Dummy Baseline)",
                "Primary_Metric": "Macro F1",
                "Value": cat.get("dummy_baseline", {}).get("macro_f1", 0.0),
                "Secondary_Metric": "Accuracy",
                "Secondary_Value": cat.get("dummy_baseline", {}).get("accuracy", 0.0),
            })
            rows.append({
                "Model_Type": "Categorizer (Logistic Regression)",
                "Primary_Metric": "Macro F1",
                "Value": cat.get("logistic_regression", {}).get("macro_f1", 0.0),
                "Secondary_Metric": "Accuracy",
                "Secondary_Value": cat.get("logistic_regression", {}).get("accuracy", 0.0),
            })
            rows.append({
                "Model_Type": "Categorizer (XGBoost)",
                "Primary_Metric": "Macro F1",
                "Value": cat.get("xgboost", {}).get("macro_f1", 0.0),
                "Secondary_Metric": "Accuracy",
                "Secondary_Value": cat.get("xgboost", {}).get("accuracy", 0.0),
            })

        # Forecaster summary
        fc = self.metrics_store.get("forecasting", {}).get("metrics", {})
        if fc:
            for m_name, m_dict in fc.items():
                rows.append({
                    "Model_Type": f"Forecaster ({m_name.replace('_', ' ').title()})",
                    "Primary_Metric": "MAE ($)",
                    "Value": m_dict.get("mae", 0.0),
                    "Secondary_Metric": "RMSE ($)",
                    "Secondary_Value": m_dict.get("rmse", 0.0),
                })

        # Personas summary
        pers = self.metrics_store.get("personas", {})
        if pers:
            rows.append({
                "Model_Type": "Clustering (K-Means Personas)",
                "Primary_Metric": "Silhouette Score",
                "Value": pers.get("silhouette_score", 0.0),
                "Secondary_Metric": "Optimal K",
                "Secondary_Value": pers.get("optimal_k", 0),
            })

        df_bench = pd.DataFrame(rows)
        target_path = self.output_dir / filename
        df_bench.to_csv(target_path, index=False)
        return str(target_path)
