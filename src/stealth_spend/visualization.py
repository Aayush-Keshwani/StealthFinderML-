"""
Comprehensive visualization module for Stealth-Spend.
Renders and exports all 20 data-backed figures specified in the project blueprint
using matplotlib and seaborn, saving publication-quality figures to disk.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Styling configuration
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


class StealthVisualizer:
    """
    Renders the full inventory of diagnostic, model evaluation, and reporting figures.
    """

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_fig01_data_quality(self, quality_df: pd.DataFrame, filename: str = "fig01_data_quality.png"):
        """Fig 01: Missing values and data-quality summary."""
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=quality_df, x="Column", y="Completeness_Pct", hue="Column", legend=False, palette="Blues_r", ax=ax)
        ax.set_ylim(0, 105)
        ax.set_title("Data Quality Summary: Column Completeness Rate (%)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Completeness (%)")
        ax.set_xlabel("Canonical Column")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_fig02_transaction_volume(self, df: pd.DataFrame, filename: str = "fig02_transaction_volume.png"):
        """Fig 02: Daily/monthly transaction volume."""
        df_copy = df.copy()
        df_copy["date"] = pd.to_datetime(df_copy["transaction_date"]).dt.to_period("M").astype(str)
        monthly_vol = df_copy.groupby("date").size()

        fig, ax = plt.subplots(figsize=(10, 5))
        monthly_vol.plot(kind="bar", color="#2b5c8f", ax=ax)
        ax.set_title("Monthly Transaction Volume", fontsize=14, fontweight="bold")
        ax.set_ylabel("Number of Transactions")
        ax.set_xlabel("Billing Month")
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_fig03_category_distribution(self, df: pd.DataFrame, filename: str = "fig03_category_distribution.png"):
        """Fig 03: Category distribution bar chart."""
        fig, ax = plt.subplots(figsize=(11, 5))
        counts = df["category"].value_counts()
        sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, palette="mako", ax=ax)
        ax.set_title("Transaction Category Distribution (Class Balance)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Transaction Count")
        ax.set_xlabel("Category")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_fig04_xgb_loss_curve(self, evals_result: Dict[str, Dict[str, list]], filename: str = "fig04_xgb_loss_curve.png"):
        """Fig 04: XGBoost training vs validation log-loss by boosting round."""
        fig, ax = plt.subplots(figsize=(9, 5))
        train_loss = evals_result.get("validation_0", {}).get("mlogloss", [])
        val_loss = evals_result.get("validation_1", {}).get("mlogloss", [])

        if train_loss:
            ax.plot(train_loss, label="Training Log-Loss", color="#1f77b4", lw=2)
        if val_loss:
            ax.plot(val_loss, label="Validation Log-Loss", color="#ff7f0e", lw=2)

        ax.set_title("XGBoost Categorizer: Training vs Validation Log-Loss per Round", fontsize=14, fontweight="bold")
        ax.set_xlabel("Boosting Round")
        ax.set_ylabel("Multi-class Log-Loss")
        ax.legend()
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_fig05_confusion_matrix(self, cm: np.ndarray, classes: List[str], filename: str = "fig05_confusion_matrix.png"):
        """Fig 05: Confusion matrix heatmap."""
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title("XGBoost Categorization Confusion Matrix", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Category")
        ax.set_ylabel("True Category")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_fig06_per_class_metrics(self, report: Dict[str, Any], filename: str = "fig06_per_class_metrics.png"):
        """Fig 06: Per-class precision, recall, and F1 bar chart."""
        classes = [k for k in report.keys() if k not in ["accuracy", "macro avg", "weighted avg"]]
        metrics_df = pd.DataFrame([
            {
                "Class": c,
                "Precision": report[c]["precision"],
                "Recall": report[c]["recall"],
                "F1-Score": report[c]["f1-score"],
            }
            for c in classes
        ])
        
        melted = metrics_df.melt(id_vars="Class", var_name="Metric", value_name="Score")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=melted, x="Class", y="Score", hue="Metric", palette="Set2", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_title("Per-Class Classification Metrics (Precision, Recall, F1)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Score")
        plt.xticks(rotation=45, ha="right")
        ax.legend(loc="lower right")
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_fig07_anomaly_scatter(self, df_scored: pd.DataFrame, filename: str = "fig07_anomaly_scatter.png"):
        """Fig 07: Date vs transaction amount scatter plot colored by anomaly score."""
        fig, ax = plt.subplots(figsize=(12, 6))
        dates = pd.to_datetime(df_scored["transaction_date"])
        amounts = df_scored["amount"].abs()
        scores = df_scored["anomaly_score"]

        scatter = ax.scatter(dates, amounts, c=scores, cmap="YlOrRd", alpha=0.85, edgecolors="grey", s=50 + scores * 120)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Anomaly Score (0 = Normal, 1 = Suspicious)")
        ax.set_title("Stealth Charge Detector: Transaction Date vs Amount", fontsize=14, fontweight="bold")
        ax.set_ylabel("Absolute Amount ($)")
        ax.set_xlabel("Transaction Date")
        plt.xticks(rotation=30)
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_fig08_anomaly_distribution(self, df_scored: pd.DataFrame, threshold: float = 0.5, filename: str = "fig08_anomaly_distribution.png"):
        """Fig 08: Anomaly score distribution with review threshold."""
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(df_scored["anomaly_score"], bins=30, kde=True, color="#d95f02", ax=ax)
        ax.axvline(threshold, color="red", linestyle="--", lw=2, label=f"Review Threshold ({threshold})")
        ax.set_title("Distribution of Composite Anomaly Scores", fontsize=14, fontweight="bold")
        ax.set_xlabel("Composite Anomaly Score")
        ax.set_ylabel("Frequency")
        ax.legend()
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_fig09_merchant_charge_timeline(self, df: pd.DataFrame, top_merchants: List[str] = None, filename: str = "fig09_merchant_charge_timeline.png"):
        """Fig 09: Merchant amount timeline revealing price spikes."""
        fig, ax = plt.subplots(figsize=(12, 6))
        if top_merchants is None:
            top_merchants = df["merchant"].value_counts().head(4).index.tolist()

        df_sub = df[df["merchant"].isin(top_merchants)].copy()
        df_sub["abs_amt"] = df_sub["amount"].abs()
        df_sub["transaction_date"] = pd.to_datetime(df_sub["transaction_date"])

        for m in top_merchants:
            m_data = df_sub[df_sub["merchant"] == m].sort_values(by="transaction_date")
            ax.plot(m_data["transaction_date"], m_data["abs_amt"], marker="o", label=m)

        ax.set_title("Historical Merchant Charge Timelines (Tracking Cadence & Spikes)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Charge Amount ($)")
        ax.set_xlabel("Date")
        ax.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_fig10_elbow_curve(self, k_values: List[int], inertias: List[float], filename: str = "fig10_elbow_curve.png"):
        """Fig 10: Elbow curve (WCSS / inertia vs K)."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_values, inertias, marker="s", color="#386cb0", lw=2)
        ax.set_title("Persona Clustering: Elbow Method (Inertia vs K)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Within-Cluster Sum of Squares (Inertia)")
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_fig11_silhouette_curve(self, k_values: List[int], silhouettes: List[float], filename: str = "fig11_silhouette_curve.png"):
        """Fig 11: Silhouette score vs K."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(k_values, silhouettes, marker="o", color="#7fc97f", lw=2)
        ax.set_title("Persona Clustering: Silhouette Score vs K", fontsize=14, fontweight="bold")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Silhouette Score")
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_fig12_pca_clusters(self, df_monthly: pd.DataFrame, filename: str = "fig12_pca_clusters.png"):
        """Fig 12: PCA 2D cluster projection."""
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.scatterplot(
            data=df_monthly,
            x="pca_x",
            y="pca_y",
            hue="persona",
            s=120,
            palette="tab10",
            ax=ax,
        )
        ax.set_title("Monthly Spending Personas (PCA 2D Projection)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_fig13_cluster_profile_heatmap(self, df_monthly: pd.DataFrame, filename: str = "fig13_cluster_profile_heatmap.png"):
        """Fig 13: Cluster profile heatmap showing normalized category shares."""
        share_cols = [c for c in df_monthly.columns if c.startswith("share_") or c in ["essential_ratio", "discretionary_ratio"]]
        grouped = df_monthly.groupby("persona")[share_cols].mean()
        grouped.columns = [c.replace("share_", "").title() for c in grouped.columns]

        fig, ax = plt.subplots(figsize=(10, max(4, len(grouped) * 1.5)))
        sns.heatmap(grouped, cmap="YlGnBu", annot=True, fmt=".2f", ax=ax)
        ax.set_title("Spending Persona Profile Heatmap (Category Shares)", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_fig14_actual_vs_forecast(self, forecast_res: Dict[str, Any], filename: str = "fig14_actual_vs_forecast.png"):
        """Fig 14: Actual vs forecast balance on held-out test period."""
        fig, ax = plt.subplots(figsize=(12, 6))
        dates = pd.to_datetime(forecast_res["dates"])
        actual = forecast_res["actual_balance"]
        pred_ml = forecast_res["pred_ml"]
        upper = forecast_res["upper_bound"]
        lower = forecast_res["lower_bound"]

        ax.plot(dates, actual, label="Actual Balance", color="#000000", lw=2)
        ax.plot(dates, pred_ml, label="ML Forecast (Gradient Boosting)", color="#1f77b4", linestyle="--", lw=2)
        ax.fill_between(dates, lower, upper, color="#1f77b4", alpha=0.2, label="95% Confidence Band")

        ax.axhline(forecast_res["threshold"], color="red", linestyle=":", lw=1.5, label=f"Low Balance Alert (<${forecast_res['threshold']:.0f})")
        ax.set_title("Cash-Flow Horizon: Actual vs Forecast Balance", fontsize=14, fontweight="bold")
        ax.set_ylabel("Balance ($)")
        ax.set_xlabel("Date")
        ax.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_fig15_learning_loss(self, evals_result: Dict[str, Dict[str, list]], filename: str = "fig15_learning_loss.png"):
        """Fig 15: Training vs validation loss."""
        return self.plot_fig04_xgb_loss_curve(evals_result, filename=filename)

    def plot_fig16_forecast_model_comparison(self, metrics: Dict[str, Dict[str, float]], filename: str = "fig16_forecast_model_comparison.png"):
        """Fig 16: Forecast MAE / RMSE comparison bar chart across models."""
        rows = []
        for model_name, m_vals in metrics.items():
            rows.append({"Model": model_name.replace("_", " ").title(), "MAE": m_vals["mae"], "RMSE": m_vals["rmse"]})
        comp_df = pd.DataFrame(rows).melt(id_vars="Model", var_name="Metric", value_name="Error ($)")

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=comp_df, x="Model", y="Error ($)", hue="Metric", palette="Blues", ax=ax)
        ax.set_title("Cash-Flow Forecasting Benchmark: Error Comparison (Holdout)", fontsize=14, fontweight="bold")
        plt.xticks(rotation=20)
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_fig17_residuals(self, forecast_res: Dict[str, Any], filename: str = "fig17_residuals.png"):
        """Fig 17: Forecast residuals over time."""
        fig, ax = plt.subplots(figsize=(10, 5))
        dates = pd.to_datetime(forecast_res["dates"])
        residuals = forecast_res["residuals"]

        ax.plot(dates, residuals, marker="o", color="#e7298a", lw=1.5)
        ax.axhline(0, color="grey", linestyle="--")
        ax.set_title("Forecast Residuals over Time (Actual - Predicted)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Residual ($)")
        plt.xticks(rotation=30)
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_fig18_monthly_spending(self, df_monthly: pd.DataFrame, filename: str = "fig18_monthly_spending.png"):
        """Fig 18: Stacked monthly category spending chart."""
        fig, ax = plt.subplots(figsize=(12, 6))
        share_cols = [c for c in df_monthly.columns if c.startswith("share_")]
        clean_labels = [c.replace("share_", "").title() for c in share_cols]

        x = np.arange(len(df_monthly))
        bottom = np.zeros(len(df_monthly))
        colors = plt.cm.tab20(np.linspace(0, 1, len(share_cols)))

        for idx, col in enumerate(share_cols):
            shares = df_monthly[col].values
            ax.bar(x, shares, bottom=bottom, label=clean_labels[idx], color=colors[idx])
            bottom += shares

        ax.set_xticks(x)
        ax.set_xticklabels(df_monthly["year_month"], rotation=45)
        ax.set_title("Monthly Spending Composition by Category Share", fontsize=14, fontweight="bold")
        ax.set_ylabel("Share of Total Expense")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_fig19_persona_timeline(self, df_monthly: pd.DataFrame, filename: str = "fig19_persona_timeline.png"):
        """Fig 19: Monthly persona timeline."""
        fig, ax = plt.subplots(figsize=(11, 4))
        personas = df_monthly["persona"].unique().tolist()
        persona_to_num = {p: i for i, p in enumerate(personas)}

        y = [persona_to_num[p] for p in df_monthly["persona"]]
        x = df_monthly["year_month"]

        ax.step(x, y, where="mid", marker="s", color="#7570b3", lw=2)
        ax.set_yticks(range(len(personas)))
        ax.set_yticklabels(personas)
        ax.set_title("Monthly Spending Persona Trajectory Timeline", fontsize=14, fontweight="bold")
        ax.set_xlabel("Billing Period")
        plt.xticks(rotation=45)
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)

    def plot_fig20_top_anomaly_reasons(self, df_scored: pd.DataFrame, filename: str = "fig20_top_anomaly_reasons.png"):
        """Fig 20: Top anomaly reasons breakdown."""
        flagged = df_scored[df_scored["is_anomaly_review"]].copy()
        if len(flagged) == 0:
            flagged = df_scored.sort_values(by="anomaly_score", ascending=False).head(10)

        # Break into individual reasons
        all_reasons = []
        for r in flagged["anomaly_reason"]:
            for item in r.split("; "):
                if item != "Routine transaction":
                    all_reasons.append(item)

        series = pd.Series(all_reasons).value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=series.values, y=series.index, hue=series.index, legend=False, palette="Reds_r", ax=ax)
        ax.set_title("Stealth Charge Review Queue: Primary Anomaly Drivers", fontsize=14, fontweight="bold")
        ax.set_xlabel("Alert Count")
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()
        return str(path)
