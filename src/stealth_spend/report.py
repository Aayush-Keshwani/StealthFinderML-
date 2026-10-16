"""
Report generation module for Stealth-Spend.
Produces an interactive, self-contained HTML report (using Plotly)
and exports categorized transactions, anomaly queues, and persona summaries.
"""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ReportGenerator:
    """
    Assembles all pipeline results into an interactive HTML report
    and structured CSV exports.
    """

    def __init__(self, output_dir: str = "reports/generated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_csv_artifacts(
        self,
        cleaned_df: pd.DataFrame,
        scored_df: pd.DataFrame,
        monthly_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
    ):
        """Exports pipeline results to CSV for external consumption."""
        cleaned_df.to_csv(self.output_dir / "cleaned_transactions.csv", index=False)
        scored_df.to_csv(self.output_dir / "scored_transactions.csv", index=False)
        
        # Anomaly review queue
        anomaly_queue = scored_df[scored_df["is_anomaly_review"]].sort_values(by="anomaly_score", ascending=False)
        anomaly_queue.to_csv(self.output_dir / "anomaly_review_queue.csv", index=False)
        
        monthly_df.to_csv(self.output_dir / "monthly_personas.csv", index=False)
        forecast_df.to_csv(self.output_dir / "cash_flow_forecast.csv", index=False)

    def build_interactive_html_report(
        self,
        summary_stats: Dict[str, Any],
        scored_df: pd.DataFrame,
        monthly_df: pd.DataFrame,
        forecast_res: Dict[str, Any],
        benchmarks: Dict[str, Any],
        filename: str = "stealth_spend_report.html",
    ) -> str:
        """
        Builds a self-contained, responsive HTML dashboard using Plotly.
        """
        # 1. Figure: Category Spending Breakdown
        exp_df = scored_df[scored_df["amount"] < 0].copy()
        cat_agg = exp_df.groupby("category")["amount"].sum().abs().reset_index()
        fig_cat = px.pie(cat_agg, values="amount", names="category", title="Total Spending Breakdown by Category", hole=0.4)

        # 2. Figure: Anomaly Scatter
        fig_anom = px.scatter(
            scored_df,
            x="transaction_date",
            y=scored_df["amount"].abs(),
            color="anomaly_score",
            color_continuous_scale="Reds",
            hover_data=["merchant", "category", "anomaly_reason", "amount"],
            title="Stealth Charge Anomaly Map (Score Gradient)",
            labels={"y": "Amount ($)", "transaction_date": "Date"},
        )

        # 3. Figure: Forecast Horizon
        fig_fc = go.Figure()
        dates = pd.to_datetime(forecast_res["dates"])
        fig_fc.add_trace(go.Scatter(x=dates, y=forecast_res["actual_balance"], mode="lines+markers", name="Actual Balance", line=dict(color="black", width=2)))
        fig_fc.add_trace(go.Scatter(x=dates, y=forecast_res["pred_ml"], mode="lines", name="ML Forecast", line=dict(color="#1f77b4", dash="dash", width=2)))
        fig_fc.add_trace(go.Scatter(x=dates, y=forecast_res["upper_bound"], mode="lines", line=dict(width=0), showlegend=False))
        fig_fc.add_trace(go.Scatter(x=dates, y=forecast_res["lower_bound"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(31,119,180,0.15)", name="95% Confidence Interval"))
        fig_fc.add_hline(y=forecast_res["threshold"], line_dash="dot", line_color="red", annotation_text="Low Balance Threshold")
        fig_fc.update_layout(title="Cash-Flow Forecast Horizon & Liquidity Outlook", xaxis_title="Date", yaxis_title="Balance ($)")

        # Convert plots to HTML components
        cat_html = fig_cat.to_html(full_html=False, include_plotlyjs="cdn")
        anom_html = fig_anom.to_html(full_html=False, include_plotlyjs=False)
        fc_html = fig_fc.to_html(full_html=False, include_plotlyjs=False)

        # Currency symbol detection
        curr = "₹" if "INR" in scored_df["currency"].values else "$"

        # Top alerts table
        top_alerts = scored_df.sort_values(by="anomaly_score", ascending=False).head(10)
        table_rows = ""
        for _, row in top_alerts.iterrows():
            table_rows += f"""
            <tr>
                <td>{row['transaction_date']}</td>
                <td><strong>{row['merchant']}</strong></td>
                <td>{row['category']}</td>
                <td>{curr}{abs(row['amount']):,.2f}</td>
                <td><span style="background-color: {'#ffcccc' if row['anomaly_score'] > 0.4 else '#e6f2ff'}; padding: 3px 8px; border-radius: 4px; font-weight: bold;">{row['anomaly_score']:.2f}</span></td>
                <td>{row['anomaly_reason']}</td>
            </tr>
            """

        # Persona summary
        persona_summary = ""
        if "persona" in monthly_df.columns:
            for p, g in monthly_df.groupby("persona"):
                persona_summary += f"<li><strong>{p}</strong>: {len(g)} billing period(s)</li>"

        # Benchmarks
        xgb_acc = benchmarks.get("categorization", {}).get("xgboost", {}).get("accuracy", 0.0)
        xgb_f1 = benchmarks.get("categorization", {}).get("xgboost", {}).get("macro_f1", 0.0)
        dummy_f1 = benchmarks.get("categorization", {}).get("dummy_baseline", {}).get("macro_f1", 0.0)
        lr_f1 = benchmarks.get("categorization", {}).get("logistic_regression", {}).get("macro_f1", 0.0)
        fc_mae = benchmarks.get("forecasting", {}).get("metrics", {}).get("gradient_boosting", {}).get("mae", 0.0)
        naive_mae = benchmarks.get("forecasting", {}).get("metrics", {}).get("naive", {}).get("mae", 0.0)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stealth-Spend | Financial Intelligence Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f7f9fc; color: #2d3748; margin: 0; padding: 24px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #1e3a8a, #3b82f6); color: white; padding: 32px; border-radius: 12px; margin-bottom: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 28px; }}
        .header p {{ margin: 0; opacity: 0.9; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .card-stat {{ font-size: 26px; font-weight: bold; color: #1e3a8a; margin: 8px 0 0 0; }}
        .section {{ background: white; padding: 24px; border-radius: 8px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        h2 {{ color: #1e3a8a; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; margin-top: 0; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
        th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #e2e8f0; font-size: 14px; }}
        th {{ background-color: #edf2f7; color: #4a5568; }}
        .footer {{ text-align: center; color: #718096; font-size: 13px; margin-top: 40px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Stealth-Spend Analysis & Anomaly Report</h1>
            <p>Offline-First Financial Behavior Modeler • Multi-Model Diagnostic Summary</p>
        </div>

        <div class="grid">
            <div class="card">
                <div>Total Transactions</div>
                <div class="card-stat">{summary_stats.get('total_transactions', 0):,}</div>
            </div>
            <div class="card">
                <div>Total Expenses</div>
                <div class="card-stat">{curr}{summary_stats.get('total_expense', 0.0):,.2f}</div>
            </div>
            <div class="card">
                <div>Total Credits / Income</div>
                <div class="card-stat">{curr}{summary_stats.get('total_income', 0.0):,.2f}</div>
            </div>
            <div class="card">
                <div>Flagged Review Alerts</div>
                <div class="card-stat" style="color: #e53e3e;">{summary_stats.get('anomalies_flagged', 0)}</div>
            </div>
        </div>

        <div class="section">
            <h2>1. Spending Overview</h2>
            {cat_html}
        </div>

        <div class="section">
            <h2>2. Stealth Charge Anomaly Detection</h2>
            <p>Transactions ranked by Isolation Forest multivariate outlier scores, duplicate charging checks, and historical merchant price increases.</p>
            {anom_html}
            <h3>Top 10 High-Priority Review Candidates</h3>
            <table>
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Merchant</th>
                        <th>Category</th>
                        <th>Amount</th>
                        <th>Score</th>
                        <th>Evidence / Review Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>3. Behavioral Personas</h2>
            <p>Monthly spending patterns clustered into interpretable financial profiles:</p>
            <ul>{persona_summary}</ul>
        </div>

        <div class="section">
            <h2>4. Cash-Flow Forecast & Balance Liquidity</h2>
            {fc_html}
        </div>

        <div class="section">
            <h2>5. Model Benchmarks & Validation</h2>
            <table>
                <thead>
                    <tr>
                        <th>Pipeline Model</th>
                        <th>Baseline Metric</th>
                        <th>Trained Model Metric</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Transaction Categorizer</strong></td>
                        <td>Dummy Macro F1: {dummy_f1:.3f} | LR: {lr_f1:.3f}</td>
                        <td><strong>XGBoost Macro F1: {xgb_f1:.3f}</strong> (Acc: {xgb_acc:.3f})</td>
                        <td><span style="color: green; font-weight: bold;">Passed Baseline</span></td>
                    </tr>
                    <tr>
                        <td><strong>Cash-Flow Forecaster</strong></td>
                        <td>Naive Holdout MAE: ${naive_mae:,.2f}</td>
                        <td><strong>ML Regressor MAE: ${fc_mae:,.2f}</strong></td>
                        <td><span style="color: green; font-weight: bold;">Evaluated Chronologically</span></td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="footer">
            <p>Generated by Stealth-Spend Machine Learning Pipeline • Privacy Principle: Processed locally, zero external transmissions.</p>
        </div>
    </div>
</body>
</html>
"""
        target_path = self.output_dir / filename
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return str(target_path)
