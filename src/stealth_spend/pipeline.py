"""
Main orchestration pipeline for Stealth-Spend.
Executes the end-to-end data ingestion, validation, categorization,
anomaly detection, persona clustering, cash-flow forecasting, and reporting workflow.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from stealth_spend.config import load_config
from stealth_spend.data_ingestion import SyntheticDataGenerator, normalize_to_canonical
from stealth_spend.validation import DataValidator
from stealth_spend.preprocessing import Preprocessor
from stealth_spend.merchant_normalization import MerchantNormalizer
from stealth_spend.categorization import TransactionCategorizer
from stealth_spend.anomaly_detection import AnomalyDetector
from stealth_spend.personas import MonthlyPersonaSegmenter
from stealth_spend.forecasting import CashFlowForecaster
from stealth_spend.visualization import StealthVisualizer
from stealth_spend.evaluation import MetricsTracker
from stealth_spend.report import ReportGenerator


class StealthSpendPipeline:
    """
    End-to-end executor for the Stealth-Spend machine learning pipeline.
    """

    def __init__(self, config_path: str = "config/settings.yml"):
        self.config = load_config(config_path)
        self.metrics_tracker = MetricsTracker(self.config["paths"]["generated_reports"])
        self.visualizer = StealthVisualizer(self.config["paths"]["figures"])
        self.report_generator = ReportGenerator(self.config["paths"]["generated_reports"])
        
        self.validator = DataValidator()
        self.preprocessor = Preprocessor()
        self.merchant_normalizer = MerchantNormalizer()
        self.categorizer = TransactionCategorizer(
            min_confidence_threshold=self.config["categorization"]["min_confidence_threshold"],
            random_seed=self.config["project"]["random_seed"],
        )
        self.anomaly_detector = AnomalyDetector(
            contamination=self.config["anomaly_detection"]["contamination"],
            zscore_threshold=self.config["anomaly_detection"]["amount_deviation_threshold_zscore"],
            random_seed=self.config["project"]["random_seed"],
        )
        self.persona_segmenter = MonthlyPersonaSegmenter(
            random_seed=self.config["project"]["random_seed"],
        )
        self.forecaster = CashFlowForecaster(
            test_days=self.config["forecasting"]["test_days"],
            low_balance_threshold=self.config["forecasting"]["low_balance_threshold"],
            random_seed=self.config["project"]["random_seed"],
        )

    def run(self, input_csv_path: Optional[str] = None, use_demo: bool = False) -> Dict[str, Any]:
        """
        Executes the entire end-to-end pipeline:
        1. Ingestion / Generation
        2. Validation & fig01, fig02, fig03
        3. Preprocessing & Leakage-Free Splitting
        4. Categorization Training & Evaluation (fig04, fig05, fig06)
        5. Anomaly Detection & Stealth Alert Scoring (fig07, fig08, fig09, fig20)
        6. Persona Segmentation (fig10, fig11, fig12, fig13, fig18, fig19)
        7. Cash-flow Forecasting & Holdout Backtesting (fig14, fig15, fig16, fig17)
        8. Report Assembly, CSV Exports, & Metric Tracking
        """
        print("=== [Phase 1] Data Ingestion & Canonical Schema Mapping ===")
        # Ensure directories exist
        for p in self.config["paths"].values():
            Path(p).mkdir(parents=True, exist_ok=True)

        # 1. Ingestion
        if use_demo or input_csv_path is None or not Path(input_csv_path).exists():
            print("Generating realistic synthetic multi-year bank statements...")
            gen = SyntheticDataGenerator(seed=self.config["project"]["random_seed"])
            raw_canonical = gen.generate(days=365)
            # Save synthetic copies
            raw_canonical.to_csv(Path(self.config["paths"]["synthetic_data"]) / "synthetic_statement.csv", index=False)
        else:
            print(f"Ingesting bank statement from: {input_csv_path}")
            if "bank_transactions" in str(input_csv_path).lower():
                print("Processing 50,000 transactions from bank_transactions.csv for model training...")
                raw_df = pd.read_csv(input_csv_path, nrows=50000)
            else:
                raw_df = pd.read_csv(input_csv_path)
            raw_canonical = normalize_to_canonical(raw_df, source_filename=Path(input_csv_path).name)

        print(f"Ingested {len(raw_canonical)} transactions.")

        # If statement has small number of rows (e.g. 300 rows in transactions.csv), we also augment or combine
        # with synthetic generator for full-year daily time-series cash-flow forecasting if needed
        is_small = len(raw_canonical) < 500
        if is_small and not use_demo:
            print("Augmenting transaction base with synthetic historical sequence for robust 365-day forecasting...")
            gen = SyntheticDataGenerator(seed=self.config["project"]["random_seed"])
            synth_df = gen.generate(days=365)
            full_df = pd.concat([raw_canonical, synth_df], ignore_index=True)
            full_df.sort_values(by="transaction_date", inplace=True)
            full_df.reset_index(drop=True, inplace=True)
        else:
            full_df = raw_canonical.copy()

        # 2. Validation
        print("=== [Phase 2] Data Validation & Quality Audit ===")
        val_report = self.validator.validate(full_df)
        quality_summary_df = self.validator.get_quality_summary_df(full_df)
        self.metrics_tracker.record_stage("validation", val_report)

        # Plot Fig 01, 02, 03
        self.visualizer.plot_fig01_data_quality(quality_summary_df)
        self.visualizer.plot_fig02_transaction_volume(full_df)
        self.visualizer.plot_fig03_category_distribution(full_df)
        print("Saved Fig 01, Fig 02, Fig 03 to reports/figures/.")

        # 3. Preprocessing
        print("=== [Phase 3] Preprocessing & Leakage-Free Splitting ===")
        clean_df = self.preprocessor.process(full_df)
        clean_df.to_csv(Path(self.config["paths"]["interim_data"]) / "interim_transactions.csv", index=False)

        # Merchant Normalization
        clean_df["merchant"] = clean_df["description_raw"].apply(self.merchant_normalizer.normalize)

        train_df, test_df = self.preprocessor.train_test_split(clean_df, test_ratio=self.config["categorization"]["test_size"])
        print(f"Chronological split: Train = {len(train_df)}, Test = {len(test_df)}")

        # 4. Categorization
        print("=== [Phase 4] Model 1: Automated Categorization (Baseline vs XGBoost) ===")
        cat_metrics = self.categorizer.train(train_df, val_df=test_df)
        self.metrics_tracker.record_stage("categorization", cat_metrics)
        self.categorizer.save(Path(self.config["paths"]["models"]) / "categorizer.joblib")

        print(f"Categorizer Results -> Dummy Macro F1: {cat_metrics['dummy_baseline']['macro_f1']:.3f} | LR Macro F1: {cat_metrics['logistic_regression']['macro_f1']:.3f} | XGBoost Macro F1: {cat_metrics['xgboost']['macro_f1']:.3f}")

        # Plot Fig 04, 05, 06
        self.visualizer.plot_fig04_xgb_loss_curve(self.categorizer.training_eval_results)
        self.visualizer.plot_fig05_confusion_matrix(
            np.array(cat_metrics["confusion_matrix"]),
            cat_metrics["classes"]
        )
        self.visualizer.plot_fig06_per_class_metrics(cat_metrics["classification_report"])
        print("Saved Fig 04, Fig 05, Fig 06 to reports/figures/.")

        # 5. Anomaly Detection
        print("=== [Phase 5] Model 2: 'Stealth Charge' Anomaly & Subscription Intelligence ===")
        self.anomaly_detector.fit(train_df)
        scored_df = self.anomaly_detector.score_transactions(clean_df)
        self.anomaly_detector.save(Path(self.config["paths"]["models"]) / "anomaly_detector.joblib")

        top_alerts = self.anomaly_detector.get_top_alerts(scored_df, top_k=self.config["anomaly_detection"]["top_k_review"])
        self.metrics_tracker.record_stage("anomaly_detection", {
            "total_scored": len(scored_df),
            "flagged_count": int(scored_df["is_anomaly_review"].sum()),
            "mean_anomaly_score": float(scored_df["anomaly_score"].mean()),
        })

        # Plot Fig 07, 08, 09, 20
        self.visualizer.plot_fig07_anomaly_scatter(scored_df)
        self.visualizer.plot_fig08_anomaly_distribution(scored_df)
        self.visualizer.plot_fig09_merchant_charge_timeline(scored_df)
        self.visualizer.plot_fig20_top_anomaly_reasons(scored_df)
        print("Saved Fig 07, Fig 08, Fig 09, Fig 20 to reports/figures/.")

        # 6. Monthly Spending Personas
        print("=== [Phase 6] Model 3: Behavioral Persona Segmentation (K-Means) ===")
        monthly_df = self.persona_segmenter.build_monthly_features(clean_df)
        cluster_eval = self.persona_segmenter.find_optimal_clusters(monthly_df, max_k=self.config["clustering"]["max_k"])
        monthly_clustered = self.persona_segmenter.fit_personas(monthly_df, n_clusters=cluster_eval["optimal_k"])
        
        self.metrics_tracker.record_stage("personas", {
            "optimal_k": cluster_eval["optimal_k"],
            "k_values": cluster_eval["k_values"],
            "silhouette_score": float(np.max(cluster_eval["silhouette_scores"])) if cluster_eval["silhouette_scores"] else 0.0,
            "persona_counts": monthly_clustered["persona"].value_counts().to_dict(),
        })

        # Plot Fig 10, 11, 12, 13, 18, 19
        self.visualizer.plot_fig10_elbow_curve(cluster_eval["k_values"], cluster_eval["inertias"])
        self.visualizer.plot_fig11_silhouette_curve(cluster_eval["k_values"], cluster_eval["silhouette_scores"])
        self.visualizer.plot_fig12_pca_clusters(monthly_clustered)
        self.visualizer.plot_fig13_cluster_profile_heatmap(monthly_clustered)
        self.visualizer.plot_fig18_monthly_spending(monthly_clustered)
        self.visualizer.plot_fig19_persona_timeline(monthly_clustered)
        print("Saved Fig 10, 11, 12, 13, 18, 19 to reports/figures/.")

        # 7. Cash-flow Forecasting
        print("=== [Phase 7] Model 4: Cash-Flow Forecasting & Liquidity Horizon ===")
        daily_ts = self.forecaster.build_daily_time_series(clean_df)
        forecast_res = self.forecaster.train_and_evaluate(daily_ts)
        self.forecaster.save(Path(self.config["paths"]["models"]) / "forecaster.joblib")
        self.metrics_tracker.record_stage("forecasting", {"metrics": forecast_res["metrics"]})

        print(f"Forecasting Results -> Naive MAE: ${forecast_res['metrics']['naive']['mae']:.2f} | ML Regressor MAE: ${forecast_res['metrics']['gradient_boosting']['mae']:.2f}")

        # Plot Fig 14, 15, 16, 17
        self.visualizer.plot_fig14_actual_vs_forecast(forecast_res)
        self.visualizer.plot_fig15_learning_loss(self.categorizer.training_eval_results)
        self.visualizer.plot_fig16_forecast_model_comparison(forecast_res["metrics"])
        self.visualizer.plot_fig17_residuals(forecast_res)
        print("Saved Fig 14, 15, 16, 17 to reports/figures/.")

        # 8. Report Generation & Exports
        print("=== [Phase 8] Exporting Datasets, Metrics, and HTML Dashboard ===")
        forecast_df = pd.DataFrame({
            "date": forecast_res["dates"],
            "actual_balance": forecast_res["actual_balance"],
            "predicted_balance": forecast_res["pred_ml"],
            "lower_95_ci": forecast_res["lower_bound"],
            "upper_95_ci": forecast_res["upper_bound"],
            "residual": forecast_res["residuals"],
        })

        self.report_generator.export_csv_artifacts(
            cleaned_df=clean_df,
            scored_df=scored_df,
            monthly_df=monthly_clustered,
            forecast_df=forecast_df,
        )

        json_metrics_path = self.metrics_tracker.export_json("metrics.json")
        csv_bench_path = self.metrics_tracker.export_summary_table("model_benchmark_summary.csv")

        summary_stats = {
            "total_transactions": len(scored_df),
            "total_expense": float(scored_df[scored_df["amount"] < 0]["amount"].abs().sum()),
            "total_income": float(scored_df[scored_df["amount"] > 0]["amount"].sum()),
            "anomalies_flagged": int(scored_df["is_anomaly_review"].sum()),
        }

        html_report_path = self.report_generator.build_interactive_html_report(
            summary_stats=summary_stats,
            scored_df=scored_df,
            monthly_df=monthly_clustered,
            forecast_res=forecast_res,
            benchmarks=self.metrics_tracker.metrics_store,
        )

        print(f"Successfully generated HTML report: {html_report_path}")
        print(f"Exported metrics JSON: {json_metrics_path}")
        print(f"Exported benchmarks CSV: {csv_bench_path}")

        return {
            "status": "COMPLETED",
            "html_report": html_report_path,
            "metrics": self.metrics_tracker.metrics_store,
            "top_alerts": top_alerts.to_dict(orient="records"),
            "figures_count": 20,
        }
