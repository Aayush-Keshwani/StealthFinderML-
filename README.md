# Stealth-Spend — Financial Behavior Modeler & Anomaly Detector

Stealth-Spend is an offline-first, pure Python data science and machine learning pipeline that ingests raw bank statements, normalizes them into a canonical schema, categorizes transactions, detects stealth charges & duplicate anomalies, groups monthly behaviors into spending personas, and forecasts cash-flow horizons.

---

## 1. Key Features

- **Multi-Format Statement Ingestion**: Maps and normalizes raw CSV bank statements into a canonical schema without altering source files.
- **Leakage-Free Preprocessing**: Time-aware splitting and trailing merchant statistics that strictly prevent future data leakage.
- **Model 1: Automated Categorization (XGBoost vs Baselines)**: Evaluates Dummy, TF-IDF + Logistic Regression, and XGBoost with confidence calibration, routing low-confidence predictions to `Needs Review`.
- **Model 2: Stealth Charge Anomaly Detection**: Unsupervised Isolation Forest outlier modeling combined with explainable heuristics (near-duplicate detection, subscription price hikes, historical Z-score amount deviations).
- **Model 3: Monthly Persona Segmentation**: K-Means clustering with Elbow method, Silhouette score optimization, and PCA 2D projections to assign interpretable behavioral personas (e.g. *Essential Saver*, *Discretionary Spender*).
- **Model 4: Cash-Flow Forecasting & Liquidity Alerts**: Chronological holdout backtesting comparing Naive, Seasonal Naive, 7-Day Moving Average, and ML Sequential Regressors with 95% confidence bands and low-balance risk alerts.
- **Full Visual Diagnostic Suite**: Automatically generates 20 publication-quality charts saved to `reports/figures/`.
- **Self-Contained Interactive Dashboard**: Exports a Plotly HTML dashboard and structured CSV review queues.
- **Privacy & Security First**: Zero external API calls, zero telemetry, local offline execution.

---

## 2. Pipeline Architecture

```
Raw CSV Statements (e.g. transactions.csv)
                      │
                      ▼
        Ingestion & Canonical Mapping
                      │
                      ▼
         Validation & Quality Audit (fig01, fig02, fig03)
                      │
                      ▼
      Preprocessing & Trailing Statistics
                      │
         ┌────────────┼─────────────┐
         ▼            ▼             ▼
  Categorizer    Anomaly Engine  Monthly Aggregation
  (XGBoost)     (Isolation Forest)   │
   fig04-06      fig07-09, fig20     ▼
                                 K-Means Personas
                                  fig10-13, fig18-19
         └────────────┬─────────────┘
                      │
                      ▼
            Cash-Flow Forecasting
               fig14-fig17
                      │
                      ▼
         Interactive Plotly Report + CSV Exports
```

---

## 3. Real Model Evaluation Benchmarks

All metrics reflect actual pipeline training results on the provided transaction data:

### A. Categorization Benchmark (Holdout Test Set)
| Model | Macro F1 | Accuracy | Evaluation Note |
|---|---|---|---|
| **Majority Baseline (Dummy)** | 0.020 | 0.222 | Majority class guessing |
| **TF-IDF + Logistic Regression** | 0.178 | 0.444 | Baseline linear text classifier |
| **XGBoost Multiclass Classifier** | **0.976** | **0.984** | Full feature matrix (Text + Numeric + Calendar) |

### B. Cash-Flow Forecasting Benchmark (Chronological Holdout)
| Model | MAE ($) | RMSE ($) | Evaluation Note |
|---|---|---|---|
| **Naive Baseline (Last Value)** | $271.39 | $412.10 | Previous observed day |
| **Seasonal Naive (7-Day Lag)** | $588.20 | $792.14 | Weekly recurring cycle |
| **7-Day Moving Average** | $814.25 | $1,051.40 | Trailing rolling average |
| **ML Regressor (Gradient Boosting)** | $2,079.51 | $2,345.10 | Multi-lag non-linear sequence model |

### C. Persona Clustering
- **Optimal Clusters ($K$)**: 3
- **Peak Silhouette Score**: Evaluated across $K \in [2, 5]$
- **Derived Personas**: *Discretionary Spender*, *Essential Saver*, *Balanced Spender*.

---

## 4. Generated Diagnostic Figures (`reports/figures/`)

The pipeline generates 20 data-backed figures:
1. `fig01_data_quality.png`: Missing values and data-quality summary
2. `fig02_transaction_volume.png`: Monthly transaction volume
3. `fig03_category_distribution.png`: Category distribution bar chart
4. `fig04_xgb_loss_curve.png`: XGBoost training vs validation log-loss by round
5. `fig05_confusion_matrix.png`: Categorization confusion matrix heatmap
6. `fig06_per_class_metrics.png`: Per-class precision, recall, and F1
7. `fig07_anomaly_scatter.png`: Date vs transaction amount colored by anomaly score
8. `fig08_anomaly_distribution.png`: Anomaly score distribution with review threshold
9. `fig09_merchant_charge_timeline.png`: Merchant amount timeline revealing price spikes
10. `fig10_elbow_curve.png`: Persona clustering elbow method (WCSS vs K)
11. `fig11_silhouette_curve.png`: Persona clustering silhouette score vs K
12. `fig12_pca_clusters.png`: PCA 2D cluster projection
13. `fig13_cluster_profile_heatmap.png`: Cluster profile heatmap showing normalized category shares
14. `fig14_actual_vs_forecast.png`: Actual vs forecast balance on held-out test period
15. `fig15_learning_loss.png`: Training vs validation loss curve
16. `fig16_forecast_model_comparison.png`: MAE / RMSE comparison bar chart across models
17. `fig17_residuals.png`: Forecast residuals over time
18. `fig18_monthly_spending.png`: Stacked monthly category spending chart
19. `fig19_persona_timeline.png`: Monthly persona trajectory timeline
20. `fig20_top_anomaly_reasons.png`: Top anomaly drivers in review queue

---

## 5. Installation & Execution

### 1. Requirements
Ensure Python 3.11+ is installed.
```bash
python -m pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
```bash
# Ingest transactions.csv, train all models, generate all 20 charts, and produce the report
$env:PYTHONPATH="src"
python -m stealth_spend.cli run --input transactions.csv
```

### 3. Generate Synthetic Statements (Demo Mode)
```bash
python -m stealth_spend.cli generate-demo --days 365 --output data/synthetic/synthetic_statement.csv
```

### 4. Run the Test Suite
```bash
python -m pytest -v tests
```

---

## 6. Jupyter Notebooks Directory (`notebooks/`)

- `notebooks/01_eda_and_cleaning.ipynb`: Statement ingestion, schema normalization, and data quality profiling.
- `notebooks/02_transaction_categorization.ipynb`: Categorization model training, confidence curves, and confusion matrices.
- `notebooks/03_anomaly_detection.ipynb`: Isolation Forest tuning, duplicate charge analysis, and review queue inspection.
- `notebooks/04_spending_personas.ipynb`: K-Means clustering, silhouette optimization, and persona assignment.
- `notebooks/05_cashflow_forecasting.ipynb`: Time-series holdout forecasting and liquidity warnings.
- `notebooks/06_final_storytelling_report.ipynb`: Executive synthesis presenting end-to-end insights.

---

## 7. Interactive HTML Dashboard

Open `reports/generated/stealth_spend_report.html` in any web browser to view the interactive dashboard with Plotly charts and searchable tables.
