# Stealth-Spend --- Financial Behavior Modeler & Anomaly Detector

## Implementation-Ready Project Specification

> **Project type:** Offline-first, pure Python data-science and
> machine-learning pipeline\
> **Primary goal:** Turn messy bank-statement CSVs into understandable
> transaction categories, suspicious-charge alerts, monthly spending
> personas, and cash-flow forecasts.\
> **Privacy principle:** Process statements locally by default. Use
> synthetic/demo data for development and never commit real statements,
> account identifiers, or trained artifacts containing sensitive
> information.

------------------------------------------------------------------------

## 1. Project Overview

Stealth-Spend is a modular machine-learning pipeline that ingests one or
more raw bank-statement CSV files, normalizes them into a consistent
transaction schema, creates useful features, trains several
complementary models, evaluates their performance, and produces a
polished visual report.

It should do more than show charts of spending. It should help a user
answer:

-   **What did I spend on?** Automatically categorize transaction
    descriptions.
-   **What looks unusual?** Identify potential duplicate payments,
    subscription price increases, unfamiliar merchant activity, and
    unusual amounts.
-   **What kind of spender am I this month?** Group monthly spending
    patterns into interpretable personas.
-   **Could my balance become tight?** Forecast future daily net cash
    flow and estimate low-balance risk.
-   **Why did the model flag something?** Provide human-readable
    evidence and confidence/uncertainty, rather than presenting every
    anomaly as fraud.

This is a decision-support and personal-finance analysis project, **not
a fraud verdict, credit decision, or financial-advice system**.

## 2. Objectives and Success Criteria

1.  Support multiple CSV statement formats through configurable column
    mapping.
2.  Preserve the original source files unchanged and maintain a
    reproducible cleaning pipeline.
3.  Categorize transactions using text and numeric features, with a
    baseline model for comparison.
4.  Detect candidate anomalies using multiple signals, not a single
    hard-coded spending threshold.
5.  Generate monthly behavior clusters that can be described in plain
    language.
6.  Forecast daily net cash flow or end-of-day balance using time-aware
    validation.
7.  Generate attractive, exportable charts and a concise HTML/PDF-style
    report (the core pipeline itself does not require a web framework).
8.  Make every model measurable: include baseline comparisons, suitable
    metrics, and limitations.
9.  Provide a safe demo mode with synthetic transactions when a user has
    no statement data.
10. Make results reproducible through configuration, fixed random seeds,
    saved metrics, and versioned model artifacts.

### Suggested acceptance targets

These are initial targets, not guaranteed outcomes. Confirm them with
real validation data and report the actual values.

  ----------------------------------------------------------------------------
  Area                    Evaluation                   Initial target /
                                                       expectation
  ----------------------- ---------------------------- -----------------------
  Transaction             Macro F1, weighted F1,       Beat a majority-class
  categorization          per-class precision/recall   baseline; aim for macro
                                                       F1 ≥ 0.75 when labels
                                                       and data support it

  Anomaly detection       Precision@K from manually    Review top 20--50
                          reviewed top alerts;         alerts; avoid claiming
                          synthetic anomaly recall     real-world fraud
                                                       accuracy without labels

  Clustering              Silhouette score plus        Choose a useful, stable
                          interpretability/stability   number of
                                                       clusters---not just the
                                                       highest score

  Forecasting             MAE, RMSE, MASE or a naïve   Beat a last-value /
                          baseline comparison          seasonal-naïve baseline
                                                       on a chronological
                                                       holdout

  Data quality            Missing-value and duplicate  All transformations
                          checks                       logged; unresolved rows
                                                       surfaced in a quality
                                                       report
  ----------------------------------------------------------------------------

------------------------------------------------------------------------

## 3. Dataset Plan

Use a staged data strategy: synthetic data first, public labeled data
for initial categorization experiments, and optionally anonymized
personal statements for final testing.

### 3.1 Required data sources

#### A. Synthetic bank statements --- required for development

Generate realistic demo CSVs with configurable merchants, dates,
credits/debits, recurring subscriptions, refunds, transfers,
duplicate-looking charges, price increases, and occasional large unusual
purchases.

**Why:** It lets the project run end-to-end without exposing personal
financial data and gives controlled examples for testing anomaly
detection. Synthetic anomalies are useful for testing code, but they do
not prove performance on real anomalies.

Suggested generated files: - `synthetic_statement_2024.csv` -
`synthetic_statement_2025.csv` - `synthetic_labeled_transactions.csv` -
`synthetic_daily_balances.csv`

#### B. Public transaction categorization data --- recommended

Find a public dataset that includes transaction descriptions and
ground-truth categories. Dataset availability, licensing, and label
quality can change, so verify the terms and data dictionary before use.
Search terms: - `bank transaction categorization dataset` -
`personal finance transaction classification` -
`merchant transaction description category dataset`

Use only a dataset whose license permits the intended academic/demo use.
Record its source URL, retrieval date, license, number of rows, columns,
and known limitations in `data/DATA_SOURCES.md`.

**Important:** A dataset containing only fraud labels is not
automatically suitable for merchant/category classification. A dataset
containing only clean numeric features may not be suitable for text
categorization.

#### C. Optional real statement CSVs --- final validation

Allow the user to add their own exported statement CSVs to `data/raw/`.
These files can have different column names and formats. Keep them local
and excluded from version control.

Do not require account numbers, full card numbers, addresses, or other
unnecessary identifiers. Remove or mask them during ingestion.

### 3.2 Canonical transaction schema

Normalize every source into this schema before feature engineering.

  -----------------------------------------------------------------------------------------
  Field                      Type                         Required? Description
  -------------------------- ---------------- --------------------- -----------------------
  `transaction_id`           string                             Yes Generated stable row
                                                                    ID; avoid account/card
                                                                    identifiers

  `transaction_date`         datetime                           Yes Transaction or posting
                                                                    date; document which is
                                                                    used

  `description_raw`          string                             Yes Original description
                                                                    kept in the local
                                                                    interim dataset

  `description_clean`        string                             Yes Normalized
                                                                    merchant/transaction
                                                                    text

  `amount`                   float                              Yes Signed amount under a
                                                                    documented convention

  `transaction_type`         category                           Yes `debit`, `credit`,
                                                                    `transfer`, `unknown`

  `merchant`                 string                              No Extracted/normalized
                                                                    merchant, if possible

  `category`                 category                For supervised Ground-truth or
                                                           training predicted category

  `currency`                 string                     Recommended ISO currency code when
                                                                    known

  `balance`                  float                               No Statement balance after
                                                                    transaction, if
                                                                    provided

  `source_file`              string                             Yes File provenance; do not
                                                                    expose in public
                                                                    reports if sensitive

  `is_recurring_candidate`   bool                           Derived Candidate recurring
                                                                    payment

  `is_duplicate_candidate`   bool                           Derived Candidate duplicate
                                                                    charge

  `anomaly_score`            float                          Derived Model score with
                                                                    documented direction

  `data_split`               category                       Derived Train/validation/test
                                                                    marker where relevant
  -----------------------------------------------------------------------------------------

**Amount convention:** choose and document one convention, such as
debits negative and credits positive. Do not silently mix bank formats.
Transfers between a user's own accounts should be identified where
possible so they are not misinterpreted as spending or income.

### 3.3 Suggested category taxonomy

Start with a manageable, editable taxonomy:

-   `Income`
-   `Housing`
-   `Utilities`
-   `Groceries`
-   `Dining`
-   `Transport`
-   `Shopping`
-   `Health`
-   `Education`
-   `Entertainment`
-   `Travel`
-   `Subscriptions`
-   `Fees`
-   `Transfers`
-   `Cash Withdrawal`
-   `Refund`
-   `Other`
-   `Needs Review`

Keep `Transfers`, `Refund`, and `Income` separate from discretionary
expenses. The taxonomy should be configurable in `config/categories.yml`
(or a simple Python config) and documented.

### 3.4 Data splits and leakage prevention

-   For classification, split by time where possible and group
    repeated/near-identical descriptions so copies of the same
    transaction pattern do not leak across train and test.
-   For forecasting, **never randomly shuffle time-series rows**. Train
    on earlier dates, validate on later dates, and reserve the latest
    period as a final test set.
-   Fit TF-IDF, scalers, imputers, and dimensionality reduction on
    training data only.
-   Avoid using future-derived values in features, including future
    merchant averages or future paycheck dates.
-   Document whether a balance is observed after a transaction or
    reconstructed from transactions.

------------------------------------------------------------------------

## 4. Technology Stack

Keep the core project as a local Python pipeline; no Flask, Django, or
other external web framework is required.

### Core data science

-   **Python 3.11 or 3.12** --- primary language
-   **pandas** --- CSV ingestion, joins, aggregation, cleaning
-   **NumPy** --- numeric operations
-   **PyYAML** --- configuration files
-   **python-dateutil** --- date parsing and recurrence support

### Machine learning

-   **scikit-learn** --- pipelines, TF-IDF, baselines, metrics,
    Isolation Forest, K-Means, PCA, scaling
-   **XGBoost** --- primary transaction categorization model
-   **LightGBM** --- optional comparison model, not required for the
    first version
-   **TensorFlow/Keras** --- optional LSTM forecasting experiment
-   **statsmodels** --- optional statistical/time-series baseline
-   **joblib** --- save/load scikit-learn and XGBoost artifacts

### Text processing

-   **scikit-learn `TfidfVectorizer`** --- initial text representation
-   **RapidFuzz** --- optional fuzzy merchant normalization and
    duplicate-description matching
-   **NLTK or spaCy** --- optional; use only if
    tokenization/lemmatization adds measurable value

### Charts and reports

-   **matplotlib** --- reliable static charts and model diagnostics
-   **seaborn** --- statistical visualizations
-   **Plotly** --- interactive HTML charts for the final report
-   **JupyterLab / Notebook** --- experiments and exploratory analysis
-   **ydata-profiling** --- optional automated data profiling; not
    required

### Engineering and quality

-   **pytest** --- unit and pipeline tests
-   **ruff** --- linting and formatting
-   **tqdm** --- progress bars
-   **logging** from the Python standard library --- traceable
    processing logs
-   **Git + `.gitignore`** --- version control and data privacy

### Install strategy

Keep `requirements.txt` lean and pin versions after confirming
compatibility in the target environment. TensorFlow and XGBoost can have
platform-specific constraints; install and test them separately if
necessary.

Suggested dependency list to begin testing:

``` text
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
plotly
jupyterlab
joblib
pyyaml
python-dateutil
rapidfuzz
pytest
ruff
tqdm
```

Optional forecasting dependencies:

``` text
tensorflow
statsmodels
```

Do not install every optional library before the core pipeline works.
Create a virtual environment and record the tested Python/library
versions.

------------------------------------------------------------------------

## 5. Recommended Repository Structure

``` text
stealth_spend_project/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── config/
│   ├── settings.yml
│   └── categories.yml
├── data/
│   ├── raw/                 # Local source CSVs; never commit
│   ├── interim/             # Parsed and normalized rows; normally never commit
│   ├── processed/           # Model-ready data; never commit private data
│   ├── synthetic/           # Reproducible generated demo data
│   └── DATA_SOURCES.md
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb
│   ├── 02_transaction_categorization.ipynb
│   ├── 03_anomaly_detection.ipynb
│   ├── 04_spending_personas.ipynb
│   ├── 05_cashflow_forecasting.ipynb
│   └── 06_final_storytelling_report.ipynb
├── src/
│   └── stealth_spend/
│       ├── __init__.py
│       ├── config.py
│       ├── data_ingestion.py
│       ├── validation.py
│       ├── preprocessing.py
│       ├── merchant_normalization.py
│       ├── feature_engine.py
│       ├── categorization.py
│       ├── anomaly_detection.py
│       ├── recurring_payments.py
│       ├── personas.py
│       ├── forecasting.py
│       ├── evaluation.py
│       ├── visualization.py
│       ├── report.py
│       └── pipeline.py
├── tests/
│   ├── test_ingestion.py
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_anomalies.py
│   └── test_forecasting.py
├── models/                  # Generated locally; do not commit private artifacts
├── reports/
│   ├── figures/
│   └── generated/
└── logs/
```

Keep raw files untouched. The pipeline should write new outputs to
`interim/` and `processed/`, and should never overwrite a user's source
statement.

------------------------------------------------------------------------

## 6. Pipeline Architecture

``` text
CSV statements / public labeled data / synthetic data
                         |
                         v
              Ingestion and schema mapping
                         |
                         v
             Validation + data-quality report
                         |
                         v
          Cleaning + merchant normalization
                         |
                         v
                Feature engineering
                         |
        +----------------+-----------------+
        |                |                 |
        v                v                 v
 Categorization     Anomaly signals     Daily/monthly
 (XGBoost)          (Isolation Forest)  aggregation
        |                |                 |
        |                v                 +------------------+
        |        Recurring/duplicate                          |
        |        charge analysis                              v
        |                                              K-Means personas
        |                                                    |
        +-----------------------+----------------------------+
                                |
                                v
                       Cash-flow forecasting
                                |
                                v
                 Evaluation + explanation layer
                                |
                                v
                Plotly/Matplotlib report + CSV exports
```

The pipeline should run without notebooks as well. Notebooks are for
exploration and storytelling; reusable logic belongs in `src/`.

------------------------------------------------------------------------

## 7. Feature Engineering

### 7.1 Text and merchant features

-   Lowercase and normalize whitespace/punctuation.
-   Preserve useful merchant tokens; do not blindly remove all numbers
    because they may distinguish merchants or installments.
-   Normalize common payment processor prefixes/suffixes where
    appropriate.
-   Build a merchant alias table so variants such as `UBER *TRIP`,
    `UBER TRIP`, and similar strings can map to a canonical merchant
    when justified.
-   Create TF-IDF word n-grams and optionally character n-grams.
-   Track unknown/new merchant descriptions for review.

### 7.2 Numeric and calendar features

-   Signed amount, absolute amount, log absolute amount.
-   Day of week, day of month, month, weekend flag.
-   Transaction count and amount by merchant/category over trailing
    windows.
-   Days since the previous transaction with the same merchant.
-   Rolling mean, median, standard deviation, and robust deviation from
    merchant history.
-   Month-over-month category change.
-   Time between similar charges.
-   Daily total debits, credits, net flow, and end-of-day balance where
    available.

**Leakage rule:** rolling features must use only information available
on or before the transaction date.

------------------------------------------------------------------------

## 8. Model 1 --- Automated Transaction Categorization

### Goal

Predict a category for each transaction using its description and
optional numeric/calendar features.

### Recommended approach

1.  Establish a baseline with `DummyClassifier` and a TF-IDF + Logistic
    Regression model.
2.  Train XGBoost using TF-IDF features and selected numeric features.
    If sparse text matrices are too large or the environment is
    constrained, compare a linear model and a compact text
    representation first.
3.  Use a stratified split only when it is appropriate; otherwise use a
    time-based holdout and grouped descriptions to reduce leakage.
4.  Tune class weights or sampling carefully if categories are
    imbalanced.
5.  Calibrate confidence if possible, and use `Needs Review` when
    confidence is low.

### Evaluation

-   Accuracy (report, but do not rely on it alone)
-   Macro F1 and weighted F1
-   Per-class precision, recall, and F1
-   Confusion matrix
-   Top-k accuracy if the user-facing workflow presents category
    suggestions
-   Calibration/confidence distribution, where implemented

### Required plots

1.  **Class distribution bar chart** --- identify class imbalance.
2.  **Top merchant tokens / TF-IDF terms** --- interpret common text
    signals.
3.  **Training vs validation log-loss** by boosting round, if supported
    by the selected XGBoost objective/API.
4.  **Confusion matrix heatmap** --- show categories confused with one
    another.
5.  **Per-class precision/recall/F1 bar chart.**
6.  **Confidence histogram** --- reveal uncertain predictions.
7.  **Learning curve** --- assess whether more labels may help.

Do not fabricate a smooth loss curve. Plot the actual recorded metrics
from training.

------------------------------------------------------------------------

## 9. Model 2 --- "Stealth Charge" Anomaly Detection

### Goal

Rank transactions that deserve review, such as unusual amounts,
suspiciously close duplicates, unfamiliar merchants, and recurring
charges that rise unexpectedly.

### Primary model

`sklearn.ensemble.IsolationForest`

### Candidate features

-   Absolute amount and robust log amount
-   Merchant/category-specific deviation from trailing historical
    behavior
-   Frequency of similar merchant charges
-   Days since last similar charge
-   Duplicate similarity and time gap
-   Local transaction density and transaction timing
-   New merchant indicator
-   Amount change relative to a detected recurring payment baseline

Fit scaling and feature transformations on training/history data only.
Consider separate models or scoring logic for distinct transaction
groups (for example, credits vs debits) if mixing them makes scores
meaningless.

### Add explainable anomaly signals

Use a combined **review score** rather than assuming one model detects
every case: - Isolation Forest outlier score - Near-duplicate score
(same/similar merchant and amount within a short interval) -
Recurring-payment increase score - New/unseen merchant signal - Robust
deviation from the merchant's historical amount

Normalize components and document the scoring logic. Keep the original
component scores so a flagged transaction can be explained.

### Evaluation

Anomaly detection is unsupervised, so do not claim precision/recall on
real statements without ground-truth labels. Evaluate using: - Manually
reviewed top-K alerts and a review spreadsheet - Synthetic injected
anomalies with known labels - Precision@K and recall on the synthetic
benchmark - Stability of the highest-ranked alerts across reasonable
parameter changes - False-positive analysis by category and merchant
type

### Required plots

1.  **Date vs transaction amount scatter plot**, colored by anomaly
    score.
2.  **Anomaly score distribution** with the selected review threshold
    marked.
3.  **Top 20 flagged transactions** as a horizontal bar chart.
4.  **Merchant amount timeline** to expose price increases or unusual
    repeat charges.
5.  **Duplicate-candidate timeline** with linked candidate pairs.
6.  **Anomaly breakdown** by category and reason.
7.  **Synthetic benchmark confusion matrix/PR curve**, only when
    injected anomalies provide labels.

Use red/orange for review alerts and neutral colors for normal
transactions, with an accessible legend. Explain that an alert is a
candidate for human review---not proof of fraud.

------------------------------------------------------------------------

## 10. Model 3 --- Behavioral Persona Segmentation

### Goal

Group months with similar spending composition into interpretable
patterns, such as `Food-Heavy Month`, `Travel Month`, or
`Low-Spend Month`. Personas are descriptive summaries, not permanent
labels for a person.

### Feature construction

Create one row per month with features such as: - Category spending
totals and shares - Total debits, total credits, net cash flow -
Transaction count and median transaction size - Recurring payment
share - Essential vs discretionary spending share (taxonomy must be
documented) - Month-over-month category changes - Number of anomaly
candidates

Exclude transfers and refunds from expense shares when appropriate, and
handle months with very low activity. Avoid using sensitive attributes.

### Recommended process

1.  Standardize numeric features.
2.  Test a range of K values.
3.  Compare K-Means with a simple baseline or DBSCAN if cluster shapes
    make K-Means unsuitable.
4.  Use the elbow curve, silhouette score, cluster size, stability, and
    human interpretability together.
5.  Use PCA only for visualization; do not imply that a 2D projection
    preserves every relationship.
6.  Generate cluster descriptions from each cluster's distinguishing
    category shares.

### Required plots

1.  **Elbow curve (WCSS/inertia vs K).**
2.  **Silhouette score vs K.**
3.  **PCA 2D projection** colored by cluster.
4.  **Cluster profile heatmap** showing normalized category shares.
5.  **Stacked monthly category spending chart.**
6.  **Cluster size bar chart.**
7.  **Monthly persona timeline** showing how a month is assigned.
8.  Optional radar chart for category shares, with a clear scale and no
    excessive overlapping labels.

Do not force a fixed number of clusters if the data is too small. If
there are too few months, display descriptive trends and mark clustering
as inconclusive.

------------------------------------------------------------------------

## 11. Model 4 --- Cash-Flow Forecasting

### Goal

Forecast future daily net cash flow or end-of-day balance and identify
periods where the projected balance may become low.

### Critical data requirement

A reliable daily balance forecast usually needs sufficient history,
consistent date coverage, and a clear balance definition. A year of data
is a useful starting point, not a guarantee. If only transactions are
available, reconstruct daily net flow and only reconstruct balances when
an opening balance is known and the transaction data is sufficiently
complete.

### Models to compare

1.  **Naïve baseline:** next day equals the latest observed value.
2.  **Seasonal-naïve baseline:** use the same weekday/weekly pattern
    where justified.
3.  **Statistical baseline:** optional moving average or a suitable
    statsmodels model.
4.  **LSTM:** optional deep-learning model for sequential patterns.

Start with baselines before the LSTM. LSTMs can overfit small personal
datasets and should not be presented as superior unless they beat the
baselines on a chronological holdout.

### Features

-   Daily net flow and/or end-of-day balance
-   Day of week and month
-   Lagged values (e.g., previous 1, 7, 14, 28 days)
-   Rolling means and standard deviations
-   Known recurring income/payment indicators
-   Paycheck timing only when it can be inferred from historical
    evidence; label inferred dates as uncertain

Do not use future actual values as model inputs. Fit scalers on training
data only, and inverse-transform predictions before reporting
currency-based errors.

### Evaluation

-   MAE and RMSE
-   MASE or another baseline-relative metric where practical
-   Compare against naïve and seasonal-naïve forecasts
-   Chronological holdout and rolling-origin backtesting
-   Error by forecast horizon (e.g., 1, 7, and 14 days)
-   Optional prediction intervals or scenario ranges; never imply
    certainty

### Required plots

1.  **Daily net cash-flow line chart** with zero reference line.
2.  **Actual vs predicted balance** on the held-out period.
3.  **Training vs validation loss** over epochs for the LSTM.
4.  **Residual/error distribution** and residuals over time.
5.  **MAE/RMSE comparison bar chart** for all baselines and candidate
    models.
6.  **Forecast horizon error chart**.
7.  **Projected balance with uncertainty band**, only if the method
    supports a defensible interval.
8.  **Low-balance risk timeline** using a user-configurable threshold,
    clearly labeled as a scenario rather than a guarantee.

Use early stopping and model checkpoints for neural training. Choose
epochs based on validation behavior rather than assuming 50--100 epochs
is always appropriate.

------------------------------------------------------------------------

## 12. Additional High-Value Features

Implement these after the core pipeline is stable.

### A. Recurring payment and subscription tracker

-   Detect likely monthly/weekly/annual recurring charges using merchant
    similarity and time gaps.
-   Estimate the expected next charge window.
-   Highlight amount increases, newly appearing recurring payments, and
    charges that stop unexpectedly.
-   Display confidence and the historical transactions supporting each
    recurring-payment candidate.

### B. Duplicate billing candidate detector

-   Compare merchant similarity, amount similarity, and time interval.
-   Treat identical amounts as a clue, not proof; legitimate repeated
    purchases are common.
-   Offer configurable time windows and show both transactions side by
    side.

### C. Merchant normalization and search

-   Map merchant aliases to a canonical merchant.
-   Allow corrections to be stored locally and reused in future runs.
-   Provide an "unknown merchants" review queue.

### D. Budget drift and spending trend analysis

-   Compare category spending against a user-defined budget or a
    trailing personal baseline.
-   Show absolute and percentage month-over-month change.
-   Use minimum-volume checks so tiny categories do not produce
    misleading percentage spikes.

### E. Explainable insights

Generate evidence-backed summaries such as: - "Dining transactions were
18% higher than the trailing three-month average." - "A recurring
candidate appears to have increased from ₹X to ₹Y." - "Three
transactions were flagged because they were unusually large relative to
this merchant's history."

Only generate a statement if its underlying calculation supports it.
Include date ranges, amounts, and comparison baselines; distinguish
observed facts from model inferences.

### F. What-if scenarios

-   Simulate a recurring bill increase, reduced discretionary spending,
    or delayed income.
-   Let users change assumptions and compare forecast trajectories.
-   Clearly label these as hypothetical scenarios, not predictions of
    guaranteed outcomes.

### G. Data-quality and model monitoring

-   Detect missing dates, inconsistent signs, duplicate rows, unknown
    categories, currency mixing, and implausible balances.
-   Track model version, training timestamp, data range, metrics, and
    feature schema.
-   Alert when incoming data differs significantly from the training
    distribution.

### H. Exportable outputs

Generate: - Cleaned transactions CSV - Categorized transactions CSV -
Anomaly review CSV with reasons and component scores - Monthly persona
CSV - Forecast CSV - Metrics JSON/CSV - Figures in PNG/SVG and an
interactive Plotly HTML report

------------------------------------------------------------------------

## 13. Visualization and UI-Free Presentation Design

The core project has no web framework, but its outputs should still look
polished. Use a consistent visual theme, readable labels, currency
formatting, and restrained colors.

### Suggested report sections

1.  **Executive summary:** date range, transaction count, total
    debits/credits, net flow, data-quality notes.
2.  **Spending overview:** category breakdown, monthly trend, top
    merchants.
3.  **Stealth alerts:** high-priority anomaly candidates with reason and
    evidence.
4.  **Recurring payments:** expected range, last amount, change,
    confidence.
5.  **Behavior personas:** cluster labels and monthly persona timeline.
6.  **Cash-flow outlook:** model comparison, forecast, low-balance
    scenarios.
7.  **Model quality:** metrics, validation split, limitations, training
    date.
8.  **Data quality and privacy:** missing fields, excluded rows,
    local-processing note.

### Chart design checklist

-   Use consistent figure dimensions and typography.
-   Add titles that state the question the chart answers.
-   Include axis labels, units, date ranges, and legends.
-   Format currency using the dataset's currency; do not assume every
    statement is in INR.
-   Use colorblind-aware palettes and do not encode meaning by color
    alone.
-   Annotate important points sparingly.
-   Use Plotly for hover details and Matplotlib/Seaborn for static
    exports.
-   Avoid 3D charts and overloaded pie charts; prefer sorted bars,
    lines, and heatmaps.
-   Save both the chart and the data used to generate it so results are
    reproducible.
-   Make all plots use actual model outputs. Never create illustrative
    curves and present them as training results.

### Suggested figure inventory

  Figure ID   Figure                                    Stage
  ----------- ----------------------------------------- ---------------------
  `fig01`     Missing values and data-quality summary   Cleaning
  `fig02`     Daily/monthly transaction volume          EDA
  `fig03`     Category distribution                     Categorization
  `fig04`     XGBoost training/validation log-loss      Categorization
  `fig05`     Confusion matrix                          Categorization
  `fig06`     Per-class metrics                         Categorization
  `fig07`     Anomaly score vs date/amount              Anomaly detection
  `fig08`     Anomaly score distribution                Anomaly detection
  `fig09`     Merchant charge timeline                  Recurring/anomalies
  `fig10`     Elbow curve                               Clustering
  `fig11`     Silhouette score vs K                     Clustering
  `fig12`     PCA cluster projection                    Clustering
  `fig13`     Cluster profile heatmap                   Clustering
  `fig14`     Actual vs forecast balance                Forecasting
  `fig15`     Training vs validation loss               LSTM
  `fig16`     Model MAE/RMSE comparison                 Forecasting
  `fig17`     Residuals over time                       Forecasting
  `fig18`     Monthly category spending                 Final report
  `fig19`     Monthly persona timeline                  Final report
  `fig20`     Top anomaly reasons                       Final report

------------------------------------------------------------------------

## 14. Evaluation and Experiment Tracking

Every notebook and training script should save: - Dataset identifier and
date range - Number of rows before and after cleaning - Feature list and
target definition - Train/validation/test date ranges - Random seed and
model hyperparameters - Metrics in machine-readable format - Model
artifact path and software versions - Charts and short interpretation -
Known failure modes and limitations

### Model-specific evaluation summary

  ------------------------------------------------------------------------
  Model             Baselines         Main metrics       Important
                                                         limitation
  ----------------- ----------------- ------------------ -----------------
  Categorizer       Majority class,   Macro F1, weighted Labels can be
                    TF-IDF + Logistic F1, per-class      noisy and
                    Regression        recall             categories
                                                         subjective

  Anomaly detector  Simple robust     Precision@K,       Unsupervised
                    deviation and     synthetic recall,  scores are not
                    duplicate         stability          fraud
                    heuristics                           probabilities

  Clustering        Several K values  Silhouette,        Clusters may be
                    and possibly      stability,         unstable with few
                    DBSCAN            interpretability   months

  Forecasting       Last value,       MAE, RMSE, MASE    Sparse or
                    seasonal naïve,                      incomplete
                    moving average                       statement history
                                                         limits forecasts
  ------------------------------------------------------------------------

------------------------------------------------------------------------

## 15. Privacy, Security, and Responsible Use

-   Keep raw statements, interim data, processed personal data, logs,
    and trained models out of Git.
-   Add `data/raw/`, `data/interim/`, `data/processed/`, `models/`, and
    private report outputs to `.gitignore`.
-   Do not upload bank statements to public notebooks, public
    repositories, or third-party services.
-   Avoid storing full account numbers, card numbers, addresses, or
    unnecessary personal identifiers.
-   Do not log raw transaction descriptions by default; use row IDs and
    redacted diagnostics.
-   Provide a `--demo` mode that uses synthetic data.
-   Allow users to delete generated artifacts and clear local outputs.
-   Keep an audit record of transformations without copying sensitive
    fields into logs.
-   Describe anomaly outputs as "review candidates," never definitive
    fraud findings.
-   Make clear that forecasts are estimates and can fail when income,
    bills, or behavior changes.
-   Check dataset licenses and applicable privacy requirements before
    distributing sample data.

------------------------------------------------------------------------

## 16. Implementation Roadmap

### Phase 1 --- Repository, synthetic data, and ingestion

-   Set up package structure, environment, configuration, logging, and
    tests.
-   Generate a deterministic synthetic dataset with known edge cases.
-   Implement column mapping, date parsing, amount-sign normalization,
    currency checks, and schema validation.
-   Save a data-quality report and initial EDA figures.

**Definition of done:** one command ingests demo CSVs, produces
canonical transactions, reports invalid rows, and never alters raw
inputs.

### Phase 2 --- Categorization

-   Build baseline TF-IDF + Logistic Regression.
-   Train XGBoost and record learning curves where supported.
-   Evaluate with macro/weighted F1, per-class metrics, confusion
    matrix, and confidence analysis.
-   Add low-confidence `Needs Review` behavior.

**Definition of done:** test split is documented; baseline and XGBoost
metrics are saved; the report shows actual evaluation charts.

### Phase 3 --- Anomaly and recurring-charge intelligence

-   Implement Isolation Forest with a reproducible feature pipeline.
-   Add duplicate candidates, recurring-payment detection, and
    merchant-history deviation.
-   Combine signals into explainable review scores.
-   Test on synthetic injected anomalies and create a review CSV.

**Definition of done:** each alert includes score, reason(s), relevant
comparison transactions, and a clear caveat.

### Phase 4 --- Spending personas

-   Aggregate category and cash-flow features by month.
-   Scale features and evaluate several cluster counts.
-   Generate PCA, elbow, silhouette, cluster profile, and persona
    timeline plots.
-   Add plain-language cluster summaries.

**Definition of done:** each cluster is described from its actual
profile; inconclusive clustering is reported honestly.

### Phase 5 --- Forecasting

-   Construct a complete daily index and document missing-date
    treatment.
-   Implement naïve and seasonal-naïve baselines.
-   Add an optional LSTM with chronological validation, early stopping,
    and inverse-transformed predictions.
-   Compare model errors and produce forecast/risk plots.

**Definition of done:** forecasting is evaluated on unseen future dates
and compared to baselines; no future leakage.

### Phase 6 --- Final report, packaging, and polish

-   Build the report generator and export CSV/JSON/HTML outputs.
-   Add unit tests and an end-to-end integration test.
-   Add command-line execution and reproducible configuration.
-   Document data setup, model limitations, privacy, and
    troubleshooting.

**Definition of done:** a fresh environment can run demo mode from start
to finish and generate all expected outputs.

------------------------------------------------------------------------

## 17. Proposed Command-Line Workflow

Implement a small command-line interface using `argparse` first; a
third-party CLI library is optional.

Example target commands (implement these as part of the project):

``` bash
# Install dependencies in a virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.venv\Scripts\Activate.ps1

# Install the project's tested dependencies
pip install -r requirements.txt

# Generate safe demo data
python -m stealth_spend.cli generate-demo

# Validate and normalize source CSVs
python -m stealth_spend.cli ingest --input data/raw --output data/interim

# Train/evaluate all models and create report artifacts
python -m stealth_spend.cli run --config config/settings.yml

# Run the test suite
pytest -q
```

These commands are the desired interface and must be implemented; they
are not available automatically from this specification. Also provide
notebook instructions for users who prefer a notebook-first workflow.

------------------------------------------------------------------------

## 18. Final Deliverables

1.  Working Python package under `src/stealth_spend/`.
2.  Five model/analysis notebooks plus a final storytelling notebook.
3.  Synthetic demo dataset generator and documented dataset sources.
4.  Tested ingestion and preprocessing pipeline.
5.  Categorization baseline and XGBoost model with saved evaluation
    metrics.
6.  Isolation Forest anomaly model plus recurring/duplicate charge
    candidates.
7.  K-Means persona analysis with cluster interpretation.
8.  Forecast baselines and optional LSTM experiment.
9.  At least 15--20 genuine, data-backed charts saved to
    `reports/figures/`.
10. Exportable CSVs, JSON metrics, and an interactive Plotly HTML
    report.
11. Unit tests, configuration files, dependency list, `.gitignore`, and
    comprehensive README.
12. Privacy and limitations documentation.

## 19. README Quickstart Outline

The final `README.md` should contain: - Project summary and feature
list - Pipeline architecture - Sample screenshots of real generated
charts - Supported CSV schema and column mapping examples - Dataset
sources and licensing notes - Installation and demo instructions - How
to run each notebook - Model evaluation results (actual values only) -
Known limitations - Privacy and security practices - Roadmap and
contribution instructions

## 20. Final Guidance

Build a reliable, explainable pipeline before optimizing for model
complexity. A strong project is not the one with the most algorithms; it
is the one that compares models against sensible baselines, avoids
leakage, explains its alerts, produces clear visual evidence, and
behaves safely with sensitive data.

**Recommended build order:** synthetic data → ingestion and EDA →
categorization baseline → XGBoost → anomaly/recurring charges → personas
→ forecasting baselines → optional LSTM → polished report.
