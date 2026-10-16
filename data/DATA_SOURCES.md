# Stealth-Spend Data Sources & Lineage

## 1. Provided Datasets
- **`transactions.csv`**: Labeled multi-category bank transaction dataset containing 300 transactions across account categories (Groceries, Housing, Dining, Shopping, Health, Transport, Travel, Telecom, Subscriptions, Salary/Income, Transfers).
  - Columns: `transaction_id`, `account_id`, `customer_id`, `timestamp`, `amount`, `currency`, `merchant_name`, `merchant_category`, `merchant_mcc`, `transaction_type`, `payment_channel`, `city`, `country`, `counterparty_iban`, `label`, `category`, `subcategory`, `is_subscription`, `is_salary`, `is_transfer`, `is_cash_withdrawal`, `balance_after_transaction`.
- **`accounts.csv`**: Account metadata with initial balances, current balances, overdraft limits.
- **`customers.csv`**: Customer demographic and profile metadata.
- **`archive/bank_transactions.csv`**: Large historical retail banking transaction log (India / INR benchmark) for scale validation.

## 2. Synthetic Generator (`stealth_spend.data_ingestion.SyntheticDataGenerator`)
Deterministic synthetic generator producing realistic multi-year bank statements with:
- Configurable recurring payments (rent, payroll, utilities, streaming).
- Injected edge-case anomalies (duplicate charges within 48h, unexpected price spikes, large outlier transactions, unknown foreign merchants).
- Preserves daily balance evolution and net cash flows for robust forecasting and persona segmentation.
