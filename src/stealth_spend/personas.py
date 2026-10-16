"""
Monthly behavioral persona segmentation module for Stealth-Spend.
Clusters spending months using K-Means and generates human-interpretable persona labels.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class MonthlyPersonaSegmenter:
    """
    Aggregates transactions into monthly spending profiles,
    evaluates K-Means clustering, and derives descriptive personas.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2, random_state=random_seed)
        self.kmeans: KMeans = None
        self.optimal_k = 3
        self.cluster_names: Dict[int, str] = {}
        self.monthly_profiles: pd.DataFrame = pd.DataFrame()

    def build_monthly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates transaction data into monthly feature rows:
        - Category spending shares
        - Total debits, total credits, net flow
        - Transaction count, median debit size
        - Essential vs Discretionary spending ratio
        """
        df_copy = df.copy()
        df_copy["transaction_date"] = pd.to_datetime(df_copy["transaction_date"])
        df_copy["year_month"] = df_copy["transaction_date"].dt.to_period("M").astype(str)

        monthly_rows = []
        for period, m_df in df_copy.groupby("year_month"):
            debits = m_df[m_df["amount"] < 0].copy()
            credits = m_df[m_df["amount"] > 0].copy()

            tot_debit = float(debits["amount"].abs().sum())
            tot_credit = float(credits["amount"].sum())
            net_flow = tot_credit - tot_debit
            tx_count = len(m_df)
            median_tx = float(debits["amount"].abs().median()) if len(debits) > 0 else 0.0

            # Exclude non-expense categories from spending share breakdown
            expense_debits = debits[~debits["category"].isin(["Income", "Transfers", "Refund"])]
            tot_expense = float(expense_debits["amount"].abs().sum())

            # Category shares
            cat_totals = expense_debits.groupby("category")["amount"].sum().abs()
            cat_shares = {}
            for cat, amt in cat_totals.items():
                cat_shares[f"share_{cat.lower()}"] = round(float(amt / (tot_expense + 1e-5)), 3)

            # Essential vs discretionary
            essential_cats = ["housing", "utilities", "groceries", "health", "transport"]
            essential_spend = float(expense_debits[expense_debits["category"].str.lower().isin(essential_cats)]["amount"].abs().sum())
            essential_ratio = round(essential_spend / (tot_expense + 1e-5), 3)

            row = {
                "year_month": period,
                "total_debit": round(tot_debit, 2),
                "total_credit": round(tot_credit, 2),
                "net_flow": round(net_flow, 2),
                "tx_count": tx_count,
                "median_tx_size": round(median_tx, 2),
                "essential_ratio": essential_ratio,
                "discretionary_ratio": round(1.0 - essential_ratio, 3),
            }
            row.update(cat_shares)
            monthly_rows.append(row)

        res_df = pd.DataFrame(monthly_rows).fillna(0.0)
        res_df.sort_values(by="year_month", inplace=True)
        res_df.reset_index(drop=True, inplace=True)
        self.monthly_profiles = res_df
        return res_df

    def find_optimal_clusters(self, df_monthly: pd.DataFrame, max_k: int = 5) -> Dict[str, Any]:
        """
        Computes inertia (elbow curve) and silhouette scores for K in [2, max_k].
        """
        feature_cols = [c for c in df_monthly.columns if c.startswith("share_") or c in ["essential_ratio", "discretionary_ratio"]]
        if len(feature_cols) == 0:
            feature_cols = ["total_debit", "tx_count", "essential_ratio"]

        X = df_monthly[feature_cols].values
        n_samples = len(X)

        max_valid_k = min(max_k, n_samples - 1)
        if max_valid_k < 2:
            return {"k_values": [1], "inertias": [0.0], "silhouette_scores": [0.0], "optimal_k": 1}

        k_values = list(range(2, max_valid_k + 1))
        inertias = []
        silhouettes = []

        X_scaled = self.scaler.fit_transform(X)

        for k in k_values:
            km = KMeans(n_clusters=k, random_state=self.random_seed, n_init=10)
            km.fit(X_scaled)
            inertias.append(float(km.inertia_))
            sil = float(silhouette_score(X_scaled, km.labels_))
            silhouettes.append(sil)

        best_idx = int(np.argmax(silhouettes))
        best_k = k_values[best_idx]

        return {
            "k_values": k_values,
            "inertias": inertias,
            "silhouette_scores": silhouettes,
            "optimal_k": best_k,
        }

    def fit_personas(self, df_monthly: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
        """
        Fits K-Means with chosen number of clusters and names each cluster.
        """
        self.optimal_k = n_clusters
        feature_cols = [c for c in df_monthly.columns if c.startswith("share_") or c in ["essential_ratio", "discretionary_ratio"]]
        if len(feature_cols) == 0:
            feature_cols = ["total_debit", "tx_count", "essential_ratio"]

        X = df_monthly[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)

        # Handle small sample counts gracefully
        actual_k = min(n_clusters, len(X))
        self.kmeans = KMeans(n_clusters=actual_k, random_state=self.random_seed, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_scaled)

        df_monthly = df_monthly.copy()
        df_monthly["cluster_id"] = cluster_labels

        # PCA 2D coordinates for plotting
        pca_coords = self.pca.fit_transform(X_scaled)
        df_monthly["pca_x"] = pca_coords[:, 0]
        df_monthly["pca_y"] = pca_coords[:, 1] if pca_coords.shape[1] > 1 else 0.0

        # Assign descriptive persona names based on distinct spending profile
        names = {}
        for c_id in range(actual_k):
            c_data = df_monthly[df_monthly["cluster_id"] == c_id]
            share_cols = [c for c in feature_cols if c.startswith("share_")]
            if share_cols and len(c_data) > 0:
                top_col = c_data[share_cols].mean().idxmax()
                top_cat = top_col.replace("share_", "").title()
                if c_data["essential_ratio"].mean() > 0.65:
                    persona_title = f"Essential Saver ({top_cat}-Dominant)"
                elif c_data["discretionary_ratio"].mean() > 0.45:
                    persona_title = f"Discretionary Spender ({top_cat}-Heavy)"
                else:
                    persona_title = f"Balanced Spender ({top_cat} Focus)"
            else:
                persona_title = f"Spending Persona #{c_id + 1}"
            names[c_id] = persona_title

        self.cluster_names = names
        df_monthly["persona"] = df_monthly["cluster_id"].map(names)
        return df_monthly
