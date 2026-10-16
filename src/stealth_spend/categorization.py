"""
Transaction categorization module for Stealth-Spend.
Trains baseline models (Dummy, LogisticRegression) and an XGBoost multiclass model
with confidence calibration and low-confidence 'Needs Review' routing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import joblib

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from stealth_spend.feature_engine import FeatureEngine


class TransactionCategorizer:
    """
    Manages training, evaluation, and inference of transaction category classifiers.
    """

    def __init__(
        self,
        min_confidence_threshold: float = 0.40,
        random_seed: int = 42,
    ):
        self.min_confidence_threshold = min_confidence_threshold
        self.random_seed = random_seed
        self.label_encoder = LabelEncoder()
        self.feature_engine = FeatureEngine(max_tfidf_features=300)
        
        self.baseline_dummy: Optional[DummyClassifier] = None
        self.baseline_lr: Optional[LogisticRegression] = None
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.training_eval_results: Dict[str, Dict[str, list]] = {}
        self.is_fitted = False

    def prepare_matrices(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extracts combined feature matrices (TF-IDF + tabular numeric) and encoded labels.
        """
        self.feature_engine.fit(train_df)
        self.label_encoder.fit(train_df["category"].astype(str))

        X_text_tr = self.feature_engine.transform_text(train_df)
        X_num_tr = self.feature_engine.transform(train_df).fillna(0.0).values
        X_train = np.hstack([X_text_tr, X_num_tr])
        y_train = self.label_encoder.transform(train_df["category"].astype(str))

        X_val, y_val = None, None
        if val_df is not None:
            X_text_val = self.feature_engine.transform_text(val_df)
            X_num_val = self.feature_engine.transform(val_df).fillna(0.0).values
            X_val = np.hstack([X_text_val, X_num_val])
            # Handle possible unseen classes in val set safely
            y_val = np.array([
                self.label_encoder.transform([c])[0] if c in self.label_encoder.classes_ else -1
                for c in val_df["category"].astype(str)
            ])
            # Filter any unseen class instances
            valid_mask = y_val >= 0
            X_val = X_val[valid_mask]
            y_val = y_val[valid_mask]

        return X_train, y_train, X_val, y_val

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Trains Dummy baseline, Logistic Regression baseline, and XGBoost classifier.
        Records boosting round validation loss metrics.
        """
        X_train, y_train, X_val, y_val = self.prepare_matrices(train_df, val_df)

        # 1. Dummy Baseline
        self.baseline_dummy = DummyClassifier(strategy="most_frequent", random_state=self.random_seed)
        self.baseline_dummy.fit(X_train, y_train)

        # 2. Logistic Regression Baseline
        self.baseline_lr = LogisticRegression(max_iter=1000, random_state=self.random_seed)
        self.baseline_lr.fit(X_train, y_train)

        # 3. XGBoost Multiclass Classifier
        num_classes = len(self.label_encoder.classes_)
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            objective="multi:softprob",
            num_class=num_classes,
            random_state=self.random_seed,
            eval_metric="mlogloss",
        )

        eval_set = [(X_train, y_train)]
        if X_val is not None and len(X_val) > 0:
            eval_set.append((X_val, y_val))

        self.xgb_model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self.training_eval_results = self.xgb_model.evals_result()
        self.is_fitted = True

        # Compute training evaluation summaries
        metrics = self.evaluate(val_df if val_df is not None else train_df)
        return metrics

    def predict_with_confidence(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts categories with probability confidence scores.
        Routes low-confidence predictions to 'Needs Review'.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before calling predict.")

        X_text = self.feature_engine.transform_text(df)
        X_num = self.feature_engine.transform(df).fillna(0.0).values
        X = np.hstack([X_text, X_num])

        probabilities = self.xgb_model.predict_proba(X)
        max_probs = np.max(probabilities, axis=1)
        pred_indices = np.argmax(probabilities, axis=1)
        predicted_classes = self.label_encoder.inverse_transform(pred_indices)

        # Apply confidence threshold
        final_predictions = []
        for pred, prob in zip(predicted_classes, max_probs):
            if prob < self.min_confidence_threshold:
                final_predictions.append("Needs Review")
            else:
                final_predictions.append(pred)

        return np.array(final_predictions), max_probs, probabilities

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluates models against test set, calculating accuracy, macro F1, weighted F1,
        and comparing Dummy baseline, Logistic Regression, and XGBoost.
        """
        X_text = self.feature_engine.transform_text(test_df)
        X_num = self.feature_engine.transform(test_df).fillna(0.0).values
        X_test = np.hstack([X_text, X_num])

        y_true_str = test_df["category"].astype(str).values
        # Encode test labels safely
        known_mask = np.isin(y_true_str, self.label_encoder.classes_)
        if not np.all(known_mask):
            test_df = test_df[known_mask].copy()
            X_test = X_test[known_mask]
            y_true_str = test_df["category"].astype(str).values

        y_test = self.label_encoder.transform(y_true_str)

        # Baseline Dummy predictions
        dummy_preds = self.baseline_dummy.predict(X_test)
        dummy_f1_macro = float(f1_score(y_test, dummy_preds, average="macro", zero_division=0))
        dummy_acc = float(accuracy_score(y_test, dummy_preds))

        # Baseline Logistic Regression predictions
        lr_preds = self.baseline_lr.predict(X_test)
        lr_f1_macro = float(f1_score(y_test, lr_preds, average="macro", zero_division=0))
        lr_acc = float(accuracy_score(y_test, lr_preds))

        # XGBoost predictions
        xgb_preds_raw = self.xgb_model.predict(X_test)
        xgb_probs = self.xgb_model.predict_proba(X_test)
        xgb_conf = np.max(xgb_probs, axis=1)

        xgb_acc = float(accuracy_score(y_test, xgb_preds_raw))
        xgb_f1_macro = float(f1_score(y_test, xgb_preds_raw, average="macro", zero_division=0))
        xgb_f1_weighted = float(f1_score(y_test, xgb_preds_raw, average="weighted", zero_division=0))

        all_labels = list(range(len(self.label_encoder.classes_)))
        cm = confusion_matrix(y_test, xgb_preds_raw, labels=all_labels)
        report = classification_report(
            y_test,
            xgb_preds_raw,
            labels=all_labels,
            target_names=self.label_encoder.classes_,
            output_dict=True,
            zero_division=0,
        )

        return {
            "dummy_baseline": {"accuracy": dummy_acc, "macro_f1": dummy_f1_macro},
            "logistic_regression": {"accuracy": lr_acc, "macro_f1": lr_f1_macro},
            "xgboost": {
                "accuracy": xgb_acc,
                "macro_f1": xgb_f1_macro,
                "weighted_f1": xgb_f1_weighted,
            },
            "confusion_matrix": cm.tolist(),
            "classes": self.label_encoder.classes_.tolist(),
            "classification_report": report,
            "confidence_scores": xgb_conf.tolist(),
        }

    def save(self, filepath: str = "models/categorizer.joblib"):
        """Saves model and artifacts."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str = "models/categorizer.joblib") -> "TransactionCategorizer":
        """Loads fitted model from disk."""
        return joblib.load(filepath)
