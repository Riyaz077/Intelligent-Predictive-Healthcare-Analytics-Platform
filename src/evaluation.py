"""Training and evaluation utilities."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             RocCurveDisplay)
from .models import build_model_pipelines

def evaluate_models(X, y, test_size=0.2, random_state=42):
    if y.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    results, fitted = [], {}
    for name, pipeline in build_model_pipelines(X_train, random_state).items():
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
        }
        try:
            if hasattr(pipeline, "predict_proba") and y.nunique() == 2:
                proba = pipeline.predict_proba(X_test)[:, 1]
                row["ROC_AUC"] = roc_auc_score(y_test, proba)
            else:
                row["ROC_AUC"] = np.nan
        except (ValueError, IndexError):
            row["ROC_AUC"] = np.nan
        results.append(row)
        fitted[name] = {
            "pipeline": pipeline, "X_test": X_test, "y_test": y_test,
            "predictions": pred, "confusion_matrix": confusion_matrix(y_test, pred),
            "classification_report": classification_report(y_test, pred, zero_division=0),
        }
    return pd.DataFrame(results).sort_values("F1_weighted", ascending=False), fitted

def permutation_importance_frame(pipeline, X_test, y_test, random_state=42):
    from sklearn.inspection import permutation_importance
    result = permutation_importance(
        pipeline, X_test, y_test, n_repeats=8, random_state=random_state,
        scoring="f1_weighted"
    )
    return pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": result.importances_mean
    }).sort_values("Importance", ascending=False)
