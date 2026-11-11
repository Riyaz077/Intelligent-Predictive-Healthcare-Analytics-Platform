"""Reusable preprocessing helpers for tabular classification."""
from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_csv(path_or_buffer) -> pd.DataFrame:
    """Load a CSV file and return a dataframe."""
    return pd.read_csv(path_or_buffer)

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build leakage-safe numeric/categorical preprocessing."""
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", num_pipe, numeric))
    if categorical:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", cat_pipe, categorical))
    if not transformers:
        raise ValueError("No usable feature columns were found.")
    return ColumnTransformer(transformers=transformers, remainder="drop")

def split_features_target(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found.")
    clean = df.dropna(subset=[target]).copy()
    X = clean.drop(columns=[target])
    y = clean[target]
    if X.empty:
        raise ValueError("No feature columns remain after selecting the target.")
    return X, y
