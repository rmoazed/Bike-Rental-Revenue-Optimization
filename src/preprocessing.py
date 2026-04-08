from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

TARGET_COLUMN = "cnt"
DROP_COLUMNS = ["cnt", "casual", "registered", "dteday", "instant"]
CATEGORICAL_COLUMNS = ["season", "weekday", "weathersit"]


def load_data(path: str) -> pd.DataFrame:
    """Load the bike sharing dataset from a CSV file."""
    return pd.read_csv(path)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features from the raw dataframe.

    This mirrors the preprocessing done in the notebook:
    - drops leakage / identifier columns
    - creates cyclical month features
    - one-hot encodes categorical features
    """
    X = df.drop(columns=DROP_COLUMNS, errors="ignore").copy()

    if "mnth" not in X.columns:
        raise ValueError("Expected column 'mnth' was not found in the input dataframe.")

    X["month_sin"] = np.sin(2 * np.pi * X["mnth"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * X["mnth"] / 12)
    X = X.drop(columns=["mnth"])

    missing_cats = [col for col in CATEGORICAL_COLUMNS if col not in X.columns]
    if missing_cats:
        raise ValueError(f"Missing categorical columns required for encoding: {missing_cats}")

    X = pd.get_dummies(X, columns=CATEGORICAL_COLUMNS, drop_first=False)
    X = X.astype(float)
    return X


def build_target(df: pd.DataFrame) -> pd.Series:
    """Return the target column used for demand prediction."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' was not found.")
    return df[TARGET_COLUMN].copy()


def align_feature_columns(X: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Align a dataframe to the exact feature-column order used in training."""
    X_aligned = X.copy()
    for col in feature_columns:
        if col not in X_aligned.columns:
            X_aligned[col] = 0.0

    extra_cols = [col for col in X_aligned.columns if col not in feature_columns]
    if extra_cols:
        X_aligned = X_aligned.drop(columns=extra_cols)

    return X_aligned[feature_columns]


def prepare_training_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Convenience function to load data and return model-ready X and y."""
    df = load_data(path)
    X = build_features(df)
    y = build_target(df)
    return X, y
