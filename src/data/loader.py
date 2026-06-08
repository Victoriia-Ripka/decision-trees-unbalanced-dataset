import os
import numpy as np
import pandas as pd
from src.experiments.config import DatasetConfig

PROCESSED_BASE = "data/processed"


def _processed_path(dataset_name: str) -> str:
    return os.path.join(PROCESSED_BASE, dataset_name, "index.csv")


def _process_raw(cfg: DatasetConfig) -> pd.DataFrame:
    """Load raw CSV and apply all preprocessing steps."""
    df = pd.read_csv(cfg.file_path)

    if cfg.drop_columns:
        df = df.drop(columns=[c for c in cfg.drop_columns if c in df.columns])

    df = df.dropna(subset=[cfg.target_column])

    for col in cfg.numerical_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    for col in cfg.categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].fillna(df[col].mode()[0])
            df[col] = pd.Categorical(df[col]).codes

    # Binary target: 1 = minority class, 0 = majority
    df["__target__"] = (
        df[cfg.target_column].astype(str).str.strip() == str(cfg.minority_label)
    ).astype(int)

    keep_cols = [
        c for c in cfg.numerical_features + cfg.categorical_features
        if c in df.columns
    ] + ["__target__"]

    return df[keep_cols]


def load_dataset(cfg: DatasetConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Return (X, y, feature_names) for a dataset.

    Flow:
      1. If data/processed/<name>/index.csv exists -> load directly (fast path).
      2. Otherwise -> load data/raw/<name>/index.csv, preprocess, save to
         data/processed/<name>/index.csv, then return arrays.

    Preprocessing applied to raw data:
      - Drop configured columns
      - Rows with missing target are removed
      - Numerical NaN -> column median
      - Categorical NaN -> column mode, then label-encoded to integers
      - Target encoded as binary (1 = minority class, 0 = majority)
    """
    processed = _processed_path(cfg.name)

    if os.path.exists(processed):
        print(f"[cache] Loading processed data from {processed}")
        df = pd.read_csv(processed)
    else:
        print(f"[raw]   Processing {cfg.file_path} ...")
        df = _process_raw(cfg)
        os.makedirs(os.path.dirname(processed), exist_ok=True)
        df.to_csv(processed, index=False)
        print(f"[raw]   Saved processed data to {processed}")

    feature_cols = [c for c in df.columns if c != "__target__"]
    X = df[feature_cols].values.astype(np.float64)
    y = df["__target__"].values.astype(int)

    return X, y, feature_cols
