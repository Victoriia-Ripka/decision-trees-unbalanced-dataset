import os
import numpy as np
import pandas as pd

from src.data.loader import load_dataset
from src.training.sampling import stratified_split, undersample
from src.training.iterative import (
    run_iterative_training,
    N_SAMPLES_FRACS,
    N_SAMPLES_LABELS,
)
from src.models.cart import CARTModel
from src.evaluation.metrics import compute_metrics
from src.experiments.config import ExperimentConfig


def run_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    """
    Full experiment pipeline for one dataset.

    1. Load + stratified split (90/10)
    2. Baseline model M_0 on unbalanced train set
    3. Undersample -> balanced train set + pool
    4. For each n_samples value: 25 repetitions of iterative training
    5. Return aggregated results as DataFrame
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {cfg.dataset.name}")
    print(f"{'='*60}")

    X, y, feature_names = load_dataset(cfg.dataset)

    total = len(y)
    minority = y.sum()
    majority = total - minority

    print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(
        f"Class distribution: "
        f"minority={minority} ({minority / total * 100:.2f}%), "
        f"majority={majority} ({majority / total * 100:.2f}%)"
    )

    X_train_full, X_test, y_train_full, y_test = stratified_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    # Baseline M_0: trained on full (unbalanced) train set
    m0 = CARTModel(random_state=cfg.random_state)
    m0.fit(X_train_full, y_train_full)
    m0_metrics = compute_metrics(y_test, m0.predict(X_test))
    print(f"\nBaseline M_0 (unbalanced): F1={m0_metrics['f1']:.4f}, "
          f"TPR={m0_metrics['tpr']:.4f}, Accuracy={m0_metrics['accuracy']:.4f}")

    X_train_bal, y_train_bal, X_pool, y_pool = undersample(
        X_train_full, y_train_full, random_state=cfg.random_state
    )
    print(f"After undersampling: train={len(X_train_bal)}, pool={len(X_pool)}")

    all_records = []

    # Baseline metrics as reference row
    all_records.append({
        **m0_metrics,
        "dataset":       cfg.dataset.name,
        "n_samples_frac": None,
        "n_samples_label": "M_0 (unbalanced)",
        "run":            -1,
        "iteration":      0,
        "train_size":     len(X_train_full),
        "pool_remaining": len(X_pool),
    })

    for frac, label in zip(N_SAMPLES_FRACS, N_SAMPLES_LABELS):
        print(f"  n_samples={label} ...", end=" ", flush=True)
        records = run_iterative_training(
            X_train_bal, y_train_bal,
            X_pool, y_pool,
            X_test, y_test,
            n_samples_frac=frac,
            n_runs=cfg.n_runs,
            random_state=cfg.random_state,
        )
        for r in records:
            r["dataset"] = cfg.dataset.name
            r["n_samples_label"] = label
        all_records.extend(records)
        print(f"done ({len(records)} records)")

    df = pd.DataFrame(all_records)
    _save_results(df, cfg)
    return df


def _save_results(df: pd.DataFrame, cfg: ExperimentConfig) -> None:
    out_dir = os.path.join(cfg.results_dir, cfg.dataset.name)
    os.makedirs(out_dir, exist_ok=True)

    raw_path = os.path.join(out_dir, "results_raw.csv")
    df.to_csv(raw_path, index=False)

    metric_cols = ["accuracy", "error", "tpr", "fpr", "precision", "f1"]
    agg = (
        df[df["run"] >= 0]
        .groupby(["n_samples_label", "iteration"])[metric_cols]
        .agg(["mean", "std"])
        .round(6)
    )
    agg_path = os.path.join(out_dir, "results_aggregated.csv")
    agg.to_csv(agg_path)

    print(f"  Results saved to {out_dir}/")
