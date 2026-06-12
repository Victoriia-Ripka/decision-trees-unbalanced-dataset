import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.loader import load_dataset
from src.training.sampling import stratified_split, undersample
from src.training.iterative import (
    run_iterative_training,
    N_SAMPLES_FRACS,
    N_SAMPLES_LABELS,
)
from src.models.cart import MODELS
from src.evaluation.metrics import compute_metrics, prefix_metrics
from src.experiments.config import ExperimentConfig
from src.experiments.plots import generate_all_plots


def run_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    """
    Full experiment pipeline for one dataset.

    1. Load + stratified split (90/10)
    2. Baseline model M_0 on unbalanced train set
    3. Undersample -> balanced train set + pool
    4. For each n_samples value: 25 repetitions of iterative training
    5. Return aggregated results as DataFrame
    """
    print(f"\n{'='*128}")
    print(f"Dataset: {cfg.dataset.name}  |  Model: {cfg.model}")
    print(f"{'='*128}")

    model_class = MODELS[cfg.model]

    X, y, feature_names = load_dataset(cfg.dataset)

    total = len(y)
    minority = int(y.sum())
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
    m0 = model_class(random_state=cfg.random_state)
    m0.fit(X_train_full, y_train_full)
    m0_metrics = compute_metrics(y_test, m0.predict(X_test))
    m0_train_metrics = compute_metrics(y_train_full, m0.predict(X_train_full))
    print(f"\nBaseline M_0 (unbalanced): F1={m0_metrics['f1']:.4f}  "
          f"TPR={m0_metrics['tpr']:.4f}  Accuracy={m0_metrics['accuracy']:.4f}")

    X_train_bal, y_train_bal, X_pool, y_pool = undersample(
        X_train_full, y_train_full, random_state=cfg.random_state
    )
    print(f"After undersampling: train={len(X_train_bal)}, pool={len(X_pool)}\n")

    all_records = []

    all_records.append({
        **m0_metrics,
        **prefix_metrics(m0_train_metrics, "train"),
        "dataset":         cfg.dataset.name,
        "model":           cfg.model,
        "n_samples_frac":  None,
        "n_samples_label": "M_0 (unbalanced)",
        "run":             -1,
        "iteration":       0,
        "train_size":      len(X_train_full),
        "pool_remaining":  len(X_pool),
    })

    outer_bar = tqdm(
        zip(N_SAMPLES_FRACS, N_SAMPLES_LABELS),
        total=len(N_SAMPLES_FRACS),
        desc=f"[{cfg.dataset.name}] n_samples",
        unit="config",
    )

    for frac, label in outer_bar:
        outer_bar.set_postfix(n_samples=label)
        records = run_iterative_training(
            X_train_bal, y_train_bal,
            X_pool, y_pool,
            X_test, y_test,
            n_samples_frac=frac,
            n_runs=cfg.n_runs,
            patience=cfg.patience,
            max_iterations=cfg.max_iterations,
            random_state=cfg.random_state,
            model_class=model_class,
        )
        for r in records:
            r["dataset"] = cfg.dataset.name
            r["model"] = cfg.model
            r["n_samples_label"] = label
        all_records.extend(records)

    print()
    df = pd.DataFrame(all_records)
    _save_results(df, cfg)
    return df


def _save_results(df: pd.DataFrame, cfg: ExperimentConfig) -> None:
    out_dir = os.path.join(cfg.results_dir, cfg.dataset.name)
    os.makedirs(out_dir, exist_ok=True)

    raw_path = os.path.join(out_dir, "results_raw.csv")
    df.to_csv(raw_path, index=False)

    metric_cols = ["accuracy", "error", "tpr", "fpr", "precision", "f1"]
    train_metric_cols = [f"train_{c}" for c in metric_cols]
    agg = (
        df[df["run"] >= 0]
        .groupby(["n_samples_label", "iteration"])[metric_cols + train_metric_cols]
        .agg(["mean", "std"])
        .round(6)
    )
    agg_path = os.path.join(out_dir, "results_aggregated.csv")
    agg.to_csv(agg_path)

    print(f"CSVs saved to {out_dir}/")
    generate_all_plots(raw_path, out_dir, cfg.dataset.name)
