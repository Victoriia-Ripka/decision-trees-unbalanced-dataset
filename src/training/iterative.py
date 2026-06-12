import numpy as np
from tqdm import tqdm
from src.models.cart import CARTModel
from src.evaluation.metrics import compute_metrics, prefix_metrics

# Hyperparameter grid for n_samples (fraction of false positives added each iteration)
N_SAMPLES_FRACS  = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 1.0]
N_SAMPLES_LABELS = ["0.5%", "1%", "2%", "5%", "10%", "15%", "25%", "35%", "rest"]

# Stop when pool drops below this fraction of original size (hard floor).
_MIN_POOL_FRACTION = 0.01


def _select_false_positives(
    model,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
) -> np.ndarray:
    """
    Return indices (into X_pool) of majority-class samples incorrectly
    predicted as the minority class (false positives).
    y_pool is all majority class, so any misprediction is a FP.
    """
    preds = model.predict(X_pool)
    return np.where(preds != y_pool)[0]


def run_single(
    X_train_init: np.ndarray,
    y_train_init: np.ndarray,
    X_pool_init: np.ndarray,
    y_pool_init: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_samples_frac: float,
    patience: int = 10,
    max_iterations: int = 500,
    random_state=None,
    model_class=CARTModel,
    _run_label: str = "",
) -> list[dict]:
    """
    One full iterative experiment for a given n_samples_frac.

    Stops when (first condition met):
    - pool is exhausted (< 1% of original size), OR
    - no false positives remain, OR
    - F1 has not improved for `patience` consecutive iterations (convergence), OR
    - max_iterations is reached (hard safety cap, should rarely trigger)
    """
    rng = np.random.default_rng(random_state)
    X_train = X_train_init.copy()
    y_train = y_train_init.copy()
    X_pool  = X_pool_init.copy()
    y_pool  = y_pool_init.copy()

    pool_size_init = len(X_pool)
    min_pool_size  = max(1, int(pool_size_init * _MIN_POOL_FRACTION))

    records = []
    best_f1 = -1.0
    no_improve = 0

    bar = tqdm(
        range(max_iterations),
        desc=f"    iter{_run_label}",
        leave=False,
        unit="iter",
        dynamic_ncols=True,
    )

    for iteration in bar:
        model = model_class(random_state=int(rng.integers(0, 1_000_000)))
        model.fit(X_train, y_train)

        metrics = compute_metrics(y_test, model.predict(X_test))
        train_metrics = compute_metrics(y_train, model.predict(X_train))
        records.append({
            **metrics,
            **prefix_metrics(train_metrics, "train"),
            "iteration":      iteration,
            "train_size":     len(X_train),
            "pool_remaining": len(X_pool),
        })

        # Convergence check
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            no_improve = 0
        else:
            no_improve += 1

        bar.set_postfix(
            f1=f"{metrics['f1']:.3f}",
            best=f"{best_f1:.3f}",
            no_imp=no_improve,
            pool=len(X_pool),
        )

        if no_improve >= patience:
            break

        if len(X_pool) <= min_pool_size:
            break

        fp_idx = _select_false_positives(model, X_pool, y_pool)

        if len(fp_idx) == 0:
            break

        if n_samples_frac >= 1.0:
            selected = fp_idx
        else:
            n_select = max(1, round(len(fp_idx) * n_samples_frac))
            selected = rng.choice(fp_idx, size=min(n_select, len(fp_idx)), replace=False)

        X_train = np.vstack([X_train, X_pool[selected]])
        y_train = np.concatenate([y_train, y_pool[selected]])

        keep = np.ones(len(X_pool), dtype=bool)
        keep[selected] = False
        X_pool = X_pool[keep]
        y_pool = y_pool[keep]

    bar.close()
    return records


def run_iterative_training(
    X_train_init: np.ndarray,
    y_train_init: np.ndarray,
    X_pool_init: np.ndarray,
    y_pool_init: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_samples_frac: float,
    n_runs: int = 25,
    patience: int = 10,
    max_iterations: int = 500,
    random_state=None,
    model_class=CARTModel,
) -> list[dict]:
    """
    Repeat run_single n_runs times for statistical averaging.
    Returns all records tagged with run index and n_samples_frac.
    """
    rng = np.random.default_rng(random_state)
    all_records = []

    runs_bar = tqdm(
        range(n_runs),
        desc=f"  runs (n_samples={n_samples_frac:.3f})",
        leave=False,
        unit="run",
        dynamic_ncols=True,
    )

    for run in runs_bar:
        seed = int(rng.integers(0, 1_000_000))
        records = run_single(
            X_train_init, y_train_init,
            X_pool_init, y_pool_init,
            X_test, y_test,
            n_samples_frac,
            patience=patience,
            max_iterations=max_iterations,
            random_state=seed,
            model_class=model_class,
            _run_label=f" run={run}",
        )
        for r in records:
            r["run"] = run
            r["n_samples_frac"] = n_samples_frac

        all_records.extend(records)

        if records:
            last = records[-1]
            runs_bar.set_postfix(
                iters=last["iteration"],
                f1=f"{last['f1']:.3f}",
            )

    return all_records
