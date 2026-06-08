import numpy as np
from src.models.cart import CARTModel
from src.evaluation.metrics import compute_metrics

# Hyperparameter grid for n_samples (fraction of false positives added each iteration)
N_SAMPLES_FRACS  = [0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.25, 0.35, 1.0]
N_SAMPLES_LABELS = ["0.5%", "1%", "2%", "5%", "10%", "15%", "25%", "35%", "rest"]


def _select_false_positives(
    model: CARTModel,
    X_pool: np.ndarray,
    y_pool: np.ndarray,
) -> np.ndarray:
    """
    Return indices (into X_pool) of majority-class samples that the model
    incorrectly predicts as the minority class (false positives).
    y_pool contains only majority-class labels, so any wrong prediction is a FP.
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
    random_state=None,
) -> list[dict]:
    """
    One full experiment for a given n_samples_frac.

    Starts from the balanced undersampled training set and iteratively
    adds n_samples_frac of the current false positives to the training set.
    Returns a list of metric dicts, one per iteration (iteration=0 is the
    first model trained on the balanced set).
    """
    rng = np.random.default_rng(random_state)
    X_train = X_train_init.copy()
    y_train = y_train_init.copy()
    X_pool  = X_pool_init.copy()
    y_pool  = y_pool_init.copy()

    records = []
    iteration = 0

    while True:
        model = CARTModel(random_state=int(rng.integers(0, 1_000_000)))
        model.fit(X_train, y_train)

        metrics = compute_metrics(y_test, model.predict(X_test))
        records.append({
            **metrics,
            "iteration":      iteration,
            "train_size":     len(X_train),
            "pool_remaining": len(X_pool),
        })

        if len(X_pool) == 0:
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

        iteration += 1

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
    random_state=None,
) -> list[dict]:
    """
    Repeat run_single n_runs times (for statistical averaging).
    Returns all records tagged with run index and n_samples_frac.
    """
    rng = np.random.default_rng(random_state)
    all_records = []

    for run in range(n_runs):
        seed = int(rng.integers(0, 1_000_000))
        records = run_single(
            X_train_init, y_train_init,
            X_pool_init, y_pool_init,
            X_test, y_test,
            n_samples_frac,
            random_state=seed,
        )
        for r in records:
            r["run"] = run
            r["n_samples_frac"] = n_samples_frac
        all_records.extend(records)

    return all_records
