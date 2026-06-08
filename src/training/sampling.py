import numpy as np
from sklearn.model_selection import train_test_split


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.1,
    random_state=None,
):
    """Split into train/test keeping class distribution (stratified 90/10)."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def undersample(
    X: np.ndarray,
    y: np.ndarray,
    random_state=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Balance training set by random undersampling of the majority class.

    Returns
    -------
    X_train, y_train : balanced training set (minority_n × 2 samples)
    X_pool, y_pool   : unused majority-class samples (candidates for future iterations)
    """
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    majority_class = classes[np.argmax(counts)]

    minority_idx = np.where(y == minority_class)[0]
    majority_idx = np.where(y == majority_class)[0]

    n_minority = len(minority_idx)
    selected_majority = rng.choice(majority_idx, size=n_minority, replace=False)
    pool_idx = np.setdiff1d(majority_idx, selected_majority)

    train_idx = np.concatenate([minority_idx, selected_majority])
    rng.shuffle(train_idx)

    return X[train_idx], y[train_idx], X[pool_idx], y[pool_idx]
