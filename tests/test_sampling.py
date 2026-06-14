# Autorzy: Viktoriia Nowotka, Paweł Łasica
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.sampling import stratified_split, undersample


def _make_dataset(n_majority=1000, n_minority=100, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n_majority + n_minority, n_features))
    y = np.array([0] * n_majority + [1] * n_minority)
    return X, y


class TestStratifiedSplit:
    def test_sizes(self):
        X, y = _make_dataset()
        X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.1, random_state=0)
        assert len(X_train) + len(X_test) == len(X)
        assert len(X_test) == pytest.approx(110, abs=2)

    def test_stratification_preserves_ratio(self):
        X, y = _make_dataset(1000, 100)
        _, _, y_train, y_test = stratified_split(X, y, test_size=0.1, random_state=0)
        train_ratio = y_train.mean()
        test_ratio  = y_test.mean()
        original_ratio = y.mean()
        assert train_ratio == pytest.approx(original_ratio, abs=0.01)
        assert test_ratio  == pytest.approx(original_ratio, abs=0.02)

    def test_reproducible_with_same_seed(self):
        X, y = _make_dataset()
        split1 = stratified_split(X, y, random_state=42)
        split2 = stratified_split(X, y, random_state=42)
        np.testing.assert_array_equal(split1[1], split2[1])  # same test set

    def test_different_seeds_give_different_splits(self):
        X, y = _make_dataset()
        _, X_test1, _, _ = stratified_split(X, y, random_state=1)
        _, X_test2, _, _ = stratified_split(X, y, random_state=2)
        assert not np.array_equal(X_test1, X_test2)


class TestUndersample:
    def test_balanced_output(self):
        X, y = _make_dataset(1000, 100)
        X_bal, y_bal, _, _ = undersample(X, y, random_state=0)
        assert (y_bal == 0).sum() == (y_bal == 1).sum()

    def test_pool_contains_only_majority(self):
        X, y = _make_dataset(1000, 100)
        _, _, X_pool, y_pool = undersample(X, y, random_state=0)
        assert np.all(y_pool == 0)  # pool is all majority class

    def test_pool_plus_train_equals_original_majority(self):
        X, y = _make_dataset(1000, 100)
        X_bal, y_bal, X_pool, y_pool = undersample(X, y, random_state=0)
        n_majority_in_train = (y_bal == 0).sum()
        assert n_majority_in_train + len(X_pool) == 1000

    def test_all_minority_kept_in_train(self):
        X, y = _make_dataset(1000, 100)
        _, y_bal, _, _ = undersample(X, y, random_state=0)
        assert (y_bal == 1).sum() == 100

    def test_reproducible(self):
        X, y = _make_dataset()
        _, y1, _, _ = undersample(X, y, random_state=7)
        _, y2, _, _ = undersample(X, y, random_state=7)
        np.testing.assert_array_equal(y1, y2)

    def test_extreme_imbalance(self):
        X, y = _make_dataset(n_majority=10000, n_minority=10)
        X_bal, y_bal, X_pool, y_pool = undersample(X, y, random_state=0)
        assert len(X_bal) == 20          # 10 minority + 10 majority
        assert len(X_pool) == 9990
