# Autorzy: Viktoriia Nowotka, Paweł Łasica
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.cart import CARTModel


def _make_linearly_separable(seed=0):
    rng = np.random.default_rng(seed)
    X0 = rng.random((100, 2))
    X1 = rng.random((100, 2)) + 2.0  # clearly separated
    X = np.vstack([X0, X1])
    y = np.array([0] * 100 + [1] * 100)
    return X, y


class TestCARTModel:
    def test_fit_predict_shape(self):
        X, y = _make_linearly_separable()
        model = CARTModel(random_state=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (200,)

    def test_perfect_on_separable_data(self):
        X, y = _make_linearly_separable()
        model = CARTModel(random_state=0)
        model.fit(X, y)
        assert np.all(model.predict(X) == y)

    def test_output_only_contains_training_labels(self):
        X, y = _make_linearly_separable()
        model = CARTModel(random_state=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_depth_positive_after_fit(self):
        X, y = _make_linearly_separable()
        model = CARTModel(random_state=0)
        model.fit(X, y)
        assert model.depth >= 1

    def test_n_leaves_at_least_two(self):
        X, y = _make_linearly_separable()
        model = CARTModel(random_state=0)
        model.fit(X, y)
        assert model.n_leaves >= 2

    def test_max_depth_respected(self):
        X, y = _make_linearly_separable()
        model = CARTModel(max_depth=2, random_state=0)
        model.fit(X, y)
        assert model.depth <= 2

    def test_reproducible_with_same_seed(self):
        X, y = _make_linearly_separable()
        m1 = CARTModel(random_state=42).fit(X, y)
        m2 = CARTModel(random_state=42).fit(X, y)
        np.testing.assert_array_equal(m1.predict(X), m2.predict(X))
