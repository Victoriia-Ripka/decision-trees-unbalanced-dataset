import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.metrics import compute_metrics


def _y(majority, minority):
    return np.array([0] * majority + [1] * minority)


class TestComputeMetrics:
    def test_perfect_classifier(self):
        y = _y(90, 10)
        m = compute_metrics(y, y)
        assert m["f1"] == pytest.approx(1.0)
        assert m["tpr"] == pytest.approx(1.0)
        assert m["fpr"] == pytest.approx(0.0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["error"] == pytest.approx(0.0)

    def test_all_wrong_minority(self):
        # Model always predicts majority class (0)
        y_true = _y(90, 10)
        y_pred = np.zeros(100, dtype=int)
        m = compute_metrics(y_true, y_pred)
        assert m["tpr"] == pytest.approx(0.0)   # no minority correctly found
        assert m["f1"] == pytest.approx(0.0)
        assert m["tp"] == 0
        assert m["fn"] == 10

    def test_all_predicted_minority(self):
        # Model always predicts minority class (1)
        y_true = _y(90, 10)
        y_pred = np.ones(100, dtype=int)
        m = compute_metrics(y_true, y_pred)
        assert m["tpr"] == pytest.approx(1.0)   # all minority found
        assert m["fpr"] == pytest.approx(1.0)   # all majority also flagged
        assert m["precision"] == pytest.approx(0.10)
        assert m["fp"] == 90
        assert m["tn"] == 0

    def test_known_values(self):
        # TP=3, TN=5, FP=2, FN=1  (minority=1, majority=0)
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0])
        m = compute_metrics(y_true, y_pred)
        assert m["tp"] == 3
        assert m["fn"] == 1
        assert m["fp"] == 2
        assert m["tn"] == 5
        assert m["tpr"] == pytest.approx(3 / 4)
        assert m["fpr"] == pytest.approx(2 / 7)
        assert m["precision"] == pytest.approx(3 / 5)
        expected_f1 = 2 * (3/5) * (3/4) / ((3/5) + (3/4))
        assert m["f1"] == pytest.approx(expected_f1)

    def test_minority_is_always_positive_class(self):
        # minority class = 0 (fewer samples), majority = 1
        y_true = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        m = compute_metrics(y_true, y_pred)
        # TP=1 (0 predicted as 0), FN=1 (0 predicted as 1)
        assert m["tp"] == 1
        assert m["fn"] == 1
        assert m["tpr"] == pytest.approx(0.5)

    def test_accuracy_formula(self):
        y_true = _y(80, 20)
        y_pred = _y(80, 20)
        y_pred[0] = 1  # 1 majority wrongly classified
        y_pred[80] = 0  # 1 minority wrongly classified
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == pytest.approx(98 / 100)
        assert m["error"] == pytest.approx(1.0 - m["accuracy"])
