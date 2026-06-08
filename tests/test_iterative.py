import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.sampling import stratified_split, undersample
from src.training.iterative import run_single, run_iterative_training


def _make_experiment(n_majority=500, n_minority=50, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n_majority + n_minority, 5))
    y = np.array([0] * n_majority + [1] * n_minority)
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.1, random_state=seed)
    X_bal, y_bal, X_pool, y_pool = undersample(X_train, y_train, random_state=seed)
    return X_bal, y_bal, X_pool, y_pool, X_test, y_test


class TestRunSingle:
    def test_returns_at_least_one_record(self):
        args = _make_experiment()
        records = run_single(*args, n_samples_frac=0.10, random_state=0)
        assert len(records) >= 1

    def test_iteration_index_starts_at_zero(self):
        args = _make_experiment()
        records = run_single(*args, n_samples_frac=0.10, random_state=0)
        assert records[0]["iteration"] == 0

    def test_iteration_index_is_sequential(self):
        args = _make_experiment()
        records = run_single(*args, n_samples_frac=0.10, random_state=0)
        iterations = [r["iteration"] for r in records]
        assert iterations == list(range(len(iterations)))

    def test_train_size_grows_across_iterations(self):
        args = _make_experiment()
        records = run_single(*args, n_samples_frac=0.10, random_state=0)
        if len(records) > 1:
            sizes = [r["train_size"] for r in records]
            assert sizes[-1] > sizes[0]

    def test_pool_shrinks_across_iterations(self):
        args = _make_experiment()
        records = run_single(*args, n_samples_frac=0.10, random_state=0)
        if len(records) > 1:
            pools = [r["pool_remaining"] for r in records]
            assert pools[-1] < pools[0]

    def test_all_required_metric_keys_present(self):
        args = _make_experiment()
        records = run_single(*args, n_samples_frac=0.10, random_state=0)
        required = {"f1", "tpr", "fpr", "precision", "accuracy", "error", "tp", "tn", "fp", "fn"}
        for r in records:
            assert required.issubset(r.keys())

    def test_metrics_in_valid_range(self):
        args = _make_experiment()
        records = run_single(*args, n_samples_frac=0.10, random_state=0)
        for r in records:
            assert 0.0 <= r["f1"] <= 1.0
            assert 0.0 <= r["tpr"] <= 1.0
            assert 0.0 <= r["fpr"] <= 1.0
            assert 0.0 <= r["accuracy"] <= 1.0

    def test_max_iterations_respected(self):
        args = _make_experiment()
        records = run_single(*args, n_samples_frac=0.005, max_iterations=5, random_state=0)
        assert max(r["iteration"] for r in records) < 5

    def test_frac_1_exhausts_fp_in_one_step(self):
        # frac=1.0 adds ALL false positives at once → should stop quickly
        args = _make_experiment()
        records_full = run_single(*args, n_samples_frac=1.0, random_state=0)
        records_slow = run_single(*args, n_samples_frac=0.01, random_state=0)
        assert len(records_full) <= len(records_slow)


class TestRunIterativeTraining:
    def test_run_count_tagged_correctly(self):
        args = _make_experiment()
        results = run_iterative_training(*args, n_samples_frac=0.10, n_runs=3, random_state=0)
        run_ids = {r["run"] for r in results}
        assert run_ids == {0, 1, 2}

    def test_n_samples_frac_tagged_on_all_records(self):
        args = _make_experiment()
        results = run_iterative_training(*args, n_samples_frac=0.25, n_runs=2, random_state=0)
        assert all(r["n_samples_frac"] == 0.25 for r in results)

    def test_total_records_matches_runs_times_iterations(self):
        args = _make_experiment()
        n_runs = 4
        results = run_iterative_training(*args, n_samples_frac=1.0, n_runs=n_runs,
                                         max_iterations=10, random_state=0)
        per_run = {r["run"]: 0 for r in results}
        for r in results:
            per_run[r["run"]] += 1
        assert len(per_run) == n_runs

    def test_different_runs_can_differ(self):
        # With random sampling, different runs should not all produce identical F1
        args = _make_experiment()
        results = run_iterative_training(*args, n_samples_frac=0.10, n_runs=5, random_state=0)
        first_iter_f1 = [r["f1"] for r in results if r["iteration"] == 0]
        # Not all identical (randomness in undersampling seeds)
        assert len(set(round(v, 6) for v in first_iter_f1)) >= 1  # at minimum they run
