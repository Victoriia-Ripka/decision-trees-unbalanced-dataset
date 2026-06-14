# Autorzy: Viktoriia Nowotka, Paweł Łasica
import argparse
import sys

from src.experiments.config import ExperimentConfig, DATASETS
from src.models.cart import MODELS
from src.experiments.runner import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iterative training on unbalanced datasets with CART decision trees."
    )
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="cart",
        help="Classifier to use: cart (native) or sklearn (default: cart).",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Dataset to run experiments on (default: all).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=25,
        help="Number of repetitions per n_samples value (default: 25).",
    )
    parser.add_argument(
        "--results-dir",
        default="experiments",
        help="Directory for output CSV files (default: experiments).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Global random seed (default: 42).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Stop iterating after this many steps without F1 improvement (default: 10).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=500,
        help="Hard cap on inner loop iterations per run (default: 500).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_names = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for name in dataset_names:
        cfg = ExperimentConfig(
            dataset=DATASETS[name],
            model=args.model,
            n_runs=args.n_runs,
            patience=args.patience,
            max_iterations=args.max_iterations,
            results_dir=args.results_dir,
            random_state=args.random_state,
        )
        try:
            run_experiment(cfg)
        except FileNotFoundError as e:
            print(f"\n[ERROR] {e}")
            print(f"  Place the dataset CSV in: {DATASETS[name].file_path}")
            sys.exit(1)


if __name__ == "__main__":
    main()
