# Autorzy: Viktoriia Nowotka, Paweł Łasica
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from src.training.iterative import N_SAMPLES_LABELS

METRIC_LABELS = {
    "f1":        "F1-score",
    "tpr":       "TPR (Sensitivity)",
    "fpr":       "FPR",
    "precision": "Precision",
    "accuracy":  "Accuracy",
}

# Consistent color per n_samples label
_CMAP = plt.cm.tab10


def _palette(labels):
    return {lbl: _CMAP(i / max(len(labels) - 1, 1)) for i, lbl in enumerate(labels)}


def _load_and_aggregate(raw_csv: str) -> tuple[pd.DataFrame, float | None]:
    """Return (aggregated_df, baseline_f1). Baseline rows have run == -1."""
    df = pd.read_csv(raw_csv)

    baseline_f1 = None
    baseline_rows = df[df["run"] == -1]
    if not baseline_rows.empty:
        baseline_f1 = baseline_rows["f1"].mean()

    df = df[df["run"] >= 0].copy()

    # Preserve N_SAMPLES_LABELS order via Categorical
    present = [l for l in N_SAMPLES_LABELS if l in df["n_samples_label"].unique()]
    df["n_samples_label"] = pd.Categorical(df["n_samples_label"], categories=present, ordered=True)

    agg = (
        df.groupby(["n_samples_label", "iteration"], observed=True)
        [list(METRIC_LABELS.keys())]
        .agg(["mean", "std"])
        .reset_index()
    )
    # Flatten multi-level columns: (f1, mean) → f1_mean
    agg.columns = [
        "_".join(c).strip("_") if isinstance(c, tuple) else c
        for c in agg.columns
    ]
    return agg, baseline_f1


def plot_metric_vs_iteration(
    agg: pd.DataFrame,
    metric: str,
    baseline_value: float | None,
    out_path: str,
    dataset_name: str,
) -> None:
    """One chart: metric vs iteration, one line per n_samples value."""
    labels = list(agg["n_samples_label"].cat.categories) if hasattr(agg["n_samples_label"], "cat") else [l for l in N_SAMPLES_LABELS if l in agg["n_samples_label"].unique()]
    palette = _palette(labels)

    fig, ax = plt.subplots(figsize=(10, 6))

    for label in labels:
        sub = agg[agg["n_samples_label"] == label].sort_values("iteration")
        mean_col = f"{metric}_mean"
        std_col  = f"{metric}_std"
        color = palette[label]

        ax.plot(sub["iteration"], sub[mean_col], label=label, color=color, linewidth=1.8)
        ax.fill_between(
            sub["iteration"],
            sub[mean_col] - sub[std_col],
            sub[mean_col] + sub[std_col],
            alpha=0.12,
            color=color,
        )

    if baseline_value is not None:
        ax.axhline(
            baseline_value,
            color="black",
            linestyle="--",
            linewidth=1.4,
            label="M₀ baseline (unbalanced)",
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(f"{dataset_name} — {METRIC_LABELS[metric]} vs Iteration")
    ax.legend(title="n_samples", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_final_comparison(
    agg: pd.DataFrame,
    baseline_value: float | None,
    out_path: str,
    dataset_name: str,
) -> None:
    """Bar chart: final F1 (last recorded iteration) for each n_samples value."""
    last = (
        agg.sort_values("iteration")
        .groupby("n_samples_label", observed=True)
        .last()
        .reset_index()
    )
    # Restore defined order
    ordered = [l for l in N_SAMPLES_LABELS if l in last["n_samples_label"].values]
    last["n_samples_label"] = pd.Categorical(last["n_samples_label"], categories=ordered, ordered=True)
    last = last.sort_values("n_samples_label")

    labels = last["n_samples_label"].tolist()
    palette = _palette(labels)
    colors = [palette[l] for l in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, last["f1_mean"], color=colors, yerr=last["f1_std"],
                  capsize=4, error_kw={"linewidth": 1})

    if baseline_value is not None:
        ax.axhline(
            baseline_value,
            color="black",
            linestyle="--",
            linewidth=1.4,
            label=f"M₀ baseline = {baseline_value:.3f}",
        )
        ax.legend(fontsize=9)

    for bar, val in zip(bars, last["f1_mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("n_samples")
    ax.set_ylabel("F1-score (final iteration, mean ± std)")
    ax.set_title(f"{dataset_name} — Final F1 by n_samples")
    ax.set_ylim(0, min(1.0, last["f1_mean"].max() + 0.08))
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_metrics(
    agg: pd.DataFrame,
    baseline_values: dict,
    out_path: str,
    dataset_name: str,
) -> None:
    """2x2 grid: F1, TPR, FPR, Precision."""
    metrics = ["f1", "tpr", "fpr", "precision"]
    labels = list(agg["n_samples_label"].cat.categories) if hasattr(agg["n_samples_label"], "cat") else [l for l in N_SAMPLES_LABELS if l in agg["n_samples_label"].unique()]
    palette = _palette(labels)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        for label in labels:
            sub = agg[agg["n_samples_label"] == label].sort_values("iteration")
            color = palette[label]
            ax.plot(sub["iteration"], sub[f"{metric}_mean"], label=label,
                    color=color, linewidth=1.6)
            ax.fill_between(
                sub["iteration"],
                sub[f"{metric}_mean"] - sub[f"{metric}_std"],
                sub[f"{metric}_mean"] + sub[f"{metric}_std"],
                alpha=0.10, color=color,
            )
        if metric in baseline_values and baseline_values[metric] is not None:
            ax.axhline(baseline_values[metric], color="black", linestyle="--",
                       linewidth=1.2, label="M₀ baseline")

        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("Iteration")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=0.3)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, title="n_samples", loc="lower center",
               ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{dataset_name} — All Metrics vs Iteration", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(raw_csv: str, out_dir: str, dataset_name: str) -> None:
    """Generate all plots from a raw results CSV and save to out_dir."""
    agg, baseline_f1 = _load_and_aggregate(raw_csv)

    os.makedirs(out_dir, exist_ok=True)

    # 1. F1 vs iteration (main plot)
    plot_metric_vs_iteration(
        agg, "f1", baseline_f1,
        os.path.join(out_dir, "f1_vs_iteration.png"),
        dataset_name,
    )

    # 2. Final F1 bar chart
    plot_final_comparison(
        agg, baseline_f1,
        os.path.join(out_dir, "final_f1_comparison.png"),
        dataset_name,
    )

    # 3. All metrics 2x2 grid
    baseline_row = pd.read_csv(raw_csv)
    baseline_row = baseline_row[baseline_row["run"] == -1]
    baseline_vals = {m: baseline_row[m].mean() if not baseline_row.empty else None
                     for m in METRIC_LABELS}
    plot_confusion_metrics(
        agg, baseline_vals,
        os.path.join(out_dir, "all_metrics_grid.png"),
        dataset_name,
    )

    print(f"  Plots saved to {out_dir}/")
