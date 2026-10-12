"""Generate figures from official-split retrieval metrics and diagnostics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def load_official_results():
    path = METRICS_DIR / "official_split_reranking.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing official-split results: {path}\n"
            "Run scripts/run_reranking.py first."
        )

    results = json.loads(path.read_text(encoding="utf-8"))

    if "test_metrics" not in results:
        raise ValueError(
            f"{path} does not contain the expected test_metrics section."
        )

    return results["test_metrics"]


def plot_metric_comparison():
    """Compare all methods on the same official held-out test split."""
    results = load_official_results()

    methods = {
        "Cosine baseline": results["cosine_baseline"],
        "Fixed reranking": results["fixed_reciprocal_reranking"],
        "Adaptive reranking": results["adaptive_reciprocal_reranking"],
    }

    metric_keys = [
        "top1_accuracy",
        "hit_rate_at_5",
        "hit_rate_at_10",
        "map_at_100",
    ]

    metric_labels = [
        "Top-1 accuracy",
        "HitRate@5",
        "HitRate@10",
        "mAP@100",
    ]

    x = np.arange(len(metric_keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))

    for index, (method_name, metrics) in enumerate(methods.items()):
        values = [100 * metrics[key] for key in metric_keys]
        offset = (index - 1) * width

        bars = ax.bar(
            x + offset,
            values,
            width,
            label=method_name,
        )

        ax.bar_label(
            bars,
            fmt="%.1f",
            padding=2,
            fontsize=8,
        )

    ax.set_ylabel("Score (%)")
    ax.set_title(
        "EuroSAT Retrieval: Official Held-Out Test Results"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 112)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output = FIGURES_DIR / "reranking_comparison.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output}")


def plot_class_metrics():
    """Plot baseline retrieval performance for each test class."""
    path = METRICS_DIR / "class_metrics.csv"

    if not path.exists():
        print(f"Skipping class metrics; missing {path}")
        return

    df = pd.read_csv(path)

    required = {
        "class",
        "num_queries",
        "top1_accuracy",
        "hit_rate_at_5",
        "hit_rate_at_10",
    }

    if not required.issubset(df.columns):
        raise ValueError(
            f"{path} is missing required columns: "
            f"{required - set(df.columns)}"
        )

    df = df.sort_values("top1_accuracy")
    x = np.arange(len(df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        x - width,
        100 * df["top1_accuracy"],
        width,
        label="Top-1",
    )
    ax.bar(
        x,
        100 * df["hit_rate_at_5"],
        width,
        label="HitRate@5",
    )
    ax.bar(
        x + width,
        100 * df["hit_rate_at_10"],
        width,
        label="HitRate@10",
    )

    ax.set_ylabel("Score (%)")
    ax.set_title(
        "Cosine Baseline Retrieval Performance by Test Class"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["class"], rotation=35, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    output = FIGURES_DIR / "class_metrics.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output}")


def plot_confusion_matrix():
    """Plot the baseline top-1 confusion matrix."""
    path = METRICS_DIR / "top1_confusion_matrix.csv"

    if not path.exists():
        print(f"Skipping confusion matrix; missing {path}")
        return

    df = pd.read_csv(path).set_index("true_class")
    matrix = df.to_numpy(dtype=int)
    names = df.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(
        matrix,
        interpolation="nearest",
        aspect="auto",
    )
    fig.colorbar(image, ax=ax, label="Query count")

    ax.set_title(
        "Cosine Baseline Top-1 Confusion Matrix on Test Split"
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    positions = np.arange(len(names))
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)

    threshold = matrix.max() / 2 if matrix.size else 0

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row, col] > threshold
                    else "black"
                ),
                fontsize=8,
            )

    fig.tight_layout()

    output = FIGURES_DIR / "top1_confusion_matrix.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_metric_comparison()
    plot_class_metrics()
    plot_confusion_matrix()

    print("Figure generation complete.")


if __name__ == "__main__":
    main()