
"""Compare baseline, fixed, adaptive, and selective satellite retrieval.

The selective threshold is chosen using validation mAP@100 only.
Final metrics are evaluated on the held-out test split.

Outputs are written to selective_split_reranking.json/.csv so the
existing official-split results remain untouched.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satellite_retrieval.config import load_config
from satellite_retrieval.data import EuroSATSubset
from satellite_retrieval.encoder import build_encoder, extract_embeddings
from satellite_retrieval.metrics import (
    hit_rate_at_k,
    mean_average_precision_at_k,
    top1_accuracy,
)
from satellite_retrieval.retrieval import build_index, search_index
from satellite_retrieval.reranking import (
    adaptive_rerank,
    build_reciprocal_sets,
    reciprocal_rerank_from_sets,
    selective_rerank,
)


def evaluate(query_labels, retrieved_labels, gallery_labels):
    """Calculate retrieval metrics using a shared gallery."""
    return {
        "top1_accuracy": float(
            top1_accuracy(query_labels, retrieved_labels)
        ),
        "hit_rate_at_5": float(
            hit_rate_at_k(query_labels, retrieved_labels, 5)
        ),
        "hit_rate_at_10": float(
            hit_rate_at_k(query_labels, retrieved_labels, 10)
        ),
        "map_at_100": float(
            mean_average_precision_at_k(
                query_labels,
                retrieved_labels,
                100,
                gallery_labels=gallery_labels,
            )
        ),
    }


def evaluate_indices(query_labels, gallery_labels, indices):
    """Evaluate retrieved gallery indices against query labels."""
    return evaluate(
        query_labels,
        gallery_labels[indices],
        gallery_labels,
    )


def main():
    config = load_config(PROJECT_ROOT / "configs" / "experiment.json")
    seed = int(config["seed"])

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    batch_size = 16 if device.type == "cuda" else 8

    print(f"Device: {device}")
    print("Loading official gallery, validation, and test splits...")

    gallery = EuroSATSubset(config, split="gallery")
    validation = EuroSATSubset(config, split="validation")
    test = EuroSATSubset(config, split="test")

    if gallery.label_names != validation.label_names:
        raise ValueError("Gallery and validation class orders differ.")

    if gallery.label_names != test.label_names:
        raise ValueError("Gallery and test class orders differ.")

    def make_loader(dataset):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    print(
        f"Gallery: {len(gallery)} | "
        f"Validation: {len(validation)} | "
        f"Test: {len(test)}"
    )

    model = build_encoder()

    print("Extracting gallery embeddings...")
    gallery_embeddings, gallery_labels = extract_embeddings(
        model, make_loader(gallery), device
    )

    print("Extracting validation embeddings...")
    val_embeddings, val_labels = extract_embeddings(
        model, make_loader(validation), device
    )

    print("Extracting test embeddings...")
    test_embeddings, test_labels = extract_embeddings(
        model, make_loader(test), device
    )

    candidate_k = min(
        max(100, int(config["retrieval"]["top_k"])),
        len(gallery),
    )
    reciprocal_k = int(config["reranking"]["reciprocal_k"])

    index = build_index(gallery_embeddings)

    print("Searching validation candidates...")
    val_scores, val_indices = search_index(
        index, val_embeddings, top_k=candidate_k
    )

    print("Searching test candidates...")
    test_scores, test_indices = search_index(
        index, test_embeddings, top_k=candidate_k
    )

    print("Building gallery reciprocal neighborhoods...")
    reciprocal_sets = build_reciprocal_sets(
        gallery_embeddings,
        reciprocal_k=reciprocal_k,
    )

    # ---------------------------------------------------------------
    # 1. Fixed reranking: choose alpha on validation mAP@100.
    # ---------------------------------------------------------------
    fixed_candidates = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    fixed_validation = {}

    print("\nTuning fixed reranking on validation data...")

    for alpha in fixed_candidates:
        _, indices = reciprocal_rerank_from_sets(
            val_scores,
            val_indices,
            reciprocal_sets,
            alpha=alpha,
            query_neighbor_k=reciprocal_k,
        )

        metrics = evaluate_indices(
            val_labels, gallery_labels, indices
        )
        fixed_validation[alpha] = metrics

        print(
            f"  alpha={alpha:.2f} | "
            f"val mAP@100={metrics['map_at_100']:.6f} | "
            f"Top-1={metrics['top1_accuracy']:.4f}"
        )

    best_fixed_alpha = max(
        fixed_candidates,
        key=lambda a: fixed_validation[a]["map_at_100"],
    )

    # ---------------------------------------------------------------
    # 2. Adaptive reranking: choose temperature on validation.
    # ---------------------------------------------------------------
    alpha_min = 0.70
    alpha_max = 0.95
    temperature_candidates = [0.05, 0.10, 0.20]
    adaptive_validation = {}

    print("\nTuning adaptive temperature on validation data...")

    for temperature in temperature_candidates:
        _, indices, alphas = adaptive_rerank(
            val_scores,
            val_indices,
            reciprocal_sets,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            confidence_temperature=temperature,
            query_neighbor_k=reciprocal_k,
        )

        metrics = evaluate_indices(
            val_labels, gallery_labels, indices
        )

        adaptive_validation[temperature] = {
            "metrics": metrics,
            "alpha_mean": float(np.mean(alphas)),
            "alpha_std": float(np.std(alphas)),
        }

        print(
            f"  temperature={temperature:.2f} | "
            f"val mAP@100={metrics['map_at_100']:.6f} | "
            f"Top-1={metrics['top1_accuracy']:.4f}"
        )

    best_temperature = max(
        temperature_candidates,
        key=lambda t: adaptive_validation[t]["metrics"][
            "map_at_100"
        ],
    )

    # ---------------------------------------------------------------
    # 3. Selective reranking: choose margin threshold on validation.
    #
    # Queries with margin < threshold are reranked.
    # Queries not selected retain their original rankings.
    # Threshold zero is the no-reranking reference.
    # ---------------------------------------------------------------
    selective_threshold_candidates = [
        0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20
    ]
    selective_validation = {}

    print("\nTuning selective threshold on validation data...")

    for threshold in selective_threshold_candidates:
        _, indices, reranked_mask = selective_rerank(
            val_scores,
            val_indices,
            reciprocal_sets,
            margin_threshold=threshold,
            alpha=best_fixed_alpha,
            query_neighbor_k=reciprocal_k,
        )

        metrics = evaluate_indices(
            val_labels, gallery_labels, indices
        )

        selective_validation[threshold] = {
            "metrics": metrics,
            "reranked_fraction": float(np.mean(reranked_mask)),
            "reranked_queries": int(np.sum(reranked_mask)),
        }

        print(
            f"  threshold={threshold:.3f} | "
            f"val mAP@100={metrics['map_at_100']:.6f} | "
            f"Top-1={metrics['top1_accuracy']:.4f} | "
            f"reranked={np.mean(reranked_mask):.1%}"
        )

    best_selective_threshold = max(
        selective_threshold_candidates,
        key=lambda t: selective_validation[t]["metrics"][
            "map_at_100"
        ],
    )

    print(
        f"\nSelected selective threshold: "
        f"{best_selective_threshold:.3f}"
    )

    # ---------------------------------------------------------------
    # 4. Final held-out evaluation.
    # All hyperparameters above were selected using validation only.
    # ---------------------------------------------------------------
    print("\nEvaluating selected methods on held-out test data...")

    baseline_test = evaluate_indices(
        test_labels, gallery_labels, test_indices
    )

    _, fixed_test_indices = reciprocal_rerank_from_sets(
        test_scores,
        test_indices,
        reciprocal_sets,
        alpha=best_fixed_alpha,
        query_neighbor_k=reciprocal_k,
    )

    fixed_test = evaluate_indices(
        test_labels, gallery_labels, fixed_test_indices
    )

    _, adaptive_test_indices, test_alphas = adaptive_rerank(
        test_scores,
        test_indices,
        reciprocal_sets,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        confidence_temperature=best_temperature,
        query_neighbor_k=reciprocal_k,
    )

    adaptive_test = evaluate_indices(
        test_labels, gallery_labels, adaptive_test_indices
    )

    _, selective_test_indices, selective_test_mask = selective_rerank(
        test_scores,
        test_indices,
        reciprocal_sets,
        margin_threshold=best_selective_threshold,
        alpha=best_fixed_alpha,
        query_neighbor_k=reciprocal_k,
    )

    selective_test = evaluate_indices(
        test_labels, gallery_labels, selective_test_indices
    )

    selective_test_fraction = float(np.mean(selective_test_mask))
    selective_test_count = int(np.sum(selective_test_mask))

    methods = {
        "cosine_baseline": baseline_test,
        "fixed_reciprocal_reranking": fixed_test,
        "adaptive_reciprocal_reranking": adaptive_test,
        "selective_reciprocal_reranking": selective_test,
    }

    results = {
        "experiment": "validation_selected_selective_official_split_evaluation",
        "dataset": "timm/eurosat-rgb",
        "encoder": "ImageNet-pretrained ResNet-18",
        "device": str(device),
        "seed": seed,
        "gallery_official_split": "train",
        "validation_official_split": "validation",
        "test_official_split": "test",
        "num_gallery": int(len(gallery)),
        "num_validation_queries": int(len(validation)),
        "num_test_queries": int(len(test)),
        "embedding_dimension": int(gallery_embeddings.shape[1]),
        "candidate_k": int(candidate_k),
        "reciprocal_k": reciprocal_k,
        "selection_metric": "validation map_at_100",
        "fixed_alpha_candidates": fixed_candidates,
        "selected_fixed_alpha": best_fixed_alpha,
        "fixed_validation_metrics": fixed_validation[best_fixed_alpha],
        "adaptive_alpha_bounds": [alpha_min, alpha_max],
        "adaptive_temperature_candidates": temperature_candidates,
        "selected_adaptive_temperature": best_temperature,
        "adaptive_validation_metrics": adaptive_validation[
            best_temperature
        ],
        "test_adaptive_alpha_summary": {
            "mean": float(np.mean(test_alphas)),
            "std": float(np.std(test_alphas)),
            "min": float(np.min(test_alphas)),
            "max": float(np.max(test_alphas)),
        },
        "selective_threshold_candidates": selective_threshold_candidates,
        "selected_selective_threshold": best_selective_threshold,
        "selective_alpha": best_fixed_alpha,
        "selective_validation_metrics": selective_validation[
            best_selective_threshold
        ],
        "test_selective_reranking_fraction": selective_test_fraction,
        "test_selective_reranked_queries": selective_test_count,
        "test_metrics": methods,
        "test_deltas_vs_baseline": {
            method: {
                metric: values[metric] - baseline_test[metric]
                for metric in baseline_test
            }
            for method, values in methods.items()
            if method != "cosine_baseline"
        },
        "protocol_note": (
            "The official training split is the gallery. Fixed alpha, "
            "adaptive temperature, and selective margin threshold are "
            "selected on validation queries by mAP@100. The official "
            "test split is evaluated after selection and is not used "
            "for parameter tuning. Selective reranking uses the "
            "top-two cosine-score margin and does not use query labels "
            "at inference time."
        ),
        "limitations": (
            "This experiment evaluates a selective reranking heuristic. "
            "It does not establish methodological novelty or generalization "
            "beyond this dataset, gallery, encoder, and split."
        ),
    }

    # Use separate filenames to preserve the existing official results.
    output_dir = PROJECT_ROOT / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "selective_split_reranking.json"
    csv_path = output_dir / "selective_split_reranking.csv"

    json_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["method", "metric", "test_value", "delta_vs_baseline"]
        )

        for method, metrics in methods.items():
            for metric, value in metrics.items():
                delta = (
                    value - baseline_test[metric]
                    if method != "cosine_baseline"
                    else 0.0
                )
                writer.writerow([method, metric, value, delta])

    print("\nHeld-out test results")
    print(
        f"{'Method':<36} {'Top-1':>9} {'Hit@5':>9} "
        f"{'Hit@10':>9} {'mAP@100':>10}"
    )

    for method, metrics in methods.items():
        print(
            f"{method:<36} "
            f"{metrics['top1_accuracy']:>9.4f} "
            f"{metrics['hit_rate_at_5']:>9.4f} "
            f"{metrics['hit_rate_at_10']:>9.4f} "
            f"{metrics['map_at_100']:>10.4f}"
        )

    print(f"\nSelected fixed alpha: {best_fixed_alpha}")
    print(f"Selected adaptive temperature: {best_temperature}")
    print(f"Selected selective threshold: {best_selective_threshold}")
    print(
        f"Test queries reranked: {selective_test_count}/"
        f"{len(test)} ({selective_test_fraction:.1%})"
    )
    print(f"Saved results: {json_path}")
    print(f"Saved comparison: {csv_path}")


if __name__ == "__main__":
    main()
