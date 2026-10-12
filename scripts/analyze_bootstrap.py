
"""Paired, class-stratified bootstrap for held-out retrieval metrics."""

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
from satellite_retrieval.retrieval import build_index, search_index
from satellite_retrieval.reranking import (
    adaptive_rerank,
    build_reciprocal_sets,
    reciprocal_rerank_from_sets,
)


def per_query_metrics(query_labels, retrieved_labels, gallery_labels):
    """Return one value per query for each reported metric."""
    query_labels = np.asarray(query_labels)
    retrieved_labels = np.asarray(retrieved_labels)
    gallery_labels = np.asarray(gallery_labels)

    relevance = retrieved_labels == query_labels[:, None]
    n_queries, width = relevance.shape

    top1 = relevance[:, 0].astype(float)
    hit5 = np.any(relevance[:, :min(5, width)], axis=1).astype(float)
    hit10 = np.any(relevance[:, :min(10, width)], axis=1).astype(float)

    k = min(100, width)
    rel_k = relevance[:, :k]
    precision = np.cumsum(rel_k, axis=1) / np.arange(1, k + 1)
    summed_precision = np.sum(precision * rel_k, axis=1)

    relevant_total = np.array([
        np.count_nonzero(gallery_labels == label)
        for label in query_labels
    ])
    denominator = np.maximum(np.minimum(relevant_total, k), 1)
    map100 = summed_precision / denominator

    if n_queries == 0:
        raise ValueError("No queries were provided.")

    return {
        "top1_accuracy": top1,
        "hit_rate_at_5": hit5,
        "hit_rate_at_10": hit10,
        "map_at_100": map100,
    }


def stratified_paired_bootstrap(
    query_labels,
    method_values,
    baseline_name="cosine_baseline",
    n_bootstrap=2000,
    seed=2026,
):
    """Bootstrap paired method-minus-baseline differences within classes."""
    query_labels = np.asarray(query_labels)
    rng = np.random.default_rng(seed)
    class_indices = [
        np.flatnonzero(query_labels == label)
        for label in np.unique(query_labels)
    ]

    results = []
    baseline = method_values[baseline_name]

    for method_name, metrics in method_values.items():
        if method_name == baseline_name:
            continue

        for metric_name, baseline_values in baseline.items():
            method_metric = metrics[metric_name]
            paired_difference = method_metric - baseline_values

            # Preserve the observed number of queries in every class.
            bootstrap_means = np.empty(n_bootstrap, dtype=float)
            for b in range(n_bootstrap):
                sampled_parts = [
                    rng.choice(indices, size=len(indices), replace=True)
                    for indices in class_indices
                ]
                sampled = np.concatenate(sampled_parts)
                bootstrap_means[b] = np.mean(paired_difference[sampled])

            lower, upper = np.quantile(
                bootstrap_means, [0.025, 0.975]
            )
            observed = float(np.mean(paired_difference))

            results.append({
                "method": method_name,
                "metric": metric_name,
                "delta_vs_baseline": observed,
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "bootstrap_replicates": n_bootstrap,
                "seed": seed,
                "interval_excludes_zero": bool(lower > 0 or upper < 0),
            })

    return results


def main():
    config = load_config(PROJECT_ROOT / "configs" / "experiment.json")
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16 if device.type == "cuda" else 8

    gallery = EuroSATSubset(config, split="gallery")
    validation = EuroSATSubset(config, split="validation")
    test = EuroSATSubset(config, split="test")

    if not (
        gallery.label_names
        == validation.label_names
        == test.label_names
    ):
        raise ValueError("Dataset class order differs between splits.")

    def loader(dataset):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    print(f"Device: {device}")
    print("Re-extracting embeddings for paired test analysis...")

    model = build_encoder()
    gallery_embeddings, gallery_labels = extract_embeddings(
        model, loader(gallery), device
    )
    val_embeddings, val_labels = extract_embeddings(
        model, loader(validation), device
    )
    test_embeddings, test_labels = extract_embeddings(
        model, loader(test), device
    )

    candidate_k = min(
        max(100, int(config["retrieval"]["top_k"])),
        len(gallery),
    )
    reciprocal_k = int(config["reranking"]["reciprocal_k"])

    index = build_index(gallery_embeddings)
    val_scores, val_indices = search_index(
        index, val_embeddings, top_k=candidate_k
    )
    test_scores, test_indices = search_index(
        index, test_embeddings, top_k=candidate_k
    )

    reciprocal_sets = build_reciprocal_sets(
        gallery_embeddings, reciprocal_k=reciprocal_k
    )

    # Reproduce validation selection for the fixed-alpha method.
    fixed_candidates = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    fixed_validation_map = {}
    for alpha in fixed_candidates:
        _, indices = reciprocal_rerank_from_sets(
            val_scores, val_indices, reciprocal_sets,
            alpha=alpha, query_neighbor_k=reciprocal_k,
        )
        labels = gallery_labels[indices]
        values = per_query_metrics(val_labels, labels, gallery_labels)
        fixed_validation_map[alpha] = float(
            np.mean(values["map_at_100"])
        )

    best_fixed_alpha = max(
        fixed_candidates, key=fixed_validation_map.get
    )

    # Reproduce validation selection for the adaptive method.
    alpha_min, alpha_max = 0.70, 0.95
    temperatures = [0.05, 0.10, 0.20]
    adaptive_validation_map = {}

    for temperature in temperatures:
        _, indices, _ = adaptive_rerank(
            val_scores, val_indices, reciprocal_sets,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            confidence_temperature=temperature,
            query_neighbor_k=reciprocal_k,
        )
        values = per_query_metrics(
            val_labels, gallery_labels[indices], gallery_labels
        )
        adaptive_validation_map[temperature] = float(
            np.mean(values["map_at_100"])
        )

    best_temperature = max(
        temperatures, key=adaptive_validation_map.get
    )

    print(
        f"Selected parameters: fixed alpha={best_fixed_alpha}, "
        f"adaptive temperature={best_temperature}"
    )

    _, fixed_indices = reciprocal_rerank_from_sets(
        test_scores, test_indices, reciprocal_sets,
        alpha=best_fixed_alpha, query_neighbor_k=reciprocal_k,
    )
    _, adaptive_indices, _ = adaptive_rerank(
        test_scores, test_indices, reciprocal_sets,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        confidence_temperature=best_temperature,
        query_neighbor_k=reciprocal_k,
    )

    retrieved_labels = {
        "cosine_baseline": gallery_labels[test_indices],
        "fixed_reciprocal_reranking": gallery_labels[fixed_indices],
        "adaptive_reciprocal_reranking": gallery_labels[adaptive_indices],
    }

    method_values = {
        name: per_query_metrics(test_labels, labels, gallery_labels)
        for name, labels in retrieved_labels.items()
    }

    output = stratified_paired_bootstrap(
        test_labels,
        method_values,
        n_bootstrap=2000,
        seed=2026,
    )

    output_dir = PROJECT_ROOT / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "paired_bootstrap_test.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0].keys()))
        writer.writeheader()
        writer.writerows(output)

    json_path = output_dir / "paired_bootstrap_test.json"
    json_path.write_text(json.dumps({
        "analysis": "class-stratified paired bootstrap",
        "test_queries": int(len(test)),
        "gallery_images": int(len(gallery)),
        "replicates": 2000,
        "seed": 2026,
        "interval": "95% percentile confidence interval",
        "comparison": "paired per-query difference versus cosine baseline",
        "selected_fixed_alpha": best_fixed_alpha,
        "selected_adaptive_temperature": best_temperature,
        "results": output,
        "interpretation_note": (
            "Intervals quantify query-sampling uncertainty conditional on "
            "this gallery, encoder, split, and validation-selected settings. "
            "They do not capture uncertainty across training seeds or datasets."
        ),
    }, indent=2), encoding="utf-8")

    print("\nPaired bootstrap: method minus cosine baseline")
    for row in output:
        print(
            f"{row['method']:<32} {row['metric']:<18} "
            f"delta={row['delta_vs_baseline']:+.6f} "
            f"95% CI [{row['ci95_lower']:+.6f}, "
            f"{row['ci95_upper']:+.6f}] "
            f"excludes zero={row['interval_excludes_zero']}"
        )

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
