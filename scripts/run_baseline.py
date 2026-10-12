
"""Run the baseline EuroSAT image-retrieval experiment."""

import json
import sys
from pathlib import Path
import csv
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


def main():
    config = load_config(PROJECT_ROOT / "configs" / "experiment.json")
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16 if device.type == "cuda" else 8

    print(f"Device: {device}")
    print("Loading gallery and query datasets...")

    gallery = EuroSATSubset(config, split="gallery")
    query = EuroSATSubset(config, split="query")

    gallery_loader = DataLoader(
        gallery, batch_size=batch_size, shuffle=False, num_workers=0
    )
    query_loader = DataLoader(
        query, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"Gallery images: {len(gallery)}")
    print(f"Query images: {len(query)}")

    model = build_encoder()

    print("Extracting gallery embeddings...")
    gallery_embeddings, gallery_labels = extract_embeddings(
        model, gallery_loader, device
    )

    print("Extracting query embeddings...")
    query_embeddings, query_labels = extract_embeddings(
        model, query_loader, device
    )

    index = build_index(gallery_embeddings)
    top_k = min(config["retrieval"]["top_k"], len(gallery))

    print(f"Searching top {top_k} results for each query...")
    scores, indices = search_index(index, query_embeddings, top_k=top_k)
    retrieved_labels = gallery_labels[indices]

        # Per-class retrieval analysis
    class_rows = []
    confusion = {}

    for label_id, class_name in enumerate(gallery.label_names):
        mask = query_labels == label_id
        class_queries = int(np.sum(mask))

        if class_queries == 0:
            continue

        class_top1 = float(
            np.mean(retrieved_labels[mask, 0] == label_id)
        )
        class_hit5 = float(
            np.mean(
                np.any(retrieved_labels[mask, :5] == label_id, axis=1)
            )
        )
        class_hit10 = float(
            np.mean(
                np.any(retrieved_labels[mask, :10] == label_id, axis=1)
            )
        )

        class_rows.append({
            "class": class_name,
            "num_queries": class_queries,
            "top1_accuracy": class_top1,
            "hit_rate_at_5": class_hit5,
            "hit_rate_at_10": class_hit10,
        })

        confusion[class_name] = {}
        for predicted_id, predicted_name in enumerate(gallery.label_names):
            confusion[class_name][predicted_name] = int(
                np.sum(
                    (query_labels[mask] == label_id)
                    & (retrieved_labels[mask, 0] == predicted_id)
                )
            )

    metrics = {
        "top1_accuracy": top1_accuracy(query_labels, retrieved_labels),
        "hit_rate_at_5": hit_rate_at_k(query_labels, retrieved_labels, 5),
        "hit_rate_at_10": hit_rate_at_k(query_labels, retrieved_labels, 10),
        "map_at_100": mean_average_precision_at_k(
            query_labels,
            retrieved_labels,
            100,
            gallery_labels=gallery_labels,
        ),
        "num_gallery": int(len(gallery)),
        "num_queries": int(len(query)),
        "embedding_dimension": int(gallery_embeddings.shape[1]),
        "device": str(device),
        "seed": int(seed),
    }

    output_dir = PROJECT_ROOT / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    class_path = output_dir / "class_metrics.csv"
    with class_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=class_rows[0].keys())
        writer.writeheader()
        writer.writerows(class_rows)

    confusion_path = output_dir / "top1_confusion_matrix.csv"
    with confusion_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        class_names = list(gallery.label_names)
        writer.writerow(["true_class"] + class_names)
        for class_name in class_names:
            writer.writerow(
                [class_name]
                + [confusion[class_name][name] for name in class_names]
            )

    print(f"Saved class metrics to: {class_path}")
    print(f"Saved top-1 confusion matrix to: {confusion_path}")
    output_path = output_dir / "baseline_metrics.json"
    output_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nBaseline metrics")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.6f}")
        else:
            print(f"{name}: {value}")

    print(f"\nSaved metrics to: {output_path}")


if __name__ == "__main__":
    main()
