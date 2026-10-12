
"""Inspect representative EuroSAT retrieval failures."""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from satellite_retrieval.config import load_config
from satellite_retrieval.data import EuroSATSubset
from satellite_retrieval.encoder import build_encoder, extract_embeddings
from satellite_retrieval.retrieval import build_index, search_index


def main():
    config = load_config(PROJECT_ROOT / "configs" / "experiment.json")
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16 if device.type == "cuda" else 8

    gallery = EuroSATSubset(config, split="gallery")
    query = EuroSATSubset(config, split="query")

    gallery_loader = DataLoader(
        gallery, batch_size=batch_size, shuffle=False, num_workers=0
    )
    query_loader = DataLoader(
        query, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = build_encoder()
    gallery_embeddings, gallery_labels = extract_embeddings(
        model, gallery_loader, device
    )
    query_embeddings, query_labels = extract_embeddings(
        model, query_loader, device
    )

    index = build_index(gallery_embeddings)
    scores, indices = search_index(
        index, query_embeddings, top_k=min(5, len(gallery))
    )
    retrieved_labels = gallery_labels[indices]

    class_names = list(gallery.label_names)
    output_dir = PROJECT_ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select representative errors from the confusion pairs.
    target_pairs = [
        ("River", "Highway"),
        ("Highway", "River"),
        ("HerbaceousVegetation", "PermanentCrop"),
        ("Industrial", "Residential"),
    ]

    examples = []
    for true_name, predicted_name in target_pairs:
        true_id = class_names.index(true_name)
        predicted_id = class_names.index(predicted_name)

        candidates = np.flatnonzero(
            (query_labels == true_id)
            & (retrieved_labels[:, 0] == predicted_id)
        )

        if len(candidates) == 0:
            print(f"No examples found: {true_name} -> {predicted_name}")
            continue

        # Choose the first matching query deterministically.
        query_idx = int(candidates[0])
        examples.append((true_name, predicted_name, query_idx))

    if not examples:
        print("No target failure examples found.")
        return

    csv_path = PROJECT_ROOT / "results" / "metrics" / "retrieval_failures.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    fig, axes = plt.subplots(
        len(examples), 6, figsize=(15, 3.2 * len(examples)),
        squeeze=False
    )

    for row_idx, (true_name, predicted_name, query_idx) in enumerate(examples):
        query_image, _ = query[query_idx]
        query_image = query_image.detach().cpu()

        # Undo ImageNet normalization for display.
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        query_image = (query_image * std + mean).clamp(0, 1)
        query_image = query_image.permute(1, 2, 0).numpy()

        axes[row_idx, 0].imshow(query_image)
        axes[row_idx, 0].set_title(
            f"QUERY\nTrue: {true_name}\nTop-1: {predicted_name}"
        )
        axes[row_idx, 0].axis("off")

        correct_in_top5 = bool(np.any(retrieved_labels[query_idx] == query_labels[query_idx]))

        rows.append({
            "query_index": query_idx,
            "true_class": true_name,
            "top1_predicted_class": predicted_name,
            "correct_class_in_top5": correct_in_top5,
        })

        for rank in range(indices.shape[1]):
            gallery_idx = int(indices[query_idx, rank])
            image, _ = gallery[gallery_idx]
            image = image.detach().cpu()
            image = (image * std + mean).clamp(0, 1)
            image = image.permute(1, 2, 0).numpy()

            result_name = class_names[int(gallery_labels[gallery_idx])]
            is_correct = result_name == true_name

            axes[row_idx, rank + 1].imshow(image)
            axes[row_idx, rank + 1].set_title(
                f"Rank {rank + 1}: {result_name}\n"
                f"Score: {scores[query_idx, rank]:.3f}"
                f"{' ✓' if is_correct else ''}"
            )
            axes[row_idx, rank + 1].axis("off")

            rows[-1][f"rank_{rank + 1}_class"] = result_name
            rows[-1][f"rank_{rank + 1}_score"] = float(
                scores[query_idx, rank]
            )
            rows[-1][f"rank_{rank + 1}_gallery_index"] = gallery_idx

    fig.tight_layout()
    figure_path = output_dir / "failure_examples.png"
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved failure visualizations: {figure_path}")
    print(f"Saved failure metadata: {csv_path}")
    print("\nRepresentative failures:")
    for row in rows:
        print(
            f"{row['true_class']} -> {row['top1_predicted_class']} | "
            f"correct class in Top-5: {row['correct_class_in_top5']}"
        )


if __name__ == "__main__":
    main()
