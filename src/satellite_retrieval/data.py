"""Memory-efficient EuroSAT loading with official dataset splits."""

from collections import Counter

from datasets import concatenate_datasets, load_dataset
from torch.utils.data import Dataset
from torchvision import transforms


class EuroSATSubset(Dataset):
    """Use official EuroSAT splits without eagerly loading all images."""

    SPLIT_ALIASES = {
        "gallery": "train",
        "query": "test",
        "train": "train",
        "validation": "validation",
        "test": "test",
    }

    def __init__(self, config: dict, split: str = "gallery"):
        allowed = {"all", "gallery", "query", "train", "validation", "test"}
        if split not in allowed:
            raise ValueError(
                f"split must be one of {sorted(allowed)}, got {split!r}."
            )

        image_size = int(config["dataset"]["image_size"])
        if image_size <= 0:
            raise ValueError("image_size must be positive.")

        datasets = load_dataset("timm/eurosat-rgb")

        if split == "all":
            self.dataset = concatenate_datasets([
                datasets["train"],
                datasets["validation"],
                datasets["test"],
            ])
            self.official_split = "all"
        else:
            self.official_split = self.SPLIT_ALIASES[split]
            self.dataset = datasets[self.official_split]

        self.split = split
        self.label_names = self.dataset.features["label"].names
        self.labels = [int(label) for label in self.dataset["label"]]
        self.class_counts = Counter(
            self.label_names[label] for label in self.labels
        )

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        if len(self.dataset) == 0:
            raise ValueError(f"The {split!r} dataset split is empty.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        image = row["image"].convert("RGB")
        label = int(row["label"])
        return self.transform(image), label
