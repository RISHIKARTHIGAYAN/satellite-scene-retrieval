
"""ImageNet-pretrained ResNet-18 feature extraction."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


def build_encoder():
    """Build ResNet-18 with its classification head removed."""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    return model


@torch.inference_mode()
def extract_embeddings(model, dataloader, device=None):
    """Extract L2-normalized 512-D embeddings and integer labels."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    embedding_batches = []
    label_batches = []

    for images, labels in dataloader:
        images = images.to(device)
        features = model(images)
        features = F.normalize(features, p=2, dim=1)

        embedding_batches.append(features.cpu().numpy())
        label_batches.append(labels.numpy())

    if not embedding_batches:
        raise ValueError("The dataloader produced no batches.")

    embeddings = np.concatenate(embedding_batches, axis=0).astype(np.float32)
    labels = np.concatenate(label_batches, axis=0)

    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise ValueError(
            f"Expected embeddings of shape (N, 512), got {embeddings.shape}."
        )

    return embeddings, labels
